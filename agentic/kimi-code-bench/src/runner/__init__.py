"""Runner — orchestrates the full bench cycle on a remote sandbox.

High-level flow:
1. Probe endpoint (unless --skip-probe), derive run mode.
2. Build a render context (urls, model, tasks, mirrors, etc).
3. Generate three artifacts: setup_and_smoke.sh, local_kimi_cli_vendor_agent.py,
   run_smoke_vendor.sh — by filling Jinja-style placeholders in templates/.
4. SSH-push artifacts to the remote sandbox.
5. Trigger `bash setup_and_smoke.sh` over SSH, stream the log back.
6. Pull jobs/ + TSV back into runs/<tag>/.
7. Run triage_all on the local copy, emit TRIAGE.md.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # kimi-code-bench/
SRC = ROOT / "src"
RUNS = ROOT / "runs"
TEMPLATES = SRC / "runner" / "templates"


def check_local_prereqs() -> int:
    """Verify the calling machine has the bare minimum (ssh + python3)."""
    errs = []
    for tool in ("ssh", "python3"):
        if shutil.which(tool) is None:
            errs.append(f"missing: {tool}")
    print("=== local prerequisites ===")
    if errs:
        for e in errs:
            print(f"  ✗ {e}")
        return 1
    print("  ✓ ssh, python3 found")
    print()
    print("Note: docker / uv / harbor are needed on the REMOTE SANDBOX,")
    print("      not on this machine. Bench installs them lazily over SSH.")
    return 0


def _render_template(name: str, ctx: dict) -> str:
    """Tiny string.Template-style renderer with $key substitution.

    Uses safe_substitute so any $foo / ${foo} not in ctx is left literal —
    needed for shell scripts that contain $HOME, $PATH etc, and Python
    f-strings inside templates.
    """
    from string import Template
    text = (TEMPLATES / name).read_text()
    return Template(text).safe_substitute(ctx)


def _ssh_cmd(remote: dict) -> list[str]:
    return [
        "ssh", "-p", str(remote["ssh_port"]),
        "-o", "LogLevel=ERROR",
        "-o", "BatchMode=yes",
        remote["ssh_user_host"],
    ]


def _ssh_cmd_detach(remote: dict) -> list[str]:
    """Like _ssh_cmd but with -n (close stdin) — prevents ssh from
    hanging when the remote bash forks a long-running setsid child."""
    return [
        "ssh", "-n", "-p", str(remote["ssh_port"]),
        "-o", "LogLevel=ERROR",
        "-o", "BatchMode=yes",
        remote["ssh_user_host"],
    ]


def _ssh_push(remote: dict, local_text: str, remote_path: str) -> None:
    cmd = _ssh_cmd(remote) + [f"cat > {shlex.quote(remote_path)} && chmod +x {shlex.quote(remote_path)}"]
    p = subprocess.run(cmd, input=local_text.encode(), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ssh push to {remote_path} failed: {p.stderr.decode()}")


def _ssh_exec(remote: dict, cmd: str, log_path: Path | None = None) -> int:
    full = _ssh_cmd(remote) + [cmd]
    if log_path is None:
        return subprocess.run(full).returncode
    with log_path.open("wb") as f:
        return subprocess.run(full, stdout=f, stderr=subprocess.STDOUT).returncode


def _ssh_pull_targz(remote: dict, remote_path: str, local_path: Path) -> None:
    """cat remote tar.gz to local file (avoids scp through jumpserver)."""
    cmd = _ssh_cmd(remote) + [f"cat {shlex.quote(remote_path)}"]
    with local_path.open("wb") as f:
        rc = subprocess.run(cmd, stdout=f).returncode
    if rc != 0:
        raise RuntimeError(f"ssh pull {remote_path} failed (rc={rc})")


def run_bench(args) -> int:
    from config import (
        DEFAULT_REMOTE,
        DEFAULT_TASKS,
        EXTENDED_TASKS,
        DEFAULT_MIRRORS,
        KIMI_CLI_REF,
        HARBOR_VERSION,
        HARBOR_AGENT_SETUP_TIMEOUT_MUL,
        HARBOR_AGENT_TIMEOUT_MUL,
        TASK_TIMEOUT_OVERRIDES,
    )
    from probe import probe_endpoint

    # Resolve config
    tag = args.tag or f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.remote_host == "local":
        print("[runner] local mode not yet supported", file=sys.stderr)
        return 2
    remote = _resolve_remote(args)

    # Resolve task list. Priority: --task-list file > --tasks keyword > comma-list
    task_list_file = getattr(args, "task_list", None)
    if task_list_file:
        try:
            task_list = [
                ln.strip() for ln in Path(task_list_file).read_text().splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
        except Exception as e:
            print(f"[runner] could not read --task-list {task_list_file}: {e}", file=sys.stderr)
            return 2
    elif args.tasks in ("smoke", "all"):  # 'all' kept as legacy alias for smoke
        task_list = list(DEFAULT_TASKS)
    elif args.tasks == "extended":
        task_list = list(EXTENDED_TASKS)
    else:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    known = set(EXTENDED_TASKS)
    unknown = set(task_list) - known
    if unknown:
        print(f"[runner] unknown task names (not in 89-task cache): {sorted(unknown)}", file=sys.stderr)
        return 2
    if not task_list:
        print("[runner] empty task list", file=sys.stderr)
        return 2
    print(f"[runner] running {len(task_list)} task(s)")

    # Resolve extra HTTP headers — kbench-level --header CLI repeats land in
    # args.header. Merged with --infer-id back-compat shortcut.
    extra_headers: dict[str, str] = {}
    for entry in getattr(args, "header", None) or []:
        if ":" in entry:
            k, v = entry.split(":", 1)
            k = k.strip(); v = v.lstrip()
            if k:
                extra_headers[k] = v
    if getattr(args, "infer_id", "") and "X-Infer-ID" not in extra_headers:
        extra_headers["X-Infer-ID"] = args.infer_id
    # Reverse-lift: if X-Infer-ID came in via --header, surface it into
    # args.infer_id so the rendered setup_and_smoke.sh health probe (which
    # only knows about VLLM_INFER_ID) can attach it. Vendor agent still gets
    # the full extra_headers dict for chat completions; this is purely for
    # the gateway health-check curl.
    if not getattr(args, "infer_id", "") and extra_headers.get("X-Infer-ID"):
        args.infer_id = extra_headers["X-Infer-ID"]

    # 1. Probe (optional)
    probe_result = None
    if not args.skip_probe:
        print("[runner] probing endpoint...")
        # When OAuth creds are provided instead of a static api_key, fish the
        # current access_token out for the probe — the probe doesn't refresh,
        # but the long-running smoke does (via kimi-cli inside the container).
        probe_key = args.api_key
        if not probe_key and getattr(args, "oauth_creds", None):
            try:
                probe_key = json.loads(Path(args.oauth_creds).read_text()).get(
                    "access_token", ""
                )
            except Exception:
                pass
        probe_result = probe_endpoint(
            args.base_url, probe_key, args.infer_id,
            args.model, extra_headers,
            strict_mode=getattr(args, "strict_thinking_spec", False),
        )
        (run_dir / "probe.json").write_text(json.dumps({
            "base_url": probe_result.base_url,
            "models_listed": probe_result.models_listed,
            "cases": [c.__dict__ for c in probe_result.cases],
            "capabilities": {
                "supports_top_level_thinking": probe_result.supports_top_level_thinking,
                "enforces_kimi_validations": probe_result.enforces_kimi_validations,
                "interleaved_thinking_strict": probe_result.interleaved_thinking_strict,
            },
        }, indent=2))
        print(probe_result.format())
        if not probe_result.cases[0].passed:
            print("[runner] endpoint not reachable, aborting", file=sys.stderr)
            return 1

    from user_config import resolve_mirrors
    mirrors = resolve_mirrors()

    # 2. Snapshot the resolved config
    snap = {
        "tag": tag,
        "base_url": args.base_url,
        "model": args.model,
        "infer_id": args.infer_id,
        "tasks": task_list,
        "preserve_thinking": bool(args.preserve_thinking),
        "kimi_cli_ref": args.kimi_cli_ref or KIMI_CLI_REF,
        "harbor_version": HARBOR_VERSION,
        "remote_host": remote["ssh_user_host"],
        "remote_port": remote["ssh_port"],
        "remote_workdir": remote["workdir"],
        "container_proxy": remote["container_proxy"],
        "mirrors": mirrors,
        "timestamp": datetime.now().isoformat(),
    }
    (run_dir / "config.snapshot.json").write_text(json.dumps(snap, indent=2))

    # 3. Render artifacts
    ctx = {
        "BASE_URL": args.base_url,
        "API_KEY": args.api_key,
        "MODEL": args.model,
        "INFER_ID": args.infer_id,
        "EXTRA_HEADERS_JSON": json.dumps(extra_headers),
        "HARBOR_MODEL": f"kimi/{args.model}",
        "KIMI_CLI_REF": snap["kimi_cli_ref"],
        "HARBOR_VERSION": HARBOR_VERSION,
        "GH_PREFIX": mirrors["gh_prefix"],
        "PYPI_INDEX": mirrors["pypi_index"],
        "REMOTE_WORKDIR": remote["workdir"],
        "CONTAINER_PROXY": remote["container_proxy"],
        "PRESERVE_THINKING": "1" if snap["preserve_thinking"] else "0",
        "TASK_LIST": "\n".join(task_list),
        "AGENT_SETUP_MUL": str(HARBOR_AGENT_SETUP_TIMEOUT_MUL),
        "AGENT_TIMEOUT_MUL": str(HARBOR_AGENT_TIMEOUT_MUL),
        # Render TASK_TIMEOUT_OVERRIDES as bash case clauses, e.g.:
        #     "polyglot-rust-c") task_timeout_mul=10 ;;
        # Avoids shell-quoting issues that come with embedding a JSON literal.
        "TASK_TIMEOUT_CASES": "\n".join(
            f'    "{name}") task_timeout_mul={mul} ;;'
            for name, mul in TASK_TIMEOUT_OVERRIDES.items()
        ),
        "CONCURRENCY": str(getattr(args, "concurrency", 1) or 1),
    }
    setup_sh = _render_template("setup_and_smoke.sh.tmpl", ctx)
    vendor_py = _render_template("local_kimi_cli_vendor_agent.py.tmpl", ctx)

    # Save rendered artifacts locally for audit
    (run_dir / "setup_and_smoke.sh.rendered").write_text(setup_sh)
    (run_dir / "local_kimi_cli_vendor_agent.py.rendered").write_text(vendor_py)

    # 4. Push to remote
    remote_setup = f"{remote['workdir']}/setup_and_smoke.sh"
    remote_vendor = f"{remote['workdir']}/local_kimi_cli_vendor_agent.py"
    print(f"[runner] pushing artifacts to {remote['ssh_user_host']}...")
    _ssh_exec(remote, f"mkdir -p {shlex.quote(remote['workdir'])}")
    _ssh_push(remote, setup_sh, remote_setup)
    _ssh_push(remote, vendor_py, remote_vendor)
    # Mark the remote workdir with this run's tag so `kbench fetch` knows
    # which run is currently active without the user repeating themselves.
    _ssh_exec(
        remote,
        f"echo {shlex.quote(tag)} > {shlex.quote(remote['workdir'] + '/.kbench-tag')}",
    )

    # 4b. OAuth credentials (optional). Push the file content, base64-encoded,
    # into a sibling file the launcher reads. Kept out of setup_and_smoke.sh
    # itself so .rendered files don't contain the secret.
    oauth_creds_path = getattr(args, "oauth_creds", None)
    if oauth_creds_path:
        import base64 as _b64
        try:
            creds_text = Path(oauth_creds_path).read_text()
        except Exception as e:
            print(f"[runner] --oauth-creds: cannot read {oauth_creds_path}: {e}", file=sys.stderr)
            return 2
        creds_b64 = _b64.b64encode(creds_text.encode()).decode()
        _ssh_push(remote, creds_b64, f"{remote['workdir']}/.kbench-oauth-b64")
        print("[runner] OAuth credentials staged (will refresh inside containers)")
    else:
        _ssh_exec(remote, f"rm -f {shlex.quote(remote['workdir'] + '/.kbench-oauth-b64')}")

    # 5. Run smoke — detached or foreground
    # Both modes need to source the oauth creds (if any) into env so
    # setup_and_smoke -> harbor run -> vendor agent.install() can write them
    # into the container.
    creds_export = (
        'if [ -f .kbench-oauth-b64 ]; then '
        'export KIMI_VENDOR_OAUTH_CREDS_B64="$(cat .kbench-oauth-b64)"; '
        'fi; '
    )
    detach = bool(getattr(args, "detach", False))
    if detach:
        # Goal: ssh exits within seconds, remote bash keeps running.
        #
        # Tricky bit: ssh's exit is gated on "all remote children have
        # closed our stdout/stderr." nohup+setsid+`& disown` orphans the
        # bash so it survives ssh disconnect, but if any descendant still
        # holds the original pipe open, ssh hangs waiting for it.
        # Hence the explicit </dev/null >/dev/null 2>&1 around the bash -c,
        # then `exit 0` to make ssh's wait return.
        launch_cmd = (
            f"cd {shlex.quote(remote['workdir'])} && "
            f"rm -f .kbench-done .kbench-exit && "
            f"touch .kbench-running && "
            f"(nohup setsid bash -c "
            f"  '{creds_export}"
            f"   bash {shlex.quote(remote_setup)} > setup.log 2>&1; "
            f"   echo $? > .kbench-exit; "
            f"   rm -f .kbench-running; "
            f"   touch .kbench-done' "
            f"  </dev/null >/dev/null 2>&1 & "
            f"  echo $! > .kbench-pid"
            f") </dev/null >/dev/null 2>&1; "
            f"cat .kbench-pid; "
            f"exit 0"
        )
        print(f"[runner] launching detached smoke (PRESERVE_THINKING={ctx['PRESERVE_THINKING']})...")
        # Use the -n variant so ssh closes stdin immediately and doesn't
        # block waiting on it.
        cmd = _ssh_cmd_detach(remote) + [launch_cmd]
        subprocess.run(cmd, timeout=60)
        print()
        print(f"=== Detached. Run tag: {tag} ===")
        print(f"  Tail live log:   kbench tail {tag}")
        print(f"  Wait + fetch:    kbench fetch {tag}")
        print(f"  Run dir:         {run_dir}")
        return 0

    # Foreground (legacy): stream log to local
    print(f"[runner] running smoke foreground (PRESERVE_THINKING={ctx['PRESERVE_THINKING']})...")
    log_path = run_dir / "setup.log"
    rc = _ssh_exec(
        remote,
        f"cd {shlex.quote(remote['workdir'])} && {creds_export}"
        f"bash {shlex.quote(remote_setup)} 2>&1 | tee {shlex.quote(remote['workdir'] + '/setup.log')}",
        log_path,
    )
    print(f"[runner] smoke exit={rc}")

    _archive_and_triage(remote, tag, run_dir)
    return 0 if rc == 0 else 1


def _archive_and_triage(remote: dict, tag: str, run_dir: Path) -> None:
    """Pack remote workdir artifacts, pull, extract, run triage."""
    print("[runner] archiving remote artifacts...")
    pack_cmd = (
        f"cd {shlex.quote(remote['workdir'])} && "
        f"REMOTE_ARCHIVE_NAME=kbench-{tag}.tar.gz && "
        f"tar czf /tmp/$REMOTE_ARCHIVE_NAME --ignore-failed-read "
        f"setup.log rewards.tsv jobs/ 2>/dev/null; "
        f"echo /tmp/$REMOTE_ARCHIVE_NAME"
    )
    _ssh_exec(remote, pack_cmd)
    _ssh_pull_targz(remote, f"/tmp/kbench-{tag}.tar.gz", run_dir / "remote.tar.gz")
    subprocess.run(["tar", "xzf", str(run_dir / "remote.tar.gz"), "-C", str(run_dir)])

    print("[runner] running triage...")
    from triage.triage_all import main as triage_main
    triage_main(str(run_dir))

    print(f"\n=== Done. Run archived at: {run_dir} ===")


def _workdir_from_snapshot(tag: str | None) -> str | None:
    """Return `remote_workdir` recorded in runs/<tag>/config.snapshot.json.

    Detach runs write the exact remote workdir they used into their local
    snapshot. tail/fetch (which take a tag but no CLI --remote-workdir by
    default) should honor that instead of re-deriving from ~/.kbench.json,
    which yields a stale workdir when the run used --remote-workdir/<tag>.
    """
    if not tag:
        return None
    snap = RUNS / tag / "config.snapshot.json"
    if not snap.exists():
        return None
    try:
        data = json.loads(snap.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    wd = data.get("remote_workdir")
    return wd if wd else None


def _resolve_remote(args) -> dict:
    """Resolve remote config: CLI > env > ~/.kbench.json > src/config DEFAULT_REMOTE.

    When called from tail/fetch with a --tag but no --remote-workdir, prefer
    the workdir the detach run actually used (recorded in the run's
    config.snapshot.json). This makes `kbench tail <tag>` and
    `kbench fetch <tag>` work out of the box after a `run --detach`.
    """
    from user_config import resolve_remote
    cli_workdir = getattr(args, "remote_workdir", None)
    if not cli_workdir:
        cli_workdir = _workdir_from_snapshot(getattr(args, "tag", None))
    return resolve_remote(
        cli_host=getattr(args, "remote_host", None),
        cli_port=getattr(args, "remote_port", None),
        cli_workdir=cli_workdir,
    )


def tail_run(args) -> int:
    """Tail the remote setup.log for a detached run."""
    remote = _resolve_remote(args)
    print(f"[tail] following {remote['workdir']}/setup.log (Ctrl+C to stop)...")
    cmd = (
        f"cd {shlex.quote(remote['workdir'])} && "
        f"tail -n {int(args.lines)} -F setup.log 2>/dev/null"
    )
    return _ssh_exec(remote, cmd)


def fetch_run(args) -> int:
    """Fetch a detached run's artifacts and triage them.

    Waits (with timeout) for the remote workdir to declare the run done
    (.kbench-done flag), then pulls jobs/ + rewards.tsv + setup.log into
    runs/<tag>/ and runs triage.
    """
    import time

    remote = _resolve_remote(args)
    tag = args.tag
    run_dir = RUNS / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Verify the workdir's current tag matches what the user asked for.
    rc_check = subprocess.run(
        _ssh_cmd(remote) + [
            f"cat {shlex.quote(remote['workdir'] + '/.kbench-tag')} 2>/dev/null"
        ],
        capture_output=True, text=True,
    )
    current_tag = rc_check.stdout.strip()
    if current_tag and current_tag != tag:
        print(f"[fetch] WARN: remote workdir tag is '{current_tag}', not '{tag}'. "
              f"Continuing anyway (artifacts will be those of whichever ran last).",
              file=sys.stderr)

    timeout = int(getattr(args, "timeout", 0))
    deadline = time.monotonic() + timeout if timeout > 0 else None
    interval = 30  # seconds between status checks
    last_log_size = -1
    while True:
        status = subprocess.run(
            _ssh_cmd(remote) + [
                f"cd {shlex.quote(remote['workdir'])} && "
                f"if [ -f .kbench-done ]; then echo DONE; cat .kbench-exit 2>/dev/null; "
                f"elif [ -f .kbench-running ]; then echo RUNNING; "
                f"else echo UNKNOWN; fi; "
                f"wc -c < setup.log 2>/dev/null"
            ],
            capture_output=True, text=True,
        )
        lines = status.stdout.strip().splitlines()
        state = lines[0] if lines else "UNKNOWN"
        if state == "DONE":
            exit_code = lines[1].strip() if len(lines) > 1 else "?"
            print(f"[fetch] remote run finished (exit={exit_code})")
            break
        if state == "RUNNING":
            log_size = lines[-1].strip() if lines else "?"
            if log_size != str(last_log_size):
                print(f"[fetch] still running... setup.log size={log_size} bytes")
                try: last_log_size = int(log_size)
                except: pass
        elif state == "UNKNOWN":
            print(f"[fetch] WARN: no run marker found in {remote['workdir']}", file=sys.stderr)
            return 2
        if deadline and time.monotonic() > deadline:
            print(f"[fetch] timed out after {timeout}s waiting for run to finish", file=sys.stderr)
            return 3
        time.sleep(interval)

    _archive_and_triage(remote, tag, run_dir)
    return 0
