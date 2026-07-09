"""kbench init / doctor subcommands.

- init: interactive prompt → write ~/.kbench.json + test SSH + test PPIO.
- doctor: audit local + remote + preset config, print pass/fail lines.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

USER_CONFIG_PATH = Path.home() / ".kbench.json"


def _prompt(msg: str, default: str = "") -> str:
    """Read a line from stdin. Returns default if user hits Enter."""
    hint = f" [{default}]" if default else ""
    try:
        val = input(f"{msg}{hint}: ").strip()
    except EOFError:
        val = ""
    return val or default


def _yn(msg: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    ans = _prompt(f"{msg} ({d})", "")
    if not ans:
        return default
    return ans.lower() in ("y", "yes")


def _test_ssh(user_host: str, port: int, timeout: int = 8) -> tuple[bool, str]:
    """Return (ok, message)."""
    cmd = [
        "ssh", "-n", "-p", str(port),
        "-o", "LogLevel=ERROR",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={timeout}",
        user_host,
        "echo kbench_ok",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    if r.returncode == 0 and "kbench_ok" in r.stdout:
        return True, "OK"
    err = (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else "unknown"
    return False, f"rc={r.returncode}: {err[:100]}"


def _test_endpoint(base_url: str, api_key: str, extra_headers: dict,
                   model: str = "", timeout: int = 15) -> tuple[bool, str]:
    """Curl /v1/models. Returns (ok, message)."""
    import urllib.request, urllib.error, socket
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    headers.update(extra_headers or {})
    req = urllib.request.Request(url, headers=headers)
    socket.setdefaulttimeout(timeout)
    try:
        with urllib.request.urlopen(req) as r:
            if r.status == 200:
                body = json.loads(r.read())
                ids = [m.get("id") for m in body.get("data", [])]
                if model and model not in ids:
                    return False, f"200 but model {model!r} not in {ids[:3]}..."
                return True, f"200, {len(ids)} model(s) exposed"
            return False, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _config_field(cli_val: str | None, env_var: str, default: str) -> str:
    """Return CLI arg > env var > default. Empty string counts as unset."""
    if cli_val:
        return cli_val
    env_val = os.environ.get(env_var, "")
    if env_val:
        return env_val
    return default


def cmd_init(args) -> int:
    """Set up ~/.kbench.json.

    Two modes:
    - **interactive** (default when stdin is a tty and --non-interactive is
      not set): prompt the user for each field.
    - **non-interactive** (--non-interactive, or auto-enabled when stdin is
      not a tty, e.g. under CI / nohup / `< /dev/null`): read every value
      from CLI flags and KBENCH_* env vars, then write. No prompts, no
      overwrite confirmation. Missing SSH spec → exit 2 with a clear message
      listing the flag / env var to set.
    """
    print("=== Kimi Code Bench setup ===")
    print(f"Will write {USER_CONFIG_PATH}\n")

    non_interactive = getattr(args, "non_interactive", False) or not sys.stdin.isatty()

    if USER_CONFIG_PATH.exists():
        if non_interactive:
            print(f"[init] {USER_CONFIG_PATH} exists — overwriting "
                  f"(non-interactive mode)")
        elif not _yn(f"{USER_CONFIG_PATH} exists. Overwrite?", default=False):
            print("Aborted.")
            return 1

    if non_interactive:
        ssh = _config_field(getattr(args, "ssh_spec", None),
                            "KBENCH_SSH_SPEC", "")
        if not ssh:
            print("[init] non-interactive mode but no SSH spec was given.")
            print("       Set --ssh-spec <spec> or KBENCH_SSH_SPEC env var.")
            print("       Example: --ssh-spec 'user@host' "
                  "or 'alice@root@10.0.0.5@jump.example.com'")
            return 2
        port = int(_config_field(getattr(args, "ssh_port", None),
                                 "KBENCH_SSH_PORT", "22"))
        workdir_root = _config_field(getattr(args, "workdir_root", None),
                                     "KBENCH_WORKDIR_ROOT", "~/kbench")
        proxy = _config_field(getattr(args, "container_proxy", None),
                              "KBENCH_CONTAINER_PROXY", "")
        ppio = _config_field(getattr(args, "ppio_key", None),
                             "KBENCH_PPIO_KEY", "")
        print(f"[init] non-interactive:")
        print(f"       ssh_user_host  = {ssh}")
        print(f"       ssh_port       = {port}")
        print(f"       workdir_root   = {workdir_root}")
        print(f"       container_proxy= {proxy or '(none)'}")
        print(f"       PPIO key       = {'(set)' if ppio else '(unset)'}")
    else:
        print("--- Remote test server ---")
        print("This is where docker + harbor + trial containers run.")
        ssh = _prompt("SSH spec (e.g. user@host or user@root@host@jump)", "")
        if not ssh:
            print("SSH spec is required. Aborted.")
            return 2
        port = int(_prompt("SSH port", "22"))
        print("Workdir root: where per-run job data is written on the server.")
        print("  ~/kbench      -> /home/<your-remote-user>/kbench (auto-isolated per user)")
        print("  /root/kbench  -> shared root workdir (only OK if you're the only user)")
        print("  $USER / {user} in the path expands to your LOCAL user name.")
        workdir_root = _prompt("Workdir root on server", "~/kbench")

        print("\n--- Container network ---")
        print("If the server is behind a corporate proxy / in a region that needs")
        print("a proxy for docker.io / archive.ubuntu.com, set the container proxy URL.")
        print("Otherwise leave blank.")
        proxy = _prompt("Container proxy URL", "")

        print("\n--- Optional: API keys ---")
        ppio = _prompt("PPIO API key (for --preset --track official, Enter to skip)", "")

    cfg = {
        "remote": {
            "ssh_user_host": ssh,
            "ssh_port": port,
            "workdir_root": workdir_root,
            "container_proxy": proxy,
        },
    }
    if ppio:
        cfg["keys"] = {"PPIO_API_KEY": ppio}

    USER_CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"\n✔ Written to {USER_CONFIG_PATH}")

    print("\n--- Sanity check ---")
    ok, msg = _test_ssh(ssh, port)
    mark = "✔" if ok else "✗"
    print(f"  {mark} SSH  {ssh}:{port}  {msg}")

    if ppio:
        ok, msg = _test_endpoint(
            "https://api.ppio.com/openai/v1", ppio,
            {"X-Fusion-Provider": "moonshot-openai"},
        )
        mark = "✔" if ok else "✗"
        print(f"  {mark} PPIO endpoint  {msg}")

    print("\nNext:")
    print("  ./kbench doctor    # full diagnostic")
    print("  ./kbench run --preset kimi-k2.6 --track official --tag k26-off-$(date +%F) --detach")
    return 0


def cmd_doctor(args) -> int:
    """Audit local + remote + preset config. Prints pass/fail per line."""
    import importlib

    print("=== kbench doctor ===\n")
    all_ok = True

    # 1. Local
    print("Local:")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info >= (3, 10):
        print(f"  ✔ python {py_ver}")
    else:
        print(f"  ✗ python {py_ver} (need 3.10+)")
        all_ok = False

    import shutil
    if shutil.which("ssh"):
        print("  ✔ ssh")
    else:
        print("  ✗ ssh (not on PATH)")
        all_ok = False

    # Loading user config: missing is fine as long as src/config.py
    # DEFAULT_REMOTE has enough to work with (team-fork default 4090 setup).
    # Non-team users clone → DEFAULT_REMOTE is empty → they must run init.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import DEFAULT_REMOTE
    from user_config import resolve_remote
    default_has_ssh = bool(DEFAULT_REMOTE.get("ssh_user_host"))

    if USER_CONFIG_PATH.exists():
        try:
            cfg = json.loads(USER_CONFIG_PATH.read_text())
            print(f"  ✔ {USER_CONFIG_PATH.name} parses OK")
        except json.JSONDecodeError as e:
            print(f"  ✗ {USER_CONFIG_PATH.name} invalid JSON: {e}")
            all_ok = False
            cfg = {}
    else:
        cfg = {}
        if default_has_ssh:
            # DEFAULT_REMOTE has a canonical team server baked in — usable
            # out of the box. Show as info so the user knows they can run
            # init later to customize (own user account, own workdir, etc.).
            print(f"  · {USER_CONFIG_PATH.name} missing — using src/config.py "
                  f"DEFAULT_REMOTE (run ./kbench init to override)")
        else:
            print(f"  ✗ {USER_CONFIG_PATH} missing (run: ./kbench init)")
            all_ok = False

    # 2. Remote — always resolve via resolve_remote() so DEFAULT_REMOTE
    # falls in when ~/.kbench.json is absent or partial.
    print("\nRemote SSH:")
    resolved = resolve_remote()
    ssh = resolved.get("ssh_user_host", "")
    if not ssh:
        print("  ✗ no ssh_user_host in ~/.kbench.json nor DEFAULT_REMOTE")
        print("    → run: ./kbench init")
        all_ok = False
    else:
        port = int(resolved.get("ssh_port", 22))
        ok, msg = _test_ssh(ssh, port)
        mark = "✔" if ok else "✗"
        print(f"  {mark} {ssh}:{port}  {msg}")
        if not ok:
            all_ok = False
        else:
            wd = resolved.get("workdir", "")
            # test parent so we can auto-mkdir the workdir itself
            wd_parent = wd.rsplit("/", 1)[0] if "/" in wd else wd
            for cmd_check, desc in [
                ("docker --version", "docker installed"),
                # bash -c so $HOME and ~ expand on the remote side
                (f"bash -c 'mkdir -p {wd} && test -w {wd_parent} && echo ok || echo readonly'",
                 f"workdir {wd} usable"),
            ]:
                r = subprocess.run(
                    ["ssh", "-n", "-p", str(port), "-o", "LogLevel=ERROR",
                     "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", ssh, cmd_check],
                    capture_output=True, text=True, timeout=15,
                )
                if r.returncode == 0 and "readonly" not in r.stdout:
                    out = r.stdout.strip().splitlines()[0] if r.stdout.strip() else "ok"
                    print(f"  ✔ {desc}  ({out[:60]})")
                else:
                    print(f"  ✗ {desc}  ({(r.stderr or r.stdout).strip()[:80]})")
                    all_ok = False

    # 3. Presets
    print("\nPresets:")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preset import list_presets, load_preset
    from user_config import resolve_api_key

    for name in list_presets():
        try:
            p = load_preset(name)
        except SystemExit:
            print(f"  ✗ {name}: load failed")
            all_ok = False
            continue
        # Check official track api_key ref resolves to non-empty
        official = p.get("tracks", {}).get("official", {})
        key_ref = official.get("api_key", "")
        if key_ref:
            resolved = resolve_api_key(key_ref)
            if resolved:
                # Hide the actual value; just say where it came from
                if key_ref.startswith("$"):
                    src = f"env {key_ref[1:]}"
                elif key_ref.startswith("~") or key_ref.startswith("$HOME"):
                    src = f"file {key_ref}"
                elif key_ref in cfg.get("keys", {}):
                    src = f"keys.{key_ref}"
                else:
                    src = "literal"
                print(f"  ✔ {name} (api_key: {src})")
            else:
                # Missing key only blocks `--track official`. Users running
                # `--track self` (self-deployed vLLM) never touch this key,
                # so surface as a warning rather than an error.
                print(f"  ⚠ {name} api_key ref {key_ref!r} does not resolve "
                      f"(only needed for --track official)")
        else:
            print(f"  · {name} (no official api_key ref)")

    print()
    if all_ok:
        print("✔ All checks passed. Ready to run.")
        print("  ./kbench run --preset kimi-k2.6 --track official --tag k26-off-$(date +%F) --detach")
        return 0
    else:
        print("✗ Some checks failed. Fix the ✗ lines above and re-run kbench doctor.")
        return 1
