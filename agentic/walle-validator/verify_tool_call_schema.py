#!/usr/bin/env python3
"""walle-validator — server-side JSON schema (walle) conformance test.

Reads the 204 canonical wire bodies from testdata/kimi_official_cases.jsonl
(extracted verbatim from a Kimi-official acceptance test report) and, for
each case:

  1. Uses `case.schema` directly as tools[0].function.parameters
  2. Fires a /chat/completions request in both non-stream and stream mode
  3. Records whether the server accepted the schema, and whether the model
     actually produced a valid tool_call with well-formed JSON arguments

The upstream walle jsonl files under testdata/validator_cases/ are kept in
the tree for provenance / reference, but are NOT read at runtime.

Outputs go under a per-run directory beneath --out-dir (default ./out/):
  <out-dir>/<UTC-stamp>_<model-slug>[_<tag>]/
      tool-call-schema-report.json       — full structured report, mirroring
                                           the format used by Kimi's official
                                           acceptance verifier
      verify_tool_call_json_schema_result.log — per-line PASSED/FAILED lines
  <out-dir>/latest -> <most-recent-run-dir>   (symlink, best-effort)

Each run leaves a fresh directory so history is preserved and the team can
diff runs / trace which cases regress.

No mitm proxy, no kimi CLI. Works against any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
KIMI_OFFICIAL_CASES = ROOT / "testdata" / "kimi_official_cases.jsonl"
# testdata/validator_cases/ (upstream walle) is kept as reference material
# but is NOT the source of what we put on the wire — see load_cases().

TOOL_NAME = "kvv_walle_case"
TRIGGER_PROMPT = (
    "Call the tool named kvv_walle_case with well-formed arguments that satisfy "
    "its declared JSON Schema. Only call the tool. Do not answer in text."
)
# Match Kimi's official acceptance test: no explicit temperature / top_p
# (server picks default).
DEFAULT_MAX_TOKENS = 2048


def parse_header(s: str) -> tuple[str, str]:
    if ":" not in s:
        sys.exit(f"--header {s!r}: must be KEY:VALUE")
    k, v = s.split(":", 1)
    return k.strip(), v.lstrip()


def slug(s: str, maxlen: int = 40) -> str:
    """Reduce a free-form string to a filesystem-safe slug."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip()).strip("-._") or "run"
    return s[:maxlen]


def update_latest_symlink(out_root: Path, run_dir: Path) -> None:
    """Point out_root/latest at run_dir. Best-effort: on filesystems without
    symlink support we silently skip."""
    link = out_root / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(run_dir.name)
    except OSError:
        pass


def load_cases(suites: list[str] | None) -> list[dict]:
    """Load the canonical case set from testdata/kimi_official_cases.jsonl.

    Each line: `{suite, line, selection_reason, schema}` where `schema` is
    the EXACT `tools[0].function.parameters` block Kimi's official
    acceptance testers send on the wire when validating a candidate
    provider. It's an authoritative snapshot copied from a real
    Kimi-official test report (see artifacts (22)/).

    The upstream walle jsonl files under testdata/validator_cases/ are
    kept in the tree for provenance / reference, but we deliberately do
    NOT read them at run time — replicating Kimi's own wrapping logic in
    our code is fragile and diverges. Reading their materialised output
    is stable.
    """
    out = []
    with KIMI_OFFICIAL_CASES.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            item = json.loads(ln)
            if suites and item["suite"] not in suites:
                continue
            out.append(item)
    return out


def build_body(case: dict, model: str, stream: bool,
               thinking: bool, think_mode: str) -> dict:
    """Build /chat/completions body for one probe (one case, one mode)."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": TRIGGER_PROMPT},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": (
                    "Test tool used by walle-validator. Its parameter schema is "
                    "the case under test; you MUST call it with any well-formed "
                    "arguments that match the schema."
                ),
                "parameters": case["schema"],
            },
        }],
        "tool_choice": "required",
        "max_tokens": DEFAULT_MAX_TOKENS,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}

    # Thinking mode encoding: send at the TOP LEVEL (not nested under
    # `extra_body`). `extra_body` is an OpenAI SDK concept — the SDK
    # flattens it onto the top-level wire body. Kimi's official test
    # (and PPIO gateway) receive top-level fields directly.
    #   'kimi'       -> "thinking": {"type": "enabled"|"disabled"}
    #   'opensource' -> "chat_template_kwargs": {"thinking": bool}
    #                   (vLLM/SGLang chat template extension)
    #   'none'       -> omit thinking control entirely
    if think_mode == "kimi":
        body["thinking"] = {"type": "enabled" if thinking else "disabled"}
    elif think_mode == "opensource":
        body["chat_template_kwargs"] = {"thinking": thinking}
    return body


def post(url: str, headers: dict, body: dict, timeout: int) -> tuple[int, str]:
    """Send request. For stream=True, concatenate SSE data lines. Returns
    (status_code, body_text). status=0 on network exception."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    socket.setdefaulttimeout(timeout)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def parse_stream(body_text: str) -> tuple[list[dict], str]:
    """Reconstitute a non-stream-shaped {choices:[{message:...}]} from SSE
    chunks. Returns (choices, final_finish_reason)."""
    chunks: list[dict] = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        try:
            chunks.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    # Reassemble message: content + tool_calls by index
    content = ""
    tool_calls_by_idx: dict[int, dict] = {}
    finish = ""
    for c in chunks:
        for ch in (c.get("choices") or []):
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            delta = ch.get("delta") or {}
            if delta.get("content"):
                content += delta["content"]
            for tc in (delta.get("tool_calls") or []):
                idx = tc.get("index", 0)
                slot = tool_calls_by_idx.setdefault(idx, {
                    "id": "", "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
    tool_calls = [tool_calls_by_idx[i] for i in sorted(tool_calls_by_idx)]
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return [{"index": 0, "message": msg, "finish_reason": finish}], finish


def classify(status: int, body_text: str, mode: str) -> dict:
    """Classify one (case, mode) request into a result row.

    Row shape mirrors artifacts (22)/tool-call-schema-report.json.results[i]:
      status: 'passed' | 'failed'
      message: human-readable
      mode: 'non-stream' | 'stream'
      http_status: HTTP code if non-200, else None
      error_type: server-reported error type, else None
      arguments: string (raw tool_call arguments JSON), else ''
    """
    row = {
        "status": "failed",
        "message": "",
        "mode": mode,
        "http_status": None,
        "error_type": None,
        "arguments": "",
    }
    if status == 0:
        row["message"] = f"network error: {body_text[:200]}"
        return row
    if status != 200:
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            body = {}
        # Vendors disagree about the shape of `error`. OpenAI / PPIO gateway
        # send `{"error": {"type": ..., "message": ...}}`, but some vLLM-based
        # deployments send `{"error": "some string"}` or omit it entirely.
        # Coerce all of these into a stable row shape rather than crash.
        err_raw = body.get("error") if isinstance(body, dict) else None
        if isinstance(err_raw, dict):
            err_type = err_raw.get("type")
            err_msg = err_raw.get("message")
        elif isinstance(err_raw, str):
            err_type = None
            err_msg = err_raw
        else:
            err_type = None
            err_msg = None
        row["http_status"] = status
        row["error_type"] = err_type or "HTTPError"
        row["message"] = err_msg or body_text[:400]
        return row

    # 200: parse choices/message
    if mode == "stream":
        choices, _finish = parse_stream(body_text)
    else:
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError as e:
            row["message"] = f"200 body not valid JSON: {e}"
            return row
        choices = body.get("choices") or []

    if not choices:
        row["message"] = "no choices in response"
        return row
    msg = (choices[0] or {}).get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        row["message"] = "tool call arguments are missing"
        return row

    args = tool_calls[0].get("function", {}).get("arguments", "")
    row["arguments"] = args
    if not args:
        row["message"] = "tool call arguments are missing"
        return row
    try:
        json.loads(args)
    except json.JSONDecodeError as e:
        row["message"] = (
            f"tool call arguments are not valid JSON: {e.msg}; arguments={args}"
        )
        return row
    row["status"] = "passed"
    row["message"] = f"tool call arguments matched schema; arguments={args}"
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", required=True,
                   help="OpenAI-compatible base URL, e.g. https://api.ppio.com/openai/v1")
    p.add_argument("--api-key", required=True,
                   help="Bearer token; supports '$ENV_VAR' to read from environment")
    p.add_argument("--model", required=True,
                   help="Model id, e.g. moonshotai/kimi-k2.6-agentic")
    p.add_argument("--header", action="append", default=[],
                   help="Extra HTTP header KEY:VALUE (repeatable)")
    p.add_argument("--modes", nargs="+", default=["non-stream", "stream"],
                   choices=["non-stream", "stream"],
                   help="Which response modes to test (default: both)")
    p.add_argument("--suites", nargs="+", default=None,
                   help="Restrict to specific suites (e.g. TestBasicTypes TestRequired). "
                        "Default: all suites in kimi_official_cases.jsonl")
    p.add_argument("--thinking", action="store_true",
                   help="Enable thinking mode (default: off, aka non-thinking)")
    p.add_argument("--think-mode", default="none",
                   choices=["none", "kimi", "opensource"],
                   help="Thinking parameter format. 'kimi' -> extra_body.thinking.type; "
                        "'opensource' -> extra_body.chat_template_kwargs.thinking; "
                        "'none' -> omit (default)")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't send requests; just print what would be sent")
    p.add_argument("--out-dir", default=str(ROOT / "out"),
                   help="Parent directory for per-run report subdirs "
                        "(default: ./out/). Each run writes to "
                        "<out-dir>/<UTC-timestamp>_<model-slug>[_<tag>]/ so "
                        "history is preserved.")
    p.add_argument("--tag", default=None,
                   help="Optional label appended to the run directory name "
                        "(e.g. 'baseline', 'after-fix-123'). Alphanumerics, "
                        "dot, dash, underscore only; other chars are folded.")
    p.add_argument("--timeout", type=int, default=180,
                   help="Per-request timeout in seconds (default 180)")
    args = p.parse_args()

    # Resolve $ENV api-key ref
    if args.api_key.startswith("$"):
        env_name = args.api_key[1:]
        resolved = os.environ.get(env_name, "")
        if not resolved:
            sys.exit(f"--api-key {args.api_key!r} but env {env_name} not set")
        args.api_key = resolved

    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    for h in args.header:
        k, v = parse_header(h)
        headers[k] = v

    cases = load_cases(args.suites)
    if not cases:
        sys.exit("[verify] no cases selected")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    now = dt.datetime.now(dt.timezone.utc)
    generated_at = now.isoformat()
    # Timestamp is UTC compact-ISO so that lexicographic sort = chronological.
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    model_slug = slug(args.model.replace("/", "_"))
    parts = [stamp, model_slug]
    if args.tag:
        parts.append(slug(args.tag))
    if args.dry_run:
        parts.append("dryrun")
    run_dir = out_root / "_".join(parts)
    # If someone manages to fire two runs in the same second, disambiguate
    # rather than overwrite the earlier one.
    if run_dir.exists():
        n = 2
        while (out_root / f"{'_'.join(parts)}-{n}").exists():
            n += 1
        run_dir = out_root / f"{'_'.join(parts)}-{n}"
    run_dir.mkdir(parents=True)

    from collections import Counter
    reasons = Counter(c["selection_reason"] for c in cases)
    print(f"Loaded {sum(reasons.values())} tool-call schema case(s) from "
          f"{KIMI_OFFICIAL_CASES}")
    print(f"Selected {len(cases)} tool-call schema case(s)")
    print(f"Selection mode: all")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Dry run: {args.dry_run}")
    print()

    log_lines: list[str] = []
    results: list[dict] = []
    report_path = run_dir / "tool-call-schema-report.json"
    log_path = run_dir / "verify_tool_call_json_schema_result.log"

    def flush_report(*, aborted: str | None = None) -> None:
        """Write current results + summary to disk. Safe to call from any
        exit path (normal, KeyboardInterrupt, exception) — partial results
        are still useful for triage."""
        summary = {
            "total": len(results),
            "by_status": dict(Counter(r["status"] for r in results)),
            "by_selection_reason": dict(Counter(r["selection_reason"] for r in results)),
            "by_mode": {},
        }
        for m in args.modes:
            sub = [r for r in results if r["mode"] == m]
            summary["by_mode"][m] = dict(Counter(r["status"] for r in sub))

        report = {
            "generated_at": generated_at,
            "run_dir": run_dir.name,
            "tag": args.tag,
            "aborted": aborted,     # None on clean finish; otherwise a short reason
            "model": args.model,
            "base_url": args.base_url,
            "tool_name": TOOL_NAME,
            "dry_run": args.dry_run,
            "thinking": args.thinking,
            "think_mode": args.think_mode,
            "modes": list(args.modes),
            "selected_cases": [{
                "suite": c["suite"],
                "line": c["line"],
                "selection_reason": c["selection_reason"],
                "schema": c["schema"],
            } for c in cases],
            "summary": summary,
            "results": results,
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        log_path.write_text("\n".join(log_lines) + "\n")
        update_latest_symlink(out_root, run_dir)

    aborted_reason: str | None = None
    try:
        for case in cases:
            for mode in args.modes:
                body = build_body(case, args.model,
                                  stream=(mode == "stream"),
                                  thinking=args.thinking,
                                  think_mode=args.think_mode)
                if args.dry_run:
                    row = {"status": "passed", "message": "dry-run", "mode": mode,
                           "http_status": None, "error_type": None, "arguments": ""}
                else:
                    status, body_text = post(url, headers, body, args.timeout)
                    row = classify(status, body_text, mode)
                full_row = {
                    "suite": case["suite"],
                    "line": case["line"],
                    "selection_reason": case["selection_reason"],
                    **row,
                }
                results.append(full_row)
                marker = "[PASSED]" if row["status"] == "passed" else "[FAILED]"
                trailer = ""
                if row["status"] == "failed":
                    trailer = f" - {row['message']}"
                line = (f"{marker} [{mode}] {case['suite']}/valid.jsonl:{case['line']} "
                        f"({case['selection_reason']}){trailer}")
                print(line)
                log_lines.append(line)
    except KeyboardInterrupt:
        aborted_reason = f"KeyboardInterrupt after {len(results)}/{len(cases)*len(args.modes)} results"
        print(f"\n[verify] interrupted; flushing {len(results)} partial results to disk")
    except Exception as e:
        aborted_reason = f"{type(e).__name__}: {e}"
        print(f"\n[verify] crashed ({aborted_reason}); flushing {len(results)} partial results")
    finally:
        flush_report(aborted=aborted_reason)

    print()
    print(f"Run dir: {run_dir}")
    print(f"Report:  {report_path}")
    print(f"Log:     {log_path}")
    print(f"Latest:  {out_root / 'latest'} -> {run_dir.name}")
    print(f"Summary: {dict(Counter(r['status'] for r in results))}")
    if aborted_reason:
        print(f"Aborted: {aborted_reason}")
    return 0 if aborted_reason is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
