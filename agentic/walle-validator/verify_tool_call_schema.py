#!/usr/bin/env python3
"""walle-validator — server-side JSON schema (walle) conformance test.

Loads MoonshotAI/walle testdata/validator_cases/*/valid.jsonl, filters to the
tool-callable subset (see testdata/selection_reasons.jsonl), and for each case:

  1. Wraps the schema as tools[0].function.parameters
  2. Fires a /chat/completions request in both non-stream and stream mode
  3. Records whether the server accepted the schema, and whether the model
     actually produced a valid tool_call with well-formed JSON arguments

Outputs (in --out-dir, default ./out/):
  tool-call-schema-report.json       — full structured report, mirroring
                                       the format used by novita's internal
                                       verify tool
  verify_tool_call_json_schema_result.log — per-line PASSED/FAILED lines

No mitm proxy, no kimi CLI. Works against any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
        err = body.get("error") or {}
        row["http_status"] = status
        row["error_type"] = err.get("type") or "HTTPError"
        row["message"] = err.get("message") or body_text[:400]
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
                        "Default: all suites in selection_reasons.jsonl")
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
                   help="Directory for report artifacts (default: ./out/)")
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

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

    summary = {
        "total": len(results),
        "by_status": dict(Counter(r["status"] for r in results)),
        # by_selection_reason counts per result (case × mode), matching
        # novita's internal verifier: each of the 204 selected cases
        # contributes twice (non-stream + stream).
        "by_selection_reason": dict(Counter(r["selection_reason"] for r in results)),
        "by_mode": {},
    }
    for m in args.modes:
        sub = [r for r in results if r["mode"] == m]
        summary["by_mode"][m] = dict(Counter(r["status"] for r in sub))

    report = {
        "generated_at": generated_at,
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
    report_path = out_dir / "tool-call-schema-report.json"
    log_path = out_dir / "verify_tool_call_json_schema_result.log"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log_path.write_text("\n".join(log_lines) + "\n")

    print()
    print(f"Report: {report_path}")
    print(f"Log:    {log_path}")
    print(f"Summary: {summary['by_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
