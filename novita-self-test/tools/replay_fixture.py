#!/usr/bin/env python3
"""Replay the official K3 image or OpenClaw JSONL fixture.

Only the model field is substituted. Authorization is sent from the environment
and never written to an artifact. The runner accepts both JSON and SSE replies
and records enough structure to diagnose finish-reason and tool-call failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .reporting import run_metadata
except ImportError:  # direct `python tools/replay_fixture.py` invocation
    from reporting import run_metadata

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {"image": 25, "openclaw": 12}
SAFE_RESPONSE_HEADERS = {
    "content-type", "date", "server", "x-request-id", "x-trace-id",
    "traceparent", "x-msh-providerdeployid", "x-msh-usage-prompt-tokens",
    "x-msh-usage-cached-tokens",
}


def safe_response_headers(headers: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() in SAFE_RESPONSE_HEADERS}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(kind: str, path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != EXPECTED_COUNTS[kind]:
        raise ValueError(f"{kind} fixture count: expected {EXPECTED_COUNTS[kind]}, got {len(records)}")
    for index, record in enumerate(records, start=1):
        if kind == "image":
            if not isinstance(record.get("transformed_request"), dict):
                raise ValueError(f"image record {index} lacks transformed_request")
            if record.get("expected_finish_reason") not in {"stop", "tool_calls"}:
                raise ValueError(f"image record {index} has invalid expected finish reason")
        elif not isinstance(record.get("request"), dict):
            raise ValueError(f"openclaw record {index} lacks request")
    return records


def request_for(kind: str, record: dict[str, Any], model: str) -> dict[str, Any]:
    source = record["transformed_request"] if kind == "image" else record["request"]
    payload = json.loads(json.dumps(source))
    payload["model"] = model
    return payload


def headers_from_args(raw: list[str], api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for item in raw:
        name, separator, value = item.partition(":")
        if not separator or not name.strip() or not value.strip():
            raise ValueError(f"invalid --header {item!r}; use Name: value")
        headers[name.strip()] = value.strip()
    extra = os.environ.get("KIMI_EXTRA_HEADERS_JSON")
    if extra:
        values = json.loads(extra)
        if not isinstance(values, dict):
            raise ValueError("KIMI_EXTRA_HEADERS_JSON must be a JSON object")
        headers.update({str(key): str(value) for key, value in values.items()})
    return headers


def parse_body(body: bytes) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        return json.loads(body), {"stream": False, "events": 0, "tool_calls": []}
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    last: dict[str, Any] | None = None
    finish_reason = None
    reasoning = False
    content = False
    tool_calls: dict[int, dict[str, Any]] = {}
    events = 0
    done = False
    for raw_line in body.splitlines():
        if not raw_line.startswith(b"data: "):
            continue
        value = raw_line[6:]
        if value == b"[DONE]":
            done = True
            continue
        try:
            event = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        events += 1
        last = event
        for choice in event.get("choices") or []:
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            reasoning = reasoning or bool(delta.get("reasoning_content") or delta.get("reasoning"))
            content = content or bool(delta.get("content"))
            for tool_call in delta.get("tool_calls") or []:
                index = int(tool_call.get("index", 0))
                function = tool_call.get("function") or {}
                item = tool_calls.setdefault(index, {"name": None, "arguments": ""})
                item["name"] = function.get("name") or item["name"]
                item["arguments"] += function.get("arguments") or ""
    return last, {
        "stream": True,
        "events": events,
        "done": done,
        "finish_reason": finish_reason,
        "has_reasoning": reasoning,
        "has_content": content,
        "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
    }


def error_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    error = (payload or {}).get("error")
    if not isinstance(error, dict):
        return None
    return {key: error.get(key) for key in ("type", "param", "code", "message")}


def run_case(kind: str, index: int, record: dict[str, Any], args: argparse.Namespace, headers: dict[str, str], root: Path) -> dict[str, Any]:
    payload = request_for(kind, record, args.model)
    case_id = str(record.get("line", index)) if kind == "openclaw" else f"image-{index:04d}"
    case_dir = root / f"{index:04d}-{case_id}"
    case_dir.mkdir(parents=True)
    (case_dir / "request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = utc_now()
    start = time.monotonic()
    status: int | None = None
    response_headers: dict[str, str] = {}
    body = b""
    transport_error = None
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            response_headers = safe_response_headers(dict(response.headers.items()))
            body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = safe_response_headers(dict(exc.headers.items()))
        body = exc.read()
    except Exception as exc:  # pragma: no cover - exercised by live failures
        transport_error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.monotonic() - start) * 1000, 3)
    (case_dir / "response.headers.json").write_text(json.dumps(response_headers, indent=2) + "\n", encoding="utf-8")
    (case_dir / "response.body").write_bytes(body)

    parsed, observation = parse_body(body)
    choices = (parsed or {}).get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason") or observation.get("finish_reason")
    tool_calls = message.get("tool_calls") or observation.get("tool_calls") or []
    expected = record.get("expected_finish_reason") if kind == "image" else "tool_calls"
    passed = status == 200 and finish_reason == expected and transport_error is None
    result = {
        "case": case_id,
        "index": index,
        "started_at": started,
        "http_status": status,
        "latency_ms": latency_ms,
        "stream": observation.get("stream", False),
        "sse_events": observation.get("events", 0),
        "done_seen": observation.get("done", False),
        "expected_finish_reason": expected,
        "finish_reason": finish_reason,
        "tool_call_count": len(tool_calls),
        "has_reasoning": bool(message.get("reasoning_content") or observation.get("has_reasoning")),
        "has_content": bool(message.get("content") or observation.get("has_content")),
        "provider_deploy_id": response_headers.get("X-Msh-ProviderDeployId"),
        "request_id": response_headers.get("x-request-id") or response_headers.get("X-Request-Id"),
        "passed": passed,
        "error": error_summary(parsed),
        "transport_error": transport_error,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(EXPECTED_COUNTS))
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL"), required=False)
    parser.add_argument("--api-key", default=os.environ.get("KIMI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("KIMI_MODEL"), required=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--header", action="append", default=[])
    args = parser.parse_args()
    if not args.base_url or not args.model:
        parser.error("--base-url and --model are required, or set KIMI_BASE_URL and KIMI_MODEL")
    if not args.api_key:
        parser.error("--api-key is required, or set KIMI_API_KEY")
    records = load_records(args.kind, args.fixture)
    if args.max_cases is not None:
        if args.max_cases < 1:
            parser.error("--max-cases must be positive")
        records = records[: args.max_cases]
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    headers = headers_from_args(args.header, args.api_key)
    metadata = run_metadata(ROOT, command=sys.argv, base_url=args.base_url, model=args.model)
    metadata.update({
        "kind": args.kind,
        "fixture": str(args.fixture),
        "fixture_sha256": sha256(args.fixture),
        "fixture_records": EXPECTED_COUNTS[args.kind],
        "attempts": len(records),
    })
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    results = []
    for index, record in enumerate(records, start=1):
        result = run_case(args.kind, index, record, args, headers, args.output_dir)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    passed = sum(bool(result["passed"]) for result in results)
    verdict = {
        "schema_version": 1,
        "status": "passed" if passed == len(results) else "failed",
        "completed": passed == len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "finish_reason_contract": "image fixture expected per-record; OpenClaw expected tool_calls",
        "ended_at": utc_now(),
    }
    (args.output_dir / "summary.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results), encoding="utf-8")
    (args.output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(
        "\n".join([
            "# K3 Fixture Replay",
            "",
            f"Status: **{verdict['status']}**",
            f"Passed: **{passed}/{len(results)}**",
            "",
            "Inspect `summary.jsonl` and the numbered case directories for failure details.",
            "",
        ])
        + "\n",
        encoding="utf-8",
    )
    return 0 if verdict["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
