#!/usr/bin/env python3
"""Replay Kimi requests with malformed historical tool-call arguments."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


VALIDATIONS_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = VALIDATIONS_DIR / "kimi_malformed_tool_history" / "cases.jsonl"
DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "Kimi-K2.6-mix-b300"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if not isinstance(case, dict) or not isinstance(case.get("request"), dict):
                raise ValueError(f"{path}:{line_no}: expected object with request object")
            if not case.get("case_id"):
                raise ValueError(f"{path}:{line_no}: missing case_id")
            cases.append(case)
    return cases


def request_headers(api_key: str | None, provider: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if provider:
        headers["X-Fusion-Provider"] = provider
    return headers


def prepare_request(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    request = copy.deepcopy(case["request"])
    if args.model:
        request["model"] = args.model
    if args.max_tokens is not None:
        request["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        request["temperature"] = args.temperature
    if args.stream is not None:
        request["stream"] = args.stream
    return request


def post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float,
) -> tuple[int, dict[str, str], bytes]:
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def parse_sse_payload(text: str) -> dict[str, Any]:
    event_count = 0
    errors: list[Any] = []
    finish_reason = None
    usage = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        event_count += 1
        try:
            parsed = json.loads(data)
        except Exception as exc:
            errors.append({
                "type": "sse_parse_error",
                "message": f"{type(exc).__name__}: {exc}",
                "data_head": data[:200],
            })
            continue
        if parsed.get("error"):
            errors.append(parsed["error"])
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices:
            finish = (choices[0] or {}).get("finish_reason")
            if finish is not None:
                finish_reason = finish
        if parsed.get("usage") is not None:
            usage = parsed["usage"]

    return {
        "sse_event_count": event_count,
        "sse_error_count": len(errors),
        "sse_errors": errors,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def summarize_response(
    status_code: int,
    headers: dict[str, str],
    payload: bytes,
    request: dict[str, Any],
) -> dict[str, Any]:
    content_type = headers.get("Content-Type", headers.get("content-type", ""))
    text = payload.decode("utf-8", errors="replace")
    result: dict[str, Any] = {
        "http_status": status_code,
        "content_type": content_type,
        "response_bytes": len(payload),
        "response_head": text[:1200],
        "json_error_present": False,
        "sse_error_count": 0,
        "sse_errors": [],
    }

    if "text/event-stream" in content_type or request.get("stream"):
        result.update(parse_sse_payload(text))
        return result

    try:
        parsed = json.loads(text)
    except Exception as exc:
        result["json_parse_error"] = f"{type(exc).__name__}: {exc}"
        return result

    error = parsed.get("error") if isinstance(parsed, dict) else None
    if error:
        result["json_error_present"] = True
        result["json_error"] = error
    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    if isinstance(choices, list) and choices:
        result["finish_reason"] = (choices[0] or {}).get("finish_reason")
    if isinstance(parsed, dict) and parsed.get("usage") is not None:
        result["usage"] = parsed["usage"]
    return result


def response_ok(result: dict[str, Any]) -> bool:
    return (
        isinstance(result.get("http_status"), int)
        and result["http_status"] < 400
        and not result.get("json_error_present")
        and not result.get("sse_error_count")
    )


def run_case(
    case: dict[str, Any],
    args: argparse.Namespace,
    headers: dict[str, str],
) -> dict[str, Any]:
    started = time.time()
    request = prepare_request(case, args)
    result = {
        "case_id": case["case_id"],
        "description": case.get("description"),
        "bad_argument_kind": case.get("bad_argument_kind"),
    }
    try:
        status_code, response_headers, payload = post_json(
            args.url,
            headers,
            request,
            args.timeout,
        )
        result.update(summarize_response(status_code, response_headers, payload, request))
    except Exception as exc:
        result["http_status"] = "CLIENT_ERROR"
        result["client_error"] = f"{type(exc).__name__}: {exc}"
    result["ok"] = response_ok(result)
    result["duration_seconds"] = round(time.time() - started, 3)
    return result


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--url", default=os.getenv("KIMI_BASE_URL", DEFAULT_URL))
    parser.add_argument("--model", default=os.getenv("KIMI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.getenv("KIMI_API_KEY"))
    parser.add_argument("--provider", default=os.getenv("KIMI_PROVIDER"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true")
    stream_group.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=None)
    parser.add_argument("--case-id", action="append", help="Run only this case id.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases.expanduser())
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in selected]
    if args.limit is not None:
        cases = cases[: args.limit]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "cases": len(cases),
                    "case_ids": [case["case_id"] for case in cases],
                    "kinds": dict(Counter(case.get("bad_argument_kind") for case in cases)),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    headers = request_headers(args.api_key, args.provider)
    results = [run_case(case, args, headers) for case in cases]
    summary = {
        "cases_path": str(args.cases),
        "url": args.url,
        "model": args.model,
        "case_count": len(results),
        "ok_count": sum(1 for result in results if result["ok"]),
        "failure_count": sum(1 for result in results if not result["ok"]),
        "status_counts": dict(Counter(str(result.get("http_status")) for result in results)),
        "bad_argument_kind_counts": dict(
            Counter(result.get("bad_argument_kind") for result in results)
        ),
        "failures": [result for result in results if not result["ok"]],
    }

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output_dir / "results.jsonl", results)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
