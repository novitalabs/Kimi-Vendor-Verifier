#!/usr/bin/env python3
"""Measure an OpenAI-compatible SSE response from request start through EOF."""

from __future__ import annotations

import argparse
import codecs
import datetime as dt
import hashlib
import http.client
import json
import os
import socket
import ssl
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


SAFE_RESPONSE_HEADERS = {
    "cache-control",
    "connection",
    "content-type",
    "date",
    "server",
    "transfer-encoding",
    "x-request-id",
    "x-trace-id",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_request(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if isinstance(value, dict) and isinstance(value.get("body"), dict):
        value = value["body"]
    if not isinstance(value, dict):
        raise ValueError(f"request must be a JSON object: {path}")
    if value.get("stream") is not True:
        raise ValueError("request must set stream=true")
    return value


class EvidenceWriter:
    def __init__(self, path: Path, start_ns: int) -> None:
        self._file = path.open("w")
        self._start_ns = start_ns

    def emit(self, kind: str, **fields: Any) -> None:
        now_ns = time.monotonic_ns()
        record = {
            "kind": kind,
            "observed_at": utc_now(),
            "elapsed_ms": round((now_ns - self._start_ns) / 1_000_000, 3),
            **fields,
        }
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class SSEParser:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buffer = ""
        self._data_lines: list[str] = []

    def feed(self, data: bytes, final: bool = False) -> list[str]:
        self._buffer += self._decoder.decode(data, final=final)
        events: list[str] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                if self._data_lines:
                    events.append("\n".join(self._data_lines))
                    self._data_lines.clear()
                continue
            if line.startswith("data:"):
                self._data_lines.append(line[5:].lstrip(" "))
        if final:
            if self._buffer.startswith("data:"):
                self._data_lines.append(self._buffer[5:].lstrip(" "))
            self._buffer = ""
            if self._data_lines:
                events.append("\n".join(self._data_lines))
                self._data_lines.clear()
        return events


def summarize_sse_event(raw: str) -> dict[str, Any]:
    if raw == "[DONE]":
        return {"event_type": "done"}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "event_type": "invalid_json",
            "byte_count": len(raw.encode()),
            "error": str(exc),
            "sha256": hashlib.sha256(raw.encode()).hexdigest(),
        }

    choices = value.get("choices") or []
    choice = choices[0] if choices else {}
    delta = choice.get("delta") or {}
    tool_calls = delta.get("tool_calls") or []
    tool_argument_chars = 0
    tool_names: list[str] = []
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        if function.get("name"):
            tool_names.append(function["name"])
        tool_argument_chars += len(function.get("arguments") or "")

    return {
        "event_type": "json",
        "response_id": value.get("id"),
        "finish_reason": choice.get("finish_reason"),
        "delta_keys": sorted(delta),
        "reasoning_chars": len(delta.get("reasoning_content") or ""),
        "content_chars": len(delta.get("content") or ""),
        "tool_names": tool_names,
        "tool_argument_chars": tool_argument_chars,
        "completion_tokens": (value.get("usage") or {}).get("completion_tokens"),
        "prompt_tokens": (value.get("usage") or {}).get("prompt_tokens"),
        "upstream_ts_us": (value.get("sla_metrics") or {}).get("ts_us"),
        "upstream_ttft_ms": (value.get("sla_metrics") or {}).get("ttft_ms"),
        "raw_bytes": len(raw.encode()),
    }


def remaining_seconds(deadline_ns: int) -> float:
    return max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)


def set_socket_deadline(conn: http.client.HTTPConnection, deadline_ns: int) -> None:
    if conn.sock is None:
        return
    remaining = remaining_seconds(deadline_ns)
    if remaining <= 0:
        raise TimeoutError("overall deadline reached")
    conn.sock.settimeout(remaining)


def run_once(
    *,
    endpoint: str,
    api_key: str,
    extra_headers: dict[str, str],
    request_body: dict[str, Any],
    request_source: Path,
    run_dir: Path,
    connect_timeout: float,
    overall_timeout: float,
    tail_threshold: float,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    start_ns = time.monotonic_ns()
    start_epoch_us = time.time_ns() // 1_000
    deadline_ns = start_ns + int(overall_timeout * 1_000_000_000)
    evidence = EvidenceWriter(run_dir / "events.jsonl", start_ns)
    parser = SSEParser()
    payload = json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode()
    raw_path = run_dir / "response.sse"

    summary: dict[str, Any] = {
        "schema_version": 1,
        "started_at": utc_now(),
        "endpoint": endpoint,
        "model": request_body.get("model"),
        "request_source": str(request_source),
        "request_sha256": hashlib.sha256(payload).hexdigest(),
        "request_bytes": len(payload),
        "connect_timeout_seconds": connect_timeout,
        "overall_timeout_seconds": overall_timeout,
        "tail_threshold_seconds": tail_threshold,
        "http_status": None,
        "response_headers": {},
        "transport_read_count": 0,
        "transport_bytes": 0,
        "sse_event_count": 0,
        "json_event_count": 0,
        "invalid_json_event_count": 0,
        "done_seen": False,
        "finish_reasons": [],
        "tool_names": [],
        "tool_argument_chars": 0,
        "first_transport_read_ms": None,
        "last_transport_read_ms": None,
        "max_transport_read_gap_ms": None,
        "max_sse_arrival_gap_ms": None,
        "max_upstream_event_gap_ms": None,
        "last_timestamped_event_arrival_ms": None,
        "last_upstream_ts_us": None,
        "eof_ms": None,
        "phase": "connecting",
        "outcome": "running",
        "issue_reproduced": False,
        "error": None,
    }

    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"unsupported endpoint: {endpoint}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection_class = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    connection_kwargs: dict[str, Any] = {
        "host": parsed.hostname,
        "port": port,
        "timeout": connect_timeout,
    }
    if parsed.scheme == "https":
        connection_kwargs["context"] = ssl.create_default_context()
    conn = connection_class(**connection_kwargs)
    last_read_ns: int | None = None
    last_sse_ns: int | None = None
    last_upstream_ts_us: int | None = None
    response: http.client.HTTPResponse | None = None
    phase = "connecting"

    def elapsed_ms(now_ns: int | None = None) -> float:
        now_ns = now_ns or time.monotonic_ns()
        return round((now_ns - start_ns) / 1_000_000, 3)

    def consume_event(raw_event: str) -> None:
        nonlocal last_sse_ns, last_upstream_ts_us
        now_ns = time.monotonic_ns()
        arrival_gap_ms = None
        if last_sse_ns is not None:
            arrival_gap_ms = (now_ns - last_sse_ns) / 1_000_000
            previous = summary["max_sse_arrival_gap_ms"] or 0
            summary["max_sse_arrival_gap_ms"] = round(max(previous, arrival_gap_ms), 3)
        last_sse_ns = now_ns

        event = summarize_sse_event(raw_event)
        summary["sse_event_count"] += 1
        if event["event_type"] == "done":
            summary["done_seen"] = True
            summary["done_arrival_ms"] = elapsed_ms(now_ns)
        elif event["event_type"] == "invalid_json":
            summary["invalid_json_event_count"] += 1
        else:
            summary["json_event_count"] += 1
            finish_reason = event.get("finish_reason")
            if finish_reason and finish_reason not in summary["finish_reasons"]:
                summary["finish_reasons"].append(finish_reason)
            for tool_name in event.get("tool_names") or []:
                if tool_name not in summary["tool_names"]:
                    summary["tool_names"].append(tool_name)
            summary["tool_argument_chars"] += event.get("tool_argument_chars") or 0
            upstream_ts_us = event.get("upstream_ts_us")
            if upstream_ts_us is not None:
                if last_upstream_ts_us is not None:
                    upstream_gap_ms = (upstream_ts_us - last_upstream_ts_us) / 1_000
                    previous = summary["max_upstream_event_gap_ms"] or 0
                    summary["max_upstream_event_gap_ms"] = round(
                        max(previous, upstream_gap_ms), 3
                    )
                last_upstream_ts_us = upstream_ts_us
                summary["last_upstream_ts_us"] = upstream_ts_us
                summary["last_timestamped_event_arrival_ms"] = elapsed_ms(now_ns)
                event["upstream_clock_lag_ms"] = round(
                    (time.time_ns() // 1_000 - upstream_ts_us) / 1_000, 3
                )
        evidence.emit("sse_event", arrival_gap_ms=arrival_gap_ms, **event)

    evidence.emit(
        "request_start",
        request_bytes=len(payload),
        request_sha256=summary["request_sha256"],
    )

    try:
        conn.connect()
        phase = "connected"
        summary["phase"] = phase
        evidence.emit("connected")
        set_socket_deadline(conn, deadline_ns)
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
            "User-Agent": "ppio-stream-lifecycle-repro/1",
            **extra_headers,
        }
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        conn.request("POST", path, body=payload, headers=headers)
        phase = "awaiting_response_headers"
        summary["phase"] = phase
        evidence.emit("request_sent")
        set_socket_deadline(conn, deadline_ns)
        response = conn.getresponse()
        phase = "streaming"
        summary["phase"] = phase
        summary["http_status"] = response.status
        summary["response_headers"] = {
            name: value
            for name, value in response.getheaders()
            if name.lower() in SAFE_RESPONSE_HEADERS
        }
        evidence.emit(
            "response_headers",
            http_status=response.status,
            headers=summary["response_headers"],
        )

        with raw_path.open("wb") as raw_file:
            while True:
                set_socket_deadline(conn, deadline_ns)
                data = response.read1(64 * 1024)
                now_ns = time.monotonic_ns()
                if not data:
                    for event in parser.feed(b"", final=True):
                        consume_event(event)
                    summary["eof_ms"] = elapsed_ms(now_ns)
                    evidence.emit("response_eof")
                    break
                raw_file.write(data)
                raw_file.flush()
                summary["transport_read_count"] += 1
                summary["transport_bytes"] += len(data)
                current_ms = elapsed_ms(now_ns)
                if summary["first_transport_read_ms"] is None:
                    summary["first_transport_read_ms"] = current_ms
                summary["last_transport_read_ms"] = current_ms
                read_gap_ms = None
                if last_read_ns is not None:
                    read_gap_ms = (now_ns - last_read_ns) / 1_000_000
                    previous = summary["max_transport_read_gap_ms"] or 0
                    summary["max_transport_read_gap_ms"] = round(
                        max(previous, read_gap_ms), 3
                    )
                last_read_ns = now_ns
                evidence.emit(
                    "transport_read",
                    byte_count=len(data),
                    read_gap_ms=read_gap_ms,
                )
                for event in parser.feed(data):
                    consume_event(event)

        if response.status != 200:
            summary["outcome"] = "http_error"
        elif summary["done_seen"]:
            summary["outcome"] = "complete"
        else:
            summary["outcome"] = "eof_without_done"
    except (TimeoutError, socket.timeout) as exc:
        if phase == "connecting":
            summary["outcome"] = "connect_timeout"
        elif phase == "awaiting_response_headers":
            summary["outcome"] = "response_headers_timeout"
        else:
            summary["outcome"] = (
                "timeout_after_done" if summary["done_seen"] else "stream_timeout_before_done"
            )
        summary["error"] = f"{type(exc).__name__}: {exc}"
        evidence.emit(
            "timeout",
            error=summary["error"],
            done_seen=summary["done_seen"],
            phase=phase,
        )
    except Exception as exc:  # Preserve the exact network or protocol failure.
        if phase != "streaming":
            summary["outcome"] = f"{phase}_error"
        else:
            summary["outcome"] = (
                "error_after_done" if summary["done_seen"] else "stream_error_before_done"
            )
        summary["error"] = f"{type(exc).__name__}: {exc}"
        evidence.emit(
            "error",
            error=summary["error"],
            done_seen=summary["done_seen"],
            phase=phase,
        )
    finally:
        conn.close()
        summary["finished_at"] = utc_now()
        summary["duration_ms"] = elapsed_ms()
        if summary.get("last_timestamped_event_arrival_ms") is not None:
            summary["tail_after_last_timestamped_event_ms"] = round(
                summary["duration_ms"] - summary["last_timestamped_event_arrival_ms"], 3
            )
        else:
            summary["tail_after_last_timestamped_event_ms"] = None
        summary["issue_reproduced"] = bool(
            summary["outcome"] in {
                "timeout_after_done",
                "stream_timeout_before_done",
                "error_after_done",
                "stream_error_before_done",
            }
            or (
                summary["tail_after_last_timestamped_event_ms"] is not None
                and summary["tail_after_last_timestamped_event_ms"]
                >= tail_threshold * 1_000
            )
        )
        summary["client_start_epoch_us"] = start_epoch_us
        evidence.emit(
            "run_complete",
            outcome=summary["outcome"],
            issue_reproduced=summary["issue_reproduced"],
        )
        evidence.close()
        write_json(run_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-access", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--overall-timeout", type=float, default=180.0)
    parser.add_argument("--tail-threshold", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    access = json.loads(args.service_access.read_text())
    if access.get("status") != "ready":
        raise SystemExit("service-access status is not ready")
    auth = access.get("authentication") or {}
    if auth.get("kind") != "env" or not auth.get("env_name"):
        raise SystemExit("service-access authentication must name an environment variable")
    api_key = os.environ.get(auth["env_name"])
    if not api_key:
        raise SystemExit(f"missing authentication environment: {auth['env_name']}")

    request_body = load_request(args.request)
    if request_body.get("model") != access.get("model"):
        raise SystemExit("request model does not match service-access model")
    endpoint = access["base_url"].rstrip("/") + "/chat/completions"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: list[dict[str, Any]] = []
    for index in range(1, args.runs + 1):
        run_name = f"run-{index:02d}"
        print(f"starting {run_name}", file=sys.stderr, flush=True)
        summary = run_once(
            endpoint=endpoint,
            api_key=api_key,
            extra_headers=access.get("extra_headers") or {},
            request_body=request_body,
            request_source=args.request.resolve(),
            run_dir=args.output_dir / run_name,
            connect_timeout=args.connect_timeout,
            overall_timeout=args.overall_timeout,
            tail_threshold=args.tail_threshold,
        )
        aggregate.append(
            {
                "run": run_name,
                "outcome": summary["outcome"],
                "issue_reproduced": summary["issue_reproduced"],
                "duration_ms": summary["duration_ms"],
                "http_status": summary["http_status"],
                "x_request_id": summary["response_headers"].get("X-Request-Id")
                or summary["response_headers"].get("x-request-id"),
                "x_trace_id": summary["response_headers"].get("X-Trace-Id")
                or summary["response_headers"].get("x-trace-id"),
                "summary": str((args.output_dir / run_name / "summary.json").resolve()),
            }
        )
        print(json.dumps(aggregate[-1], sort_keys=True), flush=True)
    write_json(args.output_dir / "aggregate.json", aggregate)
    if any(item["issue_reproduced"] for item in aggregate):
        return 1
    if any(item["outcome"] != "complete" for item in aggregate):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
