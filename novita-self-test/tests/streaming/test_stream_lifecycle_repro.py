#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "tools" / "stream_lifecycle_repro.py"
SPEC = importlib.util.spec_from_file_location("stream_lifecycle_repro", SCRIPT)
assert SPEC and SPEC.loader
REPRO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPRO)


class ChunkedSSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        now_us = time.time_ns() // 1_000
        chunks = [
            {
                "id": "local-test",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "get_weather"},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "sla_metrics": {"ttft_ms": 1, "ts_us": now_us},
            },
            {
                "id": "local-test",
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                "sla_metrics": {"ttft_ms": 1, "ts_us": now_us + 1_000},
            },
        ]
        body = "".join(f"data: {json.dumps(value)}\n\n" for value in chunks)
        body += "data: [DONE]\n\n"
        encoded = body.encode()
        self.wfile.write(f"{len(encoded):X}\r\n".encode() + encoded + b"\r\n")
        self.wfile.flush()
        if self.path == "/complete":
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        else:
            time.sleep(2)


class StreamLifecycleReproTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ChunkedSSEHandler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def run_case(self, path: str, overall_timeout: float) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            return REPRO.run_once(
                endpoint=f"http://127.0.0.1:{self.server.server_port}{path}",
                api_key="local-test-only",
                extra_headers={},
                request_body={"model": "test-model", "stream": True},
                request_source=Path("local-test.json"),
                run_dir=root / "run",
                connect_timeout=1,
                overall_timeout=overall_timeout,
                tail_threshold=0.2,
            )

    def test_complete_stream_reaches_done_and_eof(self) -> None:
        summary = self.run_case("/complete", 1)
        self.assertEqual(summary["outcome"], "complete")
        self.assertTrue(summary["done_seen"])
        self.assertFalse(summary["issue_reproduced"])

    def test_stream_stall_after_done_is_reproduced(self) -> None:
        summary = self.run_case("/hang", 0.4)
        self.assertEqual(summary["outcome"], "timeout_after_done")
        self.assertTrue(summary["done_seen"])
        self.assertTrue(summary["issue_reproduced"])


if __name__ == "__main__":
    unittest.main()
