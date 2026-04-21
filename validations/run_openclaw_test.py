#!/usr/bin/env python3
"""
Send openclaw_cases12.jsonl to ppio endpoint (streaming, with tool-calling).
Replicates run_vendor_img_test.py style with SSE stream handling.
"""
import json
import sys
import time
import argparse
import threading
from pathlib import Path

import requests


def send_request_stream(index: int, req: dict, *, api_url: str, headers: dict, model: str) -> dict:
    """Send a streaming request and collect the full response."""
    req = json.loads(json.dumps(req))
    req["model"] = model
    thinking = req.get("thinking", {}).get("type", "enabled") != "disabled"
    if "thinking" not in req:
        req["thinking"] = {"type": "enabled"}
    req["temperature"] = 1.0 if thinking else 0.6

    start = time.time()
    try:
        resp = requests.post(
            f"{api_url}/v1/chat/completions",
            headers=headers,
            json=req,
            timeout=3600,
            stream=True,
        )
        resp.encoding = "utf-8"
        if resp.status_code != 200:
            elapsed = time.time() - start
            return {
                "index": index,
                "status_code": resp.status_code,
                "elapsed": round(elapsed, 3),
                "error": resp.text[:2000],
            }

        chunks = []
        content_parts = []
        reasoning_parts = []
        tool_calls_map = {}
        usage = None
        finish_reason = None
        model_name = None

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            chunks.append(chunk)
            if not model_name and chunk.get("model"):
                model_name = chunk["model"]

            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                if delta.get("content"):
                    content_parts.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning_parts.append(delta["reasoning_content"])

                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc.get("id", ""),
                            "type": tc.get("type", "function"),
                            "function": {"name": "", "arguments": ""},
                        }
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tool_calls_map[idx]["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tool_calls_map[idx]["function"]["arguments"] += fn["arguments"]

            if chunk.get("usage"):
                usage = chunk["usage"]

        elapsed = time.time() - start
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        tool_calls = [tool_calls_map[k] for k in sorted(tool_calls_map)] if tool_calls_map else None

        result = {
            "index": index,
            "status_code": 200,
            "elapsed": round(elapsed, 3),
            "model": model_name,
            "finish_reason": finish_reason,
            "content_length": len(content),
            "reasoning_length": len(reasoning),
            "chunks": len(chunks),
        }
        if usage:
            result["usage"] = usage
        if content:
            result["content_preview"] = content[:500]
        if tool_calls:
            result["tool_calls"] = [
                {"name": tc["function"]["name"], "args_length": len(tc["function"]["arguments"])}
                for tc in tool_calls
            ]
            result["tool_calls_full"] = tool_calls
        return result

    except Exception as e:
        elapsed = time.time() - start
        return {
            "index": index,
            "status_code": None,
            "elapsed": round(elapsed, 3),
            "error": str(e),
        }


def send_request_non_stream(index: int, req: dict, *, api_url: str, headers: dict, model: str) -> dict:
    """Send a non-streaming request."""
    req = json.loads(json.dumps(req))  # deep copy, don't mutate original
    req["model"] = model
    thinking = req.get("thinking", {}).get("type", "enabled") != "disabled"
    if "thinking" not in req:
        req["thinking"] = {"type": "enabled"}
    req["temperature"] = 1.0 if thinking else 0.6
    req["stream"] = False
    req.pop("stream_options", None)

    start = time.time()
    try:
        resp = requests.post(
            f"{api_url}/v1/chat/completions",
            headers=headers,
            json=req,
            timeout=3600,
        )
        elapsed = time.time() - start
        if resp.status_code != 200:
            return {
                "index": index,
                "status_code": resp.status_code,
                "elapsed": round(elapsed, 3),
                "error": resp.text[:2000],
            }

        body = resp.json()
        choice = body.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

        result = {
            "index": index,
            "status_code": 200,
            "elapsed": round(elapsed, 3),
            "model": body.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "content_length": len(content),
        }
        if body.get("usage"):
            result["usage"] = body["usage"]
        if content:
            result["content_preview"] = content[:500]
        if tool_calls:
            result["tool_calls"] = [
                {"name": tc["function"]["name"], "args_length": len(tc["function"].get("arguments", ""))}
                for tc in tool_calls
            ]
            result["tool_calls_full"] = tool_calls
        return result

    except Exception as e:
        elapsed = time.time() - start
        return {
            "index": index,
            "status_code": None,
            "elapsed": round(elapsed, 3),
            "error": str(e),
        }



def main():
    parser = argparse.ArgumentParser(description="Test OpenClaw cases against endpoint")
    _dir = Path(__file__).resolve().parent
    parser.add_argument("--api-url", required=True, help="API base URL, e.g. https://api.example.com/")
    parser.add_argument("--api-key", required=True, help="API key for Authorization header")
    parser.add_argument("--model", default="moonshotai/kimi-k2.5-h", help="Model name")
    parser.add_argument("--input", default=str(_dir / "openclaw_cases12.jsonl"))
    parser.add_argument("--output", default=str(_dir / "results" / "openclaw_results.jsonl"))
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    model = args.model
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    cases = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    total = len(cases)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {total} cases from {args.input}")
    print(f"Endpoint: {api_url}/v1/chat/completions")
    print(f"Model: {model}")
    print()

    lock = threading.Lock()
    success = 0
    failed = 0
    results = []

    def worker(idx, case):
        nonlocal success, failed
        req = case["request"]
        src_line = case.get("line", "?")
        msgs_count = len(req.get("messages", []))
        payload_kb = len(json.dumps(req)) // 1024

        if req.get("stream") is False:
            fn = send_request_non_stream
        else:
            fn = send_request_stream
        result = fn(idx, req, api_url=api_url, headers=headers, model=model)
        result["src_line"] = src_line
        result["msgs_count"] = msgs_count
        result["payload_kb"] = payload_kb

        with lock:
            results.append(result)
            code = result.get("status_code")
            elapsed = result["elapsed"]
            if code == 200:
                success += 1
                tc_info = ""
                if result.get("tool_calls"):
                    names = [tc["name"] for tc in result["tool_calls"]]
                    tc_info = f"  tools={names}"
                usage_info = ""
                if result.get("usage"):
                    u = result["usage"]
                    usage_info = f"  tokens={u.get('total_tokens', '?')}"
                print(
                    f"[{idx:>2}/{total}] OK  {elapsed:>7.1f}s  "
                    f"line={src_line:<3} msgs={msgs_count:<3} {payload_kb:>4}KB"
                    f"{usage_info}{tc_info}"
                )
            else:
                failed += 1
                err = result.get("error", "")
                print(
                    f"[{idx:>2}/{total}] ERR status={code} {elapsed:>7.1f}s  "
                    f"line={src_line:<3} msgs={msgs_count:<3}  {str(err)[:120]}\n"
                    f"         thinking={req.get('thinking')} stream={req.get('stream')} temperature={req.get('temperature')}",
                    file=sys.stderr,
                )

    threads = []
    for i, (idx, case) in enumerate(enumerate(cases)):
        t = threading.Thread(target=worker, args=(idx, case))
        t.start()
        threads.append(t)
        print(f"[{idx:>2}/{total}] sent  (line={case.get('line', '?')})")
        if i < total - 1:
            time.sleep(args.interval)

    for t in threads:
        t.join()

    results.sort(key=lambda r: r["index"])
    with open(output_path, "w") as out_f:
        for r in results:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    bad = [r for r in results if r.get("status_code") == 200 and r.get("finish_reason") != "tool_calls"]
    print(f"\nDone. success={success} failed={failed} total={total}  results -> {output_path}")
    if bad:
        print(f"\nFAIL: {len(bad)}/{success} request(s) did not return finish_reason=tool_calls:")
        for r in bad:
            print(f"  [{r['index']:>2}] finish_reason={r.get('finish_reason')}")
    else:
        print(f"PASS: all {success} successful request(s) returned finish_reason=tool_calls")


if __name__ == "__main__":
    main()
