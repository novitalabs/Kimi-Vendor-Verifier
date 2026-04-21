#!/usr/bin/env python3
"""
Send vendor-img-testcases.jsonl to ppio non-thinking endpoint.
Replicates curl_ppio_non_thinking.sh with dataset requests.
"""
import json
import sys
import time
import argparse
import threading
from pathlib import Path

import requests

def send_request(index: int, req: dict, *, api_url: str, headers: dict, model: str) -> dict:
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
        )
        elapsed = time.time() - start
        return {
            "index": index,
            "status_code": resp.status_code,
            "elapsed": round(elapsed, 3),
            "response": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "index": index,
            "status_code": None,
            "elapsed": round(elapsed, 3),
            "error": str(e),
        }



def main():
    parser = argparse.ArgumentParser()
    _dir = Path(__file__).resolve().parent
    parser.add_argument("--api-url", required=True, help="API base URL, e.g. https://api.example.com/v1")
    parser.add_argument("--api-key", required=True, help="API key for Authorization header")
    parser.add_argument("--model", default="moonshotai/kimi-k2.5-h", help="Model name")
    parser.add_argument("--input", default=str(_dir / "vendor-img-testcaces.jsonl"))
    parser.add_argument("--output", default=str(_dir / "results" / "vendor_img_results.jsonl"))
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    model = args.model
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    requests_list = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                requests_list.append(json.loads(line))

    total = len(requests_list)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    todo = list(enumerate(requests_list))

    print(f"Loaded {total} requests from {args.input}")
    print(f"Sending to {api_url}/v1/chat/completions (interval={args.interval}s)")

    lock = threading.Lock()
    success = 0
    failed = 0
    results = []

    def worker(index, req):
        nonlocal success, failed
        result = send_request(index, req, api_url=api_url, headers=headers, model=model)
        with lock:
            results.append(result)
            code = result.get("status_code")
            elapsed = result["elapsed"]
            if code == 200:
                success += 1
                print(f"[{index:>3}/{total}] OK  {elapsed:.1f}s")
            else:
                failed += 1
                err = result.get("error") or result.get("response", "")
                print(f"[{index:>3}/{total}] ERR status={code} {elapsed:.1f}s  {str(err)[:120]}", file=sys.stderr)

    threads = []
    for i, (index, req) in enumerate(todo):
        t = threading.Thread(target=worker, args=(index, req))
        t.start()
        threads.append(t)
        print(f"[{index:>3}/{total}] sent")
        if i < len(todo) - 1:
            time.sleep(args.interval)

    for t in threads:
        t.join()

    results.sort(key=lambda r: r["index"])
    with open(output_path, "w") as out_f:
        for r in results:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. success={success} failed={failed} total={total}  results -> {output_path}")


if __name__ == "__main__":
    main()
