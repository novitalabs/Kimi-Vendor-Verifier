#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from threading import Lock


def send_request(idx, api_url, body, headers):
    try:
        req = Request(api_url, data=body, headers=headers)
        resp = urlopen(req, timeout=300)
        return idx, json.loads(resp.read()), None
    except Exception as e:
        return idx, None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Kimi K2.5 EOS Test")
    parser.add_argument("--req", default=os.path.join(os.path.dirname(__file__), "req.json"))
    parser.add_argument("--api-url", required=True, help="API base URL, e.g. http://host:8000")
    parser.add_argument("--api-key", default=None, help="API key for Authorization header")
    parser.add_argument("--model", default="moonshotai/Kimi-K2.5", help="Model name")
    parser.add_argument("--total", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.req),
            f"results_{time.strftime('%Y%m%d_%H%M%S')}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.req) as f:
        payload = json.load(f)
    payload["model"] = args.model
    payload["logprobs"] = True
    payload["top_logprobs"] = 5
    payload.pop("stream_options", None)
    body = json.dumps(payload).encode()

    api_url = args.api_url.rstrip("/") + "/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    stats_lock = Lock()
    stats = {
        "success": 0,
        "error": 0,
        "finish_reasons": {},
        "stop_total": 0,
        "stop_empty": 0,
        "empty_samples": [],
    }

    logprobs_f = open(os.path.join(args.output_dir, "logprobs_samples.jsonl"), "w")
    lp_lock = Lock()

    def on_result(idx, data, err):
        if err:
            with stats_lock:
                stats["error"] += 1
                total_done = stats["success"] + stats["error"]
                print(f"[{total_done:4d}] #{idx:4d} ERROR: {err[:60]}")
            return

        choice = data.get("choices", [{}])[0]
        reason = choice.get("finish_reason", "unknown")
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        lp = choice.get("logprobs", {})
        content_empty = not content.strip()

        with stats_lock:
            stats["success"] += 1
            stats["finish_reasons"][reason] = stats["finish_reasons"].get(reason, 0) + 1
            if reason == "stop":
                stats["stop_total"] += 1
                if content_empty:
                    stats["stop_empty"] += 1
                    stats["empty_samples"].append({
                        "idx": idx,
                        "message": msg,
                    })
            total_done = stats["success"] + stats["error"]
            stop_empty_str = f"empty={stats['stop_empty']}/{stats['stop_total']}" if stats["stop_total"] else "empty=0/0"

            # Print per-request log
            empty_flag = " [EMPTY]" if content_empty else ""
            print(f"[{total_done:4d}] #{idx:4d} reason={reason:<12} content_len={len(content):5d} reasoning_len={len(reasoning):5d}{empty_flag}  | {stop_empty_str}")

        record = {
            "idx": idx,
            "finish_reason": reason,
            "content_len": len(content),
            "reasoning_len": len(reasoning),
            "content_empty": content_empty,
        }
        if lp and lp.get("content"):
            record["last_5_token_logprobs"] = lp["content"][-5:]
        with lp_lock:
            logprobs_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logprobs_f.flush()

    print(f"=== Kimi K2.5 EOS Test ===")
    print(f"API:         {api_url}")
    print(f"Total:       {args.total}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output:      {args.output_dir}")
    print()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(send_request, i, api_url, body, headers): i for i in range(args.total)}
        for fut in as_completed(futures):
            idx, data, err = fut.result()
            on_result(idx, data, err)

    logprobs_f.close()
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"  RESULTS  (elapsed {elapsed:.1f}s)")
    print("=" * 60)
    print()
    print(f"Requests:  {stats['success'] + stats['error']}  "
          f"(success={stats['success']}, error={stats['error']})")
    print()
    print("finish_reason distribution:")
    for r, c in sorted(stats["finish_reasons"].items(), key=lambda x: -x[1]):
        pct = c / stats["success"] * 100 if stats["success"] else 0
        print(f"  {r}: {c} ({pct:.1f}%)")

    print()
    st, se = stats["stop_total"], stats["stop_empty"]
    if st:
        print(f"在 finish_reason = stop 的 {st} 条 response 中，"
              f"有 {se} 条 content 为空，比例为 {se/st*100:.1f}%")
    else:
        print("没有 finish_reason = stop 的 response")

    if stats["empty_samples"]:
        print(f"\n--- Empty content samples (up to 10) ---")
        for s in stats["empty_samples"][:10]:
            m = s["message"]
            print(f"  #{s['idx']}: content={repr((m.get('content') or '')[:80])}")
            if m.get("tool_calls"):
                print(f"         tool_calls: {len(m['tool_calls'])} call(s)")
            reasoning = m.get("reasoning") or m.get("reasoning_content")
            if reasoning:
                print(f"         reasoning: {reasoning[:120]}...")

    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "total": stats["success"] + stats["error"],
        "success": stats["success"],
        "error": stats["error"],
        "finish_reasons": stats["finish_reasons"],
        "stop_total": st,
        "stop_empty": se,
        "stop_empty_pct": round(se / st * 100, 2) if st else 0,
        "empty_samples": stats["empty_samples"][:20],
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

