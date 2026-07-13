#!/usr/bin/env python3
"""Triage one Harbor trial directory.

输入: 一个 <task>__<id>/ 目录
输出: 一行 JSON 包含分类信号

用法:
    triage_trial.py <trial-dir>
    triage_trial.py jobs/2026-06-26__16-21-35/regex-log__KtTDscA
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def parse_wire_events(jsonl_path):
    """Parse kimi-cli.txt 的 JSONL wire events, 返回事件列表 + assemble 出来的 tool_calls。"""
    if not Path(jsonl_path).exists():
        return [], [], None
    events = []
    raw = Path(jsonl_path).read_text()
    # JSONL: 一行一条 JSON-RPC 消息. 但 ToolResult 的 output 可能包含原始换行,
    # 沿用 KimiCli._parse_wire_events 的逻辑: 以 '{"jsonrpc"' 开头的行作为分隔
    buffer = ""
    for line in raw.splitlines():
        if line.lstrip().startswith('{"jsonrpc"'):
            if buffer:
                try:
                    events.append(json.loads(buffer, strict=False))
                except Exception:
                    pass
            buffer = line
        elif buffer:
            buffer += "\n" + line
    if buffer:
        try:
            events.append(json.loads(buffer, strict=False))
        except Exception:
            pass

    # 提取 method=event 的 params
    event_payloads = [e["params"] for e in events if e.get("method") == "event"]
    # 最后一条 result/error
    final = next(
        (e for e in reversed(events) if "result" in e or "error" in e), None
    )
    return events, event_payloads, final


def assemble_tool_calls(event_payloads):
    """从 ToolCall + ToolCallPart 拼出每个调用的完整 arguments + 配对的 ToolResult。
    返回 [{call_id, name, args_text, args_parsed_ok, result, result_is_error}]
    """
    calls = {}  # call_id -> dict
    order = []  # call_id in occurrence order
    for ev in event_payloads:
        t = ev.get("type")
        p = ev.get("payload", {})
        if t == "ToolCall":
            cid = p.get("id", "")
            fn = p.get("function", {})
            calls[cid] = {
                "id": cid,
                "name": fn.get("name", ""),
                "args_text": fn.get("arguments", "") or "",
                "result": None,
                "result_is_error": None,
            }
            order.append(cid)
        elif t == "ToolCallPart":
            # 找上一个 pending ToolCall 追加
            if order:
                last = calls[order[-1]]
                last["args_text"] += p.get("arguments_part") or ""
        elif t == "ToolResult":
            cid = p.get("tool_call_id", "")
            ret = p.get("return_value", {})
            if cid in calls:
                calls[cid]["result"] = ret
                calls[cid]["result_is_error"] = bool(ret.get("is_error"))

    # JSON 解析检查
    for c in calls.values():
        try:
            json.loads(c["args_text"]) if c["args_text"] else {}
            c["args_parsed_ok"] = True
        except Exception:
            c["args_parsed_ok"] = False

    return [calls[cid] for cid in order]


def collect_status_snapshots(event_payloads):
    snaps = []
    for ev in event_payloads:
        if ev.get("type") == "StatusUpdate":
            p = ev.get("payload", {})
            snaps.append(
                {
                    "ctx_tokens": p.get("context_tokens"),
                    "ctx_usage": p.get("context_usage"),
                    "tu": p.get("token_usage", {}),
                }
            )
    return snaps


def get_reasoning_chars(event_payloads):
    total = 0
    for ev in event_payloads:
        if ev.get("type") == "ContentPart":
            p = ev.get("payload", {})
            if p.get("type") == "think":
                total += len(p.get("think") or "")
    return total


def verifier_first_error(test_stdout_path):
    """从 pytest stdout 抽出第一个 FAILED 的 message。"""
    if not Path(test_stdout_path).exists():
        return None
    text = Path(test_stdout_path).read_text()
    # pytest 失败行: 'E       AssertionError: ...'
    # 截取第一段 E 行 (一般是核心 message)
    err_lines = [ln for ln in text.splitlines() if ln.startswith("E ")]
    if err_lines:
        return err_lines[0].strip()[:300]
    # fallback: short test summary 行
    m = re.search(r"FAILED [^\n]+ - (.+)", text)
    if m:
        return m.group(1).strip()[:300]
    return None


def phase_seconds(d, key):
    p = d.get(key)
    if not p:
        return None
    from datetime import datetime
    try:
        a = datetime.fromisoformat(p["started_at"].rstrip("Z"))
        b = datetime.fromisoformat(p["finished_at"].rstrip("Z"))
        return (b - a).total_seconds()
    except Exception:
        return None


# Signatures that indicate a *vendor-side* (endpoint/service) stability
# failure — i.e. the model / agent framework / task itself is not at fault.
# These belong in bucket V so users don't waste time debugging vLLM / prompt
# / verifier when the real fix is "restart the vendor's backend".
VENDOR_STABILITY_PATTERNS = (
    "inference_id not found",   # PPIO / novita gateway routing dropped mid-run
    "no available backend",     # 429 from vendor's load balancer
    "upstream connect error",   # envoy / gateway upstream failure
    "backend unhealthy",
    "503 Service Unavailable",
    "502 Bad Gateway",
    "504 Gateway Time-out",
)


def _is_vendor_stability_error(exit_reason: str) -> bool:
    if not exit_reason:
        return False
    return any(sig in exit_reason for sig in VENDOR_STABILITY_PATTERNS)


def classify(signals):
    """5-bucket 分类启发式.
    V: vendor/endpoint 稳定性问题 (backend 挂/inference_id 失效) — 不是模型/协议问题
    A: vLLM tool-call 协议错; B: vLLM 推理质量; C: agent 框架限制; D: 模型能力天花板
    通过的不分类, 返回 PASS。

    V 桶优先级最高：只要 exit_reason 里出现 vendor stability signature 就归 V，
    避免长跑中 endpoint 挂掉被误归 A/C 桶浪费 debug 精力。
    """
    if signals["reward"] == 1.0:
        return "PASS"
    if _is_vendor_stability_error(signals.get("exit_reason", "")):
        return "V"
    if signals["exception"]:
        # 异常退出 -> 大概率 C (timeout) 或基础设施
        et = signals["exception"]
        if "Timeout" in et:
            return "C"
        return "C"  # 兜底
    if signals["json_parse_errors"] > 0 or signals["dangling_calls"] > 0:
        return "A"
    # max_steps 或 context 接近满
    if signals["ctx_usage_last"] is not None and signals["ctx_usage_last"] > 0.9:
        return "C"
    if signals["steps"] >= 100:
        return "C"
    # 默认: 没协议错、没 timeout、agent 自然结束、verifier 失败 → 推理质量(B)或模型能力(D)
    # 区分 B/D 难，启发: reasoning_chars 很大但还是错 → D; 否则 B
    if signals["reasoning_chars"] > 20000:
        return "D"
    return "B"


def triage(trial_dir):
    trial_dir = Path(trial_dir)
    result = load_json(trial_dir / "result.json") or {}
    config = load_json(trial_dir / "config.json") or {}

    task = result.get("task_name") or trial_dir.parent.name
    reward = (result.get("verifier_result") or {}).get("rewards", {}).get("reward")
    exception = (result.get("exception_info") or {}).get("exception_type") if result.get("exception_info") else None

    agent_result = result.get("agent_result") or {}
    n_input = agent_result.get("n_input_tokens") or 0
    n_output = agent_result.get("n_output_tokens") or 0
    n_cache = agent_result.get("n_cache_tokens") or 0

    # wire 事件
    events, payloads, final = parse_wire_events(trial_dir / "agent" / "kimi-cli.txt")
    calls = assemble_tool_calls(payloads)
    snaps = collect_status_snapshots(payloads)

    tool_names = Counter(c["name"] for c in calls)
    json_parse_errors = sum(1 for c in calls if not c["args_parsed_ok"])
    dangling = sum(1 for c in calls if c["result"] is None)
    error_results = sum(1 for c in calls if c["result_is_error"])

    step_count = sum(1 for ev in payloads if ev.get("type") == "StepBegin")
    reasoning_chars = get_reasoning_chars(payloads)
    ctx_usage_last = snaps[-1]["ctx_usage"] if snaps else None

    # 退出原因
    if final and "error" in final:
        exit_reason = f"jsonrpc_error: {final['error'].get('message', '')[:120]}"
    elif final and "result" in final:
        exit_reason = f"jsonrpc_result: {final['result'].get('status', '')}"
    elif exception:
        exit_reason = f"exception: {exception}"
    else:
        exit_reason = "unknown"

    # verifier
    verr = verifier_first_error(trial_dir / "verifier" / "test-stdout.txt")

    # Runaway heuristic: trial burned way more than a well-behaved trial
    # should. Terminal-Bench-2 smoke tasks that PASS typically use 10-50
    # steps and 100k-1M input tokens. When kimi-cli / harbor have no hard
    # cap, a model that loops on the same wrong plan can burn 20M+ tokens
    # before agent_execution phase runs out of walltime. Flag those loudly
    # so users don't waste money silently.
    runaway = (
        step_count >= 200
        or n_input >= 5_000_000
        or (agent_result.get("phase_seconds", {}).get("agent_execution", 0) or 0) >= 1500
    )

    signals = {
        "task": task,
        "reward": reward,
        "exception": exception,
        "n_input_tokens": n_input,
        "n_output_tokens": n_output,
        "n_cache_tokens": n_cache,
        "steps": step_count,
        "tool_calls": len(calls),
        "tool_name_hist": dict(tool_names),
        "json_parse_errors": json_parse_errors,
        "dangling_calls": dangling,
        "tool_error_results": error_results,
        "reasoning_chars": reasoning_chars,
        "ctx_usage_last": ctx_usage_last,
        "exit_reason": exit_reason,
        "verifier_first_error": verr,
        "runaway": runaway,
        "phase_env_setup_s": phase_seconds(result, "environment_setup"),
        "phase_agent_setup_s": phase_seconds(result, "agent_setup"),
        "phase_agent_exec_s": phase_seconds(result, "agent_execution"),
        "phase_verifier_s": phase_seconds(result, "verifier"),
    }
    signals["bucket"] = classify(signals)
    return signals


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sig = triage(sys.argv[1])
    print(json.dumps(sig, ensure_ascii=False, indent=2))
