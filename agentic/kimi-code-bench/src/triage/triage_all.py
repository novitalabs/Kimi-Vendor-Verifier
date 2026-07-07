#!/usr/bin/env python3
"""Triage all trials in a run directory.

输入: <run-dir>/jobs/ 下的所有 trial
输出:
    1. <run-dir>/triage.jsonl  -- 每行一个 trial 的信号
    2. <run-dir>/TRIAGE.md     -- 人读分类报告

用法:
    triage_all.py v1-baseline-20260626
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from triage_trial import triage


BUCKET_DESC = {
    "PASS": "✓ 通过",
    "A": "vLLM tool-call 协议错 (parser / schema / streaming)",
    "B": "vLLM 推理质量 (sampler / prefix cache / max_tokens / 训练分布)",
    "C": "agent 框架限制 (max_steps / timeout / context 管理)",
    "D": "模型能力天花板 (训练侧标记上报)",
}


def main(run_dir):
    run_dir = Path(run_dir)
    jobs_dir = run_dir / "jobs"
    if not jobs_dir.is_dir():
        print(f"jobs dir not found: {jobs_dir}", file=sys.stderr)
        return 2

    # 找所有 trial 目录: jobs/<ts>/<task>__<id>/
    trials = []
    for job in sorted(jobs_dir.iterdir()):
        if not job.is_dir():
            continue
        for trial in job.iterdir():
            if trial.is_dir() and "__" in trial.name:
                trials.append(trial)
    print(f"found {len(trials)} trials", file=sys.stderr)

    # triage 每个
    results = []
    for t in trials:
        try:
            sig = triage(t)
            sig["_trial_dir"] = str(t.relative_to(run_dir))
            results.append(sig)
            print(f"  [{sig['bucket']:4}] {sig['task']:30} reward={sig['reward']}", file=sys.stderr)
        except Exception as e:
            print(f"  [ERR ] {t.name}: {type(e).__name__}: {e}", file=sys.stderr)

    # jsonl 落盘
    jsonl = run_dir / "triage.jsonl"
    jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
    print(f"\nwrote {jsonl}", file=sys.stderr)

    # 按 bucket 分组
    by_bucket = {}
    for r in results:
        by_bucket.setdefault(r["bucket"], []).append(r)

    # TRIAGE.md
    md = ["# v1 Triage Report", ""]
    md.append(f"Run: `{run_dir.name}`")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append("| Bucket | Count | Description |")
    md.append("|---|---|---|")
    for k in ["PASS", "A", "B", "C", "D"]:
        n = len(by_bucket.get(k, []))
        md.append(f"| **{k}** | {n} | {BUCKET_DESC[k]} |")
    md.append("")

    # 每桶 detail
    for bucket in ["PASS", "A", "B", "C", "D"]:
        items = by_bucket.get(bucket)
        if not items:
            continue
        md.append("")
        md.append(f"## Bucket {bucket} — {BUCKET_DESC[bucket]} ({len(items)})")
        md.append("")
        for r in items:
            md.append(f"### {r['task']}")
            md.append("")
            md.append(f"- reward: {r['reward']}, exception: {r['exception']}, exit: `{r['exit_reason']}`")
            md.append(f"- steps: {r['steps']}, tool_calls: {r['tool_calls']}, tool_errors: {r['tool_error_results']}, json_parse_errors: {r['json_parse_errors']}, dangling: {r['dangling_calls']}")
            md.append(f"- tokens: in={r['n_input_tokens']:,} cache={r['n_cache_tokens']:,} out={r['n_output_tokens']:,}, reasoning_chars={r['reasoning_chars']:,}")
            md.append(f"- ctx_usage_last: {r['ctx_usage_last']}")
            md.append(f"- phases (s): env={r['phase_env_setup_s']}, setup={r['phase_agent_setup_s']}, exec={r['phase_agent_exec_s']}, verify={r['phase_verifier_s']}")
            md.append(f"- tools: {r['tool_name_hist']}")
            if r["verifier_first_error"]:
                md.append("- verifier first error:")
                md.append(f"  > `{r['verifier_first_error']}`")
            md.append(f"- trial: `{r['_trial_dir']}`")
            md.append("")

    md_path = run_dir / "TRIAGE.md"
    md_path.write_text("\n".join(md))
    print(f"wrote {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "v1-baseline-20260626"))
