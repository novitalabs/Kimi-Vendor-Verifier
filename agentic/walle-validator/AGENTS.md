# walle-validator — 任务上下文

> **新人第一次进来 → 读 [`README.md`](README.md)** (完整命令示例 + 结果解读 + 使用场景)

## 在 Kimi-Vendor-Verifier 中的定位

`agentic/` 下两个必测项之一，对应 Kimi 官方对候选 vendor 的两条硬性检查:

| | Purpose |
|---|---|
| **kimi-code-bench** (sibling) | Kimi 官方通过 Terminal-Bench-2 15/89 tasks 验证 vendor 的 agentic 能力 |
| **walle-validator** (this dir) | Kimi 官方通过 walle testdata 逐 case 验证 vendor 的 JSON schema tool-call 兼容性 |

两个都过了才能通过 Kimi 官方准入。

## 一句话跑起来

```bash
cd agentic/walle-validator

./verify_tool_call_schema.py \
  --base-url "$YOUR_ENDPOINT" \
  --api-key "$YOUR_KEY" \
  --model "$YOUR_MODEL" \
  --header "$YOUR_ROUTING_HEADER" \
  --think-mode kimi
```

单次 204 case × 2 mode = 408 requests, 约 5-15 分钟。

## 数据流

1. **`testdata/kimi_official_cases.jsonl`** — 从真实 Kimi-official 测试报告 (`artifacts (22)/tool-call-schema-report.json`) 提取的 204 个 wire body。运行时的**唯一**数据源。**不改这个文件**。
2. **`testdata/validator_cases/`** — MoonshotAI/walle upstream jsonl (16 suites, 213 lines)。作 provenance 保留，runtime 不读。
3. 每 case 用 `tool_name="kvv_walle_case"` 作 tool，`parameters=<case.schema>`，两个 mode 各发一次。
4. 结果分类 (passed / failed-HTTP / failed-arguments-missing / failed-JSON-invalid) 写 `out/tool-call-schema-report.json` + `.log`。

## 结果解读的关键

**核心指标**: `http_status != null` 的 case 数应该是 **0**。
- 只要有 1 个 HTTP 4xx，说明 vendor 侧 walle 校验层有兼容性 bug，**上报 Kimi 官方一定挂**。
- HTTP 200 但 `arguments are missing` / `not valid JSON` 通常是模型采样噪声，跑 3 次取 best-of-N 消除。

## 不要做

- **不要改 `testdata/kimi_official_cases.jsonl`** — 那是 Kimi 官方发送的 ground truth wire body，改了就不是 Kimi 官方测试了
- **不要改 `testdata/validator_cases/`** — 那是 walle upstream，只用作 provenance
- 不要跑压测 (单次 408 requests 已够；成千上万可能踩 vendor ToS)
- 不要在 `verify_tool_call_schema.py` 里加 schema wrap / 变换逻辑 — Kimi 官方测试的 wrap 逻辑已经**materialised** 到 `kimi_official_cases.jsonl` 里了，任何"我们再 wrap 一次"都是复刻错误
