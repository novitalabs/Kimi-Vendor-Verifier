# walle-validator — Kimi 官方 walle 准入测试复刻

Sibling of [`agentic/kimi-code-bench/`](../kimi-code-bench/). 两个必测项之一:

| Test | 测什么 |
|---|---|
| [`kimi-code-bench`](../kimi-code-bench/) | End-to-end **agent 能力** (15/89 Terminal-Bench-2 tasks) |
| **walle-validator** (本目录) | **Kimi walle 服务端 JSON Schema 合规**——你的自部署服务能否成为 Kimi 官方供应商的准入门槛 |

## 🚀 3 步跑起来

```bash
cd agentic/walle-validator

./verify_tool_call_schema.py \
  --base-url "https://api.ppio.com/openai/v1" \
  --api-key "$PPIO_API_KEY" \
  --model "moonshotai/kimi-k2.6-agentic" \
  --header "X-Fusion-Provider:kimi-k26-agentic-guoji-b200" \
  --think-mode kimi

# 结果在 out/ 下:
#   tool-call-schema-report.json  完整结构化报告
#   verify_tool_call_json_schema_result.log  每行一条 PASSED/FAILED
```

- 单次 408 requests (204 case × 2 mode)，~5-15 分钟
- 无需 mitm proxy / 无需 kimi CLI，任何 OpenAI-compatible endpoint 都能测

## 这是什么测试

Kimi 官方 (Moonshot) 在验证一个厂商能否成为 kimi-for-coding 的 vendor 时，会用[MoonshotAI/walle](https://github.com/MoonshotAI/walle) 的 `testdata/validator_cases/` 作 tool-schema 覆盖，逐 case 发到候选 endpoint 的 `/chat/completions`，看：

1. **服务端接不接受这个 schema** (HTTP 200 vs 4xx walle 错)
2. **模型能否按 schema 生成合法的 tool_call arguments** (well-formed JSON)

**walle-validator 完整复刻这套测试**——同样的 wire body、同样的 case 覆盖、同样的报告格式，让你在**上报官方前先自测**。

## 数据来源

- **`testdata/kimi_official_cases.jsonl`** — Kimi 官方测试**实际发送**的 204 个 `tools[0].function.parameters` wire body。从一次真实的 Kimi 官方测试报告直接提取（见 `/Users/f/Documents/artifacts (22)/tool-call-schema-report.json`），逐字节保留。**这是运行时唯一的数据源**。
- **`testdata/validator_cases/`** — MoonshotAI/walle upstream 的 213 个 raw JSON schema case (16 suites)。作 provenance/参考用，runtime 不读。想更新可 `cd /tmp && git clone --depth 1 https://github.com/MoonshotAI/walle && cp -r walle/testdata/validator_cases/* <this>/testdata/validator_cases/`。
- **`testdata/selection_reasons.jsonl`** — audit trail: 204 个 case 从 213 里的 tool-callable 子集（9 个因特殊字符/超长 enum/深度递归 ref 不适合让模型现场生成 arguments 被排除）。同 `kimi_official_cases.jsonl` 冗余，保留供审查。

## 结果解读

`rewards.tsv` 每行:

```
[PASSED] [non-stream] TestBasicTypes/valid.jsonl:1 (empty_parameter_schema)
[FAILED] [stream]     TestRangeConstraints/valid.jsonl:9 (integer_parameter_schema) - tool call arguments are not valid JSON: Expecting value; arguments={"value":-}
```

### 4 种可能的每 case 状态

| 判定 | 什么情况 | 含义 |
|---|---|---|
| `passed` | HTTP 200 + tool_calls[0].function.arguments 是合法 JSON | 服务端接受 schema，模型也生成了 valid 参数 |
| `failed` (HTTP 4xx) | HTTP != 200，`http_status` 有值 | **服务端拒绝这个 schema** — walle 兼容性问题，vendor 侧要修 |
| `failed` (`arguments are missing`) | HTTP 200 但 tool_calls 为空 or arguments 空 | 服务端接受，模型没调工具 — 通常 sampling noise 或 chat template 问题 |
| `failed` (`not valid JSON`) | HTTP 200 有 arguments 但解析失败 | 模型 tool_call 生成不完整或格式错，通常是 max_tokens 截断 |

### 什么算 vendor "walle 兼容性 PASS"

**核心指标**: **`http_status` 是 null 的 case 应该 100% 通过**（即没有一个 case 被服务端 walle 拒绝）。

**次要指标**: 全部 408 request 的 pass rate，跟 Kimi 官方 baseline (~82-100%) 对齐。差距 10 个以内通常都是 sampling noise，可以跑 3 次取 best-of-N 消除。

### 报告 JSON 结构

`tool-call-schema-report.json` 顶层:

```json
{
  "generated_at": "...",
  "model": "moonshotai/kimi-k2.6-agentic",
  "base_url": "https://api.ppio.com/openai/v1",
  "tool_name": "kvv_walle_case",
  "dry_run": false,
  "thinking": false,
  "think_mode": "kimi",
  "modes": ["non-stream", "stream"],
  "selected_cases": [ ...204 entries ],
  "summary": {
    "total": 408,
    "by_status": {"passed": N, "failed": M},
    "by_selection_reason": {...},
    "by_mode": {"non-stream": {"passed":..,"failed":..}, "stream": {...}}
  },
  "results": [ ...408 entries ]
}
```

跟 Kimi 官方内部 verifier 格式 1:1 一致，双方结果可直接 `diff`。

## 标准使用场景

### 1. 自部署上报官方前自测

```bash
# 用 kimi 官方推荐的 non-thinking mode (K2.6-agentic default)
./verify_tool_call_schema.py \
  --base-url "http://<your-vllm>:<port>/v1" \
  --api-key "$YOUR_KEY" \
  --header "X-Infer-ID:<your-value>" \
  --model "$YOUR_MODEL_ID" \
  --think-mode kimi
```

结果里若 `http_status != null` 的行 > 0，vendor 侧必须先解决——**上报官方一定挂**。

### 2. 官方 baseline 对照

```bash
./verify_tool_call_schema.py \
  --base-url "https://api.ppio.com/openai/v1" \
  --api-key "$PPIO_API_KEY" \
  --model "moonshotai/kimi-k2.6-agentic" \
  --header "X-Fusion-Provider:kimi-k26-agentic-guoji-b200" \
  --think-mode kimi
```

跑 3 次取 best-of-N，作对齐目标数字。

### 3. 快速回归 (只跑一个 suite)

```bash
./verify_tool_call_schema.py ... --suites TestBasicTypes
```

## CLI reference

```
--base-url URL              OpenAI-compatible base URL, required
--api-key KEY               Bearer token; supports '$ENV_VAR', required
--model ID                  Model id, required
--header KEY:VALUE          Extra HTTP header, repeatable
--modes {non-stream,stream} Which modes to run, default both
--suites SUITE [...]        Restrict to specific suites, default all
--thinking                  Enable thinking mode (default: off)
--think-mode {none,kimi,opensource}
                            'kimi' -> top-level {"thinking":{"type":"disabled"}}
                            'opensource' -> {"chat_template_kwargs":{"thinking":false}}
                            'none' -> omit
--dry-run                   Load cases + print planned requests, don't send
--out-dir DIR               Output dir (default ./out/)
--timeout SEC               Per-request timeout (default 180)
```

## Wire params (与 Kimi 官方 acceptance test 一致)

- **top-level** `thinking: {type: "disabled"|"enabled"}` (**不用** `extra_body` 嵌套)
- **不发** `temperature` / `top_p` (让服务端用 default)
- `tool_choice: "required"` 强制模型调工具
- `max_tokens: 2048`

## 文件

```
walle-validator/
├── verify_tool_call_schema.py       CLI + probe 编排 (stdlib only)
├── testdata/
│   ├── kimi_official_cases.jsonl    ← 运行时唯一数据源
│   ├── selection_reasons.jsonl      audit trail
│   └── validator_cases/             upstream walle testdata (provenance)
├── out/                             per-run 产出 (gitignored 除 .gitkeep)
├── README.md                        本文件
└── AGENTS.md                        任务上下文，指向 quickstart
```

## 注意

- 不要跑成自动化压测 (单次 408 requests 已经足够，压测可能踩 vendor ToS)
- `out/` 里 raw JSON 不含 API key（`Authorization` 是请求头，不 dump 到 response）—— 但仍默认 gitignored
- Sampling noise 是真实的：K2.6-agentic 温度默认 1，同 case 两次跑可能出不同结果。得出结论前跑 3 次取 best-of-N
