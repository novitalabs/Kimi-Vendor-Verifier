# kimi-code-bench — Agentic 能力测试

> **新人第一次进来 → 读 [`docs/quickstart.md`](docs/quickstart.md)（5 分钟上手）**
> **已经上手要看数字 / 修理 / 深挖 → 读 [`docs/test-matrix.md`](docs/test-matrix.md)（canonical 测试矩阵）**

## 在 Kimi-Vendor-Verifier 中的定位

这是 Kimi-Vendor-Verifier 的 **agentic** 部分，与顶层的 `eval.py` (ocrbench/mmmu/aime25) 互补：

| | Verifier 顶层 (eval.py) | agentic/kimi-code-bench (kbench) |
|---|---|---|
| **测什么** | 静态题库：知识 / 数学 / OCR / 视觉 | agent 能力：完成真实终端任务的能力 |
| **数据集** | AIME / MMMU / OCRBench | Terminal-Bench-2 (15 smoke + 89 extended) |
| **判分** | 答案匹配 | 每个任务的 `test.sh`（reward 0/1） |
| **框架** | inspect-ai | Harbor + kimi-cli |
| **时长** | 分钟级到小时级 | 小时级 |

## 项目目标

让被测模型（Kimi 系列 + 其它 OpenAI-compatible 模型）的**自部署 vLLM 服务**在 Terminal-Bench-2 上追平**官方 endpoint** baseline。

## 入口指针

| 想知道什么 | 看哪里 |
|---|---|
| **5 分钟上手** | [`docs/quickstart.md`](docs/quickstart.md) |
| 当前测试矩阵 / 已知数字 / 验收标准 | [`docs/test-matrix.md`](docs/test-matrix.md) |
| 已定义的模型 preset | [`presets/README.md`](presets/README.md) |
| 怎么跑 / CLI 选项 | [`README.md`](README.md) 或 `./kbench --help` |
| 已 pin 的版本 | `src/config.py` |
| 历史 run 数字 | `runs/baseline-*/rewards.tsv` + `README.md` |

## 标准工作流 (3 步)

```bash
./kbench init       # 一次: 引导写 ~/.kbench.json + 测 SSH + 测 API endpoint
./kbench doctor     # 每次: 完整体检
./kbench run --preset <model> --track official --tag <tag> --detach
```

详见 [`docs/quickstart.md`](docs/quickstart.md) 步 1-3。

## 不要做

- 不要在没跑 `doctor` 时就开跑 smoke（白烧资源）
- 不要在没跑 `probe` 时直接开 smoke（endpoint 挂了不知道）
- 不要在 sampling noise 没 N=3 复现前下结论
- 不要把不同模型的 raw pass 直接比（K2.7-code / K2.6 能力本身有差）
- 不要改 pin 的版本 (`src/config.py::KIMI_CLI_REF/HARBOR_VERSION/TERMINAL_BENCH_2_REF`) 不重跑 baseline
