# [Kimi Vendor Verifier](https://www.kimi.com/blogs/kimi-vendor-verifier.html)

[English](README.md) | 中文

基于 [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai) 框架的模型评测工具，用于评测 Kimi 模型在各类 benchmark 上的表现。

## 支持的评测任务

| 任务 | 描述 | 数据集 |
|------|------|--------|
| **AIME 2025** | 美国数学邀请赛，评估数学推理能力 | [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) |
| **MMMU Pro Vision** | 多模态理解评测（视觉问答，每题 10 个选项） | [MMMU/MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| **OCRBench** | OCR 文字识别能力评测 | [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench) |

### 评测参数要求

| Benchmark | 模式 | Max Tokens | Epochs |
|-----------|------|------------|--------|
| OCRBench | Non-Thinking | 8192 | 1 |
| OCRBench | Thinking | 16384 | 1 |
| MMMU | Non-Thinking | 16384 | 1 |
| MMMU | Thinking | 65536 | 1 |
| AIME 2025 | Non-Thinking | 16384 | 32 |
| AIME 2025 | Thinking | 98304 | 32 |

## 环境准备

### 1. 安装依赖

```bash
uv sync && uv pip install -e .
```

### 2. 配置环境变量

```bash
export KIMI_API_KEY="your-api-key"
export KIMI_BASE_URL="your-base-url"
```

或复制 `.env.example` 到 `.env` 并填入配置。

### 3. 预检验证

在运行 benchmark 之前，需要先验证 API 的参数约束是否正确：

```bash
# Kimi 官方 API
uv run python verify_params.py --model kimi/your-model-id --think-mode kimi --all

# 开源部署 (vLLM/SGLang/KTransformers)
uv run python verify_params.py --model your-model-id --think-mode opensource --all
```

该脚本检查不可变参数（temperature、top_p 等）是否被正确约束。**所有测试必须通过后，才能进行 benchmark 评测。**

## 运行评测

### OCRBench（快速验证）

#### Non-Thinking

```bash
uv run python eval.py ocrbench --model kimi/your-model-id \
    --think-mode kimi --max-tokens 8192 --stream
```

#### Thinking

```bash
uv run python eval.py ocrbench --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 16384 --stream
```

### MMMU Pro Vision

#### Non-Thinking

```bash
uv run python eval.py mmmu --model kimi/your-model-id \
    --think-mode kimi --max-tokens 16384 --stream
```

#### Thinking

```bash
uv run python eval.py mmmu --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 65536 --stream
```

### AIME 2025

#### Non-Thinking

```bash
uv run python eval.py aime2025 --model kimi/your-model-id \
    --think-mode kimi --max-tokens 16384 --stream
```

#### Thinking

```bash
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 98304 --stream
```

> **建议**：先运行 OCRBench 快速验证部署是否正确（约 10 分钟），确认通过后再运行 MMMU、AIME 完整评测。

## 详细说明

### 可用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `benchmark` | 评测任务: `ocrbench`, `mmmu`, `aime2025` | `ocrbench` |
| `--model` | 模型标识，如 `kimi/your-model-id` | **必填** |
| `--max-tokens` | 最大输出 token 数（见评测参数要求） | **必填** |
| `--thinking` | 开启思考模式（需配合 `--think-mode kimi/opensource`） | 关闭 |
| `--think-mode` | 思考参数格式：`kimi` 或 `opensource`（vLLM/SGLang/KTransformers） | `kimi` |
| `--stream` | 启用流式传输（推荐，避免长推理超时） | 关闭 |
| `--max-connections` | 最大并发连接数 | 按 benchmark |
| `--epochs` | 采样次数 | 按 benchmark |
| `--client-timeout` | HTTP 超时时间（秒） | `86400` |

### 思考模式参数

| 模型类型 | 参数组合 | 发送的 extra_body |
|---------|---------|------------------|
| Kimi 官方 + 思考关闭 | `--think-mode kimi` | `{"thinking": {"type": "disabled"}}` |
| Kimi 官方 + 思考开启 | `--thinking --think-mode kimi` | `{"thinking": {"type": "enabled"}}` |
| 开源框架 + 思考关闭 | `--think-mode opensource` | `{"chat_template_kwargs": {"thinking": false}}` |
| 开源框架 + 思考开启 | `--thinking --think-mode opensource` | `{"chat_template_kwargs": {"thinking": true}}` |

### 查看结果

```bash
# 使用 inspect view 查看日志
uv run inspect view

# 日志保存在 logs/ 目录
```

### 恢复中断的评测

```bash
uv run inspect eval-retry logs/<log-file>.eval
```

## 注意事项

### AIME 2025 评测

AIME 评测的输出 tokens 较多，需要注意：

1. **超时设置**
   - **客户端**: 默认 `--client-timeout 86400`（24小时），一般无需修改
   - **服务端**: 确保服务端的请求超时也设置足够长
   - **网关/代理**: 如使用 nginx/ALB，需调整 `proxy_read_timeout` 等配置

2. **流式传输**
   - **强烈建议**使用 `--stream` 参数
   - 非流式请求在 thinking 模式下容易超时
   - 流式可保持连接活跃，避免中间网关超时

3. **并发控制**
   - 默认 `max_connections=100`，根据服务端承载能力调整
   - 如果出现大量 429 或 `RemoteProtocolError`，降低并发数

4. **快速验证**
   - 建议先用 `--epochs 1` 跑通全部样本，确认配置正确
   - 验证通过后再运行 `--epochs 32` 完整评测

```bash
# Step 1: 快速验证（30 samples x 1 epoch）
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 98304 --stream --epochs 1

# Step 2: 完整评测（30 samples x 32 epochs）
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 98304 --stream
```

### 自动重试机制

以下网络类错误会**自动重试**（指数退避，1-60 秒），无需手动配置：

| 错误类型 | 说明 |
|----------|------|
| `RateLimitError` / `429` | 服务端限流 |
| `APIConnectionError` | 连接失败 |
| `ReadError` / `RemoteProtocolError` | 网络读取错误 |

> 非网络类错误（如模型输出格式问题）不会重试，会直接记录到日志供后续分析。

## 项目结构

```
├── eval.py              # 主评测入口 CLI
├── verify_params.py     # 预检参数验证
├── kimi_model.py        # Kimi Model API 实现
├── aime2025.py          # AIME 2025 评测任务
├── mmmu_pro_vision.py   # MMMU Pro Vision 评测任务
├── ocr_bench.py         # OCRBench 评测任务
├── logs/                # 评测日志
└── pyproject.toml       # 项目配置
```

## 联系我们

如果您有任何问题或建议，请联系 contact-kvv@kimi.com。
