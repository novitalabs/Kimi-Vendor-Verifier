# Model Presets

一个 preset = 一个模型在两条 track（`official` / `self`）上的所有参数。团队成员只需选 preset name + 在 self track 上补自部署 endpoint 即可开跑。

## 已定义 presets

| Preset | Model | Official endpoint |
|---|---|---|
| `kimi-k2.6` | Moonshot K2.6 (K2 通用) | PPIO `moonshotai/kimi-k2.6` |
| `kimi-k2.7-code` | Moonshot K2.7-code (K-series 编码专用) | PPIO `moonshotai/kimi-k2.7-code` |

## 使用示例

```bash
# 跑官方 baseline (需要 PPIO_API_KEY 在 ~/.kbench.json 或 env)
./kbench run --preset kimi-k2.6 --track official --tag k26-off-$(date +%F) --detach

# 跑自部署 (只需再提供 self endpoint 4 元组)
./kbench run --preset kimi-k2.6 --track self \
  --self-url http://your-host:port/v1 \
  --self-key your-key \
  --self-header X-Infer-ID:your-value \
  --tag k26-self-$(date +%F) --detach
```

## Schema

```json
{
  "name": "<model display name>",
  "description": "<one-line>",
  "preserve_thinking": 0 | 1,
  "recommended_smoke_concurrency": <int>,
  "recommended_extended_concurrency": <int>,
  "tracks": {
    "official": {
      "base_url": "https://...",
      "api_key": "$ENV_VAR" | "<literal>" | "keys.<name>",
      "headers": {"K": "V", ...},
      "model_id": "provider/model",
      "notes": "..."
    },
    "self": {
      "description": "..."
    }
  }
}
```

### api_key 引用

kbench 按下面顺序解析 `api_key` 字段：

| 前缀 | 解析规则 |
|---|---|
| `"$PPIO_API_KEY"` | **优先** `~/.kbench.json` 的 `keys.PPIO_API_KEY`；**回退** 环境变量 `PPIO_API_KEY` |
| `"~/.secrets/ppio"` 或 `"$HOME/.x"` | 读文件内容并 strip |
| `"PPIO_API_KEY"`（无 $） | 只在 `~/.kbench.json` 的 `keys.PPIO_API_KEY` 里找 |
| `"sk_..."` | 字面量（不推荐——会进 git 历史） |

**建议**：preset 里统一用 `"$NAME"` 引用。团队成员跑 `./kbench init` 会写到 `~/.kbench.json` 的 keys 段，也可以设置对应的 env var —— 两条路径都能工作。

## 添加新模型

写一个新 preset json 到 `presets/<name>.json` 即可。`--preset <name>` 会自动加载。
