# 5 分钟上手 kimi-code-bench

> 🚀 **新用户第一次来 → 三步走** (每步只有 1 条命令)
>
> ```bash
> ./kbench init         # 1. 交互引导写 ~/.kbench.json + 测 SSH + 测 PPIO endpoint
> ./kbench doctor       # 2. 完整体检: 本地 + 远端 + 每个 preset
> ./kbench run --preset kimi-k2.6 --track official \
>              --tag k26-off-$(date +%Y%m%d) --detach   # 3. 后台跑 baseline
> ```
>
> 3 步都过了？跑完后 `./kbench fetch <tag>` 拉数据 + 自动 triage。跳到 [如何看结果](#如何看结果) 读数字。

给团队成员在**自己的开发机**上跑一次 agentic 能力测试。

## 你需要什么

- 开发机: macOS / Linux 均可，Python 3.10+，`ssh` client
- 一台**测试服务器**（下文详述）
- 一个**官方 endpoint**（PPIO / Moonshot / 别的 OpenAI-compatible gateway）的 API key
- 一个**自部署 vLLM endpoint**（可选，如果只想测官方就跳过）

### 测试服务器需要什么

Docker 会在这台服务器上跑 `alexgshaw/*` 任务容器，agent 也跑在容器里。规格要求：

- **Linux (x86_64)** —— Docker Desktop / Rosetta 不支持
- **Docker ≥ 20** —— 用来起 harbor 任务容器
- **磁盘 ≥ 50 GB** —— 缓存 task 镜像 + 保存 jobs 输出
- **稳定的 CPU + 网络** —— 一次 smoke 跑 40 min ~ 6h
- **网络可达**:
  - `docker.io` / `ghcr.io`（拉任务镜像）
  - `archive.ubuntu.com` / `astral.sh` / `pypi.org`（容器内装 uv + kimi-cli）
  - 你的官方 endpoint（跑 baseline）+ 自部署 endpoint（跑对比）
- **PRC 网络环境**：可以填 `--container-proxy http://<proxy>:<port>` 让容器走代理

推荐一人一台，隔离最清晰。团队共享同一台服务器也可以，见下面。

### novitalab 内部共享 4090 (可选)

novitalab 有一台 4090 测试机作为团队共享。共用规则：

- **每个成员一个 Linux 用户账号**（不用 root，避免踩到别人的容器）
- **workdir 自动隔离**：`~/.kbench.json` 里 `workdir_root` 建议填 `~/kbench` 或 `/home/$USER/kbench`，kbench 会展开成你自己的路径，多人并行跑互不干扰
- **每个成员用自己的 SSH key + 自己的 PPIO API key**——`./kbench init` 会分别填
- **docker daemon 是全机共享的**，同时开的 harbor 容器数量 ≤ 机器 CPU/内存能承受的范围（一次 smoke ~8 并发单人跑没问题；两人同时跑请先协调 concurrency，两个人一起 `--concurrency 8` 会让 docker daemon 打瓶颈）

具体 SSH 凭证、跳板配置向 novitalab 内的 kimi-code-bench 维护者索要。

## 三步走

### 步 1: 首次配置 (只需一次)

```bash
git clone <this-repo> ~/kimi-code-bench
cd ~/kimi-code-bench

./kbench init      # 交互式引导，写 ~/.kbench.json + 立即测 SSH + PPIO
```

`init` 会问你 5 个问题：

1. **SSH spec**: 测试服务器登录字符串（例：`user@host` 或 `user@root@host@jump`）
2. **SSH port**: 默认 22
3. **Workdir root**: 服务器上放 kbench 工作目录的根目录（例：`/root/kbench`，多 run 会自动加 tag 后缀）
4. **Container proxy**: 测试服务器**在中国大陆**且需要代理访问 docker.io / archive.ubuntu.com 时填 `http://<host>:<port>`；否则回车跳过
5. **PPIO API key**（可选）: 官方 baseline 需要。可以跳过后自己去 platform.moonshot.ai 或 ppio.com 申请

写完 `init` 立即 SSH + endpoint 通不通，看到 `✔ SSH ... OK` / `✔ PPIO endpoint 200` 才算成功。

`init` 之后跑 `./kbench doctor` 做完整体检（本地 python/ssh + 远端 docker/workdir 可写 + 每个 preset 的 api_key 能不能取到）：

```bash
$ ./kbench doctor
Local:
  ✔ python 3.13
  ✔ ssh
  ✔ .kbench.json parses OK
Remote SSH:
  ✔ user@host:22  OK
  ✔ docker installed  (Docker version 28.1.1)
  ✔ workdir_root /root/kbench writable  (ok)
Presets:
  ✔ kimi-k2.6 (api_key: keys.PPIO_API_KEY)
  ✔ kimi-k2.7-code (api_key: keys.PPIO_API_KEY)

✔ All checks passed. Ready to run.
```

### 步 2: 选一个模型跑官方 baseline

看现成的模型 presets：

```bash
ls presets/*.json
# kimi-k2.6.json  kimi-k2.7-code.json
```

跑官方 15 任务 smoke（`--preset` 自动填 endpoint / key / header / preserve_thinking）:

```bash
./kbench run \
  --preset kimi-k2.6 \
  --track official \
  --tag k26-off-$(date +%Y%m%d) \
  --concurrency 4 \
  --detach
```

跟进度:

```bash
./kbench tail k26-off-20260706        # 实时 tail setup.log
./kbench fetch k26-off-20260706       # 等完成 → 拉本地 → 自动 triage
```

出结果：`runs/k26-off-20260706/rewards.tsv` + `TRIAGE.md`。

### 步 3: 跑自部署对比

自部署 endpoint 参数走 `--self-*`（其它由 preset 提供）：

```bash
./kbench run \
  --preset kimi-k2.6 \
  --track self \
  --self-url http://<your-vllm>:<port>/v1 \
  --self-key <your-key> \
  --self-header X-Infer-ID:<your-value> \
  --self-header X-Custom-Route:<val> \      # 可重复，需要多个 header 就多传几次
  --self-model <exposed-model-id> \
  --tag k26-self-$(date +%Y%m%d) \
  --remote-workdir /root/kbench/k26-self/ \  # 与 official run 并行时必加，隔离 workdir
  --concurrency 8 \
  --detach

./kbench fetch k26-self-20260706
./kbench compare runs/k26-off-20260706 runs/k26-self-20260706
```

**preset 名称约定**: `--preset <name>` 里的 `<name>` 就是 `presets/<name>.json` 文件的 stem。当前可选：`kimi-k2.6`, `kimi-k2.7-code`。加新模型只需要往 `presets/` 里丢一个新 json，见 [`presets/README.md`](../presets/README.md)。

## 命令速查

| 命令 | 作用 |
|---|---|
| `./kbench init` | 交互式引导写 `~/.kbench.json` + 测 SSH + 测 API endpoint |
| `./kbench doctor` | 全面体检: 本地 python/ssh + 远端 SSH/docker/workdir + 所有 preset 的 api_key 可达性 |
| `./kbench install` | 只检查本地依赖 (ssh / python3)。已被 doctor 覆盖 |
| `./kbench probe --preset <name> --track official` | 只跑 14 case 协议探测 (< 1 min)，验证 endpoint 能力 |
| `./kbench run --preset <name> --track official ... --detach` | **总是 --detach**：后台起 smoke run，本地立刻退出。不加 detach 时本地网断了 run 也会挂 |
| `./kbench tail <tag>` | 实时看远端 setup.log。Ctrl-C 停 tail（run 继续跑） |
| `./kbench fetch <tag>` | 阻塞等 run 完成 → 拉数据 → 自动 4 桶 triage。中断可再跑，会重新从远端拉最新数据 |
| `./kbench compare <run_a> <run_b>` | 两个 run 逐任务 diff |
| `./kbench triage <run_dir>` | 重新 triage 已有 run |

## 如何看结果

跑完 `fetch` 后你的 `runs/<tag>/` 长这样：

```
runs/<tag>/
├── rewards.tsv          ← 15 (或 89) 行 task-level reward
├── TRIAGE.md            ← 自动 4 桶分类报告
├── triage.jsonl         ← 机读版
├── config.snapshot.json ← 完整参数记录（谁跑的、endpoint 是啥、pin 是啥）
├── setup.log            ← 远端全 stdout
├── remote.tar.gz        ← 远端 jobs/ 目录打包
└── jobs/                ← 解包后每个 task 的 trial 数据
    └── <task>__<id>/
        ├── result.json  ← Harbor 完整结果
        ├── agent/       ← kimi-cli 全部 wire 消息
        └── verifier/
            ├── reward.txt      ← 0 或 1
            └── test-stdout.txt ← pytest / bash verifier 的原始输出
```

### rewards.tsv 怎么读

```
task	reward_mean	n_errors	job_dir
fix-git	1.0	0	/root/...     ← ✓ PASS
regex-log	0.0	0	/root/...   ← ✗ 模型真失败 (verifier 跑了但没过)
sanitize-git-repo	0.0	1	/root/...  ← ⚠ trial-level 异常 (agent 挂了或 docker 挂了)
```

- **reward_mean = 1.0** → 该任务通过
- **reward_mean = 0.0** + **n_errors = 0** → 模型真的没解决（B/D 桶）
- **n_errors ≥ 1** → trial 本身挂了（A/C 桶）——这个 reward 不算数，需要 retry

### 4 桶分类（TRIAGE.md 里自动打）

| 桶 | 触发条件 | 修复方向 |
|---|---|---|
| **A** | vLLM 协议错（400/422 或 tool_call 解析失败） | 改 vLLM 侧 validator / tool parser |
| **B** | reward=0 但模型正常回答，只是内容不对 | sampling / chat template / reasoning 透传 |
| **C** | trial-level 异常（apt/docker/OAuth/timeout） | 提高 timeout / 降 concurrency / 网络代理 |
| **D** | reward=0 且明确是"模型答错了" | 模型能力天花板，不在部署侧修 |

### 什么是"稳定 PASS" vs sampling noise

同一个模型 + 同一个 endpoint 跑两次，同一个任务**结果可能不同**（尤其 K2.6 温度=1）：

- 两次都 1.0 → **稳定 PASS**
- 两次 1/0 交替 → **sampling noise**（模型能力刚好在门槛）
- 两次都 0.0 → **稳定 fail**（D 桶）

看 canonical baseline `runs/baseline-*/` 里 `README.md` 会明确标每个任务是稳/noise/稳 D。

### 判定"自部署对齐官方"的三个信号

1. **raw pass rate ≥ baseline 的 -1** （允许 1 个任务差异）
2. **稳定 PASS 集合大部分命中**（见 `runs/baseline-<model>-official/README.md` 里的稳 PASS 列表）
3. **失败任务集合大部分重合**（都失败在同一批 D 桶）

如果 self 有稳定 PASS 没命中或独有失败任务，看那些 trial 的 `verifier/test-stdout.txt` 分析根因。

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--preset <name>` | none | 填 `presets/*.json` 里的模型名，一键装配所有 endpoint 参数 |
| `--track official\|self` | official | preset 里定义了两条 track |
| `--tasks smoke\|extended\|<csv>` | smoke | smoke=15 任务, extended=89 任务, 或逗号分隔子集 |
| `--task-list <file>` | none | 只跑文件里列的任务 (一行一个) |
| `--concurrency <n>` | 1 | 并发。**smoke 建议 4-8; extended 官方 4，自部署 8** |
| `--detach` | off | 远端后台跑，立刻退出 |
| `--remote-workdir <path>` | 自动 | 多 run 并行时用不同 workdir 避免互相覆盖 |
| `--preserve-thinking 0\|1` | preset 决定 | 只对 K2.7-code 需要 1 |

## 添加新模型 preset

写一个新 JSON 到 `presets/<name>.json`：

```json
{
  "name": "glm-5.2",
  "description": "Zhipu GLM 5.2",
  "preserve_thinking": 0,
  "recommended_smoke_concurrency": 8,
  "recommended_extended_concurrency": 4,
  "tracks": {
    "official": {
      "base_url": "https://api.ppio.com/openai/v1",
      "api_key": "$PPIO_API_KEY",
      "headers": {"X-Fusion-Provider": "zai-openai"},
      "model_id": "zai-org/glm-5.2"
    },
    "self": {
      "description": "Provide --self-url --self-key --self-header --self-model"
    }
  }
}
```

`--preset glm-5.2` 就能用。

## 疑难

**smoke 出现 env-err (n_errors>0)？**

concurrency 太高时长任务会 timeout。看到 `k26-off-*/rewards.tsv` 里 n_errors 列有值，把那些任务丢进 retry：

```bash
grep -P '\t[1-9]\d*$' runs/<tag>/rewards.tsv | cut -f1 > /tmp/retry.txt
./kbench run --preset <name> --track official \
  --task-list /tmp/retry.txt \
  --tag <tag>-retry \
  --remote-workdir /root/kbench/<tag>-retry \
  --concurrency 3 --detach
```

**看 4 桶分类含义？**

见 [`test-matrix.md`](test-matrix.md) §6 (A 协议 / B 推理 / C 框架 / D 模型)。

**要跑更大样本？**

`--tasks extended` (89 任务, ~4-6 小时)。长上下文任务上自部署与官方通常差距最明显。

## Canonical baseline (可对比参考)

任何团队/个人跑过一次官方 track 后，`runs/<tag>/` 就是新的参考数字。要沉淀为"团队约定的官方基线"，把它 copy 到 `runs/baseline-<model>-<track>/` 并写一份 `README.md` 说明来源（run tag、best-of-N 合并方式、pin combo）。

Canonical baseline 只含小体积文本（TSV + README），会 commit 到 git；per-run 数据（jobs/、tar.gz）默认被 gitignore 忽略。
