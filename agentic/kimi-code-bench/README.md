# Kimi Code Bench

End-to-end agentic benchmark for Moonshot Kimi (and other OpenAI-compatible) LLM
endpoints. Tests 15 or 89 real terminal tasks under
[Harbor](https://github.com/laude-institute/harbor) +
[Terminal-Bench-2](https://github.com/laude-institute/terminal-bench-2) +
[kimi-cli](https://github.com/MoonshotAI/kimi-cli), and compares an
official endpoint baseline against your self-deployed vLLM.

## 🚀 New here? 3 steps to run your first benchmark

```bash
git clone <this-repo> ~/kimi-code-bench
cd ~/kimi-code-bench

./kbench init            # 1. interactive setup (SSH + PPIO key + test connectivity)
./kbench doctor          # 2. verify config (local + remote + presets)
./kbench run --preset kimi-k2.6 --track official \
             --tag k26-off-$(date +%Y%m%d) --detach   # 3. run baseline
```

📖 **[Full walkthrough → docs/quickstart.md](docs/quickstart.md)** — 5-minute team onboarding

## What you get

- **`./kbench init`**: interactive setup, writes `~/.kbench.json`, tests SSH + API endpoint
- **`./kbench doctor`**: pre-flight audit (Python, ssh, remote docker, preset keys)
- **`./kbench probe --preset X --track official`**: 14-case protocol probe (< 1 min)
- **`./kbench run ... --detach`**: launch smoke run in background on a remote sandbox
- **`./kbench tail <tag>`** / **`./kbench fetch <tag>`**: monitor / archive
- **`./kbench compare <run_a> <run_b>`**: side-by-side task-level diff

## What it measures

| Task set | # tasks | Wall-clock (concurrency 4-8) |
|---|---|---|
| `--tasks smoke` (default) | 15 | ~40-60 min |
| `--tasks extended` | 89 | ~4-6 hours |

Each task = a Docker container with an initial state (buggy repo, unoptimized
SQL, half-configured server, etc.). The agent gets an instruction, must
use Shell / ReadFile / WriteFile / Grep tools to fix it, and a
deterministic `test.sh` grades the result (reward 0 or 1).

## Documentation

| Where | What |
|---|---|
| [`docs/quickstart.md`](docs/quickstart.md) | Team onboarding, result interpretation |
| [`docs/test-matrix.md`](docs/test-matrix.md) | Canonical baselines, endpoint config, task index |
| [`docs/design.md`](docs/design.md) | Architecture rationale |
| [`presets/README.md`](presets/README.md) | Model preset schema, how to add new models |
| `AGENTS.md` | Pointer file for AI-assisted contribution |

## Canonical baselines

`runs/baseline-<model>-<track>/` contains best-of-N reference results
for known model+endpoint combinations. Each baseline directory has a
`rewards.tsv` (15 or 89 rows) and a `README.md` explaining the source
runs, pin combo, and any known noise / stable-fail patterns.

The repo ships with the directory structure but no committed baseline
data — each team is expected to publish their own baselines after running
against their reference endpoints. See
[`docs/test-matrix.md#9-what-lives-in-runs`](docs/test-matrix.md) for
the sharing convention.

## Contributing

Presets: drop a new `presets/<model>.json` (see `presets/README.md` schema).
No code changes needed — `--preset <model>` picks it up automatically.

vLLM-int compatibility patches (for self-deployed endpoints): see
[vllm-int](https://github.com/novitalabs/vllm-int) `feat/kimi_agentic_test` branch.
