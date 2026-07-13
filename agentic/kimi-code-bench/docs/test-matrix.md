# Test Matrix — Methodology & Reference

For 5-minute onboarding see [`quickstart.md`](quickstart.md). This document
describes **how** the bench is structured and **what to expect** when you
interpret results.

---

## 1. Structure: model × track

Every benchmark run is a point in this matrix:

|  | Track A (official) | Track B (self-deployed) |
|---|---|---|
| Model X | official endpoint baseline | your self-deployed vLLM |
| Model Y | ... | ... |

- **Track A** (official) — a hosted, known-stable endpoint (PPIO, Moonshot,
  vendor gateway, etc.). Purpose: establish a "what should good look like"
  reference. Encoded per model in `presets/<model>.json::tracks.official`.
- **Track B** (self) — your own vLLM / SGLang / KTransformers deployment.
  Purpose: measure how close your deployment gets to Track A on the same
  task set. Endpoint supplied via `--self-url --self-key --self-header
  --self-model` at runtime.

Adding a new model = drop a new preset file. See
[`../presets/README.md`](../presets/README.md).

---

## 2. Task sets

Two standard task sets, both from Terminal-Bench-2 (pinned commit):

| Set | Command | # tasks | Wall-clock (concurrency 4-8) |
|---|---|---|---|
| smoke | `--tasks smoke` (default) | 15 | ~40-90 min |
| extended | `--tasks extended` | 89 | ~4-6 h |

Custom subsets: `--tasks fix-git,regex-log` or `--task-list <file>`.

---

## 3. What "PASS" means

Each task is a Docker container with an initial state (buggy repo,
unoptimized SQL, half-configured server) + a deterministic `test.sh`.
The agent (`kimi-cli --wire`) reads the instruction, uses tools
(Shell/ReadFile/WriteFile/Grep) to fix the environment, then `test.sh`
grades it and writes `reward.txt` (0 or 1).

`rewards.tsv`:

```
task	reward_mean	n_errors	job_dir
fix-git	1.0	0	/root/...     ← PASS
regex-log	0.0	0	/root/...   ← model tried and failed (B/D bucket)
sanitize-git-repo	0.0	1	/root/...  ← trial-level error (A/C bucket)
```

- `reward_mean = 1.0` + `n_errors = 0` → PASS
- `reward_mean = 0.0` + `n_errors = 0` → model attempted, verifier failed
- `n_errors ≥ 1` → trial exception before verifier ran (agent crashed,
  docker pull EOF, timeout) — the reward is not meaningful, retry needed

---

## 4. 4-bucket triage

`TRIAGE.md` auto-classifies every failed trial:

| Bucket | Fingerprint | Fix location |
|---|---|---|
| **A** | HTTP 400/422 from vLLM, or tool_call didn't parse | vLLM validator / tool parser |
| **B** | reward=0 but model produced coherent output — content wrong | sampling, chat template, reasoning transport |
| **C** | trial exception (apt/curl/docker/timeout) | kbench L1/L2, concurrency, network proxy |
| **D** | reward=0 and verifier output shows the model got the answer wrong | model capability ceiling — not deployable-side fixable |

Bucket A + C are "infrastructure problems"; retry after fixing.
Bucket B + D are "the model actually failed"; treat as the real score.

---

## 5. Sampling noise vs stable behavior

With temperature > 0 the same prompt can give different rewards across runs.
Empirically:

- Some tasks are **stable PASS** (all runs 1.0) — capability well above threshold
- Some tasks are **stable D** (all runs 0.0) — capability well below threshold
- **Noise** tasks flip between runs — capability sits right at threshold

To distinguish, retry the failing tasks N=3 times:
- ≥1 pass in 3 → noise (model can, given a lucky sample)
- 0/3 → stable fail (D bucket)

The canonical baselines under `runs/baseline-*/` are best-of-N merges;
the accompanying `README.md` explicitly labels each task as stable/noise/D.

---

## 6. Alignment criteria (self vs official)

Deployment is considered "aligned" when all three hold:

1. **Raw pass rate**: within 1-2 tasks of the official baseline's pass count
2. **Stable PASS coverage**: your self run passes the majority of tasks
   that were stable PASS on official
3. **Failure overlap**: the tasks your self run fails on are mostly the
   same tasks official also fails on (shared D bucket)

If self passes a task official failed on: probably noise / lucky sample.
If self fails a task official passed on: dig into `verifier/test-stdout.txt`
and classify the bucket. B (inference quality) is the most common
deployable-side issue.

---

## 7. Concurrency guidelines

Per-server empirical caps (may differ for your deployment):

| Endpoint type | Recommended `--concurrency` for smoke | Notes |
|---|---|---|
| Official gateway (rate-limited) | 3-5 | Higher values cause long tasks to hit gateway agent-timeouts |
| Self-deployed vLLM | 4-8+ | Bounded by your KV cache + parallel decode capacity |

If you see many `n_errors=1` on long tasks (polyglot-rust-c,
db-wal-recovery, sanitize-git-repo, query-optimize, mailman, etc.),
lower concurrency and retry those tasks separately at `--concurrency 3`.

---

## 8. Pin registry

`src/config.py` holds:

```python
KIMI_CLI_REF         = "2c34efb"          # kimi-cli 1.48.0
HARBOR_VERSION       = "0.5.0"
TERMINAL_BENCH_2_REF = "53ff2b87..."
```

Bumping any of these invalidates prior baselines. `runs/baseline-*/README.md`
records which pin combo was in effect at run time.

Also in `config.py`:

- `DEFAULT_TASKS` — the 15 smoke tasks (subset of `EXTENDED_TASKS`)
- `EXTENDED_TASKS` — full 89-task set
- `TASK_TIMEOUT_OVERRIDES` — per-task agent-timeout multiplier for known-slow tasks
- `HARBOR_AGENT_SETUP_TIMEOUT_MUL` / `HARBOR_AGENT_TIMEOUT_MUL` — global multipliers

---

## 9. What lives in `runs/`

- `runs/baseline-<model>-<track>[-extended]/` — canonical best-of-N per model+track.
  Committed to git (small: TSV + README only).
- `runs/<your-tag>/` — a per-run result folder. Contains `rewards.tsv`,
  `TRIAGE.md`, `jobs/`, `remote.tar.gz`. Large (~2-20 MB per run). Local only;
  not committed.

To publish a new canonical baseline, copy the interesting parts (TSV +
README explaining source runs and best-of-N picks) into a
`runs/baseline-<name>/` and commit.
