# Kimi Code Bench — Design

## Goal

Give a single command-line tool that answers **"how does this LLM endpoint
(official or self-deployed) perform on real coding-agent tasks?"** for any
OpenAI-compatible model. Fill in a `--preset` (which model) + `--track`
(official vs self) + one-time `~/.kbench.json` (which test server) — that's it.

## Input model

Two conceptual axes:

1. **What model / where** — captured in a `presets/<name>.json`. Each preset
   defines a model plus its `official` and `self` track endpoint tuples.
2. **What test server** — captured once in `~/.kbench.json`. The bench spawns
   docker + harbor + trial containers there, then rsyncs results back.

Both axes are decoupled from the source tree — team members and open-source
users only touch these two config sources, not `src/`.

## Output

```
runs/<tag>/
├── rewards.tsv          15 or 89 rows: task | reward | n_errors | job_dir
├── TRIAGE.md            4-bucket classification (A/B/C/D) auto-generated
├── triage.jsonl         machine-readable version
├── probe.json           14-case protocol probe result (if --probe)
├── config.snapshot.json all resolved params, versions, endpoints
├── setup.log            full stdout of remote setup + trials
├── remote.tar.gz        raw remote artifacts
└── jobs/<task>__<id>/   Harbor trial data
    ├── result.json      Harbor summary
    ├── agent/           kimi-cli wire protocol dump
    └── verifier/        pytest / test.sh output + reward.txt
```

## Flow

```
kbench run --preset <model> --track official --tag <tag> --detach
  │
  ├─ resolve preset → base_url, api_key ($env or ~/.kbench.json.keys),
  │                    headers, model_id, preserve_thinking
  │
  ├─ resolve ~/.kbench.json → remote SSH spec, workdir, mirrors, proxy
  │
  ├─ probe (14 cases) — optional (--skip-probe)
  │   ├─ /v1/models reachable + baseline chat
  │   ├─ empty tools=[] accepted
  │   ├─ top-level thinking={type,keep} + strict validators
  │   ├─ interleaved thinking (reasoning_content preservation)
  │   ├─ streaming + tool_call round-trip
  │   ├─ max_tokens truncation + reasoning emission
  │   └─ prompt caching + long prompt smoke
  │
  ├─ render artifacts from templates/ + push over ssh
  │   ├─ setup_and_smoke.sh.rendered   (bash setup + per-task harbor run pool)
  │   └─ local_kimi_cli_vendor_agent.py.rendered  (Harbor adapter)
  │
  ├─ ssh nohup setsid → remote runs; kbench exits ~seconds
  │
  ├─ kbench tail <tag>   follow remote setup.log
  ├─ kbench fetch <tag>  wait for done → pull tar.gz → extract → triage
  │
  └─ result: runs/<tag>/ + auto TRIAGE.md
```

## Key design decisions

### Preset + track separation

Every "test scenario" is `model × endpoint`. Instead of tangling both into
CLI flags, `presets/*.json` names the model and enumerates its endpoint
tracks:

```json
{
  "name": "kimi-k2.6",
  "preserve_thinking": 0,
  "tracks": {
    "official": {
      "base_url": "https://api.ppio.com/openai/v1",
      "api_key": "$PPIO_API_KEY",
      "headers": {"X-Fusion-Provider": "moonshot-openai"},
      "model_id": "moonshotai/kimi-k2.6"
    },
    "self": { "description": "provide --self-* CLI flags" }
  }
}
```

Adding a new model = drop a new JSON file. No code changes.

### `~/.kbench.json` for user-specific config

The tool needs a Linux test server with docker + network access, but that
server is different per user / per team. `./kbench init` writes SSH spec,
workdir root, API keys, and (optional) network mirrors to `~/.kbench.json`.

Precedence when resolving values:
`CLI flag > env var > ~/.kbench.json > src/config.py DEFAULTS (empty)`

### Pinned upstream versions

```python
KIMI_CLI_REF         = "2c34efb"          # kimi-cli 1.48.0
HARBOR_VERSION       = "0.5.0"
TERMINAL_BENCH_2_REF = "53ff2b87..."
```

kimi-cli / harbor / task set — any drift makes rewards non-comparable to
prior baselines. Bumps happen only via explicit maintainer PR that
re-generates canonical baselines.

### Stability layers L1 + L2

Trials fail for two dominant reasons under load: (a) `apt-get`/`uv install`
inside fresh containers hitting network flakes, (b) long tasks
(`polyglot-rust-c`, etc.) blowing the default agent timeout.

- **L1**: the rendered vendor agent swaps `archive.ubuntu.com` to the TUNA
  mirror and wraps `apt-get` / `curl uv install` / `uv tool install` in
  exponential-backoff retry (up to 4 attempts).
- **L2**: `src/config.py::TASK_TIMEOUT_OVERRIDES` lets known-slow tasks pick
  a bigger `--agent-timeout-multiplier` per-task, e.g. `polyglot-rust-c: 10`.

### Concurrency + workdir isolation

`--concurrency N` runs N tasks in parallel using a bash `wait -n` pool.
Pre-pull images sequentially first (avoids `docker.io` TLS pileups), then
release the pool. Each parallel task writes to a per-task jobs subdir to
avoid `ls -td` races.

Multiple concurrent runs (e.g. official + self side-by-side on the same
server) use `--remote-workdir /root/kbench/<tag>/` for physical isolation.

### Detach + tail/fetch

Smokes take 30 min – 6 h. Foreground `ssh` blocks and dies if the laptop
sleeps. `--detach` `nohup setsid`s the remote script and returns in
seconds; `./kbench tail <tag>` follows the log, `./kbench fetch <tag>`
waits for `.kbench-done` then pulls.

### 4-bucket triage

Every failed trial is auto-classified into:

| Bucket | Cause | Fix location |
|---|---|---|
| A | vLLM protocol error (400 / tool_call parse fail) | vLLM validator / tool parser |
| B | vLLM inference quality (wrong content, format OK) | sampling / chat template / reasoning |
| C | agent framework / env (docker / apt / timeout) | kbench / harbor / concurrency |
| D | model capability ceiling | model itself, not deployment |

The triager reads each trial's `verifier/test-stdout.txt` and pattern-matches
into a bucket. Not perfect but 80%+ correct on our runs and dramatically
reduces read-through time.

### Best-of-N canonical baseline

Sampling noise is real (K2.6 shows ~30% of tasks flipping between runs).
Canonical `runs/baseline-*/` are **best-of-N merges** across 2-3 runs with
`source_run` column tracking provenance. New baseline runs supersede when
maintainers agree the deployment/pins have shifted.

## What we don't do

- **Local mode (`--remote-host local`)** — no Docker-Desktop / Rosetta path;
  runs always execute on a remote Linux box
- **Rich UI / dashboard** — everything is TSV + Markdown, machine-diffable
- **Continuous integration** — not built to run on every PR; runs are
  minutes-to-hours and require GPU-adjacent servers

## Related project layout

```
kbench                Python CLI entry point
src/
├── config.py         pinned versions, task lists, timeout overrides
├── user_config.py    ~/.kbench.json loader + api-key ref resolver
├── preset.py         presets/*.json loader
├── setup_cli.py      kbench init / doctor
├── probe/            14-case protocol probe
├── runner/           setup+smoke orchestration
│   └── templates/    bash + python vendor agent templates
├── triage/           4-bucket auto-classifier
└── compare/          side-by-side run diff
presets/*.json        model presets (add new models here)
docs/                 quickstart / test-matrix / this file
runs/                 per-run artifacts + canonical baselines
```
