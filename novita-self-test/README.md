# Novita Self Tests

This directory is the only home for Novita-owned additions to the KVV fork.
Files outside this directory are treated as Moonshot upstream content.

## Ownership

- `tests/interface/`: K3 request-shape, observability, image-limit, and rate-limit probes.
- `tests/streaming/`: Streaming Output Spec and stream-lifecycle diagnostics.
- `tests/unit/`: offline fixture and replay-runner integrity tests.
- `fixtures/`: byte-verified K3 image and OpenClaw Feishu attachments.
- `tools/`: Novita replay, metrics, observability, and stream diagnostic helpers.
- `validations/`: retained legacy standalone validation entrypoints.
- `references/`: source/reference copies that are not executable source of truth.
- `resources/beam-1m/`: recovered BEAM data and provenance used when upstream LFS
  is unavailable.
- `scripts/`: Novita endpoint-specific preflight entrypoints.
- `evidence/`: ignored local outputs; never acceptance source code.

Kimi Code Bench does not live here. Use `/Users/f/Documents/novita-kimi-code-bench`.
deepSWE execution and cross-suite acceptance reporting are coordinated by
`/Users/f/Documents/ppio-workspace/tasks/kimi-k3-vendor-acceptance`.

## Run

Official KVV suite with Novita artifact recording:

```bash
export PYTHONPATH="$PWD/novita-self-test:$PWD"
uv run pytest -p pytest_artifacts tests/params \
  --artifact-dir /path/to/run/params ...
```

Novita self tests:

```bash
export PYTHONPATH="$PWD/novita-self-test:$PWD"
uv run pytest novita-self-test/tests \
  --artifact-dir /path/to/run/novita ...
```

Fixture replay:

```bash
uv run python novita-self-test/tools/replay_fixture.py image \
  --fixture novita-self-test/fixtures/vendor-img-testcases-inhouse-3.jsonl \
  --output-dir /path/to/run/image ...
```

The acceptance console reads these artifacts; it does not own or duplicate these tests.

## BEAM LFS fallback

The upstream `beam/data/beam_1m_*` paths remain Moonshot LFS pointers until
`git lfs pull upstream` succeeds. The byte-verified fallback is tracked as Novita LFS
under `resources/beam-1m/`. Use it explicitly without changing upstream paths:

```bash
python beam/beam_generate.py \
  --chats-file novita-self-test/resources/beam-1m/beam_1m_chats.jsonl.gz \
  --questions-file novita-self-test/resources/beam-1m/beam_1m_questions.jsonl \
  ...
```
