"""Pinned versions / defaults for Kimi Code Bench.

Single source of truth — bump these only when re-validating the pinned
combo against a known-good run.
"""

from __future__ import annotations

# Pinned upstream versions
KIMI_CLI_REF = "2c34efb"                     # MoonshotAI/kimi-cli @ kimi-cli 1.48.0
HARBOR_VERSION = "0.5.0"                      # laude-institute/harbor
TERMINAL_BENCH_2_REF = "53ff2b87d621bdb97b455671f2bd9728b7d86c11"

# Default remote sandbox — INTENTIONALLY EMPTY.
# Team members configure their own test server via `./kbench init`, which
# writes ~/.kbench.json. See docs/quickstart.md.
# `./kbench doctor` will fail with a clear message if the config is missing.
DEFAULT_REMOTE = {
    "ssh_user_host": "",
    "ssh_port": 22,
    "workdir": "",
    "container_proxy": "",
}

# Default 15 smoke tasks (matches kimi-cli/tests_ai/accuracy_smoke/
# terminal_bench_2_tasks_default.txt). The "smoke" task set for fast CI signal.
DEFAULT_TASKS = [
    "fix-git",
    "regex-log",
    "cancel-async-tasks",
    "sqlite-db-truncate",
    "build-cython-ext",
    "git-leak-recovery",
    "sanitize-git-repo",
    "fix-code-vulnerability",
    "configure-git-webserver",
    "query-optimize",
    "polyglot-c-py",
    "polyglot-rust-c",
    "nginx-request-logging",
    "headless-terminal",
    "pypi-server",
]

# Full 89-task set from terminal_bench_2_cache. Includes everything in
# DEFAULT_TASKS plus 74 more (GPU-heavy / long-running / domain-specific).
# Use --tasks extended to run the whole set; expect 8-20 hours for full pass.
EXTENDED_TASKS = [
    "adaptive-rejection-sampler",
    "bn-fit-modify",
    "break-filter-js-from-html",
    "build-cython-ext",
    "build-pmars",
    "build-pov-ray",
    "caffe-cifar-10",
    "cancel-async-tasks",
    "chess-best-move",
    "circuit-fibsqrt",
    "cobol-modernization",
    "code-from-image",
    "compile-compcert",
    "configure-git-webserver",
    "constraints-scheduling",
    "count-dataset-tokens",
    "crack-7z-hash",
    "custom-memory-heap-crash",
    "db-wal-recovery",
    "distribution-search",
    "dna-assembly",
    "dna-insert",
    "extract-elf",
    "extract-moves-from-video",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "filter-js-from-html",
    "financial-document-processor",
    "fix-code-vulnerability",
    "fix-git",
    "fix-ocaml-gc",
    "gcode-to-text",
    "git-leak-recovery",
    "git-multibranch",
    "gpt2-codegolf",
    "headless-terminal",
    "hf-model-inference",
    "install-windows-3.11",
    "kv-store-grpc",
    "large-scale-text-editing",
    "largest-eigenval",
    "llm-inference-batching-scheduler",
    "log-summary-date-ranges",
    "mailman",
    "make-doom-for-mips",
    "make-mips-interpreter",
    "mcmc-sampling-stan",
    "merge-diff-arc-agi-task",
    "model-extraction-relu-logits",
    "modernize-scientific-stack",
    "mteb-leaderboard",
    "mteb-retrieve",
    "multi-source-data-merger",
    "nginx-request-logging",
    "openssl-selfsigned-cert",
    "overfull-hbox",
    "password-recovery",
    "path-tracing",
    "path-tracing-reverse",
    "polyglot-c-py",
    "polyglot-rust-c",
    "portfolio-optimization",
    "protein-assembly",
    "prove-plus-comm",
    "pypi-server",
    "pytorch-model-cli",
    "pytorch-model-recovery",
    "qemu-alpine-ssh",
    "qemu-startup",
    "query-optimize",
    "raman-fitting",
    "regex-chess",
    "regex-log",
    "reshard-c4-data",
    "rstan-to-pystan",
    "sam-cell-seg",
    "sanitize-git-repo",
    "schemelike-metacircular-eval",
    "sparql-university",
    "sqlite-db-truncate",
    "sqlite-with-gcov",
    "torch-pipeline-parallelism",
    "torch-tensor-parallelism",
    "train-fasttext",
    "tune-mjcf",
    "video-processing",
    "vulnerable-secret",
    "winning-avg-corewars",
    "write-compressor",
]

# Network mirrors used when remote host is in PRC (china-friendly defaults).
# Off when remote host reaches GitHub / PyPI directly.
DEFAULT_MIRRORS = {
    "gh_prefix": "https://ghfast.top/",
    "pypi_index": "https://pypi.tuna.tsinghua.edu.cn/simple/",
}

# Harbor timeouts (multipliers vs Harbor's built-in defaults).
HARBOR_AGENT_SETUP_TIMEOUT_MUL = 4
HARBOR_AGENT_TIMEOUT_MUL = 4

# L2 stability fix: tasks that historically time out at multiplier=4. The
# vendor script reads this dict and uses the per-task override instead of
# the global HARBOR_AGENT_TIMEOUT_MUL.
# Evidence (as of 2026-06-30):
#   - polyglot-rust-c: env-err in v5, v6, v7 first, v7 retry (even at mul=6).
#     Single-task runtime on K2.7 was ~27 min, and K2.6 is slower.
TASK_TIMEOUT_OVERRIDES = {
    "polyglot-rust-c": 10,
}
