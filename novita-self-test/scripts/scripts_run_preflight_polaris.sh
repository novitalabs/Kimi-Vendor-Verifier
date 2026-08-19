#!/usr/bin/env bash
set -u
cd /Users/f/Documents/Kimi-Vendor-Verifier
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/preflight_polaris_$TS"
mkdir -p "$LOGDIR"
ln -sfn "$(pwd)/$LOGDIR" logs/preflight_polaris_latest
STATUS="$LOGDIR/STATUS.txt"

export KIMI_BASE_URL="https://kimi-k3-2tw39pfk.polaris.ppio.com/v1"
export KIMI_API_KEY="a3f8b1c7e9d2a5f60b8c3d4e7f9a1b2c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a"
export MODEL_NAME="/data/models/moonshotai/Kimi-K3"

log() { echo "[$(date '+%H:%M:%S')] $*" >> "$STATUS"; }

run_suite() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    tail -1 "$LOGDIR/$name.log" >> "$STATUS"; log "DONE  $name"
  else
    tail -1 "$LOGDIR/$name.log" >> "$STATUS"; log "FAIL  $name (exit $?)"
  fi
}

log "开始预检 polaris"

run_suite params uv run pytest tests/params \
  --smoke-model "$MODEL_NAME" --think-mode opensource -ra -v

run_suite k3_features uv run pytest tests/k3_features \
  --smoke-model "$MODEL_NAME" --think-mode opensource -ra -v

run_suite prompt_tokens uv run pytest tests/prompt_tokens -n 4 \
  --smoke-model "$MODEL_NAME" --think-mode opensource -ra -v

run_suite tool_call_json_schema uv run pytest tests/tool_call_json_schema -n 2 \
  --base-url "$KIMI_BASE_URL" --api-key "$KIMI_API_KEY" \
  --smoke-model "$MODEL_NAME" --think-mode opensource \
  --thinking --reruns 2 --reruns-delay 2 \
  --tool-json-report="$LOGDIR/tool-call-schema-report.json" -ra -v

log "全部完成"
