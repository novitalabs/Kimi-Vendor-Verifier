#!/usr/bin/env bash
# Preflight for maiyi kimik3-k3-vendor deployment (GB300-task endpoint).
# Endpoint: public NodePort on host-172-16-0-18, image nightly-65b7662d3f,
# source branch kimi-k3-vendor-gb300 @ 65b7662d3f.
# Full K3 prompt-token preflight: 44 text cases + 11 vision cases = 55 cases.
set -u
cd /Users/f/Documents/Kimi-Vendor-Verifier
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/preflight_maiyi_gb300_$TS"
mkdir -p "$LOGDIR"
ln -sfn "$(pwd)/$LOGDIR" logs/preflight_maiyi_gb300_latest
STATUS="$LOGDIR/STATUS.txt"

export KIMI_BASE_URL="http://111.48.159.177:30004/v1"
export KIMI_API_KEY="a3f8b1c7e9d2a5f60b8c3d4e7f9a1b2c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a"
export MODEL_NAME="kimi-k3"

log() { echo "[$(date '+%H:%M:%S')] $*" >> "$STATUS"; }

health() {
  curl -s -m 10 "$KIMI_BASE_URL/models" | grep -q "\"$MODEL_NAME\"" \
    && log "HEALTH ok" || log "HEALTH FAIL"
}

run_suite() {
  local name="$1"; shift
  health
  log "START $name"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then tail -1 "$LOGDIR/$name.log" >> "$STATUS"; log "DONE  $name"
  else tail -1 "$LOGDIR/$name.log" >> "$STATUS"; log "FAIL  $name (exit $?)"; fi
}

log "开始预检 maiyi gb300 (111.48.159.177:30004, image nightly-65b7662d3f)"

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
