#!/usr/bin/env bash
# 轮询 43.179.238.173:2229 就绪（连续6次200）后自动跑 K3 阶段1预检（opensource 模式）
set -u
cd /Users/f/Documents/Kimi-Vendor-Verifier
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/preflight_vllm_$TS"
mkdir -p "$LOGDIR"
ln -sfn "$LOGDIR" logs/preflight_vllm_latest
STATUS="$LOGDIR/STATUS.txt"

BASE="http://43.179.238.173:2229"
export KIMI_BASE_URL="$BASE/v1"
export KIMI_API_KEY="EMPTY"
export MODEL_NAME="/nvme1/models/Kimi-K3"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$STATUS"; }

log "开始轮询健康状态（连续6次200才开跑）..."

streak=0
while true; do
  code=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null || echo ERR)
  if [ "$code" = "200" ]; then
    streak=$((streak+1)); echo "$(date '+%H:%M:%S') healthy $streak/6" >> "$STATUS"
    [ "$streak" -ge 6 ] && break
  else
    [ "$streak" -gt 0 ] && log "健康状态中断，重新计数"
    streak=0; echo "$(date '+%H:%M:%S') unhealthy (HTTP $code)" >> "$STATUS"
  fi
  sleep 30
done

log "服务稳定就绪 6/6，开始预检"

run_suite() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    tail -1 "$LOGDIR/$name.log" >> "$STATUS"
    log "DONE  $name"
  else
    log "FAIL  $name (exit $?)"
  fi
  code=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null || echo ERR)
  if [ "$code" != "200" ]; then
    log "ABORT: 服务在 $name 后又不健康 (HTTP $code)，停止后续套件"
    exit 1
  fi
}

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
