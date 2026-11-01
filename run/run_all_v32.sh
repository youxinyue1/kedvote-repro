#!/usr/bin/env bash
# v3.2 全量跑批：五模型串行 V0 逐节点回忆（断点续跑，幂等）
# 顺序从快到慢：3b -> 7b -> 14b -> 沈仲Q8 -> 扁仓Q8
set -uo pipefail
cd ${CODE_DIR:-/data/exp}
LOG=${EXP_ROOT:-/data/result/exp}/run_all_v32.log
mkdir -p ${EXP_ROOT:-/data/result/exp}
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "==== run_all_v32.sh 开始（五模型 V0）===="

# 链路预检：单次小请求确认 Ollama 兼容端点活着
if ! curl -s --max-time 30 http://127.0.0.1:11434/v1/chat/completions \
     -d '{"model":"qwen2.5:3b","messages":[{"role":"user","content":"回答一个字：好"}],"max_tokens":8}' \
     | grep -q '"content"'; then
  log "!! Ollama /v1/chat/completions 预检失败，退出"
  exit 1
fi
log "链路预检通过"

for m in m3 m2 m1 m4b m4; do
  log "---- 开始 $m ----"
  if python3 ${CODE_DIR:-/data/exp}/theory_dag_exp.py --method "$m" \
       --root ${EXP_ROOT:-/data/result/exp} --layers ${CODE_DIR:-/data/exp}/node_layers.json 2>&1 | tee -a "$LOG"; then
    log "---- $m 完成 ----"
  else
    log "!! $m 异常退出（断点已保存，可重跑续传）"
  fi
done

log "==== 全部完成 ===="
for d in ${EXP_ROOT:-/data/result/exp}/method_m*; do
  n=$(tail -n +2 "$d/edges.csv" 2>/dev/null | wc -l)
  log "$d: $n 条边"
done
