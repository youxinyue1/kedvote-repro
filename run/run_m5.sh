#!/usr/bin/env bash
# M5（V2 提示增强）+ B 轴最小消融集（role_off / cot_off）串行跑批
set -uo pipefail
LOG=${EXP_ROOT:-/data/result/exp}/run_m5.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "==== M5 + B轴消融开始 ===="
for m in m5 m5_role_off m5_cot_off; do
  log "---- $m ----"
  python3 ${CODE_DIR:-/data/exp}/theory_dag_m5.py --method "$m" \
    --root ${EXP_ROOT:-/data/result/exp} --layers ${CODE_DIR:-/data/exp}/node_layers.json 2>&1 | tee -a "$LOG" \
    && log "---- $m 完成 ----" || log "!! $m 异常"
done
log "==== M5 全部完成 ===="
for m in m5 m5_role_off m5_cot_off; do
  log "$m: $(tail -n +2 ${EXP_ROOT:-/data/result/exp}/method_$m/edges.csv 2>/dev/null | wc -l) 条边"
done
