#!/bin/bash
# run_all_v4.sh —— v3.3 六模型 V0 逐节点回忆跑批（b 版：修复预检 heredoc）
set -u
ROOT=${EXP_ROOT:-/data/result/exp}
LOG=$ROOT/run_all_v4.log
mkdir -p "$ROOT"
exec >>"$LOG" 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

log "================ run_all_v4 (b) 启动 ================"
python3 ${CODE_DIR:-/data/exp}/preflight.py
[ $? -ne 0 ] && { log "!! 预检失败，中止"; exit 1; }
log "预检通过"

ORDER="m9c m9b m8b m9 m10"
for m in $ORDER; do
  log "---- method_$m 开始 ----"
  python3 ${CODE_DIR:-/data/exp}/theory_dag_exp.py --method "$m" --root "$ROOT" --layers ${CODE_DIR:-/data/exp}/node_layers.json
  log "---- method_$m 完成 ----"
done
log "================ run_all_v4 (b) 结束 ================"
