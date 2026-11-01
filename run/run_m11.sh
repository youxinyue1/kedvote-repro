#!/bin/bash
# run_m11.sh —— m11 Yi-1.5-34B V0 逐节点回忆跑批（替换失败的 Baichuan2-13B 槽位）
set -u
ROOT=${EXP_ROOT:-/data/result/exp}
LOG=$ROOT/run_m11.log
mkdir -p "$ROOT"
exec >>"$LOG" 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

log "================ run_m11 (yi:34b) 启动 ================"
# 预检：模型必须已完整拉取
ollama list | grep -q "yi:34b" || { log "!! yi:34b 未就绪，中止"; exit 1; }
log "预检通过：yi:34b 在位"

python3 ${CODE_DIR:-/data/exp}/theory_dag_exp.py --method m11 --root "$ROOT" --layers ${CODE_DIR:-/data/exp}/node_layers.json
[ $? -ne 0 ] && { log "!! m11 跑批异常退出"; exit 1; }
log "================ run_m11 (yi:34b) 结束 ================"
