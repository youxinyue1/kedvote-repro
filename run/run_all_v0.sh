#!/usr/bin/env bash
# V0 逐节点回忆执行：切服务 -> 跑 -> 停，逐档串行
# 用法: bash run_all_v0.sh            # 默认 m1 m2 m3 m4
#       bash run_all_v0.sh m1         # 只跑 m1
#       bash run_all_v0.sh m2 m3 m4   # 指定若干档
if [ $# -eq 0 ]; then
  set -- m1 m2 m3 m4
fi
for m in "$@"; do
  echo "===== RUN $m $(date '+%F %T') ====="
  if ! bash ${CODE_DIR:-/data/exp}/servers.sh start "$m"; then
    echo "$m start FAILED"
    continue
  fi
  /data/miniconda/bin/python ${CODE_DIR:-/data/exp}/theory_dag_exp.py \
    --method "$m" --root ${EXP_ROOT:-/data/result/exp} --layers ${CODE_DIR:-/data/exp}/node_layers.json --force
  echo "$m rc=$?"
  bash ${CODE_DIR:-/data/exp}/servers.sh stop "$m"
done
echo "===== FINAL STATUS ====="
bash ${CODE_DIR:-/data/exp}/servers.sh status
echo RUN_ALL_DONE
