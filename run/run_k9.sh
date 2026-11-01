#!/bin/bash
# K=9 重复解码稳健性实验：m1 (qwen2.5:14b) V0 提示, temperature=0.7, 9 次独立跑批
# 产物: ${EXP_ROOT:-/data/result/exp}/method_m1_k1 ... method_m1_k9
cd ${CODE_DIR:-/data/exp}
for k in 1 2 3 4 5 6 7 8 9; do
  echo "===== RUN k=$k start $(date +%H:%M:%S) ====="
  python3 theory_dag_exp.py --method m1 --temperature 0.7 --suffix _k${k} --force
done
echo "===== K9_ALL_DONE $(date +%H:%M:%S) ====="
