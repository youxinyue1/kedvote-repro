#!/usr/bin/env bash
# ============================================================
# 因果推断实验 · 推理服务管理脚本（远程 ${CODE_DIR:-/data/exp}/servers.sh）
#
# 单卡 RTX4090 24G，M1-M4 无法同时热驻 -> 同一时刻仅启动 1 个模型，
# start 某档会自动 stop 其它档（互斥切换），保证顺序批处理不 OOM。
#
# 端口固定（写死在调用脚本里，切换模型不改 URL）:
#   m1=18081  Qwen2.5-14B-Instruct Q4_K_M   (llama-server)
#   m2=18082  Qwen2.5-7B-Instruct  Q4_K_M   (llama-server)
#   m3=18083  Qwen2.5-3B-Instruct  Q4_K_M   (llama-server)
#   m4=18084  Zhongjing-LLaMA-base 13B fp16 8-bit (zserver.py + transformers)
#
# 用法: bash servers.sh status
#       bash servers.sh start   m1|m2|m3|m4
#       bash servers.sh stop    m1|m2|m3|m4|all
#       bash servers.sh restart m1|m2|m3|m4
# 产物: pid/log 在 /data/result/servers/（随 pull /data/result 一起备份）
# ============================================================
set -u
BASE=/data/result/servers
mkdir -p "$BASE"
export LD_LIBRARY_PATH=/data/llama/bin:/usr/local/cuda/lib64
LLAMA_BIN=/data/llama/bin/llama-server
ZPY=/data/miniconda/bin/python
ZSERVER=/data/zserver.py
IDS="m1 m2 m3 m4"

declare -A PORT=( [m1]=18081 [m2]=18082 [m3]=18083 [m4]=18084 )
declare -A MODEL=(
  [m1]="/data/models/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
  [m2]="/data/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
  [m3]="/data/models/qwen2.5-3b-instruct-q4_k_m.gguf"
  [m4]="/data/models/Zhongjing-LLaMA-base"
)
declare -A CTX=( [m1]=8192 [m2]=8192 [m3]=8192 [m4]=0 )

pfx()  { echo "$BASE/$1.pid"; }
lgx()  { echo "$BASE/$1.log"; }

isup() {
  local p
  p=$(cat "$(pfx "$1")" 2>/dev/null) || return 1
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null
}

stop_one() {
  local id=$1 p
  if isup "$id"; then
    p=$(cat "$(pfx "$id")")
    echo "stopping $id (pid $p)"
    kill "$p" 2>/dev/null
    for _ in $(seq 1 40); do kill -0 "$p" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$p" 2>/dev/null
  else
    echo "$id not running"
  fi
  rm -f "$(pfx "$id")"
}

launch_one() {
  local id=$1
  local port=${PORT[$id]} tmo=420 i=0 code
  if isup "$id"; then
    echo "$id already running (pid $(cat "$(pfx "$id")"))"
    return 1
  fi
  # 单卡互斥：先停掉其它所有档
  local x
  for x in $IDS; do [ "$x" = "$id" ] || stop_one "$x"; done
  if [ "$id" = "m4" ]; then
    nohup env "$ZPY" -u "$ZSERVER" --host 127.0.0.1 --port "$port" --model "${MODEL[$id]}" \
      > "$(lgx "$id")" 2>&1 &
  else
    nohup env "$LLAMA_BIN" -m "${MODEL[$id]}" -c "${CTX[$id]}" \
      --host 127.0.0.1 --port "$port" -np 1 -b 1024 \
      > "$(lgx "$id")" 2>&1 &
  fi
  echo $! > "$(pfx "$id")"
  echo "$id launching (pid $!, port $port, log $(lgx "$id")) ..."
  while [ $i -lt $((tmo * 2)) ]; do
    if ! kill -0 "$(cat "$(pfx "$id")" 2>/dev/null)" 2>/dev/null; then
      echo "$id FAILED: process exited early"
      tail -40 "$(lgx "$id")"
      rm -f "$(pfx "$id")"
      return 1
    fi
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 2 "http://127.0.0.1:$port/health" 2>/dev/null)
    if [ "$code" = "200" ]; then
      echo "$id READY (health 200, waited $((i/2))s)"
      return 0
    fi
    sleep 0.5
    i=$((i + 1))
    if [ $((i % 40)) -eq 0 ]; then echo "  ... still loading ($((i/2))s)"; fi
  done
  echo "$id TIMEOUT after ${tmo}s"
  return 1
}

case "${1:-}" in
  status)
    echo "== model services status =="
    for id in $IDS; do
      if isup "$id"; then
        code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:${PORT[$id]}/health" 2>/dev/null)
        echo "$id  RUNNING  pid=$(cat "$(pfx "$id")")  port=${PORT[$id]}  /health=$code"
      else
        echo "$id  stopped"
      fi
    done
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/GPU memory: /'
    ;;
  start)
    [ -n "${2:-}" ] || { echo "usage: servers.sh start m1|m2|m3|m4"; exit 2; }
    launch_one "$2"
    ;;
  stop)
    if [ "${2:-}" = "all" ]; then
      for id in $IDS; do stop_one "$id"; done
      echo "all stopped"
    elif [ -n "${2:-}" ]; then
      stop_one "$2"
    else
      echo "usage: servers.sh stop m1|m2|m3|m4|all"; exit 2
    fi
    ;;
  restart)
    [ -n "${2:-}" ] || { echo "usage: servers.sh restart m1|m2|m3|m4"; exit 2; }
    stop_one "$2"
    launch_one "$2"
    ;;
  *)
    echo "usage: servers.sh {status|start <m1|m2|m3|m4>|stop <m1|m2|m3|m4|all>|restart <m1|m2|m3|m4>}"
    exit 2
    ;;
esac
