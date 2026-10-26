#!/bin/bash
# Detached runner. Start it with setsid so closing Cursor does not stop it.
set -u
cd /home/sdnu/nutriagent
mkdir -p logs cache/vlm results/vlm
exec >> logs/v2_remaining.log 2>&1
echo "===== $(date '+%F %T %Z') pid=$$ ====="
exec 9>/var/tmp/v2_remaining.lock
if ! flock -n 9; then
  echo "already running"
  exit 0
fi
echo $$ > logs/v2_remaining.pid
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/sdnu/nutriagent:/home/sdnu/nutriagent/src"
export QWEN_BASE="http://127.0.0.1:18080"
export QWEN_MODEL="qwen3.8-27b"
export QWEN_API_KEY_FILE="/home/sdnu/qwen38-vlm-api/api.key"
export CUDA_VISIBLE_DEVICES=""
export D6_FORCE_CPU=1
export D6_THREADS=16
export OMP_NUM_THREADS=16
PY="/var/tmp/nutri-venv/bin/python"

if [[ -f logs/d6_dinov2.pid ]] && kill -0 "$(cat logs/d6_dinov2.pid)" 2>/dev/null; then
  echo "dinov2 already running pid=$(cat logs/d6_dinov2.pid)"
elif [[ -f weights/dinov2/d6_dinov2_small_head.pth && -f results/diagnosis/d6_dinov2/summary.json ]]; then
  echo "dinov2 already finished"
else
  setsid "$PY" experiments/diagnosis/d6_capacity/dinov2_ablation.py --log1p --epochs 60 --bs 32 \
    >> logs/d6_dinov2.log 2>&1 9>&- < /dev/null &
  echo $! > logs/d6_dinov2.pid
  echo "started dinov2 pid=$(cat logs/d6_dinov2.pid)"
fi

tries=0
while [[ "$tries" -lt 40 ]]; do
  tries=$((tries + 1))
  echo "pipeline attempt ${tries}"
  set +e
  "$PY" -m experiments.run_v2_remaining
  rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    echo "pipeline finished"
    exit 0
  fi
  if [[ "$rc" -eq 1 ]]; then
    echo "pipeline fatal exit 1"
    exit 1
  fi
  echo "pipeline exit ${rc}; sleep 60"
  sleep 60
done
echo "pipeline gave up after ${tries} attempts"
exit 1
