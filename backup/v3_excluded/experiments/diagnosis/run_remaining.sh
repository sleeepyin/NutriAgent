#!/bin/bash
# Detached runner for the experiments that were not in R0 or the first
# diagnosis pass. Ignore SIGHUP so closing Cursor does not stop it.
set -u
trap '' HUP
ROOT=/home/sdnu/nutriagent
cd "$ROOT"
mkdir -p logs results/appendix results/diagnosis/d5_nvreal results/diagnosis/d6_dinov2 \
  results/action data/nvreal/derived
echo $$ > logs/remaining.pid
exec 9>/var/tmp/remaining.lock
if ! flock -n 9; then
  echo "$(date '+%F %T') already running" >> logs/remaining_status.txt
  exit 0
fi

status() { echo "$(date '+%F %T') $*" | tee -a logs/remaining_status.txt; }

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_HOME=/var/tmp/torch-cache
mkdir -p "$TORCH_HOME"

PY=/var/tmp/nutri-venv/bin/python
status "start pid=$$ gpu=${CUDA_VISIBLE_DEVICES}"
status "SAM 3D skipped: third_party/sam3d and weights/sam3d/checkpoints are empty"

run() {
  local name="$1"
  shift
  status "start ${name}"
  if "$PY" "$@"; then
    status "done ${name}"
  else
    status "FAILED ${name} exit=$?"
  fi
}

run A1 -m experiments.action.a1_group_routing
run A2 -m experiments.action.a2_name_prior --skip-query
run A5 -m experiments.action.a5_oracle_bound --skip-query
run weighted-conformal -m experiments.appendix.weighted_conformal
run D5-remaining experiments/diagnosis/d5_nvreal_shift/run_d5_remaining.py

status "start D6"
d6_tmp=logs/d6_rerun.log
if "$PY" experiments/diagnosis/d6_capacity/dinov2_ablation.py --bs 32 --epochs 60 --log1p > "$d6_tmp" 2>&1; then
  cp "$d6_tmp" results/diagnosis/d6_dinov2/train_dino.log
  status "done D6"
else
  status "FAILED D6 exit=$? (transcript logs/d6_rerun.log)"
fi
status "remaining finished"
