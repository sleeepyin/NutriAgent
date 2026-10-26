#!/bin/bash
# Detached runner for MetaFood3D, ECUSTFD, and NV-Real. Ignore SIGHUP.
set -u
trap '' HUP
ROOT=/home/sdnu/nutriagent
cd "$ROOT"
mkdir -p logs results/diagnosis/d1_oracle results/diagnosis/d2_density results/diagnosis/d4_ecustfd results/diagnosis/d5_nvreal
echo $$ > logs/other_datasets.pid
exec 9>/var/tmp/other_datasets.lock
if ! flock -n 9; then
  echo "$(date '+%F %T') already running" >> logs/other_datasets_status.txt
  exit 0
fi

status() { echo "$(date '+%F %T') $*" | tee -a logs/other_datasets_status.txt; }

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_HOME=/var/tmp/torch-cache
mkdir -p "$TORCH_HOME"

PY=/var/tmp/nutri-venv/bin/python
MIRROR=https://mirrors.aliyun.com/pypi/simple/
status "start pid=$$ gpu=${CUDA_VISIBLE_DEVICES}"
if ! "$PY" -c 'import xlrd' >/dev/null 2>&1; then
  status "install xlrd"
  "$PY" -m pip install --prefix /var/tmp/nutri-venv --trusted-host mirrors.aliyun.com xlrd -i "$MIRROR"
fi

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

run D1 experiments/diagnosis/d1_mf3d_oracle/oracle.py
run D4 experiments/diagnosis/d4_ecustfd_scale/reference_scale.py
run D5 experiments/diagnosis/d5_nvreal_shift/run_d5.py
run D2 experiments/diagnosis/d2_mf3d_density/learned_density.py
status "other datasets finished"
