#!/bin/bash
# Detached Nutrition5K R0 runner. Ignore SIGHUP so closing Cursor does not stop it.
set -euo pipefail
trap '' HUP
ROOT=/home/sdnu/nutriagent
cd "$ROOT"
mkdir -p logs results/estimation/r0 data/nutrition5k/cache data/nutrition5k/raw/yolo_crops
echo $$ > logs/n5k_r0.pid
# Home is NFS and rejects flock. Keep the lock on the local disk.
exec 9>/var/tmp/n5k_r0.lock
if ! flock -n 9; then
  echo "$(date '+%F %T') already running" >> logs/n5k_r0_status.txt
  exit 0
fi

status() { echo "$(date '+%F %T') $*" | tee -a logs/n5k_r0_status.txt; }

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export TORCH_HOME=/var/tmp/torch-cache
mkdir -p "$TORCH_HOME"

status "start pid=$$ gpu=${CUDA_VISIBLE_DEVICES}"
/usr/bin/python3 experiments/estimation/build_n5k_cache.py &
CACHE_PID=$!

VENV=/var/tmp/nutri-venv
if [[ ! -x "$VENV/bin/python" ]]; then
  /usr/bin/python3 -m venv --system-site-packages --without-pip "$VENV"
fi
# pypi.org is unreachable over IPv6 from this machine. Aliyun resolves on IPv4.
MIRROR=https://mirrors.aliyun.com/pypi/simple/
export PIP_DEFAULT_TIMEOUT=120
export PIP_DISABLE_PIP_VERSION_CHECK=1
PY="$VENV/bin/python"
pip_prefix() { "$PY" -m pip install --prefix "$VENV" --trusted-host mirrors.aliyun.com "$@"; }
# Driver is CUDA 12.8. A newer cu130 wheel imports but reports cuda unavailable.
# pypi.org is IPv6-unreachable here; Aliyun has the cu128 wheels.
WHEEL_DIR=/var/tmp/torch-wheels
mkdir -p "$WHEEL_DIR"
fetch_wheel() {
  local url="$1" dest="$2"
  wget -c --timeout=60 --read-timeout=180 --tries=50 --waitretry=10 --retry-connrefused \
    -O "$dest" "$url"
}
cuda_ok() {
  "$PY" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1
}
if ! cuda_ok; then
  status "install torch 2.11.0+cu128 from Aliyun"
  TORCH_WHL="$WHEEL_DIR/torch-2.11.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
  VISION_WHL="$WHEEL_DIR/torchvision-0.26.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
  fetch_wheel "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl" "$TORCH_WHL"
  fetch_wheel "https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.26.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl" "$VISION_WHL"
  "$PY" -m pip uninstall -y torch torchvision >/dev/null 2>&1 || true
  pip_prefix --force-reinstall "$TORCH_WHL" "$VISION_WHL" --extra-index-url "$MIRROR"
fi
# The venv sees system pandas 1.5, which cannot load against numpy 2.2.
pip_prefix 'numpy>=1.26,<2.3' 'scipy>=1.11' 'scikit-learn>=1.3' 'pandas>=2.2,<3' pillow joblib matplotlib -i "$MIRROR"
"$PY" -c 'import torch,pandas,sklearn; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count(), "pandas", pandas.__version__); assert torch.cuda.is_available()'

status "waiting for cache pid=${CACHE_PID}"
wait "$CACHE_PID"
status "cache ready, start training"
"$VENV/bin/python" experiments/estimation/run_r0.py
status "R0 finished"
