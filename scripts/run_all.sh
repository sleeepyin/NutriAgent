#!/bin/bash
# One reproduction sequence on physical GPU 1. Safe to re-enter after a crash.
set -euo pipefail
cd /home/sdnu/nutriagent-digi-v4-repro
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
PY=/home/sdnu/nutriagent-digi-v4-repro/.venv/bin/python
if [[ -f results/repro/holdout_metrics.json ]]; then
  echo "holdout already recorded"
  exit 0
fi
wait_file() {
  local dest="$1"
  local minimum="$2"
  local url="$3"
  local ticks=0
  local resumes=0
  while [[ ! -f "$dest" ]] || [[ "$(stat -c%s "$dest")" -lt "$minimum" ]]; do
    if ! pgrep -f "wget .*$(basename "$dest")" >/dev/null; then
      resumes=$((resumes + 1))
      if [[ "$resumes" -gt 5 ]]; then
        echo "weight download died: $dest" >&2
        exit 1
      fi
      echo "resuming $(basename "$dest")" >&2
      wget -c --timeout=60 --tries=5 -O "$dest" "$url" >> "outputs/logs/dl_$(basename "$dest").log" 2>&1 &
    fi
    ticks=$((ticks + 1))
    if [[ "$ticks" -gt 180 ]]; then
      echo "timed out waiting for $dest" >&2
      exit 1
    fi
    sleep 20
  done
}
mkdir -p outputs/weights outputs/logs
wait_file outputs/weights/swin_base_patch4_window7_224.pth 352792251 \
  "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth"
"$PY" scripts/train_swin.py
wait_file outputs/weights/jx_vit_base_resnet50_384.pth 395854632 \
  "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth"
"$PY" scripts/train_dpt.py
if [[ ! -f outputs/weights/food2k_resnet101.pth ]] || [[ "$(stat -c%s outputs/weights/food2k_resnet101.pth)" -lt 50000000 ]]; then
  if pgrep -f "wget .*food2k_resnet101.pth" >/dev/null; then
    wait_file outputs/weights/food2k_resnet101.pth 186927881 \
      "https://hf-mirror.com/T0MYYY/dpf-nutrition/resolve/main/food2k_resnet101.pth"
  fi
fi
"$PY" scripts/train_dpf.py
"$PY" scripts/evaluate_once.py
echo "reproduction sequence finished"
