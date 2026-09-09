#!/usr/bin/env bash
# Cloud Agent install for the MM-CFA / NutriAgent reproducibility package.
# Idempotent: safe to re-run. Installs the CPU dev toolchain into the user site
# so the plain `python3` interpreter can run the code and the agent_extension
# pipeline. GPU training (RTX 3090, CUDA 12.8) is out of scope for CPU agents;
# per requirements.txt "CPU inference works with any torch>=2.0".
set -euo pipefail

PY="${PYTHON:-python3}"

echo "[install] python: $($PY --version 2>&1) at $(command -v "$PY")"

# Repo-pinned scientific stack. The versions in requirements.txt were frozen on
# Python 3.10 but all publish cp312 wheels, so they install cleanly on the
# default (3.12) image as well.
$PY -m pip install --user --no-input \
  numpy==2.2.6 \
  scipy==1.15.3 \
  scikit-learn==1.7.2 \
  pandas==2.3.3 \
  pillow==12.3.0 \
  joblib==1.5.3 \
  matplotlib==3.10.9 \
  pytest

# requirements.txt pins torch==2.11.0+cu128 / torchvision==0.26.0+cu128 (GPU).
# Cloud Agents are CPU-only, so install the matching CPU builds from the
# official PyTorch CPU wheel index instead.
$PY -m pip install --user --no-input \
  torch==2.11.0 \
  torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cpu

echo "[install] verifying imports..."
$PY - <<'VERIFY'
import importlib
for m in ("numpy","scipy","sklearn","pandas","matplotlib","PIL","joblib","torch","torchvision"):
    mod = importlib.import_module(m)
    print(f"  {m:12s} {getattr(mod,'__version__','?')}")
VERIFY

echo "[install] done."
