#!/bin/bash
# SAM 3D Objects 环境安装 (适配本服务器: github被墙→gh-proxy, 3090 24GB→compile=False)
# 运行: bash setup_sam3d.sh   (约 30-60 min, 主要是 pip 下载)
set -e
source /opt/miniconda3/etc/profile.d/conda.sh

echo "=== [1/6] 创建 conda 环境 sam3d (python 3.11) ==="
conda env list | grep -q "^sam3d " || conda create -n sam3d python=3.11 -y
conda activate sam3d

echo "=== [2/6] PyTorch 2.5.1 + cu121 ==="
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "=== [3/6] CUDA 工具链 (gsplat 编译需要) ==="
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-toolkit
conda install -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SAM3D_CODE="${SAM3D_CODE:-$ROOT/third_party/sam3d}"
SAM3D_WEIGHTS="${SAM3D_WEIGHTS:-$ROOT/weights/sam3d}"
if [[ ! -d "$SAM3D_CODE" ]]; then
  echo "把 SAM 3D 代码放到 $SAM3D_CODE 后再运行本脚本" >&2
  exit 1
fi

echo "=== [4/6] 打补丁: git 依赖走 gh-proxy, 剔除 dataclasses(会破坏 py311) ==="
cd "$SAM3D_CODE"
sed -i 's|git+https://github.com|git+https://gh-proxy.com/https://github.com|g' requirements.txt requirements.p3d.txt requirements.inference.txt
sed -i '/^dataclasses==0.6/d' requirements.txt
sed -i '/^flash_attn==/d' requirements.p3d.txt   # 默认 backend=sdpa, 无需 flash_attn

echo "=== [5/6] 安装 sam3d_objects + 依赖 ==="
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e .
# pytorch3d: 先试预编译 wheel, 失败则源码编译(约20min)
pip install "pytorch3d @ git+https://gh-proxy.com/https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" || \
  pip install --no-build-isolation "git+https://gh-proxy.com/https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
# gsplat: 源码构建必须 --no-build-isolation (隔离环境无 torch), 只编译 sm_86 (3090) 提速
export CUDA_HOME=/opt/miniconda3/envs/sam3d
export TORCH_CUDA_ARCH_LIST="8.6"
pip install --no-build-isolation "gsplat @ git+https://gh-proxy.com/https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"
pip install -e '.[inference]'

echo "=== [6/6] hydra patch (走 gh-proxy) + 权重链接 ==="
python - << 'PYEOF'
import hydra, urllib.request, os
assert hydra.__version__ == "1.3.2", f"hydra {hydra.__version__} != 1.3.2"
p = os.path.join(os.path.dirname(hydra.__file__), "core", "utils.py")
urllib.request.urlretrieve("https://gh-proxy.com/https://raw.githubusercontent.com/gleize/hydra/78f00766b5f37672aa7232ebbf01bdd74246bd60/hydra/core/utils.py", p)
print("hydra patched:", p)
PYEOF
mkdir -p checkpoints "$SAM3D_WEIGHTS"
[ -e checkpoints/hf ] || ln -s "$SAM3D_WEIGHTS/checkpoints" checkpoints/hf

echo ""
echo "=== 安装完成! 验证: ==="
SAM3D_CODE="$SAM3D_CODE" python -c "
import os, sys
code = os.environ['SAM3D_CODE']
sys.path.insert(0, code); sys.path.insert(0, os.path.join(code, 'notebook'))
os.environ['CUDA_HOME']=os.environ['CONDA_PREFIX']; os.environ['LIDRA_SKIP_INIT']='true'
from inference import Inference
print('import OK — 运行实验: conda activate sam3d && python experiments/diagnosis/d3_mf3d_sam3d/sam3d_volume.py --limit 10')
"
