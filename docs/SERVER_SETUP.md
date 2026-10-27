# 内网服务器最小搭建

面向：**2×4090D 24GB**、可访问外网下数据、通过 GitHub 与执行助手同步。

## 0. Clone 与依赖

```bash
git clone <YOUR_GITHUB_REPO_URL> vdr-agent
cd vdr-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 有 GPU / 要跑 VLM 时再装（按需取消注释 requirements 或）:
# pip install -e ".[vlm]"
```

## 1. 目录与环境变量

建议数据放在大盘，仓库内用符号链接或纯环境变量：

```bash
export VDR_ROOT=$PWD
export VDR_N5K_ROOT=/data/nutrition5k          # 自行修改
export VDR_FNDDS_PATH=/data/fndds/2019-2020/foods.json

mkdir -p "$VDR_N5K_ROOT"/{imagery,metadata,dish_ids} \
         /data/fndds/2019-2020
```

若直接使用仓库内已有元数据/catalog：

```bash
# 仓库已含 dish_ids + metadata + foods.json 时：
export VDR_N5K_ROOT=$PWD/data/nutrition5k
export VDR_FNDDS_PATH=$PWD/data/fndds/2019-2020/foods.json
# imagery 仍须下到 $VDR_N5K_ROOT/imagery/
```

## 2. 下载 Nutrition5K

官方：https://github.com/google-research-datasets/Nutrition5k  
许可：CC-BY 4.0（以官方 README 为准）。

### 2.1 全量包（约 181.4 GB，可选）

```bash
# HTTP
wget -c https://storage.cloud.google.com/nutrition5k_dataset/nutrition5k_dataset.tar.gz
# 或 gsutil
# gsutil -m cp gs://nutrition5k_dataset/nutrition5k_dataset.tar.gz .
tar -xzf nutrition5k_dataset.tar.gz -C /data/
```

### 2.2 推荐：只拉 overhead RGB-D（评测主路径常够用）

```bash
# 需安装 Google Cloud SDK / gsutil
gsutil -m cp -r \
  "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead" \
  "$VDR_N5K_ROOT/imagery/"
```

元数据与 split 若仓库已带，可不再下；否则：

```bash
gsutil -m cp -r \
  "gs://nutrition5k_dataset/nutrition5k_dataset/metadata" \
  "gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids" \
  "$VDR_N5K_ROOT/"
```

侧视视频 `imagery/side_angles/` 体积大，**MVP 可不下**。

## 3. FNDDS / catalog

**优先**：直接使用仓库内 `data/fndds/2019-2020/foods.json`（5624 条，~4MB）。

若需从官方重生成：

1. 从 USDA FNDDS / FoodData Central 下载 Survey Food JSON 或 At-a-Glance xlsx。  
2. 放入 `data/fndds/2019-2020/`。  
3. 运行：

```bash
python scripts/build_fndds_catalog.py
export VDR_FNDDS_PATH=$PWD/data/fndds/2019-2020/foods.json
```

**不要**把 66MB 原始 FDC JSON push 回 GitHub（已在 `.gitignore`）。

## 4. 冒烟命令（无 VLM 也可跑）

```bash
source .venv/bin/activate
python scripts/validate_schema_examples.py
python scripts/fndds_retrieve_smoke.py
python scripts/n5k_metadata_summary.py
python scripts/rule_verify_toy_smoke.py
python scripts/b1_toy_smoke.py
python scripts/eval_toy_smoke.py
```

期望：schema / rule / b1 / eval 通过；retrieve 显示 `n_foods=5624`；N5K 打印 rgb_train=4059、rgb_test=709。

imagery 就绪后：自行抽若干 `realsense_overhead/*/rgb.png`（文件名以实际为准）做 B1+VLM 联调（脚本后续经 GitHub 迭代）。

## 5. 2×4090D 上的 VLM 建议

| 建议 | 说明 |
|------|------|
| 模型 | **Qwen2.5-VL-7B**（或同级开源 7B–8B VL） |
| 精度 | bf16 / fp16；单卡 24GB 推理通常可负担 7B；两卡可数据并行评测 |
| 禁止 | 旧营养回归头/权重；默认关闭体积估计模块 |
| 输出 | 先只要成分假设列表（名称+粗份量），再走 FNDDS 工具，勿端到端只报 kcal |

显存紧张时：降 batch=1、关 flash 以外的缓存、不要同时载两个 7B。控制器训练用小 MLP，勿回传进 VLM。

本机仓库 `.venv` 没有 torch，不能用来跑 B0/B1。VLM 解释器是 `/var/tmp/nutri-venv/bin/python`。评测时用 `CUDA_VISIBLE_DEVICES=0`，让加载器的 `device_map="cuda:0"` 落在物理 GPU 0。不要把 `CUDA_VISIBLE_DEVICES` 设成 1。GPU 1 上若已有 llama-server，不要停。权重默认目录是 `/var/tmp/vdr-models/Qwen2.5-VL-7B-Instruct`。

## 6. 同步回 GitHub

```bash
mkdir -p results notes
# 写入 results/xxx.json 与 notes/YYYY-MM-DD.md
git add results notes docs/STATUS.md
git commit -m "[smoke] server env ok; n5k imagery partial"
git push
```

然后在聊天告知执行助手仓库 URL 或「看 main」。协议见 [`COLLAB_GITHUB.md`](COLLAB_GITHUB.md)。
