#!/bin/bash
# MM-CFA 一键复现 (主管线部分; 跨域/外部数据集见 README §4B/§4C)
# 前置: 解压到 /home 且 conda env nutri 已激活; GPU 可选 (默认跳过重训, 用包内权重)
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}

echo "=== [1/7] 生产推理 + 区间验证 (test_100_samples_final) ==="
(cd Nutri_0811 && $PY test_100_samples_final.py)

echo "=== [2/7] 重量共形基线 (conformal_deep_experiment_final) ==="
(cd Nutri_0811 && $PY conformal_deep_experiment_final.py)

echo "=== [3/7] AEC-S 诚实协议 + 重标定变体 (aec_s_v2_experiment) ==="
(cd Nutri_0811 && $PY aec_s_v2_experiment.py)

echo "=== [4/7] 营养素共形 AEC-S2m-N (nutrient_conformal_final) ==="
(cd Nutri_0811 && $PY nutrient_conformal_final.py)

echo "=== [5/7] CQR 基线对比 (nutrient_cqr_experiment) ==="
(cd Nutri_0811 && $PY nutrient_cqr_experiment.py)

echo "=== [6/7] 加权共形基线 (nutrient_weighted_conformal_experiment) ==="
(cd Nutri_0811 && $PY nutrient_weighted_conformal_experiment.py)

echo "=== [7/7] 论文图表生成 (make_paper_figures) ==="
(cd Nutri_0811 && $PY make_paper_figures.py)

echo ""
echo "=== DONE — 预期数字对照 README §4A ==="
echo "  生产推理: coverage ~91.1% | AEC-S2m: 90.4%/76g/Winkler 171"
echo "  AEC-S2m-N: 89.7-90.6%, nWinkler 2.42 | CQR best: 2.55 (被支配)"
echo "  加权共形: 被 AEC-S2m-5W 支配 | 图表: Nutri_0811/figures/ ×5"
