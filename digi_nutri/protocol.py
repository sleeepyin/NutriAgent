"""Frozen Plan A protocol. Fixed before training."""

from __future__ import annotations

NUTRIENTS = ("energy", "fat", "carb", "protein")
HEADLINE_METRICS = ("mae", "dataset_pmae", "plate_pmae")

IMAGE_SIZE = 384
EPOCHS = 60
OPTIMIZER = "AdamW"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.9, 0.999)
SCHEDULER = "cosine"
ETA_MIN = 0.0
HFLIP_P = 0.5
SEED = 20261125

SMOOTH_L1_BETA = 1.0
MASS_FLOOR_G = 5.0
LOSS_MASS_WEIGHT = 0.5
LOSS_DENSITY_WEIGHT = 0.5

RAW_UNIT_MM = 0.1
HEIGHT_MAX_MM = 150.0
BORDER_PIXELS = 1
MORPH_CLOSE_K = 3

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EXPECTED_TEST_N = 507
PARAM_LIMIT = 60_000_000
RGB_BACKBONE = "resnet101"
HEIGHT_BACKBONE = "resnet18"
CAB_REDUCTION = 16
GAMMA_INIT = 0.0
HEAD_HIDDEN = 256
TOKEN_DIM = 256
CAB_ATTN_DIM = 256
MAX_INGREDIENTS = 48

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261125
BOOTSTRAP_CI = (2.5, 97.5)
RESHOOT_MAX_GAP_SECONDS = 120
RESHOOT_NUTRITION_ATOL = 1e-3

PLATE_PMAE_RULE = "mean of 100*|error|/|y| over plates with |y| > 0"
DATASET_PMAE_RULE = "100 * sum(|error|) / sum(|y|)"

FROZEN_B1_TRAIN_MAE = {
    "energy": 11.6004,
    "fat": 0.5260,
    "carb": 1.0343,
    "protein": 0.6469,
}
FROZEN_B1_TRAIN_MASS_MAE = 6.8761
FROZEN_TING_TRAIN_MAE = {
    "energy": 6.2960,
    "fat": 0.3339,
    "carb": 0.8170,
    "protein": 0.4493,
}
FROZEN_B1_PARAMS = 43_550_533
FROZEN_TING_PARAMS = 44_877_893
FROZEN_SIGMA_MEAN = (
    254.84454729738562,
    12.699703116920844,
    19.336471637981116,
    18.096426388162673,
)
FROZEN_SIGMA_MASS = 218.04175744371824

RGB_ROOT = "/home/sdnu/nutriagent/data/nutrition5k/imagery/realsense_overhead"
DEPTH_ROOT = (
    "/home/sdnu/nutriagent-digi/data/official_depth/"
    "nutrition5k_dataset/imagery/realsense_overhead"
)
META_DIR = "/home/sdnu/nutriagent/data/nutrition5k/metadata"
SPLIT_DIR = "/home/sdnu/nutriagent/data/nutrition5k/dish_ids/splits"
CACHE_DIR = "/host/var/tmp/digi-nutri-plan-a-cache"
B1_CHECKPOINT = "/home/sdnu/nutriagent-digi-v2/outputs/checkpoints/b1.pt"
TING_CHECKPOINT = "/home/sdnu/nutriagent-digi-v2/outputs/checkpoints/t_ing.pt"

STATUS_COMMIT = "3880898533880c75c3bc772225fc93d76940ac04"
FROZEN_WEIGHT_COMMIT = "d28b89a58aa82666b0543c0caad63a4a8c1462a4"

AUDIT_PATH = "results/plan_a/d_table_audit.json"
TRAIN_METRICS = "results/plan_a/train_metrics.json"
HOLDOUT_METRICS = "results/plan_a/holdout_metrics.json"
PLATE_GROUPS_PATH = "results/plan_a/plate_groups_test.json"
CONFIG_PATH = "configs/plan_a_frozen.json"
VOCAB_PATH = "configs/ingredient_vocab.json"
REPORT_PATH = "results/plan_a/REPORT.md"

PASS = "q0_pass"
FAIL = "q0_fail_stop_wait_triste"
TRAIN_FAIL = "train_gate_fail_stop"
TRAIN_PASS = "train_gate_pass"
