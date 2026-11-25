"""Frozen same-setting rerun. Fixed before training and before the holdout."""

from __future__ import annotations

NUTRIENTS = ("energy", "fat", "carb", "protein")
MODELS = ("b1", "t_geo", "t_ing")
TEACHERS = ("t_geo", "t_ing")
HEADLINE_METRICS = ("mae", "dataset_pmae", "plate_pmae")

IMAGE_SIZE = 384
EPOCHS = 60
BATCH_SIZE = 32
OPTIMIZER = "AdamW"
HEAD_LR = 1e-4
BACKBONE_LR = 3e-5
WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.9, 0.999)
SCHEDULER = "cosine"
ETA_MIN = 0.0
DROPOUT_P = 0.3
HFLIP_P = 0.5
VFLIP_P = 0.5
ROTATION_DEGREES = (0, 90, 180, 270)
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
EXPECTED_TRAIN_N = 2754
PARAM_LIMIT = 60_000_000
RGB_BACKBONE = "resnet101"
HEIGHT_BACKBONE = "resnet18"
CAB_REDUCTION = 16
GAMMA_INIT = 0.0
HEAD_HIDDEN = 256
TOKEN_DIM = 256
CAB_ATTN_DIM = 256
MAX_INGREDIENTS = 48
SPP_GRIDS = (1, 2, 4)
VOLUME_DIM = 4 + 1 + 4 + 16

B1_STRUCTURE_PARAMS = 43_550_533
TING_STRUCTURE_PARAMS = 44_877_893
PLAN_A_TGEO_WITHOUT_VOLUME = 56_562_054

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261125
BOOTSTRAP_CI = (2.5, 97.5)
RESHOOT_MAX_GAP_SECONDS = 120
RESHOOT_NUTRITION_ATOL = 1e-3

PLATE_PMAE_RULE = "mean of 100*|error|/|y| over plates with |y| > 0"
DATASET_PMAE_RULE = "100 * sum(|error|) / sum(|y|)"

PASS = "q0_pass_stop_wait_triste"
FAIL = "q0_fail_stop_wait_triste"

RGB_ROOT = "/home/sdnu/nutriagent/data/nutrition5k/imagery/realsense_overhead"
DEPTH_ROOT = (
    "/home/sdnu/nutriagent-digi/data/official_depth/"
    "nutrition5k_dataset/imagery/realsense_overhead"
)
META_DIR = "/home/sdnu/nutriagent/data/nutrition5k/metadata"
SPLIT_DIR = "/home/sdnu/nutriagent/data/nutrition5k/dish_ids/splits"
CACHE_DIR = "/host/var/tmp/digi-nutri-plan-a-cache"

STATUS_COMMIT = "c0abef27e35285a5bebdab24168b5475506154a1"
TRAIN_DEVICE = "cuda:1"

TRAIN_EXCLUSION_IDS = (
    "dish_1558109714",
    "dish_1557862384",
    "dish_1558109511",
    "dish_1564159636",
)

CONFIG_PATH = "configs/rerun_frozen.json"
VOCAB_PATH = "configs/ingredient_vocab.json"
TRAIN_METRICS = "results/rerun/train_metrics.json"
HOLDOUT_METRICS = "results/rerun/holdout_metrics.json"
PLATE_GROUPS_PATH = "results/rerun/plate_groups_test.json"
HYGIENE_PATH = "results/rerun/split_hygiene.json"
REPORT_PATH = "results/rerun/REPORT.md"
PREDICTIONS_TRAIN = "results/rerun/predictions_train.csv"
PREDICTIONS_HOLDOUT = "results/rerun/predictions_holdout.csv"

TEXT_ENCODER = (
    "learned embedding of the static dictionary token plus a linear map of the "
    "USDA per-gram prior; the OpenAI CLIP package is not used"
)


def snapshot() -> dict:
    return {
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": OPTIMIZER,
        "head_lr": HEAD_LR,
        "backbone_lr": BACKBONE_LR,
        "fusion_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "betas": list(ADAM_BETAS),
        "scheduler": SCHEDULER,
        "eta_min": ETA_MIN,
        "dropout_p": DROPOUT_P,
        "dropout_where": "regression heads only",
        "hflip_p": HFLIP_P,
        "vflip_p": VFLIP_P,
        "rotation_degrees": list(ROTATION_DEGREES),
        "seed": SEED,
        "smooth_l1_beta": SMOOTH_L1_BETA,
        "mass_floor_g": MASS_FLOOR_G,
        "loss_mass_weight": LOSS_MASS_WEIGHT,
        "loss_density_weight": LOSS_DENSITY_WEIGHT,
        "raw_unit_mm": RAW_UNIT_MM,
        "height_max_mm": HEIGHT_MAX_MM,
        "border_pixels": BORDER_PIXELS,
        "morph_close": MORPH_CLOSE_K,
        "exclude_border_zeros": True,
        "nutrients": list(NUTRIENTS),
        "param_limit": PARAM_LIMIT,
        "rgb_backbone": RGB_BACKBONE,
        "height_backbone": HEIGHT_BACKBONE,
        "cab_reduction": CAB_REDUCTION,
        "gamma_init": GAMMA_INIT,
        "head_hidden": HEAD_HIDDEN,
        "volume_dim": VOLUME_DIM,
        "spp_grids": list(SPP_GRIDS),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_percentiles": list(BOOTSTRAP_CI),
        "reshoot_max_gap_seconds": RESHOOT_MAX_GAP_SECONDS,
        "reshoot_nutrition_atol": RESHOOT_NUTRITION_ATOL,
        "plate_pmae": PLATE_PMAE_RULE,
        "dataset_pmae": DATASET_PMAE_RULE,
        "pass_verdict": PASS,
        "fail_verdict": FAIL,
        "pass_rule": (
            "On the official 507-plate holdout, the 95% plate-grouped CI of "
            "(teacher - this run's B1) lies entirely below 0 for energy, fat, "
            "carb, and protein, on MAE, dataset PMAE, and per-plate PMAE. "
            "Both teachers are required. Mass is recorded and is not a pass "
            "condition. Either verdict stops. Distillation stays locked."
        ),
        "text_encoder": TEXT_ENCODER,
        "status_commit": STATUS_COMMIT,
        "max_ingredients": MAX_INGREDIENTS,
        "train_device": TRAIN_DEVICE,
        "comparator": "b1 trained in this same-setting run",
    }
