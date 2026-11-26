"""Frozen T_multi run. Fixed before training and before the holdout."""

from __future__ import annotations

NUTRIENTS = ("energy", "fat", "carb", "protein")
MODEL = "t_multi"
HEADLINE_METRICS = ("mae", "dataset_pmae", "plate_pmae")
N_HEADLINE_INTERVALS = 12

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
EXPECTED_TEST_GROUPS = 507
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

# Counted on the untrained module. Training refuses a count that differs.
T_MULTI_PARAMS = 57_895_814
RGB_ENCODER_PARAMS = 42_500_160
FROZEN_B1_PARAMS = 43_550_533

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

STATUS_COMMIT = "8f7e0294cbe5543e72268d2ff2be3e46aa43982f"
TRAIN_DEVICE = "cuda:1"
PHYSICAL_GPU = "1"

TRAIN_EXCLUSION_IDS = (
    "dish_1558109714",
    "dish_1557862384",
    "dish_1558109511",
    "dish_1564159636",
)

# #91 same-setting B1. Identity is the file hash, not a nearby checkpoint.
FROZEN_B1_CHECKPOINT = "/home/sdnu/nutriagent-digi-v2-rerun/outputs/checkpoints/b1.pt"
FROZEN_B1_BYTES = 523464419
FROZEN_B1_SHA256 = "1f3e91156a195af0f4e5f036742ddd08b77be06c3023b82ef14dbf484565e3d2"
FROZEN_B1_CONFIG_SHA256 = "58c6d031266e057fb87f7cee8fbf46c77f8d48f86b538b08621eff88c1291f79"
FROZEN_B1_EPOCH = 60
FROZEN_B1_PREDICTIONS = (
    "/home/sdnu/nutriagent-digi-v2-rerun/results/rerun/predictions_holdout.csv"
)
FROZEN_B1_TRAIN_METRICS = (
    "/home/sdnu/nutriagent-digi-v2-rerun/results/rerun/train_metrics.json"
)
FROZEN_B1_PLATE_GROUPS = (
    "/home/sdnu/nutriagent-digi-v2-rerun/results/rerun/plate_groups_test.json"
)

# Full-precision MAE from the #91 holdout file. The registered 4-decimal
# figures are the rounded display of these values.
FROZEN_B1_HOLDOUT_MAE = {
    "energy": 47.85338618465495,
    "fat": 3.4685667804293114,
    "carb": 4.676291537150249,
    "protein": 4.601337233362243,
    "mass": 28.264109927253074,
}
FROZEN_B1_HOLDOUT_MAE_ROUNDED = {
    "energy": 47.8534,
    "fat": 3.4686,
    "carb": 4.6763,
    "protein": 4.6013,
    "mass": 28.2641,
}
FROZEN_B1_TRAIN_MAE = {
    "energy": 19.310226560895494,
    "fat": 1.188727033455727,
    "carb": 1.931819591755715,
    "protein": 1.723478975547476,
    "mass": 13.077475172240966,
}
FROZEN_SIGMA_MEAN = [
    254.84454729738562,
    12.699703116920844,
    19.336471637981116,
    18.096426388162673,
]
FROZEN_SIGMA_MASS = 218.04175744371824

CONFIG_PATH = "configs/tmulti_frozen.json"
VOCAB_PATH = "configs/ingredient_vocab.json"
TRAIN_METRICS = "results/tmulti/train_metrics.json"
HOLDOUT_METRICS = "results/tmulti/holdout_metrics.json"
PLATE_GROUPS_PATH = "results/tmulti/plate_groups_test.json"
HYGIENE_PATH = "results/tmulti/split_hygiene.json"
REPORT_PATH = "results/tmulti/REPORT.md"
PREDICTIONS_TRAIN = "results/tmulti/predictions_train.csv"
PREDICTIONS_HOLDOUT = "results/tmulti/predictions_holdout.csv"

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
        "n_headline_intervals": N_HEADLINE_INTERVALS,
        "pass_rule": (
            "On the official 507-plate holdout, all 12 headline 95% "
            "plate-grouped CIs of (T_multi - frozen #91 same-setting B1) "
            "lie entirely below 0 for energy, fat, carb, and protein, on MAE, "
            "dataset PMAE, and per-plate PMAE. The upper end is strictly below 0. "
            "Mass is recorded and is not a pass condition. Either verdict stops. "
            "Distillation stays locked."
        ),
        "text_encoder": TEXT_ENCODER,
        "status_commit": STATUS_COMMIT,
        "max_ingredients": MAX_INGREDIENTS,
        "train_device": TRAIN_DEVICE,
        "physical_gpu": PHYSICAL_GPU,
        "model": MODEL,
        "comparator": "frozen #91 same-setting B1",
        "frozen_b1_sha256": FROZEN_B1_SHA256,
        "frozen_b1_config_sha256": FROZEN_B1_CONFIG_SHA256,
        "frozen_b1_holdout_mae": dict(FROZEN_B1_HOLDOUT_MAE),
        "frozen_b1_train_mae": dict(FROZEN_B1_TRAIN_MAE),
        "t_multi_params": T_MULTI_PARAMS,
        "init": (
            "RGB encoder and regression heads from the frozen #91 B1; "
            "mass-head columns for the height readout start at 0; "
            "height encoder is a new ImageNet ResNet-18; "
            "ingredient branch is a new T_ing structure; gamma starts at 0"
        ),
    }
