"""Frozen v3 step 1. Fixed before T_multi training and before the holdout."""

from __future__ import annotations

NUTRIENTS = ("energy", "fat", "carb", "protein")
MACRO_NUTRIENTS = ("fat", "carb", "protein")
ABSOLUTE_PMAE_KEYS = ("energy", "mass", "fat", "carb", "protein")
B1_MODEL = "b1"
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
RESIDUAL_LOSS_WEIGHT = 0.1
ATWATER_FAT = 9.0
ATWATER_CARB = 4.0
ATWATER_PRO = 4.0

RAW_UNIT_MM = 0.1
HEIGHT_MAX_MM = 150.0
BORDER_PIXELS = 1
MORPH_CLOSE_K = 3

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EXPECTED_TEST_N = 507
EXPECTED_TRAIN_N = 2754
EXPECTED_TRAIN_FILE_N = 2758
EXPECTED_TEST_GROUPS = 507
PARAM_LIMIT = 70_000_000
RGB_BACKBONE = "convnext_small"
HEIGHT_BACKBONE = "resnet18"
# huggingface.co was unreachable. Same timm convnext_small module.
# Official timm tag convnext_small.fb_in1k, ImageNet-1k.
CONVNEXT_WEIGHT_URL = "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth"
CONVNEXT_WEIGHT_TAG = "convnext_small.fb_in1k"
CONVNEXT_WEIGHT_PATH = "outputs/weights/convnext_small_1k_224_ema.pth"
CONVNEXT_WEIGHT_SHA256 = "fe969be0ad46c62ace0981c933ff47af59b9753365d71466efa6896571a879ad"
RGB_CHANNELS = 768
HEIGHT_CHANNELS = 512
CAB_REDUCTION = 16
GAMMA_INIT = 0.0
HEAD_HIDDEN = 256
RESIDUAL_HIDDEN = 64
TOKEN_DIM = 256
CAB_ATTN_DIM = 256
MAX_INGREDIENTS = 48
SPP_GRIDS = (1, 2, 4)
VOLUME_DIM = 4 + 1 + 4 + 16

# Counted on the untrained modules before any T_multi step. Vocab size 556.
T_MULTI_PARAMS = 62_254_886
B1_PARAMS = 49_849_701
RGB_ENCODER_PARAMS = 49_454_688
HEIGHT_ENCODER_PARAMS = 11_176_512

ENERGY_DATASET_PMAE_MAX = 13.5
MEAN5_DATASET_PMAE_MAX = 15.5

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

STATUS_COMMIT = "120340a6d32b184b4049ddf9d832e63befe0ad85"
PHYSICAL_GPU = "1"

TRAIN_EXCLUSION_IDS = (
    "dish_1558109714",
    "dish_1557862384",
    "dish_1558109511",
    "dish_1564159636",
)

FROZEN_SIGMA_MEAN = [
    254.84454729738562,
    12.699703116920844,
    19.336471637981116,
    18.096426388162673,
]
FROZEN_SIGMA_MASS = 218.04175744371824

CONFIG_PATH = "configs/v3_step1_frozen.json"
VOCAB_PATH = "configs/ingredient_vocab.json"
TRAIN_METRICS = "results/v3_step1/train_metrics.json"
HOLDOUT_METRICS = "results/v3_step1/holdout_metrics.json"
PLATE_GROUPS_PATH = "results/v3_step1/plate_groups_test.json"
HYGIENE_PATH = "results/v3_step1/split_hygiene.json"
REPORT_PATH = "results/v3_step1/REPORT.md"
PREDICTIONS_TRAIN = "results/v3_step1/predictions_train.csv"
PREDICTIONS_HOLDOUT = "results/v3_step1/predictions_holdout.csv"

TEXT_ENCODER = (
    "learned embedding of the static dictionary token plus a linear map of the "
    "USDA per-gram prior; no separate text-encoder package is used"
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
        "residual_loss_weight": RESIDUAL_LOSS_WEIGHT,
        "atwater_fat": ATWATER_FAT,
        "atwater_carb": ATWATER_CARB,
        "atwater_pro": ATWATER_PRO,
        "energy_density": "9*rho_fat + 4*rho_carb + 4*rho_pro + delta_rho_cal",
        "hard_atwater": False,
        "raw_unit_mm": RAW_UNIT_MM,
        "height_max_mm": HEIGHT_MAX_MM,
        "border_pixels": BORDER_PIXELS,
        "morph_close": MORPH_CLOSE_K,
        "exclude_border_zeros": True,
        "foreground": "height > 0",
        "nutrients": list(NUTRIENTS),
        "absolute_pmae_keys": list(ABSOLUTE_PMAE_KEYS),
        "energy_dataset_pmae_max": ENERGY_DATASET_PMAE_MAX,
        "mean5_dataset_pmae_max": MEAN5_DATASET_PMAE_MAX,
        "param_limit": PARAM_LIMIT,
        "rgb_backbone": RGB_BACKBONE,
        "height_backbone": HEIGHT_BACKBONE,
        "convnext_weight_url": CONVNEXT_WEIGHT_URL,
        "convnext_weight_tag": CONVNEXT_WEIGHT_TAG,
        "convnext_weight_path": CONVNEXT_WEIGHT_PATH,
        "convnext_weight_sha256": CONVNEXT_WEIGHT_SHA256,
        "rgb_channels": RGB_CHANNELS,
        "height_channels": HEIGHT_CHANNELS,
        "cab_reduction": CAB_REDUCTION,
        "gamma_init": GAMMA_INIT,
        "head_hidden": HEAD_HIDDEN,
        "residual_hidden": RESIDUAL_HIDDEN,
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
            "On the official 507-plate holdout, judged once: all 12 "
            "plate-grouped 95% CIs of (T_multi - this round's ConvNeXt-Small B1) "
            "on energy, fat, carb, and protein, for MAE, dataset PMAE, and "
            "per-plate PMAE, lie entirely below 0. The upper end is strictly below 0. "
            "Absolute gate uses dataset PMAE only: T_multi energy <= 13.5 and the "
            "mean of energy, mass, fat, carb, and protein dataset PMAEs <= 15.5. "
            "Mass intervals are recorded and are not a separate pass condition. "
            "Either gate failing fails Q0. A pass also stops. Student training stays locked."
        ),
        "text_encoder": TEXT_ENCODER,
        "status_commit": STATUS_COMMIT,
        "max_ingredients": MAX_INGREDIENTS,
        "physical_gpu": PHYSICAL_GPU,
        "model": MODEL,
        "b1_model": B1_MODEL,
        "comparator": "this round's ConvNeXt-Small B1",
        "t_multi_params": T_MULTI_PARAMS,
        "b1_params": B1_PARAMS,
        "init": (
            "B1 is a new ImageNet convnext_small. T_multi copies that B1 RGB "
            "encoder and the mass head, and copies the fat, carb, and protein "
            "rows of the B1 density head. Volume-feature columns start at 0. "
            "The height encoder is a new ImageNet ResNet-18. Gamma starts at 0. "
            "The ingredient branch is new, with the attention projection at 0 so "
            "the density features match the RGB features at epoch 0. The calorie "
            "residual head is new and starts at 0."
        ),
        "distillation_started": False,
        "train_gate": False,
    }
