"""Frozen DiGI-Nutri v2 Q0 protocol.

These constants are the training and evaluation contract. They are fixed
before any holdout prediction is computed.
"""

from __future__ import annotations

NUTRIENTS = ("energy", "fat", "carb", "protein")
MODELS = ("b1", "t_geo", "t_ing")
TEACHERS = ("t_geo", "t_ing")
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
BACKBONE_PREFERENCE = ("resnet101", "resnet50")
CAB_ATTN_DIM = 256
TOKEN_DIM = 256
HEAD_HIDDEN = 256
MAX_INGREDIENTS = 48

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261125
BOOTSTRAP_CI = (2.5, 97.5)

# No official Nutrition5k reshoot map is published with the depth split.
# Two dishes in the evaluated split share a physical-plate group when their
# published (energy, mass, fat, carb, protein) tuples match within 1e-3 and
# the unix timestamps embedded in their dish ids differ by at most 120 seconds.
# The link is transitively closed. Every other dish stays in its own dish_id
# group. This threshold is fixed before training.
RESHOOT_MAX_GAP_SECONDS = 120
RESHOOT_NUTRITION_ATOL = 1e-3

PLATE_PMAE_RULE = "mean of 100*|error|/|y| over plates with |y| > 0"
DATASET_PMAE_RULE = "100 * sum(|error|) / sum(|y|)"

RGB_ROOT = "/home/sdnu/nutriagent/data/nutrition5k/imagery/realsense_overhead"
DEPTH_ROOT = (
    "/home/sdnu/nutriagent-digi/data/official_depth/"
    "nutrition5k_dataset/imagery/realsense_overhead"
)
META_DIR = "/home/sdnu/nutriagent/data/nutrition5k/metadata"
SPLIT_DIR = "/home/sdnu/nutriagent/data/nutrition5k/dish_ids/splits"
CACHE_DIR = "/host/var/tmp/digi-nutri-v2-cache"
HOLDOUT_METRICS = "results/v2_q0/holdout_metrics.json"
TRAIN_METRICS = "results/v2_q0/train_metrics.json"

STATUS_COMMIT = "faad495acb679eb04eb54b710fd9add4f3144f4a"
TEXT_ENCODER = (
    "learned embedding of the static dictionary token plus a linear map of the "
    "USDA per-gram prior; the OpenAI CLIP package is not used"
)


def snapshot() -> dict:
    return {
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "optimizer": OPTIMIZER,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "betas": list(ADAM_BETAS),
        "scheduler": SCHEDULER,
        "eta_min": ETA_MIN,
        "hflip_p": HFLIP_P,
        "seed": SEED,
        "smooth_l1_beta": SMOOTH_L1_BETA,
        "mass_floor_g": MASS_FLOOR_G,
        "loss_mass_weight": LOSS_MASS_WEIGHT,
        "loss_density_weight": LOSS_DENSITY_WEIGHT,
        "raw_unit_mm": RAW_UNIT_MM,
        "height_max_mm": HEIGHT_MAX_MM,
        "border_pixels": BORDER_PIXELS,
        "morph_close": MORPH_CLOSE_K,
        "nutrients": list(NUTRIENTS),
        "param_limit": PARAM_LIMIT,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_percentiles": list(BOOTSTRAP_CI),
        "reshoot_max_gap_seconds": RESHOOT_MAX_GAP_SECONDS,
        "reshoot_nutrition_atol": RESHOOT_NUTRITION_ATOL,
        "plate_pmae": PLATE_PMAE_RULE,
        "dataset_pmae": DATASET_PMAE_RULE,
        "pass_rule": (
            "upper endpoint of the 95% plate-grouped CI of (teacher - b1) "
            "is strictly below 0 for energy, fat, carb, and protein, "
            "on MAE, dataset PMAE, and per-plate PMAE"
        ),
        "text_encoder": TEXT_ENCODER,
        "status_commit": STATUS_COMMIT,
        "max_ingredients": MAX_INGREDIENTS,
    }
