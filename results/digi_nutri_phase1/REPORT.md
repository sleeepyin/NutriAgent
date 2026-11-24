# DiGI-Nutri Phase 1

Training is in progress. The official holdout has not been evaluated, and no Q0/Q1/Q2 verdict is claimed.

The geometry teacher is training on ResNet-101, 256 px, batch 32, AdamW 1e-4, 30 epochs. Four RGB students use that same budget after both teachers are frozen. The ingredient teacher is names-only CLIP ViT-B/32 loaded from the official Azure checkpoint.

Official depth test split: 507 dishes, 507 plate groups (no reshoot map, so each dish_id is its own plate). Train ids excluded because the local official files are missing or empty: dish_1558109714, dish_1557862384, dish_1558109511, dish_1564159636. Kept train dishes: 2754.

STATUS.md was not edited. Phase 2 was not started. The pull request base is digi-nutri, not main.

