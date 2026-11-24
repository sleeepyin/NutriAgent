# DiGI-Nutri Phase 1

Training is in progress. The official holdout has not been evaluated, and no Q0/Q1/Q2 verdict is claimed.

The geometry teacher is training on ResNet-101, 256 px, batch 32, AdamW 1e-4, 30 epochs. Four RGB students use that same budget after both teachers are frozen.

The ingredient teacher is names-only. `FrozenCLIPTextEncoder` tries Hugging Face `CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")` first. huggingface.co is unreachable here, and the local Hugging Face cache has no CLIP weights, so that call fails. This run then used the fallback: the official Azure ViT-B/32 file `outputs/clip_cache/ViT-B-32.pt` (downloaded 2026-11-24 22:59 +0800). `outputs/digi_nutri_phase1/text_cache.pt` was written at 23:04:48 +0800, after the geometry-teacher probe and before epoch 1, by that fallback. The cache was not rebuilt for this note.

Official depth test split: 507 dishes, 507 plate groups (no reshoot map, so each dish_id is its own plate). Kept train dishes: 2754 of 2758. Excluded train ids, checked on disk:

- `dish_1558109714`: no local official `rgb.png` (and no `depth_raw.png`)
- `dish_1557862384`: no local official `rgb.png` (and no `depth_raw.png`)
- `dish_1558109511`: no local official `rgb.png` (and no `depth_raw.png`)
- `dish_1564159636`: `rgb.png` is present; the extracted `depth_raw.png` is zero bytes

The index still counts those as rgb 3 and depth 1, because a zero-byte depth file is not indexed. Per-dish reasons are in `data_index_report.json`.

STATUS.md was not edited. Phase 2 was not started. The pull request base is digi-nutri, not main.
