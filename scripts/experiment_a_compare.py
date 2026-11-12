#!/usr/bin/env python3
"""Train and score the five Experiment A mass settings on the audited Nutrition5k pairs.

Settings, in the report's language:

- 单张俯视
- 单张侧视
- 双视角简单特征融合
- 双视角几何融合
- 双视角+食物语义约束

The test manifest is opened only after the checkpoint is written. Every test
dish gets one mass. Nothing is dropped because a prediction looks uncertain.
Depth is never read. The frozen Mass estimator under results/mass_estimator
is not loaded and not overwritten.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_a_common import (  # noqa: E402
    BATCH_SIZE,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CONSISTENCY_WEIGHT,
    EPOCHS,
    GRAD_CLIP,
    HEAD_HIDDEN,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    LR_BACKBONE,
    LR_HEAD,
    MASS_UNIT_G,
    POS_WEIGHT_CAP,
    RESULT_DIR,
    SEED,
    SEMANTIC_WEIGHT,
    SETTING_FILE,
    SETTINGS,
    VIEW_LOSS_WEIGHT,
    WEIGHT_DECAY,
    id_digest,
    mass_bias,
    paired_mae_difference,
    write_json,
)

FROZEN_RESULT_NAMES = {
    "mass_estimator",
    "heldout_plan",
    "heldout_eval",
    "heldout_energy_audit",
    "heldout_massonly_energy_path",
    "food_id_density",
    "food_id_semantic_audit",
    "qphi",
    "qphi_counterfactual",
    "p0d",
    "audit",
    "coverage_early_stop",
    "phase1_cross_domain",
    "weak_steps",
}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _assert_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    hits = FROZEN_RESULT_NAMES.intersection(resolved.parts)
    if hits:
        raise SystemExit(f"refusing to write into frozen results {sorted(hits)} via {resolved}")
    if resolved.name == "phase1_cross_domain":
        raise SystemExit("refusing to create results/phase1_cross_domain")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _assert_rgb(path: Path) -> None:
    folded = [part.casefold() for part in path.parts]
    if any("depth" in part for part in folded):
        raise SystemExit(f"RGB-only comparison refuses a depth path: {path}")


def _read_rows(manifest: Path, limit: int) -> list[dict]:
    rows = _load_jsonl(manifest)
    if limit > 0:
        rows = rows[:limit]
    for row in rows:
        overhead = ROOT / row["overhead_rgb"]
        side = ROOT / row["side_frame"]
        _assert_rgb(overhead)
        _assert_rgb(side)
        if not overhead.is_file() or not side.is_file():
            raise SystemExit(f"missing RGB pair for {row['dish_id']}")
        if not (float(row["mass_g"]) > 0):
            raise SystemExit(f"non-positive mass in manifest: {row['dish_id']}")
    return rows


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _transforms():
    from torchvision.transforms import functional as TF

    def apply(image, train: bool, flip: bool):
        if train and flip:
            image = TF.hflip(image)
        image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

    return apply


def _build_encoder():
    import torch.nn as nn
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    encoder.heads = nn.Identity()
    return encoder


def _encode(encoder, images):
    import torch

    tokens = encoder._process_input(images)
    batch = tokens.shape[0]
    cls = encoder.class_token.expand(batch, -1, -1)
    return encoder.encoder(torch.cat([cls, tokens], dim=1))


class MassHead:
    """Placeholder so type checkers can see the name. The real class is built with torch."""


def _mass_head_cls():
    import torch.nn as nn
    import torch.nn.functional as F

    class Head(nn.Module):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, HEAD_HIDDEN),
                nn.GELU(),
                nn.Linear(HEAD_HIDDEN, 1),
            )

        def forward(self, features):
            return F.softplus(self.net(features).squeeze(-1))

        def reset_to_mean(self, mean_g: float) -> None:
            first, _, last = self.net
            nn.init.normal_(first.weight, std=0.01)
            nn.init.zeros_(first.bias)
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, mass_bias(mean_g))

    return Head


def _models():
    import torch
    import torch.nn as nn

    Head = _mass_head_cls()

    class SingleView(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = _build_encoder()
            self.head = Head(768)

        def forward(self, image):
            tokens = _encode(self.encoder, image)
            return self.head(tokens[:, 0])

    class SimpleFusion(nn.Module):
        """Concatenated global tokens. Probes do not send gradients into the encoder."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = _build_encoder()
            self.fused = Head(1536)
            self.probe_top = Head(768)
            self.probe_side = Head(768)

        def forward(self, top, side):
            top_cls = _encode(self.encoder, top)[:, 0]
            side_cls = _encode(self.encoder, side)[:, 0]
            fused = self.fused(torch.cat([top_cls, side_cls], dim=-1))
            return fused, self.probe_top(top_cls.detach()), self.probe_side(side_cls.detach())

    class GeometricFusion(nn.Module):
        """Spatial cross-attention between the two views, plus per-view mass heads.

        The view embedding only marks top versus side. It is not a camera pose.
        No depth and no extrinsics are used.
        """

        def __init__(self) -> None:
            super().__init__()
            self.encoder = _build_encoder()
            self.view_embed = nn.Embedding(2, 768)
            self.cross = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
            self.norm_top = nn.LayerNorm(768)
            self.norm_side = nn.LayerNorm(768)
            self.head_top = Head(768)
            self.head_side = Head(768)
            self.fused = Head(1536)

        def forward(self, top, side):
            top_tokens = _encode(self.encoder, top)
            side_tokens = _encode(self.encoder, side)
            mass_top = self.head_top(top_tokens[:, 0])
            mass_side = self.head_side(side_tokens[:, 0])
            patches_top = top_tokens[:, 1:, :] + self.view_embed.weight[0]
            patches_side = side_tokens[:, 1:, :] + self.view_embed.weight[1]
            attended_top, _ = self.cross(patches_top, patches_side, patches_side, need_weights=False)
            attended_side, _ = self.cross(patches_side, patches_top, patches_top, need_weights=False)
            pooled_top = self.norm_top(patches_top + attended_top).mean(dim=1)
            pooled_side = self.norm_side(patches_side + attended_side).mean(dim=1)
            fused = self.fused(torch.cat([pooled_top, pooled_side], dim=-1))
            return fused, mass_top, mass_side

    class SemanticFusion(GeometricFusion):
        def __init__(self, n_labels: int) -> None:
            super().__init__()
            self.semantic = nn.Linear(1536, n_labels)

        def forward(self, top, side):
            top_tokens = _encode(self.encoder, top)
            side_tokens = _encode(self.encoder, side)
            mass_top = self.head_top(top_tokens[:, 0])
            mass_side = self.head_side(side_tokens[:, 0])
            patches_top = top_tokens[:, 1:, :] + self.view_embed.weight[0]
            patches_side = side_tokens[:, 1:, :] + self.view_embed.weight[1]
            attended_top, _ = self.cross(patches_top, patches_side, patches_side, need_weights=False)
            attended_side, _ = self.cross(patches_side, patches_top, patches_top, need_weights=False)
            pooled_top = self.norm_top(patches_top + attended_top).mean(dim=1)
            pooled_side = self.norm_side(patches_side + attended_side).mean(dim=1)
            pooled = torch.cat([pooled_top, pooled_side], dim=-1)
            return self.fused(pooled), mass_top, mass_side, self.semantic(pooled)

    return SingleView, SimpleFusion, GeometricFusion, SemanticFusion


def _ingredient_index(rows: list[dict]) -> tuple[dict[str, int], list[float]]:
    counts: dict[str, int] = {}
    for row in rows:
        seen = {item["id"] for item in row.get("ingredients", [])}
        for ingr_id in seen:
            counts[ingr_id] = counts.get(ingr_id, 0) + 1
    labels = sorted(counts)
    index = {label: position for position, label in enumerate(labels)}
    total = len(rows)
    weights = []
    for label in labels:
        positive = counts[label]
        negative = total - positive
        raw = 1.0 if positive == 0 else negative / positive
        weights.append(min(POS_WEIGHT_CAP, max(1.0, raw)))
    return index, weights


def _loader(rows: list[dict], train: bool, label_index: dict[str, int] | None, seed: int):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    apply = _transforms()

    class Pairs(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            row = rows[index]
            # One coin flip for the pair, so a mirror still shows the same dish.
            flip = bool(train) and random.random() < 0.5
            with Image.open(ROOT / row["overhead_rgb"]) as handle:
                top = apply(handle.convert("RGB"), train=flip, flip=True)
            with Image.open(ROOT / row["side_frame"]) as handle:
                side = apply(handle.convert("RGB"), train=flip, flip=True)
            mass = float(row["mass_g"])
            if label_index is None:
                return top, side, mass, row["dish_id"]
            target = torch.zeros(len(label_index), dtype=torch.float32)
            for item in row.get("ingredients", []):
                position = label_index.get(item["id"])
                if position is not None:
                    target[position] = 1.0
            return top, side, mass, target, row["dish_id"]

    def _seed_worker(worker_id: int) -> None:
        random.seed(seed + worker_id * 17)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Pairs(),
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=0,
        pin_memory=True,
        generator=generator if train else None,
        worker_init_fn=_seed_worker if train else None,
        persistent_workers=False,
    )


def _move_batch(batch, device, semantic: bool):
    import torch

    if semantic:
        top, side, mass, ingredients, dish_ids = batch
        ingredients = ingredients.to(device, non_blocking=True)
    else:
        top, side, mass, dish_ids = batch
        ingredients = None
    return (
        top.to(device, non_blocking=True),
        side.to(device, non_blocking=True),
        mass.to(device, non_blocking=True).float(),
        ingredients,
        list(dish_ids),
    )


def _loss(setting: str, outputs, mass, ingredients, pos_weight):
    import torch
    import torch.nn.functional as F

    def unit(pred: torch.Tensor) -> torch.Tensor:
        return (pred - mass).abs().mean() / MASS_UNIT_G

    if setting == "单张俯视" or setting == "单张侧视":
        return unit(outputs)
    if setting == "双视角简单特征融合":
        fused, probe_top, probe_side = outputs
        view = 0.5 * (unit(probe_top) + unit(probe_side))
        return unit(fused) + VIEW_LOSS_WEIGHT * view
    if setting == "双视角几何融合":
        fused, mass_top, mass_side = outputs
        view = 0.5 * (unit(mass_top) + unit(mass_side))
        consistency = (mass_top - mass_side).abs().mean() / MASS_UNIT_G
        return unit(fused) + VIEW_LOSS_WEIGHT * view + CONSISTENCY_WEIGHT * consistency
    fused, mass_top, mass_side, logits = outputs
    view = 0.5 * (unit(mass_top) + unit(mass_side))
    consistency = (mass_top - mass_side).abs().mean() / MASS_UNIT_G
    semantic = F.binary_cross_entropy_with_logits(logits.float(), ingredients.float(), pos_weight=pos_weight)
    return unit(fused) + VIEW_LOSS_WEIGHT * view + CONSISTENCY_WEIGHT * consistency + SEMANTIC_WEIGHT * semantic


def _predict_mass(setting: str, outputs):
    if setting in {"单张俯视", "单张侧视"}:
        return outputs, None, None
    if setting == "双视角+食物语义约束":
        fused, mass_top, mass_side, _logits = outputs
        return fused, mass_top, mass_side
    fused, mass_top, mass_side = outputs
    return fused, mass_top, mass_side


def _build_model(setting: str, n_labels: int, mean_g: float):
    SingleView, SimpleFusion, GeometricFusion, SemanticFusion = _models()
    if setting == "单张俯视" or setting == "单张侧视":
        model = SingleView()
        model.head.reset_to_mean(mean_g)
    elif setting == "双视角简单特征融合":
        model = SimpleFusion()
        model.fused.reset_to_mean(mean_g)
        model.probe_top.reset_to_mean(mean_g)
        model.probe_side.reset_to_mean(mean_g)
    elif setting == "双视角几何融合":
        model = GeometricFusion()
        model.fused.reset_to_mean(mean_g)
        model.head_top.reset_to_mean(mean_g)
        model.head_side.reset_to_mean(mean_g)
    elif setting == "双视角+食物语义约束":
        model = SemanticFusion(n_labels)
        model.fused.reset_to_mean(mean_g)
        model.head_top.reset_to_mean(mean_g)
        model.head_side.reset_to_mean(mean_g)
    else:
        raise SystemExit(f"unknown setting {setting}")
    return model


def _optimizer(model):
    import torch

    head = []
    backbone = []
    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            backbone.append(param)
        else:
            head.append(param)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": LR_BACKBONE},
            {"params": head, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def _run_epoch(model, loader, setting: str, device, optimizer, pos_weight, train: bool, scaler=None) -> dict[str, float]:
    import torch

    model.train(train)
    total_loss = 0.0
    total_abs = 0.0
    seen = 0
    for batch in loader:
        top, side, mass, ingredients, _dish_ids = _move_batch(batch, device, setting == "双视角+食物语义约束")
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=train):
                if setting == "单张俯视":
                    outputs = model(top)
                elif setting == "单张侧视":
                    outputs = model(side)
                else:
                    outputs = model(top, side)
            loss = _loss(setting, outputs, mass, ingredients, pos_weight)
        if train:
            if not torch.isfinite(loss):
                raise SystemExit(f"{setting} produced a non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        pred, _top, _side = _predict_mass(setting, outputs)
        batch_n = int(mass.shape[0])
        total_loss += float(loss.detach()) * batch_n
        total_abs += float((pred.detach().float() - mass).abs().sum())
        seen += batch_n
    return {"loss": total_loss / seen, "mass_mae_g": total_abs / seen, "n": seen}


def _evaluate(model, rows: list[dict], setting: str, device) -> list[dict]:
    import torch

    model.eval()
    # The semantic head is not used at test time. An empty index only satisfies
    # the batch shape; ingredient labels are not read from the test manifest.
    label_index = {} if setting == "双视角+食物语义约束" else None
    loader = _loader(rows, train=False, label_index=label_index, seed=SEED)
    predictions = []
    offset = 0
    for batch in loader:
        semantic = setting == "双视角+食物语义约束"
        top, side, mass, _ingredients, dish_ids = _move_batch(batch, device, semantic)
        with torch.no_grad():
            if setting == "单张俯视":
                outputs = model(top)
            elif setting == "单张侧视":
                outputs = model(side)
            else:
                outputs = model(top, side)
        pred, pred_top, pred_side = _predict_mass(setting, outputs)
        pred_list = pred.detach().float().cpu().tolist()
        top_list = None if pred_top is None else pred_top.detach().float().cpu().tolist()
        side_list = None if pred_side is None else pred_side.detach().float().cpu().tolist()
        mass_list = mass.detach().float().cpu().tolist()
        for index, dish_id in enumerate(dish_ids):
            value = float(pred_list[index])
            if value != value:  # NaN
                raise SystemExit(f"non-finite mass for {dish_id}")
            record = {
                "dish_id": dish_id,
                "mass_g": float(mass_list[index]),
                "pred_mass_g": value,
                "abs_err_g": abs(value - float(mass_list[index])),
                "pred_top_g": None if top_list is None else float(top_list[index]),
                "pred_side_g": None if side_list is None else float(side_list[index]),
            }
            if setting == "单张俯视":
                record["pred_top_g"] = value
            elif setting == "单张侧视":
                record["pred_side_g"] = value
            predictions.append(record)
        offset += len(dish_ids)
    if len(predictions) != len(rows):
        raise SystemExit(f"{setting} predicted {len(predictions)} dishes, manifest has {len(rows)}")
    by_id = {row["dish_id"]: row for row in predictions}
    if len(by_id) != len(predictions):
        raise SystemExit(f"{setting} emitted duplicate dish ids")
    ordered = []
    for row in rows:
        if row["dish_id"] not in by_id:
            raise SystemExit(f"{setting} missed {row['dish_id']}")
        ordered.append(by_id[row["dish_id"]])
    return ordered


def _write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_one(setting: str, args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("training expects CUDA")
    out_dir = _assert_output_dir(args.out_dir)
    audit_path = out_dir / "scale_audit.json" if (out_dir / "scale_audit.json").is_file() else RESULT_DIR / "scale_audit.json"
    if not audit_path.is_file():
        raise SystemExit("scale audit is missing; refusing to train before the pair count")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    train_rows = _read_rows(RESULT_DIR / "manifest_rgb_train.jsonl", args.limit_train)
    if args.limit_train == 0 and len(train_rows) != audit["counts"]["usable_rgb_train"]:
        raise SystemExit("train manifest length does not match the scale audit")
    mean_g = sum(float(row["mass_g"]) for row in train_rows) / len(train_rows)
    label_index: dict[str, int] = {}
    pos_weight_list: list[float] = []
    if setting == "双视角+食物语义约束":
        label_index, pos_weight_list = _ingredient_index(train_rows)
        if not label_index:
            raise SystemExit("semantic setting found no training ingredient ids")
    _set_seed(SEED)
    device = torch.device(args.device)
    model = _build_model(setting, len(label_index), mean_g).to(device)
    optimizer = _optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    pos_weight = None
    if pos_weight_list:
        pos_weight = torch.tensor(pos_weight_list, dtype=torch.float32, device=device)
    loader = _loader(
        train_rows,
        train=True,
        label_index=label_index if setting == "双视角+食物语义约束" else None,
        seed=SEED,
    )
    scaler = torch.amp.GradScaler("cuda")
    history = []
    for epoch in range(args.epochs):
        stats = _run_epoch(model, loader, setting, device, optimizer, pos_weight, train=True, scaler=scaler)
        scheduler.step()
        stats["epoch"] = epoch
        history.append(stats)
        print(
            f"{setting} epoch {epoch + 1}/{args.epochs} loss {stats['loss']:.4f} train_mae_g {stats['mass_mae_g']:.3f}",
            flush=True,
        )
    clean = _run_epoch(
        model,
        _loader(train_rows, train=False, label_index=label_index if setting == "双视角+食物语义约束" else None, seed=SEED),
        setting,
        device,
        optimizer,
        pos_weight,
        train=False,
    )
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{SETTING_FILE[setting]}.pt"
    torch.save(
        {
            "setting": setting,
            "state_dict": model.state_dict(),
            "train_mass_mean_g": mean_g,
            "n_labels": len(label_index),
            "label_index": label_index,
            "epochs": args.epochs,
            "seed": SEED,
        },
        checkpoint,
    )
    # Test labels are read only after the checkpoint exists.
    if not checkpoint.is_file():
        raise SystemExit("checkpoint was not written")
    test_rows = _read_rows(RESULT_DIR / "manifest_rgb_test.jsonl", args.limit_test)
    if args.limit_test == 0 and len(test_rows) != audit["counts"]["usable_rgb_test"]:
        raise SystemExit("test manifest length does not match the scale audit")
    predictions = _evaluate(model, test_rows, setting, device)
    pred_path = out_dir / "predictions" / f"{SETTING_FILE[setting]}.jsonl"
    _write_predictions(pred_path, predictions)
    mae = sum(row["abs_err_g"] for row in predictions) / len(predictions)
    write_json(
        out_dir / "logs" / f"{SETTING_FILE[setting]}.json",
        {
            "setting": setting,
            "epochs": args.epochs,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "train_n": len(train_rows),
            "test_n": len(test_rows),
            "train_id_digest": id_digest(row["dish_id"] for row in train_rows),
            "test_id_digest": id_digest(row["dish_id"] for row in test_rows),
            "train_mass_mean_g": mean_g,
            "clean_train_mass_mae_g": clean["mass_mae_g"],
            "test_mass_mae_g": mae,
            "n_ingredient_labels": len(label_index),
            "history": history,
            "checkpoint": str(checkpoint),
            "predictions": str(pred_path),
        },
    )
    print(f"{setting} test_mass_mae_g {mae:.4f} n {len(predictions)}", flush=True)


def _load_predictions(path: Path) -> dict[str, dict]:
    rows = _load_jsonl(path)
    by_id = {row["dish_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit(f"duplicate predictions in {path}")
    return by_id


def write_report(out_dir: Path) -> None:
    import numpy as np

    out_dir = _assert_output_dir(out_dir)
    audit = json.loads((RESULT_DIR / "scale_audit.json").read_text(encoding="utf-8"))
    counts = audit["counts"]
    prediction_maps = {}
    for setting in SETTINGS:
        path = out_dir / "predictions" / f"{SETTING_FILE[setting]}.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing predictions for {setting}: {path}")
        prediction_maps[setting] = _load_predictions(path)
    dish_ids = list(prediction_maps["单张俯视"])
    for setting, rows in prediction_maps.items():
        if set(rows) != set(dish_ids):
            raise SystemExit(f"{setting} does not cover the same test dishes as 单张俯视")
    dish_ids = sorted(dish_ids)
    if len(dish_ids) != counts["usable_rgb_test"]:
        raise SystemExit("prediction count does not match the audited rgb_test pair count")

    def column(setting: str, field: str) -> list[float]:
        return [float(prediction_maps[setting][dish_id][field]) for dish_id in dish_ids]

    top_abs = column("单张俯视", "abs_err_g")
    train_rows = _load_jsonl(RESULT_DIR / "manifest_rgb_train.jsonl")
    train_mean = sum(float(row["mass_g"]) for row in train_rows) / len(train_rows)
    test_mass = column("单张俯视", "mass_g")
    constant_abs = [abs(mass - train_mean) for mass in test_mass]
    rows = []
    for setting in SETTINGS:
        abs_err = column(setting, "abs_err_g")
        diff = paired_mae_difference(abs_err, top_abs, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
        entry = {
            "setting": setting,
            "n": len(abs_err),
            "mass_mae_g": float(np.mean(abs_err)),
            "mass_median_ae_g": float(np.median(abs_err)),
            "versus_top": None if setting == "单张俯视" else diff,
        }
        rows.append(entry)
    side_abs = column("单张侧视", "abs_err_g")
    single_gap = np.abs(np.asarray(column("单张俯视", "pred_mass_g")) - np.asarray(column("单张侧视", "pred_mass_g")))
    consistency = {
        "单张俯视与单张侧视": {
            "mean_abs_g": float(single_gap.mean()),
            "median_abs_g": float(np.median(single_gap)),
        }
    }
    for setting in ("双视角简单特征融合", "双视角几何融合", "双视角+食物语义约束"):
        gap = np.abs(np.asarray(column(setting, "pred_top_g")) - np.asarray(column(setting, "pred_side_g")))
        consistency[setting] = {"mean_abs_g": float(gap.mean()), "median_abs_g": float(np.median(gap))}
    simple = next(row for row in rows if row["setting"] == "双视角简单特征融合")
    geometric = next(row for row in rows if row["setting"] == "双视角几何融合")
    simple_beats = bool(simple["versus_top"]["stably_beats_top"])
    geometric_beats = bool(geometric["versus_top"]["stably_beats_top"])
    stop = not (simple_beats and geometric_beats)
    if stop:
        decision = (
            "停止。简单融合与几何融合没有都稳定胜过单张俯视。"
            "不把第二视角写成已经成立的增益，也不为了多模态改方案。"
            "请 Triste 判断。实验 B、实验 C 和完整联合网络不启动。Q3 仍未解。"
        )
    else:
        decision = (
            "简单融合和几何融合的质量 MAE 都稳定低于单张俯视。"
            "这只说明在这份 Nutrition5k 子集上，这两种双视角融合优于俯视基线。"
            "实验 B、实验 C 和完整联合网络仍然锁定，本轮不启动。Q3 仍未解。"
        )
    payload = {
        "test_n": len(dish_ids),
        "test_id_digest": id_digest(dish_ids),
        "train_mass_mean_g": train_mean,
        "constant_train_mean_mass_mae_g": float(np.mean(constant_abs)),
        "settings": rows,
        "consistency": consistency,
        "simple_fusion_stably_beats_top": simple_beats,
        "geometric_fusion_stably_beats_top": geometric_beats,
        "stop": stop,
        "decision": decision,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "interval": "percentile 2.5 to 97.5"},
        "stable_rule": "A setting stably beats 单张俯视 when the paired bootstrap 95% interval of (its mass MAE - 单张俯视 mass MAE) lies entirely below 0. Both 双视角简单特征融合 and 双视角几何融合 must do so. Otherwise stop.",
        "live_claim_unchanged": "质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。",
        "q3": "未解。本轮只跑了实验 A。",
        "depth_in_main_table": False,
        "side_is_metric_stereo": False,
    }
    write_json(out_dir / "comparison.json", payload)
    _write_report_md(out_dir / "report.md", audit, payload)
    print(decision, flush=True)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _write_report_md(path: Path, audit: dict, payload: dict) -> None:
    counts = audit["counts"]
    lines = [
        "# 实验 A：Nutrition5k 双视角质量对照",
        "",
        "评价范围只包括官方归档里同时有顶视 RGB 和侧视 RGB、能按 dish_id 对齐、侧视帧能解码、并且质量大于 0 克的盘。",
        "侧视不是度量双相机，没有使用公布位姿。主表只有 RGB。深度只在规模核对的附录里计数，没有进入任何模型。",
        "",
        "## 规模",
        "",
        f"可用配对 {counts['usable_pairs']} 盘。其中官方 rgb_train {counts['usable_rgb_train']} 盘，官方 rgb_test {counts['usable_rgb_test']} 盘，不在官方 RGB 划分里的 {counts['usable_outside_rgb_split']} 盘没有进入训练或主表。",
        "纳入规则、互斥排除原因和官方文件哈希见 `scale_audit.md`。",
        "",
        "## 五项对照",
        "",
        "五个设置共用 ImageNet ViT-B/16、同一种 224 输入、同一个随机种子、同样的 12 个 epoch。测试集没有参与选结构、选学习率或早停。每盘都有一个质量预测，没有按置信度弃权。",
        "",
        "单张俯视是主基线。简单特征融合把两个视角的全局特征拼在一起。几何融合用空间交叉注意力让两个视角的图像块互相对照，并惩罚两个视角各自质量预测的不一致；视角嵌入只区分俯视和侧视，不是相机外参。食物语义约束在几何融合上多加一个配料多标签损失，权重在训练前固定。测试时不读取配料头，也不因配料不确定而跳过盘子。",
        "",
        f"训练集质量均值常数预测在测试集上的质量 MAE 是 {_fmt(payload['constant_train_mean_mass_mae_g'])} 克。这只是参照，不是第六个方法。",
        "",
        "| 设置 | 测试盘数 | 质量 MAE（克） | 质量误差中位数（克） | 相对单张俯视的 MAE 差（克） | 95% 区间 | 是否稳定低于单张俯视 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["settings"]:
        if row["versus_top"] is None:
            diff = "0"
            interval = "—"
            stable = "基线"
        else:
            diff = _fmt(row["versus_top"]["mae_difference_g"])
            interval = f"[{_fmt(row['versus_top']['ci95_low_g'])}, {_fmt(row['versus_top']['ci95_high_g'])}]"
            stable = "是" if row["versus_top"]["stably_beats_top"] else "否"
        lines.append(
            f"| {row['setting']} | {row['n']} | {_fmt(row['mass_mae_g'])} | {_fmt(row['mass_median_ae_g'])} | {diff} | {interval} | {stable} |"
        )
    lines.extend(
        [
            "",
            "MAE 差等于该方法的质量 MAE 减去单张俯视的质量 MAE。负值表示该方法更低。区间是按盘配对、10000 次重抽样的 2.5% 到 97.5% 分位，种子 0。稳定低于单张俯视，指这个区间整个落在 0 以下。",
            "",
            "## 跨视角一致性",
            "",
            "一致性是同一盘上两个质量预测的绝对差，单位是克。单张俯视和单张侧视各有自己的模型。三种双视角模型则比较它们自己的俯视支路和侧视支路。简单融合的支路是停梯度探针，不改融合头的梯度。几何融合和语义约束的支路参与一致性损失。",
            "",
            "| 对照 | 平均绝对差（克） | 中位绝对差（克） |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, stats in payload["consistency"].items():
        lines.append(f"| {name} | {_fmt(stats['mean_abs_g'])} | {_fmt(stats['median_abs_g'])} |")
    lines.extend(
        [
            "",
            "## 停止还是继续",
            "",
            payload["decision"],
            "",
            "停止规则事先定好：只有简单融合和几何融合都稳定胜过单张俯视，才算双视角在这项对照里成立。食物语义约束的结果不单独推翻这条规则。无论本表如何，实验 B、实验 C 和完整联合网络都没有在本轮打开。",
            "",
            "Q3 仍未解。开跑实验 A 只是这项检验的开始。",
            "",
            "已有外测声称不因本实验改写：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。",
            "",
            "少样本 / 分层质量校准不是本实验。本表也没有在 505 或 Cafe2 上调方法。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment A five-way RGB mass comparison")
    parser.add_argument("--setting", choices=SETTINGS)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    args = parser.parse_args()
    if args.report_only:
        write_report(args.out_dir)
        return 0
    if args.setting is None:
        raise SystemExit("pass --setting or --report-only")
    train_one(args.setting, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
