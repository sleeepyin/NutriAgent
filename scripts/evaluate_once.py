"""One official holdout read, plus train metrics and the shared FPS ruler."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repro.data import FusionPlates, SwinPlates
from repro.groups import plate_groups
from repro.metadata import load_dishes
from repro.metrics import summarize
from repro.models import DPFNutrition, DepthModule, SwinNutrition
from repro.protocol import ROOT as REPO
from repro.protocol import load_config
from repro.runtime import load_checkpoint, require_gpu
from repro.splits import audit_split


def predict_rows(model, loader, device: torch.device, kind: str) -> dict[str, np.ndarray]:
    model.eval()
    rows = {}
    with torch.no_grad():
        for batch in loader:
            if kind == "swin":
                image, _target, dish_ids = batch
                pred = model(image.to(device, non_blocking=True))
            else:
                image, depth, _target, dish_ids = batch
                pred = model(image.to(device, non_blocking=True), depth.to(device, non_blocking=True))
            pred = pred.detach().float().cpu().numpy()
            for index, dish_id in enumerate(dish_ids):
                rows[dish_id] = pred[index]
    return rows


def stack(rows: dict[str, np.ndarray], dish_ids: list[str]) -> np.ndarray:
    missing = [dish_id for dish_id in dish_ids if dish_id not in rows]
    if missing:
        raise RuntimeError(f"missing predictions {missing[:4]}")
    return np.stack([rows[dish_id] for dish_id in dish_ids])


def load_swin(config: dict, device: torch.device) -> SwinNutrition:
    spec = config["swin"]
    model = SwinNutrition(drop_path_rate=0.0, ffm_channels=int(spec["ffm_channels"]))
    state = load_checkpoint(REPO / "outputs" / "swin" / "last.pt")["model"]
    model.load_state_dict(state)
    return model.to(device).eval()


def load_dpf(device: torch.device) -> DPFNutrition:
    model = DPFNutrition()
    state = load_checkpoint(REPO / "outputs" / "dpf" / "last.pt")["model"]
    model.load_state_dict(state)
    return model.to(device).eval()


def load_dpt(config: dict, device: torch.device) -> DepthModule:
    model = DepthModule()
    model.load_state_dict(load_checkpoint(REPO / "outputs" / "dpt" / "last.pt")["model"])
    return model.to(device).eval()


def measure_fps(fn, batch: int, steps: int, warmup: int) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (batch * steps) / (time.perf_counter() - start)


def write_report(path: Path, payload: dict, columns: list[str]) -> None:
    lines = [
        "# Same-split retrain report",
        "",
        "DPF-Nutrition and Swin-Nutrition were each trained once on the 2754 kept official training plates and read once on `official_depth_test`.",
        "These numbers are not the papers' published test-set numbers, and they are not a new pass line.",
        "",
        f"Holdout plates: {payload['n_test']}. Groups: {payload['n_test_groups']}. Train/holdout overlap: {payload['train_holdout_overlap']}.",
        "",
    ]
    for name in ("swin_nutrition", "dpf_nutrition"):
        block = payload["methods"][name]
        lines.append(f"## {name}")
        lines.append("")
        if block.get("status") == "stopped":
            lines.append(block["reason"])
            lines.append("")
            continue
        lines.append("| Item | Dataset PMAE | MAE | Plate PMAE |")
        lines.append("|---|---:|---:|---:|")
        for key in columns:
            item = block["holdout"][key]
            lines.append(
                f"| {key} | {item['dataset_pmae']:.4f} | {item['mae']:.4f} | {item['plate_pmae']:.4f} |"
            )
        lines.append(f"| five-item mean | {block['holdout']['mean5_dataset_pmae']:.4f} |  |  |")
        lines.append("")
        train = block["train"]
        lines.append(
            "Train-set dataset PMAE, eval mode, no augmentation: "
            + ", ".join(f"{key} {train[key]['dataset_pmae']:.4f}" for key in columns)
            + f", mean {train['mean5_dataset_pmae']:.4f}."
        )
        lines.append("")
    fps = payload["fps"]
    lines.extend(
        [
            "## FPS ruler",
            "",
            f"Hardware: {fps.get('hardware', 'unavailable')}. Physical GPU {fps['physical_gpu']}. Batch {fps['batch_size']}. Resolution {fps['height']}×{fps['width']}.",
            "The timed region is model forward only, inputs already on GPU, eval mode, no grad, after warmup.",
            "DPF's forward includes the depth module and the fusion module. Predicted depth is clipped, replicated, and ImageNet-normalized with the fusion helper before the fusion module. This is not a speed claim.",
            "",
        ]
    )
    for name in ("swin_nutrition", "dpf_nutrition"):
        value = fps["methods"].get(name)
        if isinstance(value, float):
            lines.append(f"- {name}: {value:.3f} images/s")
        else:
            lines.append(f"- {name}: {value}")
    lines.append("")
    loads = payload.get("weight_load", {})
    lines.append("## Pretrained weight load")
    lines.append("")
    for name, item in loads.items():
        if "matched" in item:
            lines.append(f"- {name}: {item['matched']}/{item['total']} tensors. {item.get('note', '')}")
        else:
            lines.append(f"- {name}: {item}")
    lines.append("")
    lines.append("Deviations fixed before training are in `docs/DEVIATIONS.md`.")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    final = REPO / "results" / "repro" / "holdout_metrics.json"
    if final.exists():
        raise SystemExit("refusing a second official holdout eval because holdout_metrics.json already exists")
    device = require_gpu()
    config = load_config()
    if not (REPO / "outputs" / "swin" / "DONE").exists():
        raise SystemExit("Swin training is not complete")
    dpf_done = (REPO / "outputs" / "dpf" / "DONE").exists()
    dpf_stop_path = REPO / "results" / "repro" / "dpf_stop.json"
    if not dpf_done and not dpf_stop_path.exists():
        raise SystemExit("DPF is neither complete nor explicitly stopped")
    audit = audit_split(config)
    dishes = load_dishes(config["paths"]["cafe1"], config["paths"]["cafe2"])
    columns = list(config["label_order"])
    groups = plate_groups(audit.test_ids, dishes, config)
    swin = load_swin(config, device)
    swin_train = predict_rows(
        swin,
        DataLoader(SwinPlates(audit.train_ids, dishes, config, train=False), batch_size=int(config["swin"]["batch_size"]), shuffle=False, num_workers=2),
        device,
        "swin",
    )
    swin_test = predict_rows(
        swin,
        DataLoader(SwinPlates(audit.test_ids, dishes, config, train=False), batch_size=int(config["swin"]["batch_size"]), shuffle=False, num_workers=2),
        device,
        "swin",
    )
    methods = {
        "swin_nutrition": {
            "status": "trained",
            "holdout": summarize(stack(swin_test, audit.test_ids), _targets(dishes, audit.test_ids, columns), tuple(columns)),
            "train": summarize(stack(swin_train, audit.train_ids), _targets(dishes, audit.train_ids, columns), tuple(columns)),
        }
    }
    dpf_test = None
    if dpf_done:
        dpf = load_dpf(device)
        cache = REPO / "outputs" / "cache" / "dpt_depth_holdout"
        _ensure_holdout_cache(config, audit.test_ids, device, cache)
        dpf_train = predict_rows(
            dpf,
            DataLoader(FusionPlates(audit.train_ids, dishes, config, False, REPO / "outputs" / "cache" / "dpt_depth"), batch_size=int(config["dpf"]["fusion_module"]["batch_size"]), shuffle=False, num_workers=2),
            device,
            "dpf",
        )
        dpf_test = predict_rows(
            dpf,
            DataLoader(FusionPlates(audit.test_ids, dishes, config, False, cache), batch_size=int(config["dpf"]["fusion_module"]["batch_size"]), shuffle=False, num_workers=2),
            device,
            "dpf",
        )
        methods["dpf_nutrition"] = {
            "status": "trained",
            "holdout": summarize(stack(dpf_test, audit.test_ids), _targets(dishes, audit.test_ids, columns), tuple(columns)),
            "train": summarize(stack(dpf_train, audit.train_ids), _targets(dishes, audit.train_ids, columns), tuple(columns)),
        }
        del dpf
    else:
        methods["dpf_nutrition"] = json.loads(dpf_stop_path.read_text())
        dpf = None
    torch.cuda.empty_cache()
    fps = _fps(config, device, swin, dpf_done)
    del swin
    csv_path = REPO / "results" / "repro" / "predictions_holdout.csv"
    _write_csv(csv_path, audit.test_ids, groups, dishes, columns, swin_test, dpf_test)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    payload = {
        "split": "official_depth_test",
        "weight_load": _weight_load(config),
        "n_test": len(audit.test_ids),
        "n_train": len(audit.train_ids),
        "n_test_groups": len(set(groups.values())),
        "train_holdout_overlap": 0,
        "holdout_evaluated_once": True,
        "predictions_sha256": digest,
        "methods": methods,
        "fps": fps,
        "status_commit": config["status_commit"],
    }
    report = REPO / "results" / "repro" / "REPORT.md"
    write_report(report, payload, columns)
    final.write_text(json.dumps(payload, indent=2) + "\n")
    print("holdout written", digest, flush=True)


def _weight_load(config: dict) -> dict:
    """Recount pretrained files. This does not touch the running Swin process."""
    swin = SwinNutrition(drop_path_rate=0.0, ffm_channels=int(config["swin"]["ffm_channels"]))
    swin_count = swin.imagenet_coverage(REPO / config["weights"]["swin_imagenet1k"]["path"])
    record = {
        "swin_nutrition": {
            "matched": swin_count["matched"],
            "total": swin_count["total"],
            "note": (
                "Recount with the same ImageNet loader, without restarting training. "
                "train_swin.py printed 'swin imagenet tensors 325'."
            ),
        }
    }
    del swin
    encoder = REPO / config["weights"]["dpt_hybrid_encoder"]["path"]
    minimum = int(config["weights"]["dpt_hybrid_encoder"]["min_bytes"])
    size = encoder.stat().st_size if encoder.exists() else 0
    if size < minimum:
        record["dpf_depth_encoder"] = {"note": f"encoder file incomplete ({size} bytes, need {minimum})"}
        return record
    depth = DepthModule()
    try:
        matched, total = depth.load_encoder(encoder)
    except RuntimeError as exc:
        record["dpf_depth_encoder"] = {"note": str(exc)}
        return record
    record["dpf_depth_encoder"] = {
        "matched": matched,
        "total": total,
        "note": "Every encoder tensor except num_batches_tracked is present with a matching shape.",
    }
    return record


def _targets(dishes: dict, dish_ids: list[str], columns: list[str]) -> np.ndarray:
    from repro.metadata import label_vector

    return np.asarray([label_vector(dishes[dish_id], tuple(columns)) for dish_id in dish_ids], dtype=np.float64)


def _ensure_holdout_cache(config: dict, dish_ids: list[str], device: torch.device, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    missing = [dish_id for dish_id in dish_ids if not (cache / f"{dish_id}.npy").exists()]
    if not missing:
        return
    from repro.data import predict_fusion_depth, read_rgb

    model = load_dpt(config, device)
    rgb_root = Path(config["paths"]["rgb_root"])
    with torch.no_grad():
        for dish_id in missing:
            depth = predict_fusion_depth(model, read_rgb(rgb_root / dish_id / "rgb.png"), config)
            np.save(cache / f"{dish_id}.npy", depth.astype(np.float32))


def _write_csv(path, dish_ids, groups, dishes, columns, swin_rows, dpf_rows) -> None:
    from repro.metadata import label_vector

    header = ["dish_id", "group_id"]
    header += [f"gt_{name}" for name in columns]
    header += [f"swin_{name}" for name in columns]
    header += [f"dpf_{name}" for name in columns]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for dish_id in dish_ids:
            gt = label_vector(dishes[dish_id], tuple(columns))
            swin = swin_rows[dish_id]
            if dpf_rows is None:
                dpf = [""] * len(columns)
            else:
                dpf = [f"{value:.8f}" for value in dpf_rows[dish_id]]
            writer.writerow(
                [dish_id, groups[dish_id]]
                + [f"{value:.8f}" for value in gt]
                + [f"{value:.8f}" for value in swin]
                + dpf
            )


def _fps(config: dict, device: torch.device, swin: SwinNutrition, dpf_done: bool) -> dict:
    ruler = config["fps_ruler"]
    batch = int(ruler["batch_size"])
    height = int(ruler["height"])
    width = int(ruler["width"])
    props = torch.cuda.get_device_properties(device)
    payload = {
        "hardware": props.name,
        "physical_gpu": int(config["gpu"]["physical_index"]),
        "batch_size": batch,
        "height": height,
        "width": width,
        "methods": {},
    }
    image = torch.rand(batch, 3, height, width, device=device)
    try:
        payload["methods"]["swin_nutrition"] = measure_fps(
            lambda: swin(image), batch, int(ruler["timed_steps"]), int(ruler["warmup_steps"])
        )
    except RuntimeError as exc:
        payload["methods"]["swin_nutrition"] = f"measurement failed: {exc}"
    if not dpf_done:
        payload["methods"]["dpf_nutrition"] = "not measured because fusion training stopped"
        return payload
    try:
        dpt = load_dpt(config, device)
        dpf = load_dpf(device)

        from repro.data import depth_to_resnet

        fusion = config["dpf"]["fusion_module"]

        def step():
            depth = dpt(image)
            depth3 = depth_to_resnet(
                depth,
                float(fusion["depth_vis_max_meters"]),
                fusion["imagenet_mean"],
                fusion["imagenet_std"],
            )
            return dpf(image, depth3)

        payload["methods"]["dpf_nutrition"] = measure_fps(
            step, batch, int(ruler["timed_steps"]), int(ruler["warmup_steps"])
        )
    except RuntimeError as exc:
        payload["methods"]["dpf_nutrition"] = f"measurement failed: {exc}"
    return payload


if __name__ == "__main__":
    main()
