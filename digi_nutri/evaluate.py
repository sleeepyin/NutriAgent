"""One official holdout read. Student scoring never loads depth or names."""

from __future__ import annotations

import logging
import subprocess

import torch
from torch.utils.data import DataLoader

from digi_nutri.datasets import (
    PrivilegedTrainDataset,
    RGBStudentTestDataset,
    collate_privileged,
    collate_student_test,
    student_test_records,
)
from digi_nutri.holdout_guard import assert_holdout_available, publish_holdout
from digi_nutri.jsonio import read_json, write_json
from digi_nutri.models.geometry_teacher import build_geometry_teacher
from digi_nutri.models.ingredient_teacher import build_ingredient_teacher
from digi_nutri.models.student import build_student
from digi_nutri.protocol import HOLDOUT_METRICS, RESULTS, WORK_ROOT
from digi_nutri.report import render_report
from digi_nutri.scoring import assemble_holdout_metrics

LOGGER = logging.getLogger("digi_nutri")
_STUDENT_KEYS = {"rgb", "y", "dish_id", "plate_group"}


def _load(model, path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _collect(dataset, collate, step, device, batch_size: int):
    size = batch_size
    while True:
        loader = DataLoader(
            dataset,
            batch_size=size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )
        rows = []
        try:
            for batch in loader:
                rows.extend(step(batch))
            return rows
        except RuntimeError as exc:
            if device.type != "cuda" or "out of memory" not in str(exc).lower() or size == 1:
                raise
            size = max(1, size // 2)
            torch.cuda.empty_cache()
            LOGGER.info("holdout microbatch reduced to %s", size)


@torch.inference_mode()
def predict_students(paths, records, image_size, hidden, fusion, device, batch_size, labels_per_variant):
    dataset = RGBStudentTestDataset(student_test_records(records), image_size)
    predictions = {}
    for variant, path in paths.items():
        model = build_student(
            n_labels=labels_per_variant[variant],
            pretrained=False,
            use_checkpoint=False,
            hidden=hidden,
            fusion=fusion,
        )
        _load(model, path, device)
        model.disable_aux_head()

        def step(batch, model=model):
            extra = set(batch) - _STUDENT_KEYS
            if extra:
                raise RuntimeError(f"student holdout batch has extra fields {sorted(extra)}")
            pred = model.forward_test(batch["rgb"].to(device))["nutrients"].float().cpu()
            return pred.tolist()

        predictions[variant] = _collect(dataset, collate_student_test, step, device, batch_size)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return predictions


@torch.inference_mode()
def predict_geometry_teacher(path, records, image_size, hidden, depth_fill, device, batch_size):
    dataset = PrivilegedTrainDataset(
        records,
        image_size,
        depth_fill=depth_fill,
        include_depth=True,
        include_ingredients=False,
    )
    model = _load(
        build_geometry_teacher(pretrained=False, use_checkpoint=False, hidden=hidden),
        path,
        device,
    )

    def step(batch, model=model):
        pred = model(
            batch["rgb"].to(device),
            batch["depth"].to(device),
        )["nutrients"].float().cpu()
        return pred.tolist()

    rows = _collect(dataset, collate_privileged, step, device, batch_size)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


@torch.inference_mode()
def predict_ingredient_teacher(
    path, records, image_size, hidden, text_cache, vocab, device, batch_size
):
    dataset = PrivilegedTrainDataset(
        records,
        image_size,
        include_depth=False,
        include_ingredients=True,
        text_cache=text_cache,
        vocab_index={name: index for index, name in enumerate(vocab)},
    )
    model = _load(
        build_ingredient_teacher(
            pretrained=False,
            use_checkpoint=False,
            hidden=hidden,
            text_dim=int(next(iter(text_cache.values())).shape[-1]),
        ),
        path,
        device,
    )

    def step(batch, model=model):
        pred = model(
            batch["rgb"].to(device),
            batch["name_emb"].to(device),
            batch["name_mask"].to(device),
        )["nutrients"].float().cpu()
        return pred.tolist()

    rows = _collect(dataset, collate_privileged, step, device, batch_size)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", *args],
            cwd=WORK_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def git_info() -> dict:
    return {
        "head": _git("HEAD"),
        "branch": _git("--abbrev-ref", "HEAD"),
        "main": _git("main"),
        "origin_main": _git("origin/main"),
        "digi_nutri": _git("digi-nutri"),
        "origin_digi_nutri": _git("origin/digi-nutri"),
    }


def _last_epoch(curves: dict) -> dict:
    last = {}
    for name, payload in curves.items():
        history = payload.get("history") or []
        if history:
            row = history[-1]
            last[name] = {"loss": row.get("loss"), "mae": row.get("mae"), "epoch": row.get("epoch")}
    return last


def run_official_holdout(
    test_records,
    frozen: dict,
    checkpoint_dir,
    text_cache,
    device,
) -> dict:
    assert_holdout_available(HOLDOUT_METRICS)
    labels = [
        {
            "dish_id": record["dish_id"],
            "plate_group": record["plate_group"],
            "y": list(record["y"]),
        }
        for record in test_records
    ]
    image_size = int(frozen["image_size"])
    hidden = int(frozen["hidden"])
    batch_size = int(frozen["batch_size"])
    vocab = list(frozen["vocab"])
    root = checkpoint_dir
    predictions = {
        "geometry_teacher": predict_geometry_teacher(
            root / "geometry_teacher.pt",
            test_records,
            image_size,
            hidden,
            frozen["depth_fill"],
            device,
            batch_size,
        ),
        "ingredient_teacher": predict_ingredient_teacher(
            root / "ingredient_teacher.pt",
            test_records,
            image_size,
            hidden,
            text_cache,
            vocab,
            device,
            batch_size,
        ),
    }
    label_counts = {"b1": 0, "b2a": 0, "b2b": len(vocab), "b3": len(vocab)}
    predictions.update(
        predict_students(
            {name: root / f"{name}.pt" for name in ("b1", "b2a", "b2b", "b3")},
            test_records,
            image_size,
            hidden,
            frozen["fusion"],
            device,
            batch_size,
            label_counts,
        )
    )
    metrics = assemble_holdout_metrics(
        labels,
        predictions,
        n_boot=int(frozen["bootstrap_samples"]),
        seed=int(frozen["bootstrap_seed"]),
        eps=float(frozen["pmae_eps"]),
        alpha=float(frozen["ci_alpha"]),
    )
    metrics["config"] = frozen
    metrics["git"] = git_info()
    curves_path = RESULTS / "train_curves.json"
    if curves_path.is_file():
        metrics["train_last_epoch"] = _last_epoch(read_json(curves_path))
    assert_holdout_available(HOLDOUT_METRICS)
    publish_holdout(HOLDOUT_METRICS, metrics)
    report = render_report(metrics)
    (RESULTS / "REPORT.md").write_text(report, encoding="utf-8")
    (RESULTS / "verdict.txt").write_text(metrics["verdict"] + "\n", encoding="utf-8")
    write_json(RESULTS / "verdict.json", {"verdict": metrics["verdict"], "decision": metrics["decision"]})
    LOGGER.info("holdout verdict %s", metrics["verdict"])
    return metrics
