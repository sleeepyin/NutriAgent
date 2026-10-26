"""Hold-out images for the single shared VLM pass.

Nutrition5K uses the evaluation half of the 698-dish test split (seed 42),
the same half D5 holds out from the calibrator. ECUSTFD and MetaFood3D use
the action-layer rows outside the policy split. NV-Real uses the official
validation images. One MetaFood3D object contributes the _000 frame.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mmcfa.data_build import build_ecustfd, build_mf3d
from mmcfa.paths import Paths


def _plain(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, (np.floating, float)):
            number = float(value)
            out[key] = number
        elif isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            out[key] = int(value)
        elif value is None or isinstance(value, (str, bool)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def n5k_eval_indices(n: int, seed: int = 42) -> np.ndarray:
    idx = np.arange(n)
    _cal, ev = train_test_split(idx, test_size=0.5, random_state=seed)
    return np.sort(np.asarray(ev))


def n5k_cal_indices(n: int, seed: int = 42) -> np.ndarray:
    idx = np.arange(n)
    cal, _ev = train_test_split(idx, test_size=0.5, random_state=seed)
    return np.sort(np.asarray(cal))


def _attach_ecustfd(ec, paths: Paths) -> None:
    raw = pd.read_csv(paths.expect("ecustfd_csv"))
    cat = raw.groupby("inst")["cat"].first().to_dict()
    for row in ec:
        row["cat"] = str(cat[row["episode_id"].split("|", 1)[1]])


def align_mf3d(paths: Paths) -> list[dict]:
    """Keep each label-table row's own category. One object id is reused by two foods."""
    labels = pd.read_csv(paths.expect("mf3d_labels"))
    rows = build_mf3d(paths)
    if len(labels) != len(rows):
        raise RuntimeError(f"MetaFood3D rows {len(rows)} != labels {len(labels)}")
    counts = labels["obj"].astype(str).value_counts()
    for row, rec in zip(rows, labels.itertuples(index=False)):
        obj = str(rec.obj)
        if row["episode_id"].split("|", 1)[1] != obj or abs(float(row["y"]) - float(rec.weight_g)) > 1e-3:
            raise RuntimeError(f"MetaFood3D alignment failed at {obj}")
        row["cat"] = str(rec.cat)
        row["sample_id"] = f"{obj}__{rec.cat}" if int(counts[obj]) > 1 else obj
    return rows


def _first_top_image(inst: str, paths: Paths) -> tuple[str, str]:
    ann = paths.data_ecustfd / "Annotations"
    jpg_dir = paths.data_ecustfd / "JPEGImages"
    xmls = sorted(ann.glob(f"{inst}T*.xml")) or sorted(ann.glob(f"{inst}*.xml"))
    if not xmls:
        return "", ""
    xml = xmls[0]
    for suffix in (".JPG", ".jpg", ".jpeg"):
        candidate = jpg_dir / (xml.stem + suffix)
        if candidate.is_file():
            return str(candidate), str(xml)
    return "", str(xml)


def _squash(text: str) -> str:
    out = []
    for ch in text.lower():
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def _mf3d_view(cat: str, obj: str, render: Path) -> str:
    cdir = render / cat
    if not cdir.is_dir():
        return ""
    names = [entry for entry in os.listdir(cdir) if (cdir / entry).is_dir()]
    hits = [entry for entry in names if entry.lower() == obj.lower()]
    if not hits:
        target = _squash(obj)
        hits = [entry for entry in names if _squash(entry) == target]
    if len(hits) != 1:
        return ""
    parent = cdir / hits[0]
    for sub in os.listdir(parent):
        original = parent / sub / "Original"
        if not original.is_dir():
            continue
        frames = sorted(p for p in original.iterdir() if p.suffix.lower() == ".png")
        prefer = [p for p in frames if p.stem.endswith("_000")]
        chosen = prefer[0] if prefer else (frames[0] if frames else None)
        if chosen is not None:
            return str(chosen)
    return ""


def build_holdout(paths: Paths | None = None) -> list[dict]:
    paths = paths or Paths.resolve()
    rows: list[dict] = []
    crops = np.load(paths.expect("deep_crops"))
    dish = np.asarray(crops["dish_id_test"]).astype(str)
    y = np.asarray(crops["y_test"], dtype=float)
    for index in n5k_eval_indices(len(y)):
        rows.append(
            _plain(
                {
                    "dataset": "n5k",
                    "sample_id": dish[index],
                    "group_id": dish[index],
                    "split": "eval",
                    "image": "",
                    "image_source": "",
                    "y_g": float(y[index]),
                    "cat": "",
                    "episode_id": f"n5k|{dish[index]}",
                    "has_depth": 1,
                    "n5k_index": int(index),
                    "nv_index": -1,
                    "ann_xml": "",
                }
            )
        )

    ec = build_ecustfd(paths)
    mf = align_mf3d(paths)
    _attach_ecustfd(ec, paths)
    for row in ec:
        if row["split"] == "policy":
            continue
        inst = row["episode_id"].split("|", 1)[1]
        image, xml = _first_top_image(inst, paths)
        rows.append(
            _plain(
                {
                    "dataset": "ecustfd",
                    "sample_id": inst,
                    "group_id": inst,
                    "split": str(row["split"]),
                    "image": image,
                    "image_source": "top_view" if image else "",
                    "y_g": float(row["y"]),
                    "cat": row["cat"],
                    "episode_id": row["episode_id"],
                    "has_depth": 0,
                    "n5k_index": -1,
                    "nv_index": -1,
                    "ann_xml": xml,
                }
            )
        )
    render = paths.expect("mf3d_render")
    for row in mf:
        if row["split"] == "policy":
            continue
        obj = row["episode_id"].split("|", 1)[1]
        image = _mf3d_view(row["cat"], obj, render)
        rows.append(
            _plain(
                {
                    "dataset": "mf3d",
                    "sample_id": row["sample_id"],
                    "group_id": obj,
                    "split": str(row["split"]),
                    "image": image,
                    "image_source": "frame_000" if image else "",
                    "y_g": float(row["y"]),
                    "cat": row["cat"],
                    "episode_id": row["episode_id"],
                    "has_depth": 0,
                    "n5k_index": -1,
                    "nv_index": -1,
                    "ann_xml": "",
                }
            )
        )

    image_dir = (
        paths.data_nvreal / "raw" / "nutritionverse-manual" / "nutritionverse-manual" / "images"
    )
    with np.load(paths.expect("nv_dish_cache"), allow_pickle=True) as nv:
        split = np.asarray(nv["split"]).astype(str)
        files = np.asarray(nv["file_name"]).astype(str)
        dishes = np.asarray(nv["dish_id"])
        weights = np.asarray(nv["y"], dtype=float)
        for index, name in enumerate(files):
            if split[index] != "val":
                continue
            path = image_dir / name
            rows.append(
                _plain(
                    {
                        "dataset": "nvreal",
                        "sample_id": name,
                        "group_id": str(int(dishes[index])),
                        "split": "val",
                        "image": str(path) if path.is_file() else "",
                        "image_source": "official_val" if path.is_file() else "",
                        "y_g": float(weights[index]),
                        "cat": "",
                        "episode_id": f"nvreal|{int(dishes[index])}|{name}",
                        "has_depth": 0,
                        "n5k_index": -1,
                        "nv_index": int(index),
                        "ann_xml": "",
                    }
                )
            )
    return rows
