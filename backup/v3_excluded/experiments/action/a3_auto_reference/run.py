"""A3: VLM coin box versus the annotated coin and a Hough circle, on ECUSTFD."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np

from experiments.vlm.geometry import load_coin_table
from experiments.vlm.scoring import block, clean
from experiments.vlm.store import load_record, read_manifest
from mmcfa.paths import Paths
from mmcfa.runtime import atomic_write_json, log, read_json


def _iou(a, b) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    if area <= 0:
        return 0.0
    return float(inter / area)


def _xml_coin(path: str):
    if not path or not Path(path).is_file():
        return None
    root = ET.parse(path).getroot()
    found = None
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name != "coin":
            continue
        box = obj.find("bndbox")
        found = [float(box.findtext(tag)) for tag in ("xmin", "ymin", "xmax", "ymax")]
    return found


def _diameter(box) -> float:
    return float(max(box[2] - box[0], box[3] - box[1]))


def vlm_coin(record: dict, min_conf: float = 0.3):
    best = None
    for ref in record.get("references") or []:
        if "coin" not in str(ref.get("object", "")):
            continue
        box = ref.get("bbox") or []
        if len(box) != 4 or float(ref.get("conf") or 0) < min_conf:
            continue
        if best is None or float(ref["conf"]) > best[0]:
            best = (float(ref["conf"]), box)
    if best is None:
        return None
    return best[1]


def hough_coin(path: str):
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        return None, f"cv2 unavailable: {exc}"
    image = cv2.imread(path)
    if image is None:
        return None, "imread failed"
    gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
    height, width = gray.shape[:2]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=120,
        param2=30,
        minRadius=12,
        maxRadius=max(16, int(0.12 * min(height, width))),
    )
    if circles is None:
        return None, ""
    x, y, radius = [float(v) for v in circles[0][0]]
    return [x - radius, y - radius, x + radius, y + radius], ""


def _summarize(name: str, y, pred, detected, size_err) -> dict:
    size = [v for v in size_err if v is not None and np.isfinite(v)]
    return {
        "name": name,
        "detection_rate_iou50": float(np.mean(detected)) if detected else None,
        "n": len(detected),
        "size_rel_error_on_hits": float(np.mean(size)) if size else None,
        "n_size": len(size),
        "end_to_end": block(y, pred),
    }


def main() -> int:
    paths = Paths.resolve()
    coins = load_coin_table(paths)
    rows = [row for row in read_manifest() if row["dataset"] == "ecustfd"]
    y, ann_p, vlm_p, hough_p, none_p = [], [], [], [], []
    vlm_hit, hough_hit = [], []
    vlm_size, hough_size = [], []
    missing = 0
    hough_note = ""
    for row in rows:
        factors = coins.factors(row["sample_id"])
        truth = _xml_coin(row.get("ann_xml") or "")
        record = load_record("ecustfd", row["sample_id"])
        if factors is None or truth is None or record is None:
            missing += 1
            continue
        truth_d = _diameter(truth)
        y.append(float(row["y_g"]))
        ann_p.append(coins.predict(row["sample_id"], factors["coin_px"], use_ref=True))
        no_ref = coins.predict(row["sample_id"], None, use_ref=False)
        none_p.append(no_ref)
        box = vlm_coin(record)
        if box is None:
            vlm_hit.append(0.0)
            vlm_p.append(no_ref)
            vlm_size.append(None)
        else:
            vlm_hit.append(1.0 if _iou(box, truth) >= 0.5 else 0.0)
            vlm_p.append(coins.predict(row["sample_id"], _diameter(box), use_ref=True))
            vlm_size.append(abs(_diameter(box) - truth_d) / truth_d)
        hough_box, note = hough_coin(row["image"])
        if note and not hough_note:
            hough_note = note
        if hough_box is None:
            hough_hit.append(0.0)
            hough_p.append(no_ref)
            hough_size.append(None)
        else:
            hough_hit.append(1.0 if _iou(hough_box, truth) >= 0.5 else 0.0)
            hough_p.append(coins.predict(row["sample_id"], _diameter(hough_box), use_ref=True))
            hough_size.append(abs(_diameter(hough_box) - truth_d) / truth_d)
    published = read_json(paths.results_diagnosis / "d4_ecustfd" / "summary.json", {})
    report = {
        "n_manifest": len(rows),
        "n_scored": len(y),
        "n_missing": missing,
        "subset": "ECUSTFD hold-out, first top view, D4 leave-one-out geometry",
        "annotated_coin": _summarize("annotated", y, ann_p, [1.0] * len(y), [0.0] * len(y)),
        "vlm_coin": _summarize("vlm", y, vlm_p, vlm_hit, vlm_size),
        "hough_coin": _summarize("hough", y, hough_p, hough_hit, hough_size),
        "no_coin": {"end_to_end": block(y, none_p)},
        "d4_full_set_coin_mape_pct": (published.get("e3_coin") or {}).get("mape"),
        "hough_note": hough_note,
        "rule": (
            "Food area and peer k, rho stay on the annotated top views. "
            "Only the test coin diameter changes. Hough keeps circles whose "
            "radius is between 12 px and 12% of the short side, and uses the strongest circle."
        ),
    }
    dest = paths.results_action / "a3_auto_reference" / "report.json"
    atomic_write_json(dest, clean(report))
    log(
        f"A3 n={len(y)} vlm_det={report['vlm_coin']['detection_rate_iou50']} "
        f"hough_det={report['hough_coin']['detection_rate_iou50']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
