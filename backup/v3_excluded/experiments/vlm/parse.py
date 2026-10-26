"""Turn one VLM object into the cache schema. Coordinates become original-image pixels."""
from __future__ import annotations

FORMS = {"rigid", "prepared", "liquid", "stacked"}


def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "yes", "1", "y"}


def _box(raw, width: int, height: int):
    if isinstance(raw, dict):
        keys = ("x0", "y0", "x1", "y1")
        if all(k in raw for k in keys):
            raw = [raw[k] for k in keys]
        elif all(k in raw for k in ("xmin", "ymin", "xmax", "ymax")):
            raw = [raw["xmin"], raw["ymin"], raw["xmax"], raw["ymax"]]
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    vals = [_float(v) for v in raw]
    if any(v is None for v in vals):
        return None
    if max(vals) <= 1.5:
        vals = [v * 1000.0 for v in vals]
    x0, y0, x1, y1 = vals
    if max(vals) <= 1000.5:
        x0, x1 = x0 / 1000.0 * width, x1 / 1000.0 * width
        y0, y1 = y0 / 1000.0 * height, y1 / 1000.0 * height
    elif max(x1, y1) > max(width, height) * 1.25:
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def parse_prediction(obj: dict, width: int, height: int) -> dict:
    name = str(obj.get("food_name") or obj.get("name") or "").strip()
    form = str(obj.get("form") or "").strip().lower()
    if form not in FORMS:
        form = "unknown"
    weight = _float(obj.get("weight_guess_g", obj.get("weight_g")))
    refs = []
    raw_refs = obj.get("references") or []
    if isinstance(raw_refs, dict):
        raw_refs = [raw_refs]
    if isinstance(raw_refs, list):
        for item in raw_refs:
            if not isinstance(item, dict):
                continue
            box = _box(item.get("bbox_1000", item.get("bbox")), width, height)
            size = _float(item.get("size_mm"))
            conf = _float(item.get("conf", item.get("confidence")))
            if conf is None:
                conf = 0.0
            refs.append(
                {
                    "object": str(item.get("object") or "other").strip().lower()[:40],
                    "bbox": box if box is not None else [],
                    "size_mm": size if size is not None and size > 0 else None,
                    "conf": max(0.0, min(1.0, conf)),
                }
            )
    scene = str(obj.get("scene") or "").strip()[:160]
    return {
        "food_name": name[:80],
        "form": form,
        "references": refs,
        "weight_guess_g": weight if weight is not None and weight > 0 else None,
        "scene": scene,
        "out_of_domain": _bool(obj.get("out_of_domain", False)),
    }


def usable(record: dict) -> bool:
    return bool(record.get("food_name")) and record.get("weight_guess_g") is not None


def self_check() -> None:
    raw = {
        "food_name": "apple",
        "form": "rigid",
        "references": [
            {"object": "Coin", "bbox_1000": [0, 0, 100, 200], "size_mm": 25, "conf": 0.8}
        ],
        "weight_guess_g": "180",
        "scene": "lab",
        "out_of_domain": "false",
    }
    parsed = parse_prediction(raw, 800, 600)
    assert parsed["food_name"] == "apple"
    assert parsed["weight_guess_g"] == 180
    assert parsed["out_of_domain"] is False
    assert parsed["references"][0]["bbox"] == [0.0, 0.0, 80.0, 120.0]
    unit = parse_prediction(
        {"food_name": "soup", "form": "nope", "weight_guess_g": 1, "references": [
            {"object": "coin", "bbox_1000": [0.1, 0.1, 0.2, 0.4], "size_mm": 25, "conf": 1}
        ]},
        1000,
        1000,
    )
    assert unit["form"] == "unknown"
    assert unit["references"][0]["bbox"] == [100.0, 100.0, 200.0, 400.0]
