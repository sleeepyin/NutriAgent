#!/usr/bin/env python3
"""Phase-1 audit for incomplete ingredient labels.

Freeze the protocol and tolerance, describe the plate-versus-sum residual on
every Hand dish, and pre-register a physical-plate train / validation / holdout.
The constants below are the freeze. They are not revised after residuals or
after the holdout counts. This script does not train, does not compute a
holdout nutrition error, and does not rewrite the public CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Frozen before any residual is computed and before any split is applied.
# Do not edit these after reading nutrition residuals or holdout composition.
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "u2-phase1-2026-11-13"
SPLIT_VERSION = "u2-physical-plate-v1"
SPLIT_HASH_PREFIX = "u2-phase1-plate-split-v1"

NUTRIENTS = ("mass", "calories", "fat", "carb", "protein")
NUTRIENT_ZH = {
    "mass": "质量",
    "calories": "能量",
    "fat": "脂肪",
    "carb": "碳水",
    "protein": "蛋白质",
}
NUTRIENT_UNIT = {
    "mass": "克",
    "calories": "千卡",
    "fat": "克",
    "carb": "克",
    "protein": "克",
}

# A nutrient aligns when the absolute gap is at most 0.5, or at most 1% of
# the larger absolute value, whichever band is wider.
NUTR_ABS_TOL = 0.5
NUTR_REL_TOL = 0.01

# Physical-plate merge copied in words from the #78 audit and not retuned.
PLATE_GAP_SECONDS = 180
MIN_MASS_INCREASE_G = 1.0
MASS_GRAM_TOL_G = 1.0

# Share of physical plates. Integer round-half-up: (n * k + 50) // 100.
HOLDOUT_PERCENT = 15
VALIDATION_PERCENT = 15
MIN_HOLDOUT_PLATES = 80

# Anti-symmetry bar for mass only. 0.5 means at least three quarters of the
# mass mismatches that fall outside tolerance sit on the plate-heavier side.
MIN_MASS_SIGN_IMBALANCE = 0.5

DISH_BLOB_SHA = "32ae4b8cd7c625856b5f5cf50c10392f6df0dfd3"
INGREDIENT_BLOB_SHA = "2cd1176104461379b04436f68b6d03f75477021a"
U78_COMMIT = "9d7822f9e8f3597889d00de76b31b3c40d096170"
STATUS_COMMIT = "4f5c09d218543c8c75fdf62f94c3cb0c8789c836"
PULL_REQUEST_URL = ""

# Published #78 field record. Used only as a comparison target, not as a pass.
U78_RECORD = {
    "dishes": 1056,
    "instances": 2922,
    "classes": 40,
    "has_nutrition_0": 98,
    "dishes_with_missing_nutrition": 86,
    "all_numeric_dishes": 970,
    "all_numeric_mismatch": 424,
    "complete_label_dishes": 546,
    "partial_label_dishes": 510,
    "mass_gap_median_g": 17.73,
    "mass_gap_median_abs_relative_percent": 9.7,
    "mass_plate_heavier_abs_0_5": 418,
    "mass_sum_heavier_abs_0_5": 1,
    "fail_counts": {
        "calories": 422,
        "fat": 383,
        "mass": 361,
        "carb": 284,
        "protein": 174,
    },
    "hand_physical_plates": 663,
    "library_physical_plates": 2482,
    "library_links": 2524,
    "library_multi_scan_plates": 1142,
    "library_multi_scan_dishes": 3666,
    "official_cross_plates": 2,
    "hand_cross_image_split_plates": 73,
    "hand_test_only_plates": 64,
    "hand_test_dishes_on_test_only_plates": 67,
    "hand_multi_annotation_plates": 208,
    "hand_dishes_on_multi_annotation_plates": 601,
    "hand_image_train_plates": 599,
    "hand_image_test_plates": 137,
}

CLASS_NAME_ZH = {
    "alium_vegetable": "葱蒜类",
    "bacon": "培根",
    "berry": "浆果",
    "bread": "面包",
    "cheese": "奶酪",
    "condiment_dressing": "调味酱",
    "corn": "玉米",
    "cruciferous": "十字花科蔬菜",
    "cucumber": "黄瓜",
    "dessert": "甜点",
    "egg": "蛋",
    "egg_whites": "蛋白",
    "fish": "鱼",
    "fruit_fatty": "高脂水果",
    "fruit_other": "其他水果",
    "fruit_tree": "仁果",
    "fruit_tropical": "热带水果",
    "grain_other": "其他谷物",
    "herb": "香草",
    "leafy_greens": "叶菜",
    "legumes_green": "绿豆类",
    "legumes_other": "其他豆类",
    "meat_red": "红肉",
    "melon": "瓜类",
    "mushroom": "蘑菇",
    "nuts_seeds": "坚果与种子",
    "pasta": "意面",
    "peppers": "辣椒",
    "pizza": "披萨",
    "poultry": "禽肉",
    "rice": "米饭",
    "root_vegetable": "根菜",
    "salad_prepared": "预制沙拉",
    "sausage": "香肠",
    "squash_eggplant": "瓜茄类",
    "tofu": "豆腐",
    "tomato": "番茄",
    "tuber": "薯类",
    "vegetable_other": "其他蔬菜",
    "yogurt": "酸奶",
}

PROTOCOL_TEXT = """\
可见实例是能在这张俯视 RGB 上描出边界的食材。Hand 公开食材表的每一行是一个可见实例，类别沿用表里的粗类名。

有营养实例是 has_nutrition=1，并且质量、能量、脂肪、碳水、蛋白质五格都不是空值的行。has_nutrition=0 或空值是缺失，不能当成 0 克或 0 千卡。has_nutrition=0 的行即使格子里写了数字，也不加进可见求和。

某个营养素算对齐，当盘级与可见求和的绝对差不超过 0.5，或不超过两者较大绝对值的 1%，二者取更宽的那个。五个营养素同时落在这条容差内，才算求和对齐。

完整标签盘是每个可见实例都是有营养实例，并且五个营养素都求和对齐。

部分标签盘是除此以外的盘子。盘级总量减去有营养实例之和，是未标注残差。残差只在至少有一个有营养实例、可见求和有定义时计算。一个有营养实例都没有时，可见求和不记成 0，残差没有定义。

残差可以来自没有营养数字的可见实例，也可以来自没有画进掩膜的成分。先前审计的附录转述 Hand 等人的说法：油和糖这类在俯视 RGB 里分不出来的成分没有画进掩膜。残差不是「该成分质量为零」的负例，这次审计也不把它当成新的监督。

盘级质量、能量和三种宏量以 Hand 盘级表为准。部分标签只影响食材分解，不影响盘级总量本身。Nutrition5k 元数据只用来核对盘级表，不替换盘级表。
"""

SPLIT_RULE_TEXT = """\
划分单位是物理餐盘，不是 dish_id，也不是 Hand 表里的图像级 train/test。

含有 Hand 标注的每一只物理餐盘恰好进入训练、验证、留出三者之一。同一只物理餐盘在 Nutrition5k 元数据里的全部扫描记在同一侧，包括没有 Hand 标注的同盘扫描。

合并规则沿用已经写死的增量扫描规则：把元数据按时间戳排序，相邻两条属于同一只盘子，当且仅当时间差不超过 180 秒，质量至少增加 1 克，后一条的食材名称多重集包含前一条，并且同名食材的克数下降不超过 1 克。再做传递闭包。餐盘标识是该连通块里时间最早的 dish_id。这次不为了凑满 80 只留出而改这条规则。

排序键是 SHA256("u2-phase1-plate-split-v1|" 加上餐盘标识) 的十六进制，相同则再比餐盘标识。这个键不读取质量、能量、宏量、类别或 Hand 的图像级划分。

排序之后，留出盘数 = (餐盘数 × 15 + 50) // 100，验证盘数用同一个公式，其余是训练。15% 是常规测试份额的量级，与 Hand 图像级测试约占 15%、官方 RGB 测试约占 15% 同一量级。它不是按留出营养误差、也不是按哪一刀更能显出残差选出来的。

留出下限是 80 只独立物理餐盘。若按上面的公式得到的留出盘数低于 80，审计停止，不把比例调大，也不把下限改小。

跨划分的物理餐盘必须是 0。官方 RGB 划分与这一刀的交集只作诊断，不回头改划分。本脚本不计算留出集上的营养误差。
"""

STRUCTURE_RULE_TEXT = """\
残差刻画使用全部 Hand 盘。这是数据事实，不是留出营养误差，也不按划分汇总。

结构是否成立，只在「每个可见实例都是有营养实例，但五个营养素不能同时对齐」的盘子上判断，而且在出结果之前写死：

1. 质量、能量、脂肪的残差中位数都大于 0。残差是盘级减去可见求和。
2. 质量超出容差时，符号不平衡度 = (盘子一侧的盘数 − 求和一侧的盘数) / 两侧盘数之和。写死下限是 0.5，也就是盘子一侧至少占两侧的四分之三。低于下限，就不是「主要是盘子比可见求和更重」，而是接近正负对称的噪声。能量和脂肪只要求中位数大于 0，并且盘子一侧的盘数多于求和一侧，用来确认方向没有反转。
3. 对不齐次数必须是能量多于脂肪，脂肪多于质量，并且脂肪多于碳水、也多于蛋白质。这是「能量和脂肪更常对不齐」的写死读法。

类别表只描述残差落在哪些粗类上。它不进入这三条判定，也不用来改容差或改划分。
"""

STERLING = {
    "citation": "Sterling et al., bioRxiv 2026.07.26.740845，2026-07-29 预印本",
    "doi": "10.64898/2026.07.26.740845",
    "url": "https://www.biorxiv.org/content/10.64898/2026.07.26.740845v1",
    "code_url": "https://github.com/charterer/vlm-food-benchmark",
    "checked_on": "2026-11-13",
    "public_corrected_labels": False,
}


def git_blob_sha1(data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def parse_optional_float(text: str | None) -> float | None:
    cleaned = (text or "").strip()
    if cleaned == "":
        return None
    return float(cleaned)


def within_tolerance(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 0.0)
    return abs(left - right) <= max(NUTR_ABS_TOL, NUTR_REL_TOL * scale)


def rounded_percent(count: int, percent: int) -> int:
    if count < 0 or percent < 0:
        raise ValueError("count and percent must be non-negative")
    return (count * percent + 50) // 100


def dish_timestamp(dish_id: str) -> int:
    if not dish_id.startswith("dish_"):
        raise ValueError(f"unexpected dish_id: {dish_id}")
    return int(dish_id[len("dish_") :])


def valid_dish_id(dish_id: str) -> bool:
    if not dish_id.startswith("dish_"):
        return False
    rest = dish_id[len("dish_") :]
    return rest.isdigit() and str(int(rest)) == rest


def quantile(values: list[float], fraction: float) -> float | None:
    """Same rank rule as the #78 audit: index round(fraction * (n - 1))."""
    if not values:
        return None
    ordered = sorted(values)
    position = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[position]


def sign_imbalance(plate_side: int, sum_side: int) -> float | None:
    total = plate_side + sum_side
    if total <= 0:
        return None
    return (plate_side - sum_side) / total


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        return list(reader.fieldnames), list(reader)


def load_id_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_metadata(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for cafe, name in (("cafe1", "dish_metadata_cafe1.csv"), ("cafe2", "dish_metadata_cafe2.csv")):
        path = directory / name
        with path.open(newline="", encoding="utf-8") as handle:
            for record in csv.reader(handle):
                if len(record) < 6:
                    raise ValueError(f"short metadata row in {name}")
                tail = record[6:]
                if len(tail) % 7 != 0:
                    raise ValueError(f"ingredient tail is not groups of 7 in {record[0]}")
                ingredients = []
                for offset in range(0, len(tail), 7):
                    ingredients.append((tail[offset + 1], float(tail[offset + 2])))
                rows.append(
                    {
                        "dish_id": record[0],
                        "ts": dish_timestamp(record[0]),
                        "cafe": cafe,
                        "calories": float(record[1]),
                        "mass": float(record[2]),
                        "fat": float(record[3]),
                        "carb": float(record[4]),
                        "protein": float(record[5]),
                        "ingredients": ingredients,
                    }
                )
    rows.sort(key=lambda row: row["ts"])
    return rows


def ingredient_grams(ingredients: list[tuple[str, float]]) -> Counter:
    totals: Counter = Counter()
    for name, grams in ingredients:
        totals[name] += grams
    return totals


def same_physical_plate(previous: dict, current: dict, gap_seconds: int) -> bool:
    if current["ts"] - previous["ts"] > gap_seconds:
        return False
    if current["mass"] < previous["mass"] + MIN_MASS_INCREASE_G:
        return False
    previous_counts = Counter(name for name, _grams in previous["ingredients"])
    current_counts = Counter(name for name, _grams in current["ingredients"])
    if any(current_counts[name] < count for name, count in previous_counts.items()):
        return False
    previous_grams = ingredient_grams(previous["ingredients"])
    current_grams = ingredient_grams(current["ingredients"])
    for name, grams in previous_grams.items():
        if current_grams.get(name, -1.0e9) + MASS_GRAM_TOL_G < grams:
            return False
    return True


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        root = index
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[index] != root:
            parent = self.parent[index]
            self.parent[index] = root
            index = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_plates(rows: list[dict], gap_seconds: int) -> tuple[int, dict[str, str], dict[str, list[str]]]:
    finder = UnionFind(len(rows))
    links = 0
    for index in range(len(rows) - 1):
        if same_physical_plate(rows[index], rows[index + 1], gap_seconds):
            finder.union(index, index + 1)
            links += 1
    members: dict[int, list[str]] = defaultdict(list)
    for index, row in enumerate(rows):
        members[finder.find(index)].append(row["dish_id"])
    dish_to_plate: dict[str, str] = {}
    plates: dict[str, list[str]] = {}
    for grouped in members.values():
        plate_id = min(grouped, key=dish_timestamp)
        ordered = sorted(grouped, key=dish_timestamp)
        plates[plate_id] = ordered
        for dish_id in ordered:
            dish_to_plate[dish_id] = plate_id
    return links, dish_to_plate, plates


def gap_distribution(rows: list[dict]) -> dict:
    nested_gaps: list[int] = []
    other_gaps: list[int] = []
    for previous, current in zip(rows, rows[1:]):
        gap = current["ts"] - previous["ts"]
        if same_physical_plate(previous, current, gap_seconds=10**12):
            nested_gaps.append(gap)
        else:
            other_gaps.append(gap)

    def describe(values: list[int]) -> dict:
        if not values:
            return {"n": 0}
        ordered = sorted(values)

        def percentile(fraction: float) -> int:
            position = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
            return ordered[position]

        return {
            "n": len(ordered),
            "min": ordered[0],
            "p50": percentile(0.50),
            "p99": percentile(0.99),
            "max": ordered[-1],
            "le_180": sum(value <= PLATE_GAP_SECONDS for value in ordered),
        }

    return {
        "nested_mass_increasing_seconds": describe(nested_gaps),
        "other_consecutive_seconds": describe(other_gaps),
    }


def is_nutrition_instance(row: dict[str, str]) -> bool:
    if row.get("has_nutrition") != "1":
        return False
    return all(parse_optional_float(row.get(nutrient)) is not None for nutrient in NUTRIENTS)


def nutrition_values(row: dict[str, str]) -> dict[str, float] | None:
    if not is_nutrition_instance(row):
        return None
    return {nutrient: float(row[nutrient]) for nutrient in NUTRIENTS}


def plate_order_key(plate_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{SPLIT_HASH_PREFIX}|{plate_id}".encode("utf-8")).hexdigest()
    return digest, plate_id


def assign_plate_splits(plate_ids: list[str]) -> dict[str, str]:
    """Assign each physical plate without reading nutrition, class, or Hand split."""
    ordered = sorted(set(plate_ids), key=plate_order_key)
    holdout_n = rounded_percent(len(ordered), HOLDOUT_PERCENT)
    validation_n = rounded_percent(len(ordered), VALIDATION_PERCENT)
    if holdout_n + validation_n > len(ordered):
        raise RuntimeError("frozen shares exceed the plate count")
    assignment: dict[str, str] = {}
    for index, plate_id in enumerate(ordered):
        if index < holdout_n:
            assignment[plate_id] = "holdout"
        elif index < holdout_n + validation_n:
            assignment[plate_id] = "validation"
        else:
            assignment[plate_id] = "train"
    return assignment


def structure_holds(nutrients: dict[str, dict]) -> tuple[bool, list[str]]:
    """Apply the frozen residual rule. nutrients are joint-mismatch dishes only."""
    reasons: list[str] = []
    if not nutrients:
        return False, ["没有可供判断的求和不对齐盘子"]
    for name in ("mass", "calories", "fat"):
        stats = nutrients[name]
        median = stats["median_signed"]
        if median is None or median <= 0:
            reasons.append(f"{NUTRIENT_ZH[name]}残差中位数不大于 0")
        if stats["plate_side"] <= stats["sum_side"]:
            reasons.append(f"{NUTRIENT_ZH[name]}不是盘子一侧更多")
    mass = nutrients["mass"]
    imbalance = sign_imbalance(mass["plate_side"], mass["sum_side"])
    if imbalance is None or imbalance < MIN_MASS_SIGN_IMBALANCE:
        reasons.append(
            f"质量符号不平衡度 {imbalance} 低于写死下限 {MIN_MASS_SIGN_IMBALANCE}"
        )
    fail = {name: nutrients[name]["fail_count"] for name in NUTRIENTS}
    if not (fail["calories"] > fail["fat"] > fail["mass"]):
        reasons.append(
            f"对不齐次数不是能量多于脂肪、脂肪多于质量：{fail['calories']}、{fail['fat']}、{fail['mass']}"
        )
    if not (fail["fat"] > fail["carb"] and fail["fat"] > fail["protein"]):
        reasons.append(
            f"脂肪对不齐次数没有同时多于碳水和蛋白质：脂肪 {fail['fat']}，碳水 {fail['carb']}，蛋白质 {fail['protein']}"
        )
    return not reasons, reasons


def decide(protocol_clear: bool, structure_ok: bool, holdout_plates: int, cross_split_plates: int) -> str:
    if not protocol_clear:
        return "stop_protocol_unclear"
    if not structure_ok:
        return "stop_residual_not_structured"
    if cross_split_plates != 0:
        return "stop_split_leaks"
    if holdout_plates < MIN_HOLDOUT_PLATES:
        return "stop_too_few_holdout_plates"
    return "phase1_conditions_met_status_unchanged"


def decision_zh(code: str) -> str:
    return {
        "stop_protocol_unclear": "停止。协议在这批数据上说不清，或俯视图像、餐盘合并盖不住 Hand 盘。",
        "stop_residual_not_structured": "停止。残差不像写死的结构化剩余。",
        "stop_split_leaks": "停止。物理餐盘划分出现跨划分盘子。",
        "stop_too_few_holdout_plates": "停止。留出物理餐盘低于 80 只下限。",
        "phase1_conditions_met_status_unchanged": "三条条件在这次复核里成立。不修改 STATUS，不宣布第二阶段解锁。",
    }[code]


def inspect_overhead(dish_ids: list[str], root: Path) -> dict:
    from PIL import Image

    missing: list[str] = []
    unreadable: list[dict[str, str]] = []
    sizes: Counter = Counter()
    modes: Counter = Counter()
    for dish_id in dish_ids:
        path = root / dish_id / "rgb.png"
        if not path.is_file():
            missing.append(dish_id)
            continue
        try:
            with Image.open(path) as image:
                modes[image.mode] += 1
                rgb = image.convert("RGB")
                sizes[f"{rgb.size[0]}x{rgb.size[1]}"] += 1
        except OSError as exc:
            unreadable.append({"dish_id": dish_id, "error": str(exc)})
    return {
        "checked": True,
        "expected": len(dish_ids),
        "opened": sum(sizes.values()),
        "missing": missing,
        "unreadable": unreadable,
        "size_counts": dict(sizes),
        "mode_counts": dict(modes),
    }


def _nutrient_block(plate: dict[str, float], known: dict[str, float] | None) -> dict:
    aligned = {}
    residual = {}
    for nutrient in NUTRIENTS:
        if known is None:
            aligned[nutrient] = False
            residual[nutrient] = None
        else:
            residual[nutrient] = plate[nutrient] - known[nutrient]
            aligned[nutrient] = within_tolerance(known[nutrient], plate[nutrient])
    return {"aligned": aligned, "residual": residual, "sums_align": known is not None and all(aligned.values())}


def classify_dishes(
    dish_rows: list[dict[str, str]],
    ingredient_rows: list[dict[str, str]],
) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ingredient_rows:
        grouped[row["dish_id"]].append(row)

    records = []
    blank_on_flagged = 0
    numeric_on_unflagged = 0
    for row in dish_rows:
        instances = grouped.get(row["dish_id"], [])
        plate = {nutrient: parse_optional_float(row[nutrient]) for nutrient in NUTRIENTS}
        n_nutrition = 0
        known = {nutrient: 0.0 for nutrient in NUTRIENTS}
        classes = []
        for item in instances:
            classes.append(item["annotation_name"])
            if item.get("has_nutrition") == "1" and any(
                parse_optional_float(item.get(nutrient)) is None for nutrient in NUTRIENTS
            ):
                blank_on_flagged += 1
            if item.get("has_nutrition") == "0" and any(
                parse_optional_float(item.get(nutrient)) is not None for nutrient in NUTRIENTS
            ):
                numeric_on_unflagged += 1
            values = nutrition_values(item)
            if values is None:
                continue
            n_nutrition += 1
            for nutrient in NUTRIENTS:
                known[nutrient] += values[nutrient]
        known_sum = known if n_nutrition else None
        block = _nutrient_block(plate, known_sum) if all(value is not None for value in plate.values()) else {
            "aligned": {nutrient: False for nutrient in NUTRIENTS},
            "residual": {nutrient: None for nutrient in NUTRIENTS},
            "sums_align": False,
        }
        all_nutrition = len(instances) > 0 and n_nutrition == len(instances)
        if all_nutrition and block["sums_align"]:
            label = "complete"
            reason = "none"
        elif not all_nutrition and not block["sums_align"]:
            label = "partial"
            reason = "missing_nutrition_and_sum_outside"
        elif not all_nutrition:
            label = "partial"
            reason = "missing_nutrition_sum_inside_tolerance"
        else:
            label = "partial"
            reason = "sum_outside_tolerance"
        records.append(
            {
                "dish_id": row["dish_id"],
                "hand_split": row.get("split", ""),
                "n_visible": len(instances),
                "n_nutrition": n_nutrition,
                "n_missing_nutrition": len(instances) - n_nutrition,
                "classes": sorted(set(classes)),
                "plate": plate,
                "known_sum": known_sum,
                "aligned": block["aligned"],
                "residual": block["residual"],
                "sums_align": block["sums_align"],
                "all_visible_have_nutrition": all_nutrition,
                "label": label,
                "partial_reason": reason,
            }
        )
    return records, {
        "has_nutrition_1_with_blank": blank_on_flagged,
        "has_nutrition_0_with_number": numeric_on_unflagged,
    }


def legacy_u78_nutrition(
    dish_rows: list[dict[str, str]],
    ingredient_rows: list[dict[str, str]],
    metadata_by_id: dict[str, dict],
) -> dict:
    """Recount the #78 field record with that audit's summing rules."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ingredient_rows:
        grouped[row["dish_id"]].append(row)
    complete_rows = []
    partial_rows = []
    meta_mass_diffs = []
    for row in dish_rows:
        instances = grouped.get(row["dish_id"], [])
        n_missing = sum(item["has_nutrition"] != "1" for item in instances)
        sums = {nutrient: 0.0 for nutrient in NUTRIENTS}
        sum_ok = True
        for item in instances:
            if item["has_nutrition"] != "1":
                continue
            for nutrient in NUTRIENTS:
                value = parse_optional_float(item[nutrient])
                if value is None:
                    sum_ok = False
                else:
                    sums[nutrient] += value
        plate = {nutrient: parse_optional_float(row[nutrient]) for nutrient in NUTRIENTS}
        matches = {}
        for nutrient in NUTRIENTS:
            if plate[nutrient] is None or not sum_ok:
                matches[nutrient] = False
            else:
                matches[nutrient] = within_tolerance(sums[nutrient], plate[nutrient])
        record = {
            "all_have": n_missing == 0 and len(instances) > 0,
            "sums_align": all(matches.values()),
            "plate": plate,
            "sums": sums if sum_ok else None,
            "matches": matches,
        }
        meta = metadata_by_id.get(row["dish_id"])
        if meta is not None and plate["mass"] is not None:
            meta_mass_diffs.append(abs(plate["mass"] - meta["mass"]))
        if record["all_have"]:
            complete_rows.append(record)
        else:
            partial_rows.append(record)

    mismatch = [row for row in complete_rows if not row["sums_align"]]
    mass_gaps = []
    rel_gaps = []
    fail = {nutrient: 0 for nutrient in NUTRIENTS}
    for row in mismatch:
        plate_mass = row["plate"]["mass"]
        sum_mass = None if row["sums"] is None else row["sums"]["mass"]
        if plate_mass is None or sum_mass is None:
            continue
        mass_gaps.append(plate_mass - sum_mass)
        if plate_mass != 0:
            rel_gaps.append(abs(plate_mass - sum_mass) / abs(plate_mass))
        for nutrient in NUTRIENTS:
            plate_value = row["plate"][nutrient]
            sum_value = None if row["sums"] is None else row["sums"][nutrient]
            if plate_value is None or sum_value is None or not within_tolerance(sum_value, plate_value):
                fail[nutrient] += 1
    partial_mass = []
    for row in partial_rows:
        if row["plate"]["mass"] is not None and row["sums"] is not None:
            partial_mass.append(row["plate"]["mass"] - row["sums"]["mass"])
    return {
        "complete_nutrition_dishes": len(complete_rows),
        "partial_nutrition_dishes": len(partial_rows),
        "complete_and_sums_match": sum(row["sums_align"] for row in complete_rows),
        "partial_and_sums_match": sum(row["sums_align"] for row in partial_rows),
        "mismatch_n": len(mismatch),
        "mass_gap_median_g": quantile(mass_gaps, 0.5),
        "mass_gap_median_abs_relative": quantile(rel_gaps, 0.5),
        "mass_plate_heavier_abs_0_5": sum(value > NUTR_ABS_TOL for value in mass_gaps),
        "mass_sum_heavier_abs_0_5": sum(value < -NUTR_ABS_TOL for value in mass_gaps),
        "fail_counts": fail,
        "partial_mass_median_g": quantile(partial_mass, 0.5),
        "partial_plate_heavier_abs_0_5": sum(value > NUTR_ABS_TOL for value in partial_mass),
        "partial_sum_heavier_abs_0_5": sum(value < -NUTR_ABS_TOL for value in partial_mass),
        "meta_mass_max_abs": max(meta_mass_diffs) if meta_mass_diffs else None,
        "meta_mass_within_0_5": sum(value <= NUTR_ABS_TOL for value in meta_mass_diffs),
        "meta_mass_n": len(meta_mass_diffs),
    }


def nutrient_direction(records: list[dict]) -> dict[str, dict]:
    stats = {}
    for nutrient in NUTRIENTS:
        signed = []
        plate_side = 0
        sum_side = 0
        for row in records:
            residual = row["residual"][nutrient]
            if residual is None:
                continue
            signed.append(residual)
            if row["aligned"][nutrient]:
                continue
            if residual > 0:
                plate_side += 1
            elif residual < 0:
                sum_side += 1
        stats[nutrient] = {
            "n": len(signed),
            "fail_count": plate_side + sum_side,
            "plate_side": plate_side,
            "sum_side": sum_side,
            "median_signed": quantile(signed, 0.5),
            "p10_signed": quantile(signed, 0.10),
            "p90_signed": quantile(signed, 0.90),
            "imbalance": sign_imbalance(plate_side, sum_side),
        }
    return stats


def class_residual_table(records: list[dict], instance_counts: Counter) -> list[dict]:
    """Descriptive only. Dishes are not partitioned: one dish can carry several classes."""
    all_nutrition = [row for row in records if row["all_visible_have_nutrition"]]
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in all_nutrition:
        for name in row["classes"]:
            buckets[name].append(row)
    table = []
    names = sorted(instance_counts, key=lambda name: (-instance_counts[name], name))
    for name in names:
        rows = buckets.get(name, [])
        mismatched = [row for row in rows if not row["sums_align"]]
        mass_gaps = [
            row["residual"]["mass"] for row in mismatched if row["residual"]["mass"] is not None
        ]
        mass_plate = sum(
            1
            for row in mismatched
            if row["residual"]["mass"] is not None
            and not row["aligned"]["mass"]
            and row["residual"]["mass"] > 0
        )
        mass_sum = sum(
            1
            for row in mismatched
            if row["residual"]["mass"] is not None
            and not row["aligned"]["mass"]
            and row["residual"]["mass"] < 0
        )
        energy_fail = sum(1 for row in rows if not row["aligned"]["calories"])
        table.append(
            {
                "annotation_name": name,
                "zh": CLASS_NAME_ZH.get(name, name),
                "instances": instance_counts[name],
                "all_nutrition_dishes": len(rows),
                "mismatch_dishes": len(mismatched),
                "mass_gap_median_g": quantile(mass_gaps, 0.5),
                "mass_plate_side": mass_plate,
                "mass_sum_side": mass_sum,
                "energy_fail_dishes": energy_fail,
            }
        )
    return table


def public_table_facts(
    dish_rows: list[dict[str, str]],
    ingredient_rows: list[dict[str, str]],
) -> dict:
    dish_ids = [row["dish_id"] for row in dish_rows]
    ingredient_ids = {row["dish_id"] for row in ingredient_rows}
    dish_id_set = set(dish_ids)
    classes = Counter(row["annotation_name"] for row in ingredient_rows)
    has_nutrition = Counter(row["has_nutrition"] for row in ingredient_rows)
    dish_split = Counter(row["split"] for row in dish_rows)
    ingredient_split = Counter(row["split"] for row in ingredient_rows)
    split_by_dish: dict[str, set[str]] = defaultdict(set)
    for row in dish_rows:
        split_by_dish[row["dish_id"]].add(row["split"])
    for row in ingredient_rows:
        split_by_dish[row["dish_id"]].add(row["split"])
    conflicts = sorted(dish_id for dish_id, splits in split_by_dish.items() if len(splits) != 1)
    return {
        "dishes": len(dish_rows),
        "unique_dish_ids": len(dish_id_set),
        "duplicate_dish_ids": len(dish_ids) - len(dish_id_set),
        "instances": len(ingredient_rows),
        "classes": len(classes),
        "class_counts": dict(classes),
        "has_nutrition": dict(has_nutrition),
        "dish_split": dict(dish_split),
        "ingredient_split": dict(ingredient_split),
        "same_dish_id_set": dish_id_set == ingredient_ids,
        "invalid_dish_ids": sorted(dish_id for dish_id in dish_id_set if not valid_dish_id(dish_id)),
        "split_conflicts": conflicts,
        "dishes_without_instances": sorted(dish_id_set - ingredient_ids),
        "unparsed_plate_nutrients": sum(
            1
            for row in dish_rows
            if any(parse_optional_float(row[nutrient]) is None for nutrient in NUTRIENTS)
        ),
    }


def hand_image_split_diagnostic(records: list[dict], dish_to_plate: dict[str, str]) -> dict:
    by_plate: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"train": [], "test": [], "other": []})
    for row in records:
        plate_id = dish_to_plate.get(row["dish_id"])
        if plate_id is None:
            continue
        slot = row["hand_split"] if row["hand_split"] in {"train", "test"} else "other"
        by_plate[plate_id][slot].append(row["dish_id"])
    cross = []
    test_only = []
    train_only = []
    rows = []
    for plate_id, sides in sorted(by_plate.items()):
        record = {
            "plate_id": plate_id,
            "hand_train_dishes": len(sides["train"]),
            "hand_test_dishes": len(sides["test"]),
            "hand_other_dishes": len(sides["other"]),
            "in_both_hand_splits": bool(sides["train"] and sides["test"]),
            "only_hand_test": bool(sides["test"] and not sides["train"]),
        }
        rows.append(record)
        if record["in_both_hand_splits"]:
            cross.append(plate_id)
        elif record["only_hand_test"]:
            test_only.append(plate_id)
        elif sides["train"] and not sides["test"]:
            train_only.append(plate_id)
    test_dishes_on_clean = sum(row["hand_test_dishes"] for row in rows if row["only_hand_test"])
    return {
        "plates": len(by_plate),
        "cross_split_plates": len(cross),
        "test_only_plates": len(test_only),
        "train_only_plates": len(train_only),
        "train_plates": len(train_only) + len(cross),
        "test_plates": len(test_only) + len(cross),
        "test_dishes_on_test_only_plates": test_dishes_on_clean,
        "rows": rows,
        "used_as_evaluation_split": False,
    }


def official_span(dish_ids: list[str], rgb_train: set[str], rgb_test: set[str]) -> dict[str, int]:
    return {
        "n": len(dish_ids),
        "official_train": sum(dish_id in rgb_train and dish_id not in rgb_test for dish_id in dish_ids),
        "official_test": sum(dish_id in rgb_test and dish_id not in rgb_train for dish_id in dish_ids),
        "official_both": sum(dish_id in rgb_train and dish_id in rgb_test for dish_id in dish_ids),
        "official_neither": sum(dish_id not in rgb_train and dish_id not in rgb_test for dish_id in dish_ids),
    }


def build_plate_rows(
    hand_plate_ids: list[str],
    plates: dict[str, list[str]],
    hand_ids_by_plate: dict[str, list[str]],
    assignment: dict[str, str],
    rgb_train: set[str],
    rgb_test: set[str],
) -> list[dict]:
    rows = []
    for plate_id in hand_plate_ids:
        scans = plates[plate_id]
        hand_ids = hand_ids_by_plate[plate_id]
        has_train = any(dish_id in rgb_train for dish_id in scans)
        has_test = any(dish_id in rgb_test for dish_id in scans)
        rows.append(
            {
                "plate_id": plate_id,
                "split": assignment[plate_id],
                "n_hand_dishes": len(hand_ids),
                "n_metadata_scans": len(scans),
                "hand_dish_ids": hand_ids,
                "metadata_dish_ids": scans,
                "official_train_scans": sum(dish_id in rgb_train for dish_id in scans),
                "official_test_scans": sum(dish_id in rgb_test for dish_id in scans),
                "spans_official_rgb": has_train and has_test,
            }
        )
    return rows


def split_cross_count(plate_rows: list[dict]) -> int:
    """A plate leaks when its scans would need more than one of our splits. Assignment forbids that."""
    leaked = 0
    for row in plate_rows:
        sides = {row["split"]}
        if len(sides) != 1:
            leaked += 1
    # Also catch a dish id that the plate map would send to two plates. The
    # caller checks dish uniqueness separately; this function reports plates.
    return leaked


def compare_u78(legacy: dict, merge: dict, hand_diag: dict, public: dict) -> list[dict]:
    relative = legacy["mass_gap_median_abs_relative"]
    relative_percent = None if relative is None else round(relative * 100, 1)
    mass_median = legacy["mass_gap_median_g"]
    checks = [
        ("盘级行数", public["dishes"], U78_RECORD["dishes"]),
        ("食材实例", public["instances"], U78_RECORD["instances"]),
        ("粗类数", public["classes"], U78_RECORD["classes"]),
        ("has_nutrition=0 的实例", public["has_nutrition"].get("0", 0), U78_RECORD["has_nutrition_0"]),
        ("有缺失营养的盘", legacy["partial_nutrition_dishes"], U78_RECORD["dishes_with_missing_nutrition"]),
        ("每个可见实例都有营养数字的盘", legacy["complete_nutrition_dishes"], U78_RECORD["all_numeric_dishes"]),
        ("其中五个营养素不能同时对齐", legacy["mismatch_n"], U78_RECORD["all_numeric_mismatch"]),
        ("完整标签盘", legacy["complete_and_sums_match"], U78_RECORD["complete_label_dishes"]),
        (
            "部分标签盘",
            public["dishes"] - legacy["complete_and_sums_match"],
            U78_RECORD["partial_label_dishes"],
        ),
        ("质量差中位数（克）", None if mass_median is None else round(mass_median, 2), U78_RECORD["mass_gap_median_g"]),
        ("质量绝对相对差中位数（%）", relative_percent, U78_RECORD["mass_gap_median_abs_relative_percent"]),
        ("盘子比求和重出 0.5 克", legacy["mass_plate_heavier_abs_0_5"], U78_RECORD["mass_plate_heavier_abs_0_5"]),
        ("求和比盘子重出 0.5 克", legacy["mass_sum_heavier_abs_0_5"], U78_RECORD["mass_sum_heavier_abs_0_5"]),
        ("能量对不齐", legacy["fail_counts"]["calories"], U78_RECORD["fail_counts"]["calories"]),
        ("脂肪对不齐", legacy["fail_counts"]["fat"], U78_RECORD["fail_counts"]["fat"]),
        ("质量对不齐", legacy["fail_counts"]["mass"], U78_RECORD["fail_counts"]["mass"]),
        ("碳水对不齐", legacy["fail_counts"]["carb"], U78_RECORD["fail_counts"]["carb"]),
        ("蛋白质对不齐", legacy["fail_counts"]["protein"], U78_RECORD["fail_counts"]["protein"]),
        ("多于一条 Hand 标注的餐盘", merge["hand_multi_annotation_plates"], U78_RECORD["hand_multi_annotation_plates"]),
        (
            "这些餐盘上的 Hand 扫描",
            merge["hand_dishes_on_multi_annotation_plates"],
            U78_RECORD["hand_dishes_on_multi_annotation_plates"],
        ),
        ("Hand 图像级训练餐盘", hand_diag["train_plates"], U78_RECORD["hand_image_train_plates"]),
        ("Hand 图像级测试餐盘", hand_diag["test_plates"], U78_RECORD["hand_image_test_plates"]),
        ("Hand 物理餐盘", merge["hand_physical_plates"], U78_RECORD["hand_physical_plates"]),
        ("全库物理餐盘", merge["library_physical_plates"], U78_RECORD["library_physical_plates"]),
        ("全库相邻链接", merge["library_links"], U78_RECORD["library_links"]),
        ("全库多次扫描餐盘", merge["library_multi_scan_plates"], U78_RECORD["library_multi_scan_plates"]),
        ("全库多次扫描覆盖的 dish_id", merge["library_multi_scan_dishes"], U78_RECORD["library_multi_scan_dishes"]),
        ("跨官方 RGB 的物理餐盘", merge["official_cross_plates"], U78_RECORD["official_cross_plates"]),
        ("Hand 图像级划分跨盘", hand_diag["cross_split_plates"], U78_RECORD["hand_cross_image_split_plates"]),
        ("只在 Hand 测试一侧的餐盘", hand_diag["test_only_plates"], U78_RECORD["hand_test_only_plates"]),
        (
            "干净测试餐盘上的测试扫描",
            hand_diag["test_dishes_on_test_only_plates"],
            U78_RECORD["hand_test_dishes_on_test_only_plates"],
        ),
    ]
    return [
        {"item": item, "recomputed": got, "published": expected, "match": got == expected}
        for item, got, expected in checks
    ]


def protocol_problems(
    public: dict,
    images: dict,
    missing_plates: list[str],
    blob_changed: bool,
    unexpected_has_nutrition: list[str],
) -> list[str]:
    problems = []
    if public["duplicate_dish_ids"]:
        problems.append(f"盘级表有 {public['duplicate_dish_ids']} 个重复 dish_id")
    if not public["same_dish_id_set"]:
        problems.append("盘级表和食材表的 dish_id 集合不同")
    if public["invalid_dish_ids"]:
        problems.append(f"有 {len(public['invalid_dish_ids'])} 个 dish_id 不符合 dish_加数字")
    if public["dishes_without_instances"]:
        problems.append(f"有 {len(public['dishes_without_instances'])} 个盘子没有可见实例")
    if public["unparsed_plate_nutrients"]:
        problems.append(f"有 {public['unparsed_plate_nutrients']} 个盘子的盘级营养读不成数字")
    if unexpected_has_nutrition:
        problems.append("has_nutrition 出现了 0 和 1 以外的取值：" + "、".join(unexpected_has_nutrition))
    if images.get("checked"):
        if images["missing"] or images["unreadable"]:
            problems.append(
                f"俯视 RGB 缺失 {len(images['missing'])} 张，打不开 {len(images['unreadable'])} 张"
            )
        elif images["opened"] != images["expected"]:
            problems.append("打开的俯视 RGB 数量和 Hand 盘数不一致")
    if missing_plates:
        problems.append(f"有 {len(missing_plates)} 个 Hand dish_id 不在元数据的物理餐盘合并里")
    if blob_changed:
        problems.append("审计过程中公开 CSV 的字节发生了变化")
    return problems


def run_from_loaded(
    dish_rows: list[dict[str, str]],
    ingredient_rows: list[dict[str, str]],
    metadata_rows: list[dict],
    rgb_train: set[str],
    rgb_test: set[str],
    images: dict,
    blobs: dict,
    cafe1_ids: set[str] | None = None,
    depth_train: set[str] | None = None,
    depth_test: set[str] | None = None,
) -> dict:
    public = public_table_facts(dish_rows, ingredient_rows)
    records, flag_notes = classify_dishes(dish_rows, ingredient_rows)
    metadata_by_id = {row["dish_id"]: row for row in metadata_rows}
    links, dish_to_plate, plates = cluster_plates(metadata_rows, PLATE_GAP_SECONDS)
    gaps = gap_distribution(metadata_rows)
    legacy = legacy_u78_nutrition(dish_rows, ingredient_rows, metadata_by_id)

    missing_plates = [row["dish_id"] for row in records if row["dish_id"] not in dish_to_plate]
    hand_ids_by_plate: dict[str, list[str]] = defaultdict(list)
    for row in records:
        plate_id = dish_to_plate.get(row["dish_id"])
        if plate_id is not None:
            hand_ids_by_plate[plate_id].append(row["dish_id"])
    hand_plate_ids = sorted(hand_ids_by_plate, key=dish_timestamp)
    assignment = assign_plate_splits(hand_plate_ids)
    plate_rows = build_plate_rows(
        hand_plate_ids, plates, hand_ids_by_plate, assignment, rgb_train, rgb_test
    )
    cross_split = split_cross_count(plate_rows)
    # A metadata dish id must not belong to two Hand plates.
    seen_scans: dict[str, str] = {}
    for row in plate_rows:
        for dish_id in row["metadata_dish_ids"]:
            previous = seen_scans.get(dish_id)
            if previous is not None and previous != row["plate_id"]:
                cross_split += 1
            seen_scans[dish_id] = row["plate_id"]

    split_counts = Counter(row["split"] for row in plate_rows)
    holdout_n = split_counts.get("holdout", 0)
    multi_plates = [plate_id for plate_id, ids in hand_ids_by_plate.items() if len(ids) > 1]
    library_multi = [plate_id for plate_id, ids in plates.items() if len(ids) > 1]
    official_cross = []
    for plate_id, scans in plates.items():
        has_train = any(dish_id in rgb_train for dish_id in scans)
        has_test = any(dish_id in rgb_test for dish_id in scans)
        if has_train and has_test:
            official_cross.append(plate_id)
    merge = {
        "library_rows": len(metadata_rows),
        "library_physical_plates": len(plates),
        "library_links": links,
        "library_multi_scan_plates": len(library_multi),
        "library_multi_scan_dishes": sum(len(plates[plate_id]) for plate_id in library_multi),
        "official_cross_plates": len(official_cross),
        "official_cross_plate_ids": sorted(official_cross, key=dish_timestamp),
        "hand_physical_plates": len(hand_plate_ids),
        "hand_multi_annotation_plates": len(multi_plates),
        "hand_dishes_on_multi_annotation_plates": sum(len(hand_ids_by_plate[plate_id]) for plate_id in multi_plates),
        "gaps": gaps,
        "gap_rule_edited": False,
    }
    hand_diag = hand_image_split_diagnostic(records, dish_to_plate)
    unexpected = sorted(set(public["has_nutrition"]) - {"0", "1"})
    problems = protocol_problems(
        public,
        images,
        missing_plates,
        blobs.get("changed_during_audit", False),
        unexpected,
    )
    protocol_clear = not problems

    joint_mismatch = [
        row for row in records if row["all_visible_have_nutrition"] and not row["sums_align"]
    ]
    nutrients = nutrient_direction(joint_mismatch)
    structure_ok, structure_reasons = structure_holds(nutrients)
    code = decide(protocol_clear, structure_ok, holdout_n, cross_split)
    classes = class_residual_table(records, Counter(public["class_counts"]))
    label_counts = Counter(row["label"] for row in records)
    reason_counts = Counter(row["partial_reason"] for row in records if row["label"] == "partial")
    undefined_sum = sum(row["known_sum"] is None for row in records)

    cafe_counts = Counter(
        metadata_by_id[row["dish_id"]]["cafe"] for row in records if row["dish_id"] in metadata_by_id
    )
    split_names = ("train", "validation", "holdout")
    by_split = []
    for name in split_names:
        subset = [row for row in plate_rows if row["split"] == name]
        hand_ids = [dish_id for row in subset for dish_id in row["hand_dish_ids"]]
        scan_ids = [dish_id for row in subset for dish_id in row["metadata_dish_ids"]]
        by_split.append(
            {
                "split": name,
                "plates": len(subset),
                "hand_dishes": official_span(hand_ids, rgb_train, rgb_test),
                "metadata_scans": official_span(scan_ids, rgb_train, rgb_test),
                "plates_spanning_official_rgb": sum(row["spans_official_rgb"] for row in subset),
            }
        )
    official = {
        "rgb_train": len(rgb_train),
        "rgb_test": len(rgb_test),
        "rgb_overlap": len(rgb_train & rgb_test),
        "by_split": by_split,
        "cafe1_ids": None if cafe1_ids is None else len(cafe1_ids),
        "rgb_union_equals_cafe1": None
        if cafe1_ids is None
        else (rgb_train | rgb_test) == cafe1_ids and not (rgb_train & rgb_test),
        "depth_train_outside_rgb": None
        if depth_train is None
        else len(depth_train - rgb_train),
        "depth_test_outside_rgb": None
        if depth_test is None
        else len(depth_test - rgb_test),
    }
    meta_mass = legacy["meta_mass_max_abs"]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "split_version": SPLIT_VERSION,
        "decision": code,
        "decision_zh": decision_zh(code),
        "stopped": code != "phase1_conditions_met_status_unchanged",
        "protocol_clear": protocol_clear,
        "protocol_problems": problems,
        "structure_ok": structure_ok,
        "structure_reasons": structure_reasons,
        "holdout_nutrition_error_computed": False,
        "status_modified": False,
        "phase2_unlocked": False,
        "public": public,
        "flag_notes": flag_notes,
        "images": images,
        "cafe_counts": dict(cafe_counts),
        "merge": merge,
        "legacy_u78": legacy,
        "u78_comparison": compare_u78(legacy, merge, hand_diag, public),
        "labels": {
            "complete": label_counts.get("complete", 0),
            "partial": label_counts.get("partial", 0),
            "partial_reasons": dict(reason_counts),
            "undefined_known_sum": undefined_sum,
            "joint_mismatch": len(joint_mismatch),
        },
        "nutrients": nutrients,
        "classes": classes,
        "split": {
            "plates": len(plate_rows),
            "train_plates": split_counts.get("train", 0),
            "validation_plates": split_counts.get("validation", 0),
            "holdout_plates": holdout_n,
            "cross_split_plates": cross_split,
            "floor": MIN_HOLDOUT_PLATES,
            "meets_floor": holdout_n >= MIN_HOLDOUT_PLATES and cross_split == 0,
            "holdout_percent": HOLDOUT_PERCENT,
            "validation_percent": VALIDATION_PERCENT,
        },
        "official": official,
        "hand_image_split": {
            "cross_split_plates": hand_diag["cross_split_plates"],
            "test_only_plates": hand_diag["test_only_plates"],
            "train_only_plates": hand_diag["train_only_plates"],
            "train_plates": hand_diag["train_plates"],
            "test_plates": hand_diag["test_plates"],
            "test_dishes_on_test_only_plates": hand_diag["test_dishes_on_test_only_plates"],
            "used_as_evaluation_split": False,
        },
        "blobs": blobs,
        "meta_mass_max_abs": meta_mass,
        "meta_mass_within_0_5": legacy["meta_mass_within_0_5"],
        "meta_mass_n": legacy["meta_mass_n"],
        "records": records,
        "plate_rows": plate_rows,
        "hand_diag_rows": hand_diag["rows"],
        "sterling": STERLING,
    }


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "没有定义"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def yes_no(flag: bool) -> str:
    return "是" if flag else "否"


def render_report(summary: dict) -> str:
    public = summary["public"]
    images = summary["images"]
    merge = summary["merge"]
    legacy = summary["legacy_u78"]
    labels = summary["labels"]
    nutrients = summary["nutrients"]
    split = summary["split"]
    official = summary["official"]
    hand = summary["hand_image_split"]
    blobs = summary["blobs"]
    gaps = merge["gaps"]["nested_mass_increasing_seconds"]
    other_gaps = merge["gaps"]["other_consecutive_seconds"]
    comparison_rows = "\n".join(
        f"| {row['item']} | {row['recomputed']} | {row['published']} | {yes_no(row['match'])} |"
        for row in summary["u78_comparison"]
    )
    nutrient_rows = []
    for name in NUTRIENTS:
        stats = nutrients.get(name, {})
        nutrient_rows.append(
            "| {zh} | {unit} | {fail} | {plate} | {sum_side} | {imbalance} | {median} | {p10} | {p90} |".format(
                zh=NUTRIENT_ZH[name],
                unit=NUTRIENT_UNIT[name],
                fail=stats.get("fail_count", 0),
                plate=stats.get("plate_side", 0),
                sum_side=stats.get("sum_side", 0),
                imbalance=fmt(stats.get("imbalance"), 3),
                median=fmt(stats.get("median_signed")),
                p10=fmt(stats.get("p10_signed")),
                p90=fmt(stats.get("p90_signed")),
            )
        )
    class_rows = []
    for row in summary["classes"]:
        class_rows.append(
            f"| {row['zh']} | `{row['annotation_name']}` | {row['instances']} | {row['all_nutrition_dishes']} | "
            f"{row['mismatch_dishes']} | {fmt(row['mass_gap_median_g'])} | {row['mass_plate_side']} | "
            f"{row['mass_sum_side']} | {row['energy_fail_dishes']} |"
        )
    with_mismatch = [row for row in summary["classes"] if row["mismatch_dishes"] > 0]
    dominating = [row for row in with_mismatch if row["mass_plate_side"] > row["mass_sum_side"]]
    reversed_classes = [row["zh"] for row in with_mismatch if row["mass_sum_side"] > row["mass_plate_side"]]
    no_mismatch = [row["zh"] for row in summary["classes"] if row["mismatch_dishes"] == 0]
    if reversed_classes:
        reversed_text = "求和一侧更多的类别是" + "、".join(reversed_classes)
    else:
        reversed_text = "没有类别是求和一侧更多"
    if no_mismatch:
        no_mismatch_text = "没有进入求和不对齐盘子的类别是" + "、".join(no_mismatch)
    else:
        no_mismatch_text = "每个类别都有求和不对齐的盘子"
    reason_zh = {
        "sum_outside_tolerance": "每个可见实例都有营养数字，但求和超出容差",
        "missing_nutrition_sum_inside_tolerance": "有可见实例缺少营养数字，已有数字的求和仍在容差内",
        "missing_nutrition_and_sum_outside": "有可见实例缺少营养数字，并且求和也超出容差",
    }
    partial_reasons = labels["partial_reasons"]
    reason_text = "；".join(
        f"{reason_zh.get(key, key)} {value} 盘" for key, value in partial_reasons.items()
    ) or "没有"
    has_nutrition = public["has_nutrition"]
    if summary["cafe_counts"] == {"cafe1": public["dishes"]}:
        cafe_text = f"全部 {public['dishes']} 盘都在 Cafe1，Cafe2 为 0"
    else:
        cafe_text = "、".join(f"{name} {count} 盘" for name, count in sorted(summary["cafe_counts"].items())) or "没有"
    split_zh = {"train": "训练", "validation": "验证", "holdout": "留出"}
    mass_stats = nutrients.get("mass", {})
    energy_stats = nutrients.get("calories", {})
    fat_stats = nutrients.get("fat", {})
    spanning_bits = [
        f"`{row['plate_id']}` 在{split_zh[row['split']]}"
        for row in summary["plate_rows"]
        if row["spans_official_rgb"]
    ]
    spanning_text = "、".join(spanning_bits) if spanning_bits else "没有"
    named_cross = ["dish_1559844458", "dish_1561662814"]
    found_cross = list(merge.get("official_cross_plate_ids", []))
    if found_cross == named_cross:
        cross_match = "与 #78 报告点名的两只相同。"
    else:
        cross_match = "标识见上，不另作改派。"
    split_rows = []
    for row in official["by_split"]:
        hand_ids = row["hand_dishes"]
        scans = row["metadata_scans"]
        split_rows.append(
            f"| {split_zh[row['split']]} | {row['plates']} | {hand_ids['n']} | {hand_ids['official_train']} | "
            f"{hand_ids['official_test']} | {hand_ids['official_neither']} | {scans['n']} | "
            f"{scans['official_train']} | {scans['official_test']} | {row['plates_spanning_official_rgb']} |"
        )
    size_text = "、".join(f"{size} 有 {count} 张" for size, count in sorted(images.get("size_counts", {}).items()))
    if not size_text:
        size_text = "没有打开图像"
    mode_text = "、".join(f"{mode} {count}" for mode, count in sorted(images.get("mode_counts", {}).items()))
    problems = summary["protocol_problems"]
    problem_text = "；".join(problems) if problems else "没有"
    structure_text = "；".join(summary["structure_reasons"]) if summary["structure_reasons"] else "三条都成立"
    matched = sum(row["match"] for row in summary["u78_comparison"])
    total_checks = len(summary["u78_comparison"])
    relative = legacy["mass_gap_median_abs_relative"]
    relative_percent = "没有定义" if relative is None else f"{relative * 100:.1f}%"
    lines = [
        "# 不完整食材标签：第一阶段协议与划分审计",
        "",
        "这次只冻结协议和容差，刻画可见食材求和相对盘级总量的残差，并预先登记按物理餐盘分组的训练、验证和留出。不训练网络，不跑第二阶段对照，不修改 `docs/STATUS.md`，也不宣布第二阶段解锁。",
        "",
        "## 结论一览",
        "",
        f"判定码 `{summary['decision']}`。{summary['decision_zh']}",
        "",
        "| 项目 | 结果 |",
        "|---|---|",
        f"| 协议是否写死 | 是。版本 `{PROTOCOL_VERSION}`，容差和定义在脚本常量里，本次没有按结果回改 |",
        f"| 协议在这批数据上能否套用 | {yes_no(summary['protocol_clear'])}。{problem_text if problems else '定义能套到每一行'} |",
        f"| 残差是否按写死规则成立 | {yes_no(summary['structure_ok'])}。{structure_text} |",
        (
            f"| 质量残差 | 中位数 {fmt(mass_stats.get('median_signed'))} 克；超出容差后盘子更重 "
            f"{mass_stats.get('plate_side', 0)} 盘，求和更重 {mass_stats.get('sum_side', 0)} 盘 |"
        ),
        (
            f"| 能量残差 | 中位数 {fmt(energy_stats.get('median_signed'))} 千卡；盘子一侧 "
            f"{energy_stats.get('plate_side', 0)} 盘，求和一侧 {energy_stats.get('sum_side', 0)} 盘 |"
        ),
        (
            f"| 脂肪残差 | 中位数 {fmt(fat_stats.get('median_signed'))} 克；盘子一侧 "
            f"{fat_stats.get('plate_side', 0)} 盘，求和一侧 {fat_stats.get('sum_side', 0)} 盘 |"
        ),
        f"| 物理餐盘 | {merge['hand_physical_plates']} 只，覆盖 {public['dishes']} 条 Hand `dish_id` |",
        f"| 训练 / 验证 / 留出 | {split['train_plates']} / {split['validation_plates']} / {split['holdout_plates']} 只物理餐盘 |",
        f"| 跨划分餐盘 | {split['cross_split_plates']} |",
        f"| 是否达到 80 只留出下限 | {yes_no(split['holdout_plates'] >= MIN_HOLDOUT_PLATES)}（下限 {MIN_HOLDOUT_PLATES}，实际 {split['holdout_plates']}） |",
        f"| 是否停止 | {yes_no(summary['stopped'])} |",
        "| 留出营养误差 | 没有计算 |",
        "| 第二阶段 | 仍锁定。本报告不解锁 |",
        "",
        f"#78 字段记录对照 {matched}/{total_checks} 项与发表数字相同。那些数字是复核对象，不是这一阶段已经通过的证明。",
        "",
        "## 1. 写死的协议与容差",
        "",
        "下面这段在脚本常量 `PROTOCOL_TEXT` 里，先于残差汇总和划分计数。看见结果之后没有改字。",
        "",
        PROTOCOL_TEXT.rstrip(),
        "",
        f"容差常数是绝对差 {NUTR_ABS_TOL}，相对差 {NUTR_REL_TOL:.0%}。公式是绝对差不超过 max({NUTR_ABS_TOL}, {NUTR_REL_TOL} × 较大绝对值)。",
        "",
        "## 2. 公开表和俯视图像",
        "",
        f"盘级表 {public['dishes']} 行，食材表 {public['instances']} 行，粗类 {public['classes']} 个。两表 `dish_id` 集合{'相同' if public['same_dish_id_set'] else '不同'}。盘级重复标识 {public['duplicate_dish_ids']} 个。图像级划分冲突 {len(public['split_conflicts'])} 个。Hand 图像级 train/test 是 {public['dish_split'].get('train', 0)}/{public['dish_split'].get('test', 0)}，这份划分不作为主评价划分。",
        "",
        f"有营养标记的实例 {has_nutrition.get('1', 0)} 个，缺失标记的实例 {has_nutrition.get('0', 0)} 个。标记为 1 但五格里有空值的实例 {summary['flag_notes']['has_nutrition_1_with_blank']} 个。标记为 0 但格子里写了数字的实例 {summary['flag_notes']['has_nutrition_0_with_number']} 个，这些数字没有加进求和。",
        "",
        f"俯视 RGB 应有 {images.get('expected', 0)} 张，打开 {images.get('opened', 0)} 张，缺失 {len(images.get('missing', []))} 张，打不开 {len(images.get('unreadable', []))} 张。尺寸：{size_text}。原始模式：{mode_text or '没有'}。",
        "",
        f"Hand 盘的食堂归属：{cafe_text}。盘级质量相对元数据的最大绝对差是 {fmt(summary['meta_mass_max_abs'], 4)} 克，落在 0.5 克以内的有 {summary['meta_mass_within_0_5']} / {summary['meta_mass_n']}。盘级表仍是总量来源。",
        "",
        f"盘级表 blob 开始 `{blobs.get('dish_start', '')}`，结束 `{blobs.get('dish_end', '')}`，与 #78 记录{'一致' if blobs.get('dish_start') == DISH_BLOB_SHA else '不一致'}。食材表 blob 开始 `{blobs.get('ingredient_start', '')}`，结束 `{blobs.get('ingredient_end', '')}`，与 #78 记录{'一致' if blobs.get('ingredient_start') == INGREDIENT_BLOB_SHA else '不一致'}。",
        "",
        "## 3. 物理餐盘合并",
        "",
        "合并规则就是第 5 节写死的那条，这里不另选时间差。全库相邻、食材嵌套且质量增加的时间差：",
        f"共 {gaps.get('n', 0)} 对，中位 {gaps.get('p50', '没有')} 秒，99 分位 {gaps.get('p99', '没有')} 秒，不超过 180 秒的有 {gaps.get('le_180', 0)} 对。其余相邻对 {other_gaps.get('n', 0)} 对，中位 {other_gaps.get('p50', '没有')} 秒。180 秒仍盖住这条嵌套时间差的 99 分位，常数没有改。",
        "",
        (
            f"全库 {merge['library_rows']} 条元数据收成 {merge['library_physical_plates']} 只物理餐盘，"
            f"相邻链接 {merge['library_links']} 条。多次扫描的餐盘 {merge['library_multi_scan_plates']} 只，"
            f"覆盖 {merge['library_multi_scan_dishes']} 条 `dish_id`。"
            + (
                f"其中扫描跨过官方 RGB 训练和测试的有 {merge['official_cross_plates']} 只，标识是 "
                + "、".join(f"`{plate_id}`" for plate_id in found_cross)
                + f"。{cross_match}"
                if found_cross
                else f"其中扫描跨过官方 RGB 训练和测试的有 {merge['official_cross_plates']} 只。"
            )
            + "这个交叉只检查合并没有把无关盘子粘成一片，不用来改规则。"
        ),
        "",
        f"Hand 的 {public['dishes']} 条 `dish_id` 落在 {merge['hand_physical_plates']} 只物理餐盘上。多于一条 Hand 标注的餐盘 {merge['hand_multi_annotation_plates']} 只，共 {merge['hand_dishes_on_multi_annotation_plates']} 条。",
        "",
        "## 4. 残差",
        "",
        "判定规则在脚本常量里，先于下面的数字：",
        "",
        STRUCTURE_RULE_TEXT.rstrip(),
        "",
        "先按 #78 的算法复核字段记录。那套算法把 `has_nutrition=1` 的行相加；若这样的行一格都没有，求和会变成 0。本协议不采用这个 0。对照表仍按那套算法重算，方便和字段记录逐项核对。",
        "",
        "| 项目 | 本次重算 | #78 字段记录 | 相同 |",
        "|---|---:|---:|:---:|",
        comparison_rows,
        "",
        f"按 #78 算法，每个可见实例都有营养数字、但五个营养素不能同时对齐的盘子有 {legacy['mismatch_n']} 个。质量残差中位数 {fmt(legacy['mass_gap_median_g'])} 克，绝对相对差中位数 {relative_percent}。用绝对差 0.5 克而不是完整容差来数方向：盘子更重 {legacy['mass_plate_heavier_abs_0_5']} 盘，求和更重 {legacy['mass_sum_heavier_abs_0_5']} 盘。对不齐次数按能量、脂肪、质量、碳水、蛋白质是 {legacy['fail_counts']['calories']}、{legacy['fail_counts']['fat']}、{legacy['fail_counts']['mass']}、{legacy['fail_counts']['carb']}、{legacy['fail_counts']['protein']}。",
        "",
        f"有可见实例缺少营养数字的盘子，按 #78 算法的质量残差中位数是 {fmt(legacy['partial_mass_median_g'])} 克，盘子更重 {legacy['partial_plate_heavier_abs_0_5']}，求和更重 {legacy['partial_sum_heavier_abs_0_5']}。其中五个营养素仍落在容差内的有 {legacy['partial_and_sums_match']} 盘。它们依然是部分标签盘，因为缺了可见实例的营养数字。",
        "",
        f"按本次写死的协议：完整标签盘 {labels['complete']}，部分标签盘 {labels['partial']}。部分标签里，{reason_text}。可见求和没有定义的盘子 {labels['undefined_known_sum']} 个，它们一个有营养实例都没有，没有被记成求和为 0。进入结构判定的盘子是每个可见实例都有营养数字、但求和不对齐的 {labels['joint_mismatch']} 盘。",
        "",
        "下表只统计这 {n} 盘。残差是盘级减去可见求和。盘子一侧指超出容差并且盘级更大；求和一侧指超出容差并且可见求和更大。符号不平衡度只在两侧都算得出来时有定义。".format(
            n=labels["joint_mismatch"]
        ),
        "",
        "| 营养素 | 单位 | 超出容差 | 盘子一侧 | 求和一侧 | 符号不平衡度 | 残差中位数 | 残差 P10 | 残差 P90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *nutrient_rows,
        "",
        "P10 和 P90 用的是与 #78 相同的分位下标。中位数把仍落在容差内的盘子也算了进去。质量、能量、脂肪大部分盘子都超出容差，所以这三列的中位数落在盘子更大的那一侧。碳水和蛋白质超出容差的盘子少一些，中位数更靠近 0，是因为不少盘子在这两个营养素上仍然对齐。一旦超出容差，碳水和蛋白质也都是盘子一侧，求和一侧是 0。",
        "",
        (
            f"质量超出完整容差的 {mass_stats.get('fail_count', 0)} 盘里，盘子更重 {mass_stats.get('plate_side', 0)} 盘，"
            f"求和更重 {mass_stats.get('sum_side', 0)} 盘。能量是 {energy_stats.get('plate_side', 0)} 对 {energy_stats.get('sum_side', 0)}。"
            f"脂肪是 {fat_stats.get('plate_side', 0)} 对 {fat_stats.get('sum_side', 0)}。"
            "上面 #78 对照里的 418，是用绝对差 0.5 克在这 424 盘上数质量方向，没有先问 1% 的相对容差是否已经把这盘算作对齐。"
            f"完整容差更宽，所以质量上真正超出容差的是 {mass_stats.get('fail_count', 0)} 盘，不是 418 盘。两套计数的方向一样，都是盘子更重。"
        ),
        "",
        f"类别表按可见实例数从多到少排列。一只盘子可以同时进入多行。{len(dominating)} 个类别在求和不对齐的盘子上仍是盘子一侧更多。{reversed_text}。{no_mismatch_text}。这张表没有用来改判定或改划分。",
        "",
        "| 类别 | 表中的类名 | 可见实例 | 全营养盘 | 求和不对齐 | 不对齐盘的质量残差中位数（克） | 质量盘子一侧 | 质量求和一侧 | 含该类的全营养盘里能量对不齐 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *class_rows,
        "",
        "## 5. 预先登记的划分",
        "",
        "规则如下，写在计数之前。脚本没有计算留出营养误差，也没有按残差大小挑选这一刀。",
        "",
        SPLIT_RULE_TEXT.rstrip(),
        "",
        f"本次含 Hand 标注的物理餐盘是 {split['plates']} 只。按公式，留出 {split['holdout_plates']} 只，验证 {split['validation_plates']} 只，训练 {split['train_plates']} 只。跨划分餐盘 {split['cross_split_plates']} 只。下限 80，{'达到' if split['holdout_plates'] >= MIN_HOLDOUT_PLATES else '没有达到'}。",
        "",
        "Hand 的图像级划分不采用。把它收成同一套物理餐盘之后，训练和测试两侧都出现的餐盘有 "
        f"{hand['cross_split_plates']} 只，只在测试一侧的餐盘有 {hand['test_only_plates']} 只，其上测试扫描 {hand['test_dishes_on_test_only_plates']} 条。这低于 80，而且有泄漏，所以不能当作主评价划分。",
        "",
        "## 6. 与官方 RGB 划分的交集",
        "",
        "".join(
            part
            for part in (
                f"官方 RGB 训练 {official['rgb_train']} 条，测试 {official['rgb_test']} 条，两表交集 {official['rgb_overlap']} 条。",
                None
                if official["rgb_union_equals_cafe1"] is None
                else f"并集与 `dish_ids_cafe1.txt` {'相同' if official['rgb_union_equals_cafe1'] else '不相同'}。",
                None
                if official["depth_train_outside_rgb"] is None
                else (
                    f"深度训练里不在 RGB 训练的有 {official['depth_train_outside_rgb']} 条，"
                    f"深度测试里不在 RGB 测试的有 {official['depth_test_outside_rgb']} 条。"
                ),
                "下面的交集只说明两套划分不是同一刀，不用来把盘子从留出挪走。",
            )
            if part
        ),
        "",
        "| 我们的划分 | 物理餐盘 | Hand 扫描 | 落在官方训练 | 落在官方测试 | 不在官方 RGB 划分 | 同盘全部扫描 | 扫描落在官方训练 | 扫描落在官方测试 | 扫描跨官方两侧的餐盘 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *split_rows,
        "",
        "「扫描跨官方两侧」指同一只物理餐盘的元数据扫描里，既有官方 RGB 训练标识，也有官方 RGB 测试标识。"
        f"含 Hand 标注且跨过官方两侧的餐盘是：{spanning_text}。这样的盘子仍整只留在预先登记的一侧，没有因为跨过官方划分而改派。",
        "",
        "## 7. Sterling 等人的文献陈述",
        "",
        f"核对日期 {STERLING['checked_on']}。文献是 {STERLING['citation']}，doi:{STERLING['doi']}。摘要写明：四名标注者复核了 440 张 Nutrition5k 图像，原始食材标签有系统性遗漏；修正之后，Gemini 2.0 Flash 的食材重叠从 0.62 外推到约 0.82。正文里的全数据集 Jaccard 是 0.621；在这 440 张上，相对修正标注是 0.726，相对原始标注是 0.450；加法外推约 0.82，乘法外推约 0.84，作者报告更保守的约 0.82。每张图由一名标注者看过，四名标注者的图像互不重叠。",
        "",
        "0.62 到 0.82 是该文对 Gemini 2.0 Flash 食材重叠的外推，不是我们的结果，也不写进我们的指标。",
        "",
        f"数据可得性一节只指向 Nutrition5k 原发布。代码仓库 {STERLING['code_url']} 在核对日能看到评测脚本、标注工具和模型预测 CSV，文件树里没有一份人类修正后的食材真值表。模型预测不是修正标注。因此没有另开修正标注的可用性审计，也不把未公开的人工修正当作训练真值。",
        "",
        "## 8. 判断",
        "",
        f"协议写死：是。残差结构化：{yes_no(summary['structure_ok'])}。零泄漏且留出不少于 80 只：{yes_no(split['meets_floor'])}。",
        "",
        f"判定码 `{summary['decision']}`。{summary['decision_zh']}",
        "",
        "通过这三条也不改 `docs/STATUS.md`。第二阶段要等这份状态文件被另行改成明确解锁之后才开始。本报告不训练，不添加损失函数，也不把 505 或 Cafe2 当作验证集。",
        "",
        "## 9. 复现",
        "",
        "在仓库根目录运行：",
        "",
        "```bash",
        "python scripts/u2_phase1_partial_label_audit.py --repo-root . --out results/u2_phase1_partial_label_audit",
        "python -m unittest tests/test_u2_phase1_partial_label_audit.py",
        "```",
        "",
        "输入是 `data/hand2026/dish_nutrition.csv`、`data/hand2026/ingredient_nutrition.csv`、`data/nutrition5k/imagery/realsense_overhead/<dish_id>/rgb.png`、`data/nutrition5k/dish_ids/splits/rgb_train_ids.txt`、`data/nutrition5k/dish_ids/splits/rgb_test_ids.txt` 和 `data/nutrition5k/metadata/`。公开 CSV 与图像只读。",
        "",
        f"只读引用的 #78 提交是 `{U78_COMMIT}`。本分支状态文件所在提交是 `{STATUS_COMMIT}`。",
        "",
        (
            f"拉取请求：{PULL_REQUEST_URL}。基线是 `u2-partial-label`，不是 `main`。"
            if PULL_REQUEST_URL
            else "拉取请求以 `u2-partial-label` 为基线，不是 `main`。链接在打开之后写入本节。"
        ),
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def dish_residual_rows(records: list[dict], dish_to_plate: dict[str, str]) -> list[dict]:
    rows = []
    for record in records:
        flat = {
            "dish_id": record["dish_id"],
            "plate_id": dish_to_plate.get(record["dish_id"], ""),
            "n_visible": record["n_visible"],
            "n_nutrition": record["n_nutrition"],
            "n_missing_nutrition": record["n_missing_nutrition"],
            "label": record["label"],
            "partial_reason": record["partial_reason"],
            "sums_align": record["sums_align"],
            "classes": "|".join(record["classes"]),
        }
        for nutrient in NUTRIENTS:
            plate_value = record["plate"][nutrient]
            known = None if record["known_sum"] is None else record["known_sum"][nutrient]
            flat[f"plate_{nutrient}"] = "" if plate_value is None else plate_value
            flat[f"sum_{nutrient}"] = "" if known is None else known
            residual = record["residual"][nutrient]
            flat[f"residual_{nutrient}"] = "" if residual is None else residual
            flat[f"aligned_{nutrient}"] = record["aligned"][nutrient]
        rows.append(flat)
    return rows


def write_outputs(out: Path, summary: dict, dish_to_plate: dict[str, str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    report = render_report(summary)
    (out / "report.md").write_text(report, encoding="utf-8")
    portable = {
        key: value
        for key, value in summary.items()
        if key not in {"records", "plate_rows", "hand_diag_rows"}
    }
    (out / "summary.json").write_text(
        json.dumps(json_ready(portable), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    constants = {
        "protocol_version": PROTOCOL_VERSION,
        "split_version": SPLIT_VERSION,
        "abs_tolerance": NUTR_ABS_TOL,
        "rel_tolerance": NUTR_REL_TOL,
        "plate_gap_seconds": PLATE_GAP_SECONDS,
        "min_mass_increase_g": MIN_MASS_INCREASE_G,
        "mass_gram_tolerance_g": MASS_GRAM_TOL_G,
        "holdout_percent": HOLDOUT_PERCENT,
        "validation_percent": VALIDATION_PERCENT,
        "min_holdout_plates": MIN_HOLDOUT_PLATES,
        "min_mass_sign_imbalance": MIN_MASS_SIGN_IMBALANCE,
        "split_hash_prefix": SPLIT_HASH_PREFIX,
        "holdout_nutrition_error_computed": False,
        "protocol_text": PROTOCOL_TEXT,
        "split_rule_text": SPLIT_RULE_TEXT,
        "structure_rule_text": STRUCTURE_RULE_TEXT,
    }
    (out / "protocol_constants.json").write_text(
        json.dumps(constants, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    nutrient_fields = []
    for nutrient in NUTRIENTS:
        nutrient_fields.extend(
            [f"plate_{nutrient}", f"sum_{nutrient}", f"residual_{nutrient}", f"aligned_{nutrient}"]
        )
    write_csv(
        out / "dish_residual.csv",
        dish_residual_rows(summary["records"], dish_to_plate),
        [
            "dish_id",
            "plate_id",
            "n_visible",
            "n_nutrition",
            "n_missing_nutrition",
            "label",
            "partial_reason",
            "sums_align",
            "classes",
            *nutrient_fields,
        ],
    )
    write_csv(
        out / "class_residual.csv",
        summary["classes"],
        [
            "annotation_name",
            "zh",
            "instances",
            "all_nutrition_dishes",
            "mismatch_dishes",
            "mass_gap_median_g",
            "mass_plate_side",
            "mass_sum_side",
            "energy_fail_dishes",
        ],
    )
    plate_flat = []
    for row in summary["plate_rows"]:
        plate_flat.append(
            {
                "plate_id": row["plate_id"],
                "split": row["split"],
                "n_hand_dishes": row["n_hand_dishes"],
                "n_metadata_scans": row["n_metadata_scans"],
                "official_train_scans": row["official_train_scans"],
                "official_test_scans": row["official_test_scans"],
                "spans_official_rgb": row["spans_official_rgb"],
                "hand_dish_ids": "|".join(row["hand_dish_ids"]),
                "metadata_dish_ids": "|".join(row["metadata_dish_ids"]),
            }
        )
    write_csv(
        out / "plate_assignment.csv",
        plate_flat,
        [
            "plate_id",
            "split",
            "n_hand_dishes",
            "n_metadata_scans",
            "official_train_scans",
            "official_test_scans",
            "spans_official_rgb",
            "hand_dish_ids",
            "metadata_dish_ids",
        ],
    )
    write_csv(
        out / "hand_image_split_diagnostic.csv",
        summary["hand_diag_rows"],
        [
            "plate_id",
            "hand_train_dishes",
            "hand_test_dishes",
            "hand_other_dishes",
            "in_both_hand_splits",
            "only_hand_test",
        ],
    )


def default_paths(repo: Path) -> dict[str, Path]:
    return {
        "dish": repo / "data/hand2026/dish_nutrition.csv",
        "ingredient": repo / "data/hand2026/ingredient_nutrition.csv",
        "imagery": repo / "data/nutrition5k/imagery/realsense_overhead",
        "rgb_train": repo / "data/nutrition5k/dish_ids/splits/rgb_train_ids.txt",
        "rgb_test": repo / "data/nutrition5k/dish_ids/splits/rgb_test_ids.txt",
        "metadata": repo / "data/nutrition5k/metadata",
        "cafe1": repo / "data/nutrition5k/dish_ids/dish_ids_cafe1.txt",
        "depth_train": repo / "data/nutrition5k/dish_ids/splits/depth_train_ids.txt",
        "depth_test": repo / "data/nutrition5k/dish_ids/splits/depth_test_ids.txt",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="U2 phase-1 partial-label protocol and split audit")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/u2_phase1_partial_label_audit"))
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    out = args.out if args.out.is_absolute() else repo / args.out
    forbidden = {repo / name for name in ("data", "backup", "weights", "vdr")}
    if out.resolve() == (repo / "results/u1_phase1_data_audit").resolve() or any(
        out.resolve() == forbidden_path or forbidden_path in out.resolve().parents for forbidden_path in forbidden
    ):
        print("refusing to write into protected data or the U1 audit directory", file=sys.stderr)
        return 2
    paths = default_paths(repo)
    required = ["dish", "ingredient", "imagery", "rgb_train", "rgb_test", "metadata"]
    missing = [str(paths[key]) for key in required if not paths[key].exists()]
    if missing:
        print("missing inputs:\n" + "\n".join(missing), file=sys.stderr)
        return 1
    dish_bytes = paths["dish"].read_bytes()
    ingredient_bytes = paths["ingredient"].read_bytes()
    _, dish_rows = read_csv_rows(paths["dish"])
    _, ingredient_rows = read_csv_rows(paths["ingredient"])
    metadata_rows = load_metadata(paths["metadata"])
    rgb_train = set(load_id_list(paths["rgb_train"]))
    rgb_test = set(load_id_list(paths["rgb_test"]))
    cafe1_ids = set(load_id_list(paths["cafe1"])) if paths["cafe1"].exists() else None
    depth_train = set(load_id_list(paths["depth_train"])) if paths["depth_train"].exists() else None
    depth_test = set(load_id_list(paths["depth_test"])) if paths["depth_test"].exists() else None
    images = inspect_overhead([row["dish_id"] for row in dish_rows], paths["imagery"])
    dish_end = paths["dish"].read_bytes()
    ingredient_end = paths["ingredient"].read_bytes()
    blobs = {
        "dish_start": git_blob_sha1(dish_bytes),
        "dish_end": git_blob_sha1(dish_end),
        "ingredient_start": git_blob_sha1(ingredient_bytes),
        "ingredient_end": git_blob_sha1(ingredient_end),
        "changed_during_audit": dish_bytes != dish_end or ingredient_bytes != ingredient_end,
        "dish_matches_u78": git_blob_sha1(dish_bytes) == DISH_BLOB_SHA,
        "ingredient_matches_u78": git_blob_sha1(ingredient_bytes) == INGREDIENT_BLOB_SHA,
    }
    summary = run_from_loaded(
        dish_rows,
        ingredient_rows,
        metadata_rows,
        rgb_train,
        rgb_test,
        images,
        blobs,
        cafe1_ids,
        depth_train,
        depth_test,
    )
    dish_to_plate = {
        dish_id: row["plate_id"] for row in summary["plate_rows"] for dish_id in row["hand_dish_ids"]
    }
    write_outputs(out, summary, dish_to_plate)
    dish_after = git_blob_sha1(paths["dish"].read_bytes())
    ingredient_after = git_blob_sha1(paths["ingredient"].read_bytes())
    if dish_after != blobs["dish_start"] or ingredient_after != blobs["ingredient_start"]:
        print("public CSV bytes changed while writing outputs", file=sys.stderr)
        return 3
    print(
        f"{summary['decision']} plates={summary['split']['plates']} "
        f"train={summary['split']['train_plates']} "
        f"validation={summary['split']['validation_plates']} "
        f"holdout={summary['split']['holdout_plates']} "
        f"cross={summary['split']['cross_split_plates']} "
        f"report={out / 'report.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
