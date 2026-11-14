#!/usr/bin/env python3
"""U2 phase-2 minimal contrasts: B1, B2, B3, and a forced-sum diagnostic.

The constants in this file are the freeze. They were written before any
holdout nutrition error was computed. Do not edit the backbone, the input
size, the epoch budget, the loss, the bootstrap, or the label rules after
reading holdout predictions.

B3 is a supervision protocol on the same ResNet-18 backbone. It is not a new
network. The run does not load the repository's mass-estimator checkpoints,
does not read instance masks, and does not score the holdout split during
training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import u2_phase1_partial_label_audit as phase1


# ---------------------------------------------------------------------------
# Frozen before any holdout nutrition error. Do not edit after scoring.
# ---------------------------------------------------------------------------

BACKBONE_NAME = "torchvision.models.resnet18"
WEIGHTS_NAME = "ResNet18_Weights.IMAGENET1K_V1"
INPUT_SIZE = 224
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
HFLIP_PROBABILITY = 0.5
SEED = 0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_LOW = 0.025
BOOTSTRAP_HIGH = 0.975
STD_FLOOR = 1e-6
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EXPECTED_PLATES = {"train": 465, "validation": 99, "holdout": 99}
Q2_NUTRIENTS = ("calories", "fat", "carb", "protein")
REPORT_NUTRIENTS = ("calories", "fat", "carb", "protein", "mass")
MODELS = ("b1", "b2", "b3", "forced")
STOP_MODELS = ("b1", "b2", "b3")
CLASSES = tuple(sorted(phase1.CLASS_NAME_ZH))

MODEL_ZH = {
    "b1": "B1 整盘回归",
    "b2": "B2 天真食材级",
    "b3": "B3 候选协议",
    "forced": "强迫加和诊断",
}

FROZEN_PROTOCOL_TEXT = """\
三个模型共用 torchvision ResNet-18，ImageNet-1K V1 初始权重，输入是单张俯视 RGB 缩放到 224×224。最后一层是线性读出，宽度只跟该协议要监督的目标走。不加载仓库里已有的质量估计器权重。

B1 只监督 Hand 盘级表上的质量、能量、脂肪、碳水、蛋白质。损失不读取食材实例。盘级预测就是这五个输出。输出经过 ReLU。

B2 把标签当成完整。有营养实例按粗类相加；缺营养实例，包括 has_nutrition=0 和空值，贡献 0，格子里若写了数字也不加进去。40 个粗类都有监督。盘级预测是类别输出求和。类别输出经过 ReLU。

B3 的类别目标只累加有营养实例。没有有营养实例的粗类，类别目标是 0，表示这些实例不进入可见求和。残差目标是盘级减去有营养实例之和，只在至少有一个有营养实例时有定义。类别目标与残差目标相加等于盘级。一个有营养实例都没有时，这张扫描的类别损失和残差损失都关掉，可见求和不记成 0，残差也不记成 0。推理时盘级等于类别输出求和加残差输出。残差输出不加 ReLU，可以为负。测试时不读取真值残差。

强迫加和是诊断。对每个营养素，在有营养实例上按该营养素的标注值比例把盘级总量摊回去；这些标注值之和为 0 时，改为在这些有营养实例上均分。缺营养实例不接收份额。摊回后的类别目标和等于盘级。损失形式与 B2 相同。没有有营养实例的扫描不监督。这个诊断不进入停止规则。

损失在原始单位上计算，每个营养素乘以训练扫描盘级标准差的倒数。标准差只用训练扫描，分母是扫描数。B1 是五个盘级输出的加权平均绝对误差。B2 和强迫加和是全部类别格子的加权平均绝对误差。B3 是类别格子的加权平均绝对误差，加上残差五格的加权平均绝对误差，两项等权相加。

每个模型都训练满 20 个 epoch。验证集按物理餐盘等权，用能量、脂肪、碳水、蛋白质四项平均绝对误差之和选择 epoch，质量不参与选择。相同分数取更早的 epoch。留出集不参与选择。

主指标是留出集上按物理餐盘等权的平均绝对误差。同一只餐盘先把各次扫描的绝对误差平均成一个数。稳定优于指（方法 − 对照）的餐盘配对差，其 95% 区间上界小于 0。区间是物理餐盘有放回抽取 10000 次、种子 0，对每次抽取的配对差取平均，再取 2.5% 和 97.5% 的线性分位。线性分位的位置是 q × (次数 − 1)。抽到一只餐盘时使用该餐盘已经平均好的误差，不把重拍或增量扫描拆开重抽。Q2 只看 B3 对 B1、B3 对 B2，在能量、脂肪、碳水、蛋白质上的八个区间。质量并列记录。强迫加和的区间并列记录，不进入 Q2。
"""


def frozen_budget() -> dict:
    return {
        "backbone": BACKBONE_NAME,
        "weights": WEIGHTS_NAME,
        "mass_estimator_checkpoints_loaded": False,
        "input_size": INPUT_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "horizontal_flip_probability": HFLIP_PROBABILITY,
        "test_time_augmentation": False,
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_quantiles": [BOOTSTRAP_LOW, BOOTSTRAP_HIGH],
        "percentile": "linear_q_times_n_minus_1",
        "optimizer": "AdamW",
        "schedule": "cosine_annealing_epoch",
        "loss": "nutrient_weighted_l1",
        "classes": list(CLASSES),
        "class_count": len(CLASSES),
        "q2_nutrients": list(Q2_NUTRIENTS),
        "protocol_text": FROZEN_PROTOCOL_TEXT,
    }


def overhead_rgb_path(imagery_root: Path, dish_id: str) -> Path:
    if not phase1.valid_dish_id(dish_id):
        raise ValueError(f"unexpected dish_id: {dish_id}")
    return imagery_root / dish_id / "rgb.png"


def assert_split_allowed(split: str, allow_holdout: bool) -> None:
    if split == "holdout" and not allow_holdout:
        raise RuntimeError("holdout scoring is closed during training and target checks")


def load_plate_assignment(path: Path) -> dict[str, dict]:
    """Read the frozen phase-1 plate table. The split is not recomputed here."""
    _, rows = phase1.read_csv_rows(path)
    plates: dict[str, dict] = {}
    dish_owner: dict[str, str] = {}
    for row in rows:
        plate_id = row["plate_id"]
        split = row["split"]
        if split not in EXPECTED_PLATES:
            raise ValueError(f"unexpected split {split} for {plate_id}")
        dishes = [item for item in row["hand_dish_ids"].split("|") if item]
        if plate_id in plates:
            raise ValueError(f"duplicate plate_id {plate_id}")
        for dish_id in dishes:
            if not phase1.valid_dish_id(dish_id):
                raise ValueError(f"unexpected dish_id {dish_id}")
            previous = dish_owner.get(dish_id)
            if previous is not None:
                raise ValueError(f"{dish_id} is on both {previous} and {plate_id}")
            dish_owner[dish_id] = plate_id
        plates[plate_id] = {"split": split, "dishes": dishes}
    counts = {name: 0 for name in EXPECTED_PLATES}
    for plate in plates.values():
        counts[plate["split"]] += 1
    if counts != EXPECTED_PLATES:
        raise ValueError(f"frozen plate counts are {counts}, expected {EXPECTED_PLATES}")
    return plates


def dish_split_map(plates: dict[str, dict]) -> dict[str, tuple[str, str]]:
    """dish_id -> (plate_id, split)."""
    mapping = {}
    for plate_id, plate in plates.items():
        for dish_id in plate["dishes"]:
            mapping[dish_id] = (plate_id, plate["split"])
    return mapping


def cross_split_plates(plates: dict[str, dict]) -> int:
    """A plate leaks when its own row would need two splits. The file forbids that."""
    leaked = 0
    for plate in plates.values():
        if plate["split"] not in EXPECTED_PLATES:
            leaked += 1
    return leaked


def hand_image_split_leak(dish_rows: list[dict[str, str]], dish_map: dict[str, tuple[str, str]]) -> dict:
    """Count the Hand image-level split after it is collapsed onto physical plates.

    This is a diagnostic. Those image-level labels are not the training split.
    """
    sides: dict[str, set[str]] = defaultdict(set)
    column_counts: dict[str, int] = defaultdict(int)
    for row in dish_rows:
        plate_id, _split = dish_map[row["dish_id"]]
        sides[plate_id].add(row["split"])
        column_counts[row["split"]] += 1
    cross = sum(1 for present in sides.values() if len(present) > 1)
    test_only = sum(1 for present in sides.values() if present == {"test"})
    return {
        "column_counts": dict(column_counts),
        "cross_split_plates": cross,
        "test_only_plates": test_only,
        "used_as_evaluation_split": False,
    }


def _class_index() -> dict[str, int]:
    return {name: index for index, name in enumerate(CLASSES)}


def _zero_classes() -> list[list[float]]:
    return [[0.0 for _ in phase1.NUTRIENTS] for _ in CLASSES]


def _as_tuples(rows: list[list[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(row) for row in rows)


def build_targets(instances: list[dict[str, str]], plate: dict[str, float]) -> dict:
    """Build B1, B2, B3, and forced-sum targets for one dish.

    `plate` is the Hand plate row. Missing ingredient nutrition is not read as
    zero except inside the B2 sum, where that substitution is the protocol.
    """
    index = _class_index()
    plate_vec = tuple(float(plate[name]) for name in phase1.NUTRIENTS)
    b2 = _zero_classes()
    b3 = _zero_classes()
    known: list[tuple[int, list[float]]] = []
    for item in instances:
        name = item["annotation_name"]
        if name not in index:
            raise ValueError(f"unknown class {name}")
        values = phase1.nutrition_values(item)
        if values is None:
            continue
        vec = [float(values[nutrient]) for nutrient in phase1.NUTRIENTS]
        class_id = index[name]
        known.append((class_id, vec))
        for nutrient_id, value in enumerate(vec):
            b2[class_id][nutrient_id] += value
            b3[class_id][nutrient_id] += value

    defined = bool(known)
    residual = None
    forced = _zero_classes()
    if defined:
        known_sum = [sum(vec[nutrient_id] for _class_id, vec in known) for nutrient_id in range(len(phase1.NUTRIENTS))]
        residual = tuple(plate_vec[nutrient_id] - known_sum[nutrient_id] for nutrient_id in range(len(phase1.NUTRIENTS)))
        for nutrient_id, total in enumerate(known_sum):
            if total != 0.0:
                scale = plate_vec[nutrient_id] / total
                for class_id, vec in known:
                    forced[class_id][nutrient_id] += vec[nutrient_id] * scale
            else:
                share = plate_vec[nutrient_id] / len(known)
                for class_id, _vec in known:
                    forced[class_id][nutrient_id] += share

    return {
        "plate": plate_vec,
        "b2": _as_tuples(b2),
        "b3": _as_tuples(b3),
        "forced": _as_tuples(forced),
        "residual": residual,
        "residual_defined": defined,
        "n_visible": len(instances),
        "n_nutrition": len(known),
    }


def iter_examples(dish_rows: list[dict[str, str]], ingredient_rows: list[dict[str, str]], dish_map: dict[str, tuple[str, str]]) -> list[dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ingredient_rows:
        grouped[row["dish_id"]].append(row)
    examples = []
    seen = set()
    for row in dish_rows:
        dish_id = row["dish_id"]
        if dish_id in seen:
            raise ValueError(f"duplicate dish row {dish_id}")
        seen.add(dish_id)
        if dish_id not in dish_map:
            raise ValueError(f"{dish_id} is missing from the frozen plate assignment")
        plate_values = {name: phase1.parse_optional_float(row.get(name)) for name in phase1.NUTRIENTS}
        if any(value is None for value in plate_values.values()):
            raise ValueError(f"{dish_id} is missing a plate nutrient")
        plate_id, split = dish_map[dish_id]
        targets = build_targets(grouped.get(dish_id, []), plate_values)
        examples.append(
            {
                "dish_id": dish_id,
                "plate_id": plate_id,
                "split": split,
                **targets,
            }
        )
    missing = sorted(set(dish_map) - seen)
    if missing:
        raise ValueError(f"plate assignment names dishes missing from the dish table: {missing[:5]}")
    return examples


def population_std(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot scale an empty training split")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def train_loss_weights(examples: list[dict]) -> dict[str, float]:
    """Per-nutrient L1 weights from training scans only."""
    buckets = {name: [] for name in phase1.NUTRIENTS}
    for example in examples:
        if example["split"] != "train":
            continue
        for index, name in enumerate(phase1.NUTRIENTS):
            buckets[name].append(example["plate"][index])
    weights = {}
    stats = {}
    for name, values in buckets.items():
        std = population_std(values)
        weights[name] = 1.0 / max(std, STD_FLOOR)
        stats[name] = {"n": len(values), "std": std, "weight": weights[name]}
    return {"weights": weights, "train_plate_std": stats}


def vector_sum(rows: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    width = len(phase1.NUTRIENTS)
    return tuple(sum(row[index] for row in rows) for index in range(width))


def negative_supervised_targets(examples: list[dict]) -> list[str]:
    """ReLU heads are only used for targets that are defined to be non-negative."""
    problems = []
    for example in examples:
        for index, name in enumerate(phase1.NUTRIENTS):
            if example["plate"][index] < -1e-6:
                problems.append(f"{example['dish_id']} plate {name}")
        for key in ("b2", "b3", "forced"):
            if key == "b3" and not example["residual_defined"]:
                continue
            if key == "forced" and not example["residual_defined"]:
                continue
            for class_id, row in enumerate(example[key]):
                for index, name in enumerate(phase1.NUTRIENTS):
                    if row[index] < -1e-6:
                        problems.append(f"{example['dish_id']} {key} class {class_id} {name}")
    return problems


def label_facts(examples: list[dict]) -> dict:
    undefined = {name: 0 for name in EXPECTED_PLATES}
    dishes = {name: 0 for name in EXPECTED_PLATES}
    for example in examples:
        dishes[example["split"]] += 1
        if not example["residual_defined"]:
            undefined[example["split"]] += 1
    return {
        "dishes": dishes,
        "undefined_residual_dishes": undefined,
        "undefined_residual_total": sum(undefined.values()),
    }


def percentile_linear(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("empty bootstrap")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def plate_mean_errors(records: list[dict], model: str, nutrient: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record["model"] != model:
            continue
        grouped[record["plate_id"]].append(abs(record["pred"][nutrient] - record["true"][nutrient]))
    if not grouped:
        raise ValueError(f"no predictions for {model}")
    return {plate_id: sum(values) / len(values) for plate_id, values in grouped.items()}


def scan_mae(records: list[dict], model: str, nutrient: str) -> float:
    values = [
        abs(record["pred"][nutrient] - record["true"][nutrient])
        for record in records
        if record["model"] == model
    ]
    if not values:
        raise ValueError(f"no predictions for {model}")
    return sum(values) / len(values)


def paired_bootstrap(method_errors: dict[str, float], control_errors: dict[str, float], seed: int, draws: int) -> dict:
    plates = sorted(method_errors)
    if plates != sorted(control_errors):
        raise ValueError("bootstrap plates differ between models")
    paired = [method_errors[plate_id] - control_errors[plate_id] for plate_id in plates]
    point = sum(paired) / len(paired)
    rng = random.Random(seed)
    stats = []
    width = len(paired)
    for _ in range(draws):
        total = 0.0
        for _draw in range(width):
            total += paired[rng.randrange(width)]
        stats.append(total / width)
    stats.sort()
    low = percentile_linear(stats, BOOTSTRAP_LOW)
    high = percentile_linear(stats, BOOTSTRAP_HIGH)
    return {
        "point": point,
        "low": low,
        "high": high,
        "stably_better": high < 0.0,
        "n_plates": width,
    }


def enters_q2(method: str, control: str, nutrient: str) -> bool:
    return method == "b3" and control in {"b1", "b2"} and nutrient in Q2_NUTRIENTS


def build_comparisons(records: list[dict], seed: int = SEED, draws: int = BOOTSTRAP_DRAWS) -> list[dict]:
    cached = {
        (model, nutrient): plate_mean_errors(records, model, nutrient)
        for model in {record["model"] for record in records}
        for nutrient in phase1.NUTRIENTS
    }
    comparisons = []
    for method, control in (("b3", "b1"), ("b3", "b2"), ("forced", "b1"), ("forced", "b2")):
        for nutrient in REPORT_NUTRIENTS:
            summary = paired_bootstrap(cached[(method, nutrient)], cached[(control, nutrient)], seed, draws)
            comparisons.append(
                {
                    "method": method,
                    "control": control,
                    "nutrient": nutrient,
                    "enters_q2": enters_q2(method, control, nutrient),
                    **summary,
                }
            )
    return comparisons


def q2_from_comparisons(comparisons: list[dict]) -> dict:
    relevant = [item for item in comparisons if item["enters_q2"]]
    expected = {(method, control, nutrient) for method in ("b3",) for control in ("b1", "b2") for nutrient in Q2_NUTRIENTS}
    found = {(item["method"], item["control"], item["nutrient"]) for item in relevant}
    if found != expected:
        raise ValueError(f"Q2 comparisons are {found}, expected {expected}")
    passed = all(item["stably_better"] for item in relevant)
    return {
        "pass": passed,
        "code": "q2_pass_stop_no_new_algorithm" if passed else "q2_fail_stop_wait_triste",
        "n_intervals": len(relevant),
    }


def plate_mae_table(records: list[dict]) -> dict[str, dict[str, float]]:
    table = {}
    for model in sorted({record["model"] for record in records}):
        table[model] = {}
        for nutrient in phase1.NUTRIENTS:
            errors = plate_mean_errors(records, model, nutrient)
            table[model][nutrient] = sum(errors.values()) / len(errors)
    return table


def prediction_records(examples: list[dict], predictions: dict[str, dict[str, tuple[float, ...]]]) -> list[dict]:
    """Join ground truth after predictions exist.

    `predictions[model][dish_id]` is the plate vector from the network. This
    function does not read masks or a ground-truth residual.
    """
    records = []
    for example in examples:
        truth = {name: example["plate"][index] for index, name in enumerate(phase1.NUTRIENTS)}
        for model, by_dish in predictions.items():
            if example["dish_id"] not in by_dish:
                raise ValueError(f"{model} is missing {example['dish_id']}")
            pred_vec = by_dish[example["dish_id"]]
            pred = {name: float(pred_vec[index]) for index, name in enumerate(phase1.NUTRIENTS)}
            records.append(
                {
                    "model": model,
                    "dish_id": example["dish_id"],
                    "plate_id": example["plate_id"],
                    "true": truth,
                    "pred": pred,
                }
            )
    return records


def epoch_plan(n_items: int, epoch: int) -> tuple[list[int], list[bool]]:
    rng = random.Random(SEED + epoch * 100003)
    order = list(range(n_items))
    rng.shuffle(order)
    flips = [rng.random() < HFLIP_PROBABILITY for _ in order]
    return order, flips


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_report(summary: dict) -> str:
    budget = summary["budget"]
    if budget["epochs"] != EPOCHS or budget["input_size"] != INPUT_SIZE or budget["backbone"] != BACKBONE_NAME:
        raise RuntimeError("report budget does not match the frozen source constants")
    decision = q2_from_comparisons(summary["comparisons"])
    if decision["pass"] != summary["q2"]["pass"] or decision["code"] != summary["q2"]["code"]:
        raise RuntimeError("report decision does not match the comparisons")
    if summary["test_time"]["reads_ground_truth_mask"] or summary["test_time"]["reads_ground_truth_residual"]:
        raise RuntimeError("test-time ground truth was marked as read")
    if summary["split"]["cross_split_plates"] != 0:
        raise RuntimeError("report describes a leaking split")
    lines = ["# 不完整食材标签：第二阶段 B1 / B2 / B3 最小对照", ""]
    if decision["pass"]:
        lines.append(
            "Q2 成立。留出集上，B3 在能量、脂肪、碳水、蛋白质上都稳定优于 B1，也稳定优于 B2。"
            "按写死的停止规则，到这里停下，不自动开始新算法。`docs/STATUS.md` 这次没有改，下一步没有解锁。"
        )
    else:
        lines.append(
            "Q2 不成立。留出集上，B3 没有在能量、脂肪、碳水、蛋白质上同时稳定优于 B1 和 B2。"
            "按写死的停止规则，到这里停下，等待 Triste。"
            "不叠更复杂的部分标签网络，不改损失、协议、容差、合并规则或划分，也不改 `docs/STATUS.md`。"
        )
    lines.extend(["", "## 写死的骨干和预算", "", "下面这段在训练之前写在脚本常量里。看到留出营养误差之后没有改。", ""])
    lines.append(FROZEN_PROTOCOL_TEXT.strip())
    lines.extend(
        [
            "",
            "| 项目 | 取值 |",
            "|---|---|",
            f"| 骨干 | `{BACKBONE_NAME}` |",
            f"| 初始权重 | `{WEIGHTS_NAME}` |",
            "| 质量估计器权重 | 未加载 |",
            f"| 输入 | 单张俯视 RGB，{INPUT_SIZE}×{INPUT_SIZE} |",
            f"| 训练轮数 | {EPOCHS} |",
            f"| 批量 | {BATCH_SIZE} |",
            f"| 学习率 | {LEARNING_RATE:g} |",
            f"| 权重衰减 | {WEIGHT_DECAY:g} |",
            f"| 梯度裁剪 | {GRAD_CLIP:g} |",
            f"| 训练翻转 | 水平，概率 {HFLIP_PROBABILITY:g} |",
            "| 测试增强 | 无 |",
            f"| 种子 | {SEED} |",
            f"| bootstrap | {BOOTSTRAP_DRAWS} 次，种子 {SEED}，线性分位 {BOOTSTRAP_LOW:g} / {BOOTSTRAP_HIGH:g} |",
            f"| 实际优化步数（每个模型） | {summary['steps_per_model']} |",
            "",
            "验证集选中的 epoch：",
            "",
            "| 模型 | epoch | 验证选择分 |",
            "|---|---:|---:|",
        ]
    )
    for model in MODELS:
        chosen = summary["selection"][model]
        lines.append(f"| {MODEL_ZH[model]} | {chosen['epoch']} | {fmt(chosen['val_score'])} |")
    split = summary["split"]
    lines.extend(
        [
            "",
            "## 划分复核",
            "",
            "划分直接读 `results/u2_phase1_partial_label_audit/plate_assignment.csv`。脚本用第一阶段的合并和排序再算一次，只用于核对，不一致就停止，不另写一份划分。",
            "",
            "| 项目 | 结果 |",
            "|---|---:|",
            f"| 训练物理餐盘 | {split['train_plates']} |",
            f"| 验证物理餐盘 | {split['validation_plates']} |",
            f"| 留出物理餐盘 | {split['holdout_plates']} |",
            f"| 跨划分物理餐盘 | {split['cross_split_plates']} |",
            f"| 训练 / 验证 / 留出扫描 | {split['train_dishes']} / {split['validation_dishes']} / {split['holdout_dishes']} |",
            f"| 与第一阶段重算一致 | {'是' if split['matches_phase1_algorithm'] else '否'} |",
            f"| Hand 图像级列的 train/test 扫描 | {split['hand_column_counts'].get('train', 0)} / {split['hand_column_counts'].get('test', 0)} |",
            f"| Hand 图像级划分跨盘 | {split['hand_cross_split_plates']} |",
            "| 主评价是否使用 Hand 图像级划分 | 否 |",
            "",
            f"公开盘级表 blob `{summary['blobs']['dish']}`，食材表 blob `{summary['blobs']['ingredient']}`。两份都与第一阶段记录一致，脚本结束时字节没有变。",
            "",
            f"残差没有定义的扫描：训练 {summary['label_facts']['undefined_residual_dishes']['train']}，"
            f"验证 {summary['label_facts']['undefined_residual_dishes']['validation']}，"
            f"留出 {summary['label_facts']['undefined_residual_dishes']['holdout']}。"
            "这是标签构造计数。一个有营养实例都没有时，B3 和强迫加和不把可见求和写成 0。",
            "",
            "## 测试时的输入",
            "",
            "测试输入是一张俯视 RGB。前向函数只接收图像张量。脚本不打开 Hand 掩膜文件，也不把真值残差送进网络。真值盘级营养只在预测写完之后用来算绝对误差。验证集只用于选择 epoch。留出集在预算写死、训练结束之后评估一次。",
            "",
            "## 留出主指标",
            "",
            "下表是物理餐盘等权的平均绝对误差。质量不进入 Q2。",
            "",
            "| 模型 | 能量（千卡） | 脂肪（克） | 碳水（克） | 蛋白质（克） | 质量（克） |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        mae = summary["plate_mae"][model]
        lines.append(
            "| "
            + " | ".join(
                [
                    MODEL_ZH[model],
                    fmt(mae["calories"]),
                    fmt(mae["fat"]),
                    fmt(mae["carb"]),
                    fmt(mae["protein"]),
                    fmt(mae["mass"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "扫描等权的平均绝对误差写在 `summary.json` 的 `scan_mae`。Q2 不读这组数。",
            "",
            "## 分组 bootstrap",
            "",
            "点估计是同一只物理餐盘上（方法误差 − 对照误差）的平均。负值表示方法的绝对误差更低。区间上界小于 0 记为稳定优于。",
            "",
            "| 比较 | 营养素 | 点估计 | 95% 区间 | 稳定优于 | 进入 Q2 |",
            "|---|---|---:|---|---|---|",
        ]
    )
    order = {name: index for index, name in enumerate(REPORT_NUTRIENTS)}
    comparisons = sorted(
        summary["comparisons"],
        key=lambda item: (item["method"], item["control"], order[item["nutrient"]]),
    )
    for item in comparisons:
        lines.append(
            "| {method} − {control} | {nutrient} | {point} | [{low}, {high}] | {better} | {flag} |".format(
                method=MODEL_ZH[item["method"]],
                control=MODEL_ZH[item["control"]],
                nutrient=phase1.NUTRIENT_ZH[item["nutrient"]],
                point=fmt(item["point"]),
                low=fmt(item["low"]),
                high=fmt(item["high"]),
                better="是" if item["stably_better"] else "否",
                flag="是" if item["enters_q2"] else "否",
            )
        )
    lines.extend(
        [
            "",
            "## 强迫加和诊断",
            "",
            "强迫加和把残差按比例摊回有营养的可见食材，再按与 B2 相同的类别求和做盘级预测。上表里方法名为强迫加和诊断的行就是这次诊断。这些行不进入停止规则，也不参与 Q2 的通过与否。",
            "",
            "## Q2",
            "",
            f"判定码 `{decision['code']}`。进入 Q2 的区间有 {decision['n_intervals']} 个，要求八个区间的上界都小于 0。",
            "",
        ]
    )
    if decision["pass"]:
        lines.append("Q2 成立，实验到此停止。不自动开始新算法。")
    else:
        lines.append("Q2 不成立，实验到此停止，等待 Triste。")
    lines.extend(
        [
            "",
            "## 复现",
            "",
            "在仓库根目录运行：",
            "",
            "```bash",
            "python scripts/u2_phase2_b123_contrasts.py check --repo-root .",
            "python scripts/u2_phase2_b123_contrasts.py train --repo-root . --out results/u2_phase2_b123_contrasts",
            "python scripts/u2_phase2_b123_contrasts.py eval --repo-root . --out results/u2_phase2_b123_contrasts",
            "python -m unittest tests.test_u2_phase2_b123_contrasts",
            "```",
            "",
            "训练不写留出营养误差。`eval` 在 `holdout_metrics.json` 已经存在时拒绝再跑。检查点在 `outputs/u2_phase2_b123_contrasts/checkpoints/`，不作为本结果的数值记录。数值记录是本目录的 `summary.json` 和 `heldout_predictions.csv`。",
            "",
        ]
    )
    return "\n".join(lines)


def protected_output(repo: Path, out: Path) -> bool:
    resolved = out.resolve()
    protected = [
        repo / "results/u2_phase1_partial_label_audit",
        repo / "results/u1_phase1_data_audit",
        repo / "results/icdl_phase1_sequence_audit",
        repo / "results/icdl_phase2_light_models",
        repo / "data",
        repo / "backup",
        repo / "weights",
    ]
    for path in protected:
        target = path.resolve()
        if resolved == target or target in resolved.parents:
            return True
    return False


def public_blobs(repo: Path) -> dict[str, str]:
    dish = (repo / "data/hand2026/dish_nutrition.csv").read_bytes()
    ingredient = (repo / "data/hand2026/ingredient_nutrition.csv").read_bytes()
    return {
        "dish": phase1.git_blob_sha1(dish),
        "ingredient": phase1.git_blob_sha1(ingredient),
        "dish_matches_phase1": phase1.git_blob_sha1(dish) == phase1.DISH_BLOB_SHA,
        "ingredient_matches_phase1": phase1.git_blob_sha1(ingredient) == phase1.INGREDIENT_BLOB_SHA,
    }


def load_tables(repo: Path) -> tuple[list[dict], list[dict], dict[str, dict], dict[str, tuple[str, str]]]:
    blobs = public_blobs(repo)
    if not blobs["dish_matches_phase1"] or not blobs["ingredient_matches_phase1"]:
        raise RuntimeError("public Hand tables do not match the phase-1 blob hashes")
    _, dishes = phase1.read_csv_rows(repo / "data/hand2026/dish_nutrition.csv")
    _, ingredients = phase1.read_csv_rows(repo / "data/hand2026/ingredient_nutrition.csv")
    plates = load_plate_assignment(repo / "results/u2_phase1_partial_label_audit/plate_assignment.csv")
    dish_map = dish_split_map(plates)
    return dishes, ingredients, plates, dish_map


def phase1_assignment(repo: Path) -> dict[str, dict]:
    """Recompute the phase-1 map as a check. Training still reads the CSV."""
    _, dishes = phase1.read_csv_rows(repo / "data/hand2026/dish_nutrition.csv")
    _, ingredients = phase1.read_csv_rows(repo / "data/hand2026/ingredient_nutrition.csv")
    metadata = phase1.load_metadata(repo / "data/nutrition5k/metadata")
    rgb_train = set(phase1.load_id_list(repo / "data/nutrition5k/dish_ids/splits/rgb_train_ids.txt"))
    rgb_test = set(phase1.load_id_list(repo / "data/nutrition5k/dish_ids/splits/rgb_test_ids.txt"))
    images = {
        "checked": False,
        "expected": len(dishes),
        "opened": 0,
        "missing": [],
        "unreadable": [],
        "size_counts": {},
        "mode_counts": {},
    }
    blobs = {
        "dish_start": phase1.DISH_BLOB_SHA,
        "dish_end": phase1.DISH_BLOB_SHA,
        "ingredient_start": phase1.INGREDIENT_BLOB_SHA,
        "ingredient_end": phase1.INGREDIENT_BLOB_SHA,
        "changed_during_audit": False,
    }
    summary = phase1.run_from_loaded(dishes, ingredients, metadata, rgb_train, rgb_test, images, blobs)
    rebuilt = {}
    for row in summary["plate_rows"]:
        rebuilt[row["plate_id"]] = {"split": row["split"], "dishes": list(row["hand_dish_ids"])}
    return rebuilt


def assignments_match(frozen: dict[str, dict], rebuilt: dict[str, dict]) -> bool:
    if set(frozen) != set(rebuilt):
        return False
    for plate_id, plate in frozen.items():
        other = rebuilt[plate_id]
        if plate["split"] != other["split"]:
            return False
        if set(plate["dishes"]) != set(other["dishes"]):
            return False
    return True


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_prediction_csv(path: Path, records: list[dict]) -> None:
    fieldnames = ["model", "dish_id", "plate_id", "nutrient", "y_true", "y_pred"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for nutrient in phase1.NUTRIENTS:
                writer.writerow(
                    {
                        "model": record["model"],
                        "dish_id": record["dish_id"],
                        "plate_id": record["plate_id"],
                        "nutrient": nutrient,
                        "y_true": format(record["true"][nutrient], ".17g"),
                        "y_pred": format(record["pred"][nutrient], ".17g"),
                    }
                )


def read_prediction_csv(path: Path) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], row["dish_id"])
            item = grouped.get(key)
            if item is None:
                item = {
                    "model": row["model"],
                    "dish_id": row["dish_id"],
                    "plate_id": row["plate_id"],
                    "true": {},
                    "pred": {},
                }
                grouped[key] = item
            item["true"][row["nutrient"]] = float(row["y_true"])
            item["pred"][row["nutrient"]] = float(row["y_pred"])
    records = list(grouped.values())
    for record in records:
        if set(record["true"]) != set(phase1.NUTRIENTS) or set(record["pred"]) != set(phase1.NUTRIENTS):
            raise ValueError(f"incomplete prediction row for {record['dish_id']}")
    return records


def require_dir(repo: Path, out: Path) -> Path:
    if protected_output(repo, out):
        raise RuntimeError(f"refusing to write into protected path {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def cmd_check(repo: Path) -> dict:
    dishes, ingredients, plates, dish_map = load_tables(repo)
    examples = iter_examples(dishes, ingredients, dish_map)
    rebuilt = phase1_assignment(repo)
    match = assignments_match(plates, rebuilt)
    if not match:
        raise RuntimeError("frozen plate_assignment.csv does not match the phase-1 algorithm")
    if cross_split_plates(plates) != 0:
        raise RuntimeError("frozen assignment leaks")
    problems = negative_supervised_targets(examples)
    if problems:
        raise RuntimeError("supervised ReLU target is negative: " + "; ".join(problems[:8]))
    facts = label_facts(examples)
    leak = hand_image_split_leak(dishes, dish_map)
    weights = train_loss_weights(examples)
    imagery = repo / "data/nutrition5k/imagery/realsense_overhead"
    missing = [example["dish_id"] for example in examples if not overhead_rgb_path(imagery, example["dish_id"]).is_file()]
    if missing:
        raise RuntimeError(f"missing overhead RGB for {len(missing)} dishes, first {missing[:3]}")
    return {
        "plates": {name: sum(plate["split"] == name for plate in plates.values()) for name in EXPECTED_PLATES},
        "cross_split_plates": 0,
        "matches_phase1_algorithm": True,
        "label_facts": facts,
        "hand_image_split": leak,
        "train_loss_weights": weights,
        "images": len(examples) - len(missing),
        "holdout_nutrition_error_computed": False,
    }


def _import_torch():
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    return torch, resnet18, ResNet18_Weights


def build_model(kind: str, pretrained: bool):
    torch, resnet18, ResNet18_Weights = _import_torch()
    if kind not in MODELS:
        raise ValueError(kind)
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    network = resnet18(weights=weights)
    features = network.fc.in_features
    network.fc = torch.nn.Identity()
    if kind == "b1":
        head = torch.nn.Linear(features, len(phase1.NUTRIENTS))
    elif kind in {"b2", "forced"}:
        head = torch.nn.Linear(features, len(CLASSES) * len(phase1.NUTRIENTS))
    else:
        head = torch.nn.Linear(features, len(CLASSES) * len(phase1.NUTRIENTS) + len(phase1.NUTRIENTS))
    return PlateModel(kind, network, head)


class PlateModel:
    """ResNet-18 plus one linear head. The forward pass accepts an image only."""

    def __init__(self, kind: str, backbone, head):
        import torch.nn as nn

        if not isinstance(backbone, nn.Module) or not isinstance(head, nn.Module):
            raise TypeError("backbone and head must be torch modules")
        self.kind = kind
        self.backbone = backbone
        self.head = head

    def to(self, device):
        self.backbone.to(device)
        self.head.to(device)
        return self

    def train(self) -> None:
        self.backbone.train()
        self.head.train()

    def eval(self) -> None:
        self.backbone.eval()
        self.head.eval()

    def parameters(self):
        yield from self.backbone.parameters()
        yield from self.head.parameters()

    def state_dict(self) -> dict:
        return {"backbone": self.backbone.state_dict(), "head": self.head.state_dict(), "kind": self.kind}

    def load_state_dict(self, payload: dict) -> None:
        if payload["kind"] != self.kind:
            raise ValueError(f"checkpoint kind {payload['kind']} does not match {self.kind}")
        self.backbone.load_state_dict(payload["backbone"])
        self.head.load_state_dict(payload["head"])

    def __call__(self, images):
        torch, _resnet18, _weights = _import_torch()
        features = self.backbone(images)
        raw = self.head(features)
        if self.kind == "b1":
            plate = torch.relu(raw)
            return {"plate": plate}
        class_width = len(CLASSES) * len(phase1.NUTRIENTS)
        classes = torch.relu(raw[:, :class_width]).reshape(images.shape[0], len(CLASSES), len(phase1.NUTRIENTS))
        if self.kind in {"b2", "forced"}:
            return {"classes": classes, "plate": classes.sum(dim=1)}
        residual = raw[:, class_width:]
        return {"classes": classes, "residual": residual, "plate": classes.sum(dim=1) + residual}


def _weight_tensor(weights: dict[str, float], device):
    torch, _resnet18, _weights = _import_torch()
    return torch.tensor([weights[name] for name in phase1.NUTRIENTS], dtype=torch.float32, device=device)


def loss_for_batch(kind: str, outputs: dict, batch: dict, weights) -> object:
    torch, _resnet18, _weights = _import_torch()
    if kind == "b1":
        error = (outputs["plate"] - batch["plate"]).abs() * weights
        return error.mean()
    classes = outputs["classes"]
    target = batch["classes"]
    class_error = (classes - target).abs() * weights.view(1, 1, -1)
    supervise = batch["supervise"]
    if kind in {"b2", "forced"}:
        if bool(supervise.any()):
            return class_error[supervise].mean()
        return classes.sum() * 0.0
    per_example = class_error.mean(dim=(1, 2))
    residual_error = ((outputs["residual"] - batch["residual"]).abs() * weights).mean(dim=1)
    both = per_example + residual_error
    if bool(supervise.any()):
        return both[supervise].mean()
    return classes.sum() * 0.0


def stack_images(images, device, flips: list[bool]):
    torch, _resnet18, _weights = _import_torch()
    chosen = []
    for image, flip in zip(images, flips):
        chosen.append(image.flip(-1) if flip else image)
    return torch.stack(chosen).to(device, non_blocking=True)


def _stack_examples(kind: str, subset: list[dict], device, flips: list[bool], images):
    """Training batch. B1 reads plate totals only; the class tensor stays zero."""
    torch, _resnet18, _weights = _import_torch()
    stacked = stack_images(images, device, flips)
    plate = torch.tensor([example["plate"] for example in subset], dtype=torch.float32, device=device)
    if kind == "b1":
        classes = torch.zeros(len(subset), len(CLASSES), len(phase1.NUTRIENTS), dtype=torch.float32, device=device)
    else:
        key = {"b2": "b2", "b3": "b3", "forced": "forced"}[kind]
        classes = torch.tensor([example[key] for example in subset], dtype=torch.float32, device=device)
    residual_rows = []
    supervise = []
    for example in subset:
        if kind == "b3":
            defined = example["residual_defined"]
            residual_rows.append(example["residual"] if defined else (0.0, 0.0, 0.0, 0.0, 0.0))
            supervise.append(defined)
        elif kind == "forced":
            supervise.append(example["residual_defined"])
            residual_rows.append((0.0, 0.0, 0.0, 0.0, 0.0))
        else:
            supervise.append(True)
            residual_rows.append((0.0, 0.0, 0.0, 0.0, 0.0))
    return {
        "images": stacked,
        "plate": plate,
        "classes": classes,
        "residual": torch.tensor(residual_rows, dtype=torch.float32, device=device),
        "supervise": torch.tensor(supervise, dtype=torch.bool, device=device),
    }


def load_image_tensor(path: Path):
    torch, _resnet18, _weights = _import_torch()
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    array = torch.tensor(list(rgb.getdata()), dtype=torch.float32).reshape(INPUT_SIZE, INPUT_SIZE, 3) / 255.0
    array = array.permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (array - mean) / std


def preload_images(examples: list[dict], imagery: Path) -> list:
    tensors = []
    for example in examples:
        path = overhead_rgb_path(imagery, example["dish_id"])
        tensors.append(load_image_tensor(path))
    return tensors


def score_split(model, examples: list[dict], images: list, device, split: str, allow_holdout: bool) -> dict[str, float]:
    assert_split_allowed(split, allow_holdout)
    torch, _resnet18, _weights = _import_torch()
    chosen = [(example, image) for example, image in zip(examples, images) if example["split"] == split]
    if not chosen:
        raise ValueError(f"no dishes in {split}")
    model.eval()
    predictions = []
    truths = []
    plate_ids = []
    with torch.no_grad():
        for start in range(0, len(chosen), BATCH_SIZE):
            chunk = chosen[start : start + BATCH_SIZE]
            batch_examples = [example for example, _image in chunk]
            batch_images = [image for _example, image in chunk]
            pred = model(stack_images(batch_images, device, [False] * len(chunk)))["plate"].detach().cpu()
            predictions.append(pred)
            truths.append(
                torch.tensor([example["plate"] for example in batch_examples], dtype=torch.float32)
            )
            plate_ids.extend(example["plate_id"] for example in batch_examples)
    pred_all = torch.cat(predictions, dim=0)
    true_all = torch.cat(truths, dim=0)
    per_plate: dict[str, dict[str, list[float]]] = defaultdict(lambda: {name: [] for name in phase1.NUTRIENTS})
    for row, plate_id in enumerate(plate_ids):
        for index, name in enumerate(phase1.NUTRIENTS):
            per_plate[plate_id][name].append(abs(float(pred_all[row, index]) - float(true_all[row, index])))
    mae = {}
    for name in phase1.NUTRIENTS:
        plate_values = [sum(item[name]) / len(item[name]) for item in per_plate.values()]
        mae[name] = sum(plate_values) / len(plate_values)
    return mae


def selection_score(mae: dict[str, float]) -> float:
    return sum(mae[name] for name in Q2_NUTRIENTS)


def train_one(kind: str, train_examples: list[dict], val_examples: list[dict], train_images: list, val_images: list, weights: dict[str, float], device, checkpoint_path: Path) -> dict:
    torch, _resnet18, _weights = _import_torch()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = build_model(kind, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    weight_tensor = _weight_tensor(weights, device)
    history = []
    best = None
    steps = 0
    for epoch in range(EPOCHS):
        model.train()
        order, flips = epoch_plan(len(train_examples), epoch)
        for start in range(0, len(order), BATCH_SIZE):
            batch_ids = order[start : start + BATCH_SIZE]
            subset = [train_examples[index] for index in batch_ids]
            batch_images = [train_images[index] for index in batch_ids]
            batch_flips = flips[start : start + BATCH_SIZE]
            packed = _stack_examples(kind, subset, device, batch_flips, batch_images)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(packed["images"])
            loss = loss_for_batch(kind, outputs, packed, weight_tensor)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{kind} produced a non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), GRAD_CLIP)
            optimizer.step()
            steps += 1
        scheduler.step()
        val_mae = score_split(model, val_examples, val_images, device, "validation", allow_holdout=False)
        score = selection_score(val_mae)
        history.append({"epoch": epoch, "val_mae": val_mae, "val_score": score})
        if best is None or score < best["val_score"]:
            best = {
                "epoch": epoch,
                "val_score": score,
                "val_mae": val_mae,
                "state": {
                    "backbone": {key: value.detach().cpu().clone() for key, value in model.backbone.state_dict().items()},
                    "head": {key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()},
                    "kind": kind,
                },
            }
        print(f"{kind} epoch {epoch} val_score {score:.4f}", flush=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": best["state"],
            "epoch": best["epoch"],
            "val_score": best["val_score"],
            "val_mae": best["val_mae"],
            "kind": kind,
            "budget": frozen_budget(),
        },
        checkpoint_path,
    )
    return {"epoch": best["epoch"], "val_score": best["val_score"], "val_mae": best["val_mae"], "history": history, "steps": steps}


def predict_split(model, examples: list[dict], images: list, device) -> dict[str, tuple[float, ...]]:
    torch, _resnet18, _weights = _import_torch()
    model.eval()
    predicted = {}
    with torch.no_grad():
        for start in range(0, len(examples), BATCH_SIZE):
            subset = examples[start : start + BATCH_SIZE]
            batch_images = images[start : start + BATCH_SIZE]
            pred = model(stack_images(batch_images, device, [False] * len(subset)))["plate"].detach().cpu()
            for row, example in enumerate(subset):
                predicted[example["dish_id"]] = tuple(float(pred[row, index]) for index in range(len(phase1.NUTRIENTS)))
    return predicted


def cmd_train(repo: Path, out: Path, checkpoints: Path) -> None:
    require_dir(repo, out)
    if (out / "holdout_metrics.json").exists():
        raise RuntimeError("holdout metrics already exist; refusing to train again")
    dishes, ingredients, plates, dish_map = load_tables(repo)
    examples = iter_examples(dishes, ingredients, dish_map)
    if not assignments_match(plates, phase1_assignment(repo)):
        raise RuntimeError("frozen plate assignment does not match the phase-1 algorithm")
    problems = negative_supervised_targets(examples)
    if problems:
        raise RuntimeError("supervised ReLU target is negative")
    weights = train_loss_weights(examples)
    facts = label_facts(examples)
    leak = hand_image_split_leak(dishes, dish_map)
    torch, _resnet18, _weights = _import_torch()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imagery = repo / "data/nutrition5k/imagery/realsense_overhead"
    train_examples = [example for example in examples if example["split"] == "train"]
    val_examples = [example for example in examples if example["split"] == "validation"]
    holdout_ids = {example["dish_id"] for example in examples if example["split"] == "holdout"}
    if holdout_ids & {example["dish_id"] for example in train_examples}:
        raise RuntimeError("holdout dish leaked into training")
    config = {
        "budget": frozen_budget(),
        "train_loss_weights": weights,
        "label_facts": facts,
        "hand_image_split": leak,
        "device": str(device),
        "holdout_nutrition_error_computed": False,
        "steps_per_epoch": (len(train_examples) + BATCH_SIZE - 1) // BATCH_SIZE,
    }
    json_dump(out / "frozen_config.json", config)
    print(f"training on {device} without opening holdout images", flush=True)
    train_images = preload_images(train_examples, imagery)
    val_images = preload_images(val_examples, imagery)
    selection = {}
    histories = {}
    steps = None
    for kind in MODELS:
        result = train_one(
            kind,
            train_examples,
            val_examples,
            train_images,
            val_images,
            weights["weights"],
            device,
            checkpoints / f"{kind}.pt",
        )
        selection[kind] = {"epoch": result["epoch"], "val_score": result["val_score"], "val_mae": result["val_mae"]}
        histories[kind] = result["history"]
        if steps is None:
            steps = result["steps"]
        elif steps != result["steps"]:
            raise RuntimeError("models did not take the same number of optimization steps")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    json_dump(out / "val_history.json", {"selection": selection, "history": histories, "steps_per_model": steps})
    after = public_blobs(repo)
    if after["dish"] != phase1.DISH_BLOB_SHA or after["ingredient"] != phase1.INGREDIENT_BLOB_SHA:
        raise RuntimeError("public CSV bytes changed during training")
    print(f"trained steps_per_model={steps} checkpoints={checkpoints}", flush=True)


def cmd_eval(repo: Path, out: Path, checkpoints: Path) -> None:
    require_dir(repo, out)
    if (out / "holdout_metrics.json").exists():
        raise RuntimeError("holdout was already scored; refusing a second evaluation")
    config_path = out / "frozen_config.json"
    if not config_path.is_file():
        raise RuntimeError("missing frozen_config.json; train before eval")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["budget"]["epochs"] != EPOCHS or config["budget"]["backbone"] != BACKBONE_NAME:
        raise RuntimeError("frozen_config.json does not match the source budget")
    if config.get("holdout_nutrition_error_computed"):
        raise RuntimeError("config already marks the holdout as scored")
    dishes, ingredients, plates, dish_map = load_tables(repo)
    if not assignments_match(plates, phase1_assignment(repo)):
        raise RuntimeError("frozen plate assignment does not match the phase-1 algorithm")
    examples = iter_examples(dishes, ingredients, dish_map)
    holdout = [example for example in examples if example["split"] == "holdout"]
    if len({example["plate_id"] for example in holdout}) != EXPECTED_PLATES["holdout"]:
        raise RuntimeError("holdout plate count is not the frozen 99")
    torch, _resnet18, _weights = _import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imagery = repo / "data/nutrition5k/imagery/realsense_overhead"
    images = preload_images(holdout, imagery)
    predictions = {}
    for kind in MODELS:
        checkpoint = torch.load(checkpoints / f"{kind}.pt", map_location="cpu", weights_only=False)
        if checkpoint["budget"]["epochs"] != EPOCHS:
            raise RuntimeError("checkpoint budget does not match the frozen source")
        model = build_model(kind, pretrained=False).to(device)
        model.load_state_dict(checkpoint["state"])
        predictions[kind] = predict_split(model, holdout, images, device)
    records = prediction_records(holdout, predictions)
    comparisons = build_comparisons(records)
    decision = q2_from_comparisons(comparisons)
    plate_mae = plate_mae_table(records)
    scan = {model: {name: scan_mae(records, model, name) for name in phase1.NUTRIENTS} for model in MODELS}
    history = json.loads((out / "val_history.json").read_text(encoding="utf-8"))
    blobs = public_blobs(repo)
    leak = hand_image_split_leak(dishes, dish_map)
    facts = label_facts(examples)
    summary = {
        "budget": frozen_budget(),
        "q2": decision,
        "test_time": {
            "inputs": "single overhead rgb",
            "reads_ground_truth_mask": False,
            "reads_ground_truth_residual": False,
        },
        "split": {
            "train_plates": EXPECTED_PLATES["train"],
            "validation_plates": EXPECTED_PLATES["validation"],
            "holdout_plates": EXPECTED_PLATES["holdout"],
            "cross_split_plates": cross_split_plates(plates),
            "train_dishes": facts["dishes"]["train"],
            "validation_dishes": facts["dishes"]["validation"],
            "holdout_dishes": facts["dishes"]["holdout"],
            "matches_phase1_algorithm": True,
            "hand_column_counts": leak["column_counts"],
            "hand_cross_split_plates": leak["cross_split_plates"],
        },
        "blobs": blobs,
        "label_facts": facts,
        "plate_mae": plate_mae,
        "scan_mae": scan,
        "comparisons": comparisons,
        "selection": history["selection"],
        "steps_per_model": history["steps_per_model"],
        "holdout_scored_once": True,
    }
    metrics = {
        "q2": decision,
        "plate_mae": plate_mae,
        "scan_mae": scan,
        "comparisons": comparisons,
    }
    text = render_report(summary)
    write_prediction_csv(out / "heldout_predictions.csv", records)
    json_dump(out / "summary.json", summary)
    (out / "report.md").write_text(text, encoding="utf-8")
    json_dump(out / "holdout_metrics.json", metrics)
    after = public_blobs(repo)
    if after["dish"] != blobs["dish"] or after["ingredient"] != blobs["ingredient"]:
        raise RuntimeError("public CSV bytes changed during eval")
    print(decision["code"], flush=True)


def cmd_check_metrics(out: Path) -> None:
    saved = json.loads((out / "holdout_metrics.json").read_text(encoding="utf-8"))
    records = read_prediction_csv(out / "heldout_predictions.csv")
    comparisons = build_comparisons(records)
    decision = q2_from_comparisons(comparisons)
    plate_mae = plate_mae_table(records)
    if decision != saved["q2"]:
        raise RuntimeError("recomputed Q2 does not match holdout_metrics.json")
    for model, nutrients in plate_mae.items():
        for name, value in nutrients.items():
            if abs(value - saved["plate_mae"][model][name]) > 1e-6:
                raise RuntimeError(f"recomputed MAE differs for {model} {name}")
    for left, right in zip(comparisons, saved["comparisons"]):
        for key in ("point", "low", "high"):
            if abs(left[key] - right[key]) > 1e-6:
                raise RuntimeError(f"recomputed interval differs for {left['method']} {left['nutrient']} {key}")
        if left["stably_better"] != right["stably_better"] or left["enters_q2"] != right["enters_q2"]:
            raise RuntimeError("recomputed flags differ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="U2 phase-2 B1/B2/B3 contrasts")
    parser.add_argument("command", choices=("check", "train", "eval", "check-metrics"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/u2_phase2_b123_contrasts"))
    parser.add_argument("--checkpoints", type=Path, default=Path("outputs/u2_phase2_b123_contrasts/checkpoints"))
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    out = args.out if args.out.is_absolute() else repo / args.out
    checkpoints = args.checkpoints if args.checkpoints.is_absolute() else repo / args.checkpoints
    if args.command == "check":
        summary = cmd_check(repo)
        print(json.dumps({"plates": summary["plates"], "cross_split_plates": 0, "images": summary["images"], "holdout_nutrition_error_computed": False}, sort_keys=True))
        return 0
    if args.command == "train":
        cmd_train(repo, out, checkpoints)
        return 0
    if args.command == "eval":
        cmd_eval(repo, out, checkpoints)
        return 0
    cmd_check_metrics(out)
    print("metrics match the prediction table")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
