#!/usr/bin/env python3
"""Explain the frozen Mass-only energy path and the 98.6201 vs 75.3870 gap.

Reads ``results/heldout_plan``, ``results/heldout_eval``, and
``results/heldout_energy_audit``. Does not roll out a model, does not train,
and does not write those directories or ``docs/STATUS.md``.

Mass-only energy is the dish-level energy head of the frozen RGB mass
estimator. The food-table commit is FNDDS kcal/100 g times component grams.
Oracle × our grams is the label kcal/g times the committed grams. The gap
between 98.6201 and 75.3870 is measured from those frozen rows.

    python scripts/heldout_massonly_energy_path.py --self-check
    python scripts/heldout_massonly_energy_path.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "results" / "heldout_plan"
EVAL = ROOT / "results" / "heldout_eval"
AUDIT = ROOT / "results" / "heldout_energy_audit"
OUT = ROOT / "results" / "heldout_massonly_energy_path"
CAFE2 = ROOT / "data" / "nutrition5k" / "metadata" / "dish_metadata_cafe2.csv"
RGB_TRAIN_IDS = ROOT / "data" / "nutrition5k" / "dish_ids" / "splits" / "rgb_train_ids.txt"
STATUS = ROOT / "docs" / "STATUS.md"
TRAIN_META = ROOT / "results" / "mass_estimator" / "train_meta.json"

N_PAIRED = 228
N_MATCHED = 222
N_WRITES = 575
N_SWAP = 106
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
GRAM_MATCH_G = 0.001
HEAD_MATCH_KCAL = 0.001

# Frozen #63 / #65 figures. Recomputation must land on these values.
PUBLISHED = {
    "ours_energy_mae_kcal": 116.84940205562714,
    "ours_mass_mae_g": 50.94828289473687,
    "direct_energy_mae_kcal": 105.61565698189912,
    "direct_mass_mae_g": 73.74561403508773,
    "mass_only_energy_mae_kcal": 75.38695546340782,
    "mass_only_mass_mae_g": 51.710742982456146,
    "energy_minus_direct_kcal": 11.233745073728016,
    "mass_minus_direct_g": -22.797331140350856,
    "energy_minus_mass_only_kcal": 41.46244659221932,
    "mass_minus_mass_only_g": -0.7624600877192762,
}
FROZEN_AUDIT = {
    "oracle_our_grams_kcal": 98.6201298896501,
    "oracle_head_grams_kcal": 100.31523136158658,
    "head_density_mae": 0.7529871262441593,
    "food_table_density_mae": 0.6313931704691669,
    "head_cancellation_kcal": 77.825274752864,
    "food_table_cancellation_kcal": 58.16400435167886,
    "head_abs_gram_term_kcal": 61.370634379619894,
    "head_abs_density_term_kcal": 91.84159583665202,
    "food_table_abs_gram_term_kcal": 87.8229337616574,
    "food_table_abs_density_term_kcal": 87.19047264564867,
    "head_opposite_sign_rate": 0.7675438596491229,
    "food_table_opposite_sign_rate": 0.5877192982456141,
}
INPUT_SHA256 = {
    "scores": "849c9781a456788b4175202c0cc749b9e1ff01b9ed4339a903cf0ee2a7590e9e",
    "rollout": "8ce6ccaf5fc3d3cd9abdbac474ed00defb3fa8c91335a7b6eafc6b95ac4e6187",
    "direct": "076d135d3ea0067eea53b2ae3d38f48d78ac5f2ec6969794bc57d115cb1283da",
    "mass": "54d4fce9b638b8fd21f2b7532376d5361ec2fe9ad8595ecd0d4bc0bf1d7bd483",
    "eval_summary": "81198046666cc8c7aee5dbc50123b78d6f5c64d08082bad58ae20e03cc6516f2",
    "heldout_ids": "0dc24b471b217d2147f62314e1ed5dfe4cb984c134e140fba2c13510c94685fc",
    "dev505_ids": "36d4534d6e8e8a3fb539ca31ad202fbadfea3f0015fe3b1db50babb716369be6",
    "fit_ids": "d4cd748b863f54ab59723da250d59076f069129619ade2a26ea7303bf702bb0d",
    "plan_proof": "5ed107b20d220ebe2f54a4239689230fc2e051fe83bba898ccff8068edfe31d6",
    "audit_summary": "a8f0d7a3741dac3ce5073732f87390e1b7ee3250f5daa668ffaa8b2f6dcbd95c",
    "audit_dishes": "76fb5af01453848975a6751e35ebe6e54f8a2ccace58d7e3811a3e6a7cff226d",
    "audit_report": "e75f579aa1a27baf2b25573f2f70fd9be00cc099af44563adbcf83265db4d563",
    "train_meta": "cb549db6864d12da1d9fc404e9c425e0ee9dc2a270ba63e86f8de94c05e576c9",
    "status": "9f67a1fda4fa60e7fa0c727a5962c61468a8e177c3dfcbab289456bb122c24e9",
    "rgb_train_ids": "efd34a682b5483b01e6bd0a7c1267efec28440a8194985f9d3817d8ebc8169bd",
}
DEVELOPMENT_505 = {"energy_mae_kcal": 64.2792, "mass_mae_g": 32.9908}
CLAIM = "质量稳赢可外推；能量开发集赢、外测未稳。"

COLOR_ORACLE = "#1f7a4d"
COLOR_HEAD = "#1d4e89"
COLOR_FOOD = "#c2410c"

PROTECTED_PARTS = frozenset(
    {
        "heldout_plan",
        "heldout_eval",
        "heldout_energy_audit",
        "food_id_density",
        "coverage_early_stop",
        "qphi_counterfactual",
        "food_id_semantic_audit",
        "audit",
        "mass_estimator",
        "weak_steps",
        "closed_loop",
    }
)

SOURCE_PINS: tuple[tuple[str, str], ...] = (
    (
        "vdr/tools/mass_estimator.py",
        '"energy_kcal",',
    ),
    (
        "vdr/tools/mass_estimator.py",
        '"energy_kcal": model_rgb["energy_kcal"]',
    ),
    (
        "vdr/tools/mass_estimator.py",
        '"mass_g": model_rgb["mass_g"]',
    ),
    (
        "vdr/tools/mass_estimator.py",
        "mean_y = mean_z * std.reshape(1, -1) + mean.reshape(1, -1)",
    ),
    (
        "scripts/heldout_eval.py",
        'energy = finite((estimate.get("nutrients") or {}).get("energy_kcal"))',
    ),
    (
        "scripts/heldout_eval.py",
        'mass = finite(estimate.get("mass_g"))',
    ),
    (
        "scripts/heldout_eval.py",
        "estimate = estimate_mass(overhead_rgb(dish_id))",
    ),
    (
        "vdr/controller/on_policy.py",
        "nutri = _scaled(self.live_code, new_portion, self.lookup_fn, self.catalog_path)",
    ),
    (
        "vdr/controller/on_policy.py",
        'head = None if not isinstance(nutrients, Mapping) else nutrients.get("energy_kcal")',
    ),
    (
        "vdr/controller/on_policy.py",
        "dish_energy_kcal=head_f",
    ),
    (
        "vdr/weak/teacher.py",
        "scaled = scale_by_portion(per, portion_g)",
    ),
    (
        "vdr/tools/nutrition_lookup.py",
        "factor = float(portion_g) / 100.0",
    ),
)


class EnergyPathError(RuntimeError):
    """The frozen Cafe2 rows do not support this diagnostic."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def f4(value: float) -> str:
    return f"{value:.4f}"


def f3(value: float) -> str:
    return f"{value:.3f}"


def f2(value: float) -> str:
    return f"{value:.2f}"


def f1(value: float) -> str:
    return f"{value:.1f}"


def xml(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def assert_output(path: Path) -> None:
    resolved = path.resolve()
    allowed = OUT.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise EnergyPathError(f"refusing to write outside {OUT}: {path}")
    if PROTECTED_PARTS.intersection(resolved.parts):
        raise EnergyPathError(f"refusing to write into a frozen directory: {path}")


def assert_close(name: str, found: float, expected: float, tol: float = 1e-6) -> None:
    if abs(float(found) - float(expected)) > tol:
        raise EnergyPathError(f"{name} is {found}, frozen value is {expected}")


def require_pin(path: Path, needle: str) -> None:
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        raise EnergyPathError(f"{path.relative_to(ROOT)} no longer contains the scoring pin {needle!r}")


def assert_sources() -> None:
    for relative, needle in SOURCE_PINS:
        require_pin(ROOT / relative, needle)
    mass_source = (ROOT / "vdr/tools/mass_estimator.py").read_text(encoding="utf-8")
    target_block = mass_source.split("TARGET_NAMES: tuple[str, ...] = (", 1)
    if len(target_block) != 2:
        raise EnergyPathError("mass estimator no longer declares TARGET_NAMES")
    body = target_block[1].split(")", 1)[0]
    names = [part.strip().strip('",') for part in body.splitlines() if part.strip().startswith('"')]
    if names[:2] != ["mass_g", "energy_kcal"]:
        raise EnergyPathError(f"calorie head is no longer the second independent target: {names}")


def bootstrap_mean(values: np.ndarray, *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0:
        raise EnergyPathError("bootstrap needs a non-empty vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, sample.size, size=(BOOTSTRAP_DRAWS, sample.size))
    means = sample[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(sample.size),
        "mean": float(sample.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "fraction_mean_positive": float(np.mean(means > 0.0)),
        "fraction_mean_negative": float(np.mean(means < 0.0)),
        "draws": BOOTSTRAP_DRAWS,
        "seed": seed,
        "interval": "percentile",
        "unit": "plate resampled with replacement",
    }


def error_split(energy: np.ndarray, mass: np.ndarray, label_energy: np.ndarray, label_mass: np.ndarray) -> dict[str, np.ndarray]:
    """Signed calorie error = predicted kcal/g × gram error + label grams × kcal/g error."""
    if np.any(mass <= 0.0) or np.any(label_mass <= 0.0):
        raise EnergyPathError("density split needs positive predicted and label mass")
    density = energy / mass
    label_density = label_energy / label_mass
    gram = density * (mass - label_mass)
    dens = label_mass * (density - label_density)
    signed = energy - label_energy
    if float(np.max(np.abs((gram + dens) - signed))) > 1e-6:
        raise EnergyPathError("energy identity failed")
    return {
        "density": density,
        "label_density": label_density,
        "gram": gram,
        "density_term": dens,
        "signed": signed,
        "cancellation": np.abs(gram) + np.abs(dens) - np.abs(signed),
        "opposite": (mass - label_mass) * (density - label_density) < 0.0,
    }


def _index(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        dish_id = str(row["dish_id"])
        if dish_id in out:
            raise EnergyPathError(f"duplicate {key} row for {dish_id}")
        out[dish_id] = dict(row)
    return out


def group_steps(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dish_id"]), []).append(dict(row))
    for steps in grouped.values():
        steps.sort(key=lambda step: int(step["step_i"]))
    return grouped


def commit_step_index(steps: Sequence[Mapping[str, Any]]) -> int:
    """Last finite dish total at or before STOP/ABSTAIN. Same reader as #63 / #65."""
    actions = [str(step.get("action_t")) for step in steps]
    decision = next((index for index, action in enumerate(actions) if action in {"STOP", "ABSTAIN"}), len(actions) - 1)
    source: int | None = None
    for step in steps[: decision + 1]:
        dish = step.get("nutri_dish") if isinstance(step.get("nutri_dish"), Mapping) else {}
        if finite(dish.get("energy_kcal")) is None:
            continue
        source = int(step["step_i"])
    if source is None:
        raise EnergyPathError("trajectory has no finite dish total")
    return source


def index_calorie_heads(traces: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Dish-level energy head copied onto each mass write. Not the FNDDS product."""
    indexed: dict[str, dict[str, Any]] = {}
    for dish_id, steps in traces.items():
        source = commit_step_index(steps)
        heads: list[float] = []
        dish_masses: list[float] = []
        portions: list[float] = []
        backbones: list[str] = []
        for step in steps:
            if int(step["step_i"]) >= source:
                break
            if step.get("action_t") != "RE_PORTION":
                continue
            obs = step.get("obs_t") if isinstance(step.get("obs_t"), Mapping) else {}
            if obs.get("tool_failed") or obs.get("portion_src") != "mass_estimator":
                continue
            model = obs.get("nutrients_model") if isinstance(obs.get("nutrients_model"), Mapping) else None
            refs = obs.get("nutrient_refs") if isinstance(obs.get("nutrient_refs"), Mapping) else None
            rgb = None if refs is None else refs.get("model_rgb")
            if not isinstance(model, Mapping) or not isinstance(rgb, Mapping):
                raise EnergyPathError(f"{dish_id} mass write has no calorie head")
            if set(model) != {"energy_kcal", "protein_g", "carb_g", "fat_g"}:
                raise EnergyPathError(f"{dish_id} calorie head carries unexpected fields {sorted(model)}")
            head = finite(model.get("energy_kcal"))
            rgb_energy = finite(rgb.get("energy_kcal"))
            rgb_mass = finite(rgb.get("mass_g"))
            obs_mass = finite(obs.get("mass_g"))
            portion = finite(obs.get("portion_g"))
            if None in {head, rgb_energy, rgb_mass, obs_mass, portion}:
                raise EnergyPathError(f"{dish_id} mass write is not finite")
            if abs(float(head) - float(rgb_energy)) > 0.0 or abs(float(rgb_mass) - float(obs_mass)) > 0.0:
                raise EnergyPathError(f"{dish_id} model_rgb disagrees with the stored head")
            heads.append(float(head))
            dish_masses.append(float(obs_mass))
            portions.append(float(portion))
            backbones.append(str(obs.get("mass_backbone")))
        if not heads:
            raise EnergyPathError(f"{dish_id} has no mass-estimator write before the commit")
        indexed[dish_id] = {
            "calorie_head_kcal": heads[0],
            "head_span_kcal": max(heads) - min(heads),
            "n_writes": len(heads),
            "dish_mass_g": dish_masses[0],
            "dish_mass_span_g": max(dish_masses) - min(dish_masses),
            "portion_sum_g": float(sum(portions)),
            "backbones": backbones,
        }
    return indexed


def cafe2_label_check(scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Re-read cafe2. The dish-level energy column is unused. The target is the ingredient sum."""
    from vdr.data.n5k import parse_dish_metadata_line

    if not CAFE2.is_file():
        raise EnergyPathError(f"missing cafe2 metadata: {CAFE2}")
    n_rows = 0
    column_zero = 0
    max_energy_gap = 0.0
    max_mass_gap = 0.0
    seen: set[str] = set()
    with CAFE2.open(encoding="utf-8") as handle:
        for line in handle:
            dish = parse_dish_metadata_line(line)
            if dish is None:
                continue
            n_rows += 1
            seen.add(dish.dish_id)
            if dish.energy_kcal == 0.0:
                column_zero += 1
            if dish.dish_id not in scores:
                continue
            ingredient_energy = float(sum(item.energy_kcal for item in dish.ingredients))
            score = scores[dish.dish_id]
            max_energy_gap = max(max_energy_gap, abs(ingredient_energy - float(score["label_energy_kcal"])))
            max_mass_gap = max(max_mass_gap, abs(float(dish.mass_g) - float(score["label_mass_g"])))
            if finite(dish.energy_kcal) != 0.0:
                raise EnergyPathError(f"{dish.dish_id} dish-level energy column is {dish.energy_kcal}")
    if n_rows != 238 or column_zero != n_rows:
        raise EnergyPathError(f"cafe2 dish-level energy is not 0 on every parsed row ({column_zero}/{n_rows})")
    missing = sorted(set(scores) - seen)
    if missing:
        raise EnergyPathError(f"{missing[0]} is missing from cafe2 metadata")
    if max_energy_gap > 1e-6 or max_mass_gap > 1e-6:
        raise EnergyPathError(
            f"score labels differ from cafe2 ingredient sums (energy {max_energy_gap}, mass {max_mass_gap})"
        )
    return {
        "n_cafe2_rows": n_rows,
        "n_dish_energy_column_zero": column_zero,
        "heldout_label_matches_ingredient_sum": True,
        "max_abs_score_minus_ingredient_kcal": max_energy_gap,
        "max_abs_score_minus_dish_mass_g": max_mass_gap,
        "energy_target": "sum of ingredient calories; the dish-level energy column is 0 on every cafe2 row",
        "mass_target": "dish-level gram total",
    }


def load_frozen() -> dict[str, Any]:
    status_hash = sha256_file(STATUS)
    required = {
        "scores": EVAL / "scores.jsonl",
        "rollout": EVAL / "rollout.jsonl",
        "direct": EVAL / "direct_vlm.jsonl",
        "mass": EVAL / "mass_only.jsonl",
        "eval_summary": EVAL / "summary.json",
        "heldout_ids": PLAN / "heldout_dish_ids.txt",
        "dev505_ids": PLAN / "dev505_dish_ids.txt",
        "fit_ids": PLAN / "fit_dish_ids.txt",
        "plan_proof": PLAN / "proof.json",
        "audit_summary": AUDIT / "summary.json",
        "audit_dishes": AUDIT / "dishes.jsonl",
        "audit_report": AUDIT / "report.md",
        "train_meta": TRAIN_META,
        "status": STATUS,
        "rgb_train_ids": RGB_TRAIN_IDS,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise EnergyPathError("missing frozen input: " + ", ".join(missing))
    hashes = {name: sha256_file(path) for name, path in required.items()}
    for name, expected in INPUT_SHA256.items():
        if hashes[name] != expected:
            raise EnergyPathError(f"{name} hash changed; refusing to diagnose a different freeze")
    if status_hash != INPUT_SHA256["status"]:
        raise EnergyPathError("docs/STATUS.md is not the #67 text this diagnostic was written against")

    heldout_ids = read_ids(required["heldout_ids"])
    dev_ids = set(read_ids(required["dev505_ids"]))
    fit_ids = set(read_ids(required["fit_ids"]))
    rgb_train = set(read_ids(required["rgb_train_ids"]))
    if len(heldout_ids) != N_PAIRED or len(set(heldout_ids)) != N_PAIRED:
        raise EnergyPathError(f"held-out list has {len(heldout_ids)} ids")
    if set(heldout_ids) & dev_ids:
        raise EnergyPathError("held-out overlaps the development 505")
    if set(heldout_ids) & fit_ids:
        raise EnergyPathError("held-out overlaps the mass-training fit set")
    if set(heldout_ids) & rgb_train:
        raise EnergyPathError("held-out overlaps rgb_train")

    proof = json.loads(required["plan_proof"].read_text(encoding="utf-8"))
    intersections = proof.get("intersections_with_heldout") or {}
    for key in ("cafe1", "dev505", "fit", "rgb_test", "rgb_train", "rgb_train_internal_holdout"):
        if intersections.get(key) != 0:
            raise EnergyPathError(f"frozen plan proof records a {key} overlap")
    if proof.get("fit_intersection_heldout_empty") is not True:
        raise EnergyPathError("frozen plan proof does not record an empty fit intersection")

    scores = _index(load_jsonl(required["scores"]), "scores")
    mass_rows = _index(load_jsonl(required["mass"]), "mass")
    direct_rows = _index(load_jsonl(required["direct"]), "direct")
    if set(scores) != set(heldout_ids) or set(mass_rows) != set(heldout_ids):
        raise EnergyPathError("scores and Mass-only rows are not the frozen 228")
    eval_summary = json.loads(required["eval_summary"].read_text(encoding="utf-8"))
    audit_summary = json.loads(required["audit_summary"].read_text(encoding="utf-8"))
    train_meta = json.loads(required["train_meta"].read_text(encoding="utf-8"))
    if int(eval_summary["n_paired"]) != N_PAIRED:
        raise EnergyPathError("frozen eval summary is not n=228")
    if int(eval_summary["density_align_applied_dishes"]) != N_SWAP:
        raise EnergyPathError("frozen eval summary is not the 106-swap table")
    if int((eval_summary.get("hollow_re_portion") or {}).get("hollow", -1)) != 0:
        raise EnergyPathError("frozen eval summary does not record 0 hollow RE_PORTION writes")
    if train_meta.get("backbone") != "vit_b_16":
        raise EnergyPathError("frozen mass checkpoint backbone is not vit_b_16")
    if train_meta.get("targets") != ["mass_g", "energy_kcal", "protein_g", "carb_g", "fat_g"]:
        raise EnergyPathError(f"frozen mass checkpoint targets changed: {train_meta.get('targets')}")
    if train_meta.get("rgb_only") is not True or train_meta.get("depth_used") is not False:
        raise EnergyPathError("frozen mass checkpoint is not RGB-only")
    if train_meta.get("uses_rgb_test_gt") is not False or train_meta.get("holdout_used_for_selection") is not False:
        raise EnergyPathError("frozen mass checkpoint meta says a held-out slice was used")
    if int(train_meta.get("n_fit") or -1) != 2498:
        raise EnergyPathError("frozen mass checkpoint fit count is not 2498")

    traces = group_steps(load_jsonl(required["rollout"]))
    if set(traces) != set(heldout_ids):
        raise EnergyPathError("rollout dishes are not the frozen 228")
    heads = index_calorie_heads(traces)
    labels = cafe2_label_check(scores)
    audit_labels = audit_summary["labels"]
    if int(audit_labels["n_cafe2_rows"]) != 238 or int(audit_labels["n_dish_energy_column_zero"]) != 238:
        raise EnergyPathError("frozen #65 label audit is not the 238-zero cafe2 column")
    if audit_summary.get("direct_vlm_placement_max_memory") != "8GiB" or int(audit_summary.get("n_direct_rows_at_8GiB") or 0) != N_PAIRED:
        raise EnergyPathError("frozen #65 direct-vlm placement record is not 8GiB on 228/228")
    placements = {str((row.get("placement_max_memory"))) for row in direct_rows.values()}
    if placements != {"8GiB"}:
        raise EnergyPathError(f"direct_vlm.jsonl placement is {placements}")

    return {
        "heldout_ids": heldout_ids,
        "scores": scores,
        "mass_rows": mass_rows,
        "heads": heads,
        "labels": labels,
        "train_meta": train_meta,
        "eval_summary": eval_summary,
        "audit_summary": audit_summary,
        "hashes": hashes,
        "status_hash": status_hash,
        "proof_intersections": intersections,
    }


def build_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dish_id in payload["heldout_ids"]:
        score = payload["scores"][dish_id]
        mass = payload["mass_rows"][dish_id]
        head = payload["heads"][dish_id]
        allowed = {"dish_id", "energy_kcal", "mass_g", "backbone", "tool", "rgb_only", "depth_used"}
        if set(mass) != allowed:
            raise EnergyPathError(f"{dish_id} Mass-only row has fields {sorted(set(mass) - allowed)}")
        if mass.get("tool") != "estimate_mass" or mass.get("backbone") != "vit_b_16":
            raise EnergyPathError(f"{dish_id} Mass-only row is not the vit_b_16 estimate_mass head")
        if mass.get("rgb_only") is not True or mass.get("depth_used") is not False:
            raise EnergyPathError(f"{dish_id} Mass-only row is not RGB-only")
        energy = float(mass["energy_kcal"])
        grams = float(mass["mass_g"])
        if abs(float(score["mass_only_energy_kcal"]) - energy) > 0.0:
            raise EnergyPathError(f"{dish_id} score energy differs from mass_only.jsonl")
        if abs(float(score["mass_only_mass_g"]) - grams) > 0.0:
            raise EnergyPathError(f"{dish_id} score mass differs from mass_only.jsonl")
        if abs(float(head["calorie_head_kcal"]) - energy) > HEAD_MATCH_KCAL:
            raise EnergyPathError(f"{dish_id} rollout calorie head differs from Mass-only energy")
        if float(head["head_span_kcal"]) > HEAD_MATCH_KCAL or float(head["dish_mass_span_g"]) > HEAD_MATCH_KCAL:
            raise EnergyPathError(f"{dish_id} calorie head changes across mass writes")
        if abs(float(head["dish_mass_g"]) - grams) > 0.0:
            raise EnergyPathError(f"{dish_id} rollout dish mass differs from Mass-only mass")
        if any(backbone != "vit_b_16" for backbone in head["backbones"]):
            raise EnergyPathError(f"{dish_id} mass write backbone is not vit_b_16")
        label_energy = float(score["label_energy_kcal"])
        label_mass = float(score["label_mass_g"])
        if label_mass <= 0.0 or grams <= 0.0 or float(score["ours_mass_g"]) <= 0.0:
            raise EnergyPathError(f"{dish_id} has a non-positive mass")
        ours_energy = float(score["ours_energy_kcal"])
        ours_mass = float(score["ours_mass_g"])
        label_density = label_energy / label_mass
        head_density = energy / grams
        records.append(
            {
                "dish_id": dish_id,
                "plate_complete": bool(score["plate_complete"]),
                "n_hypotheses": int(score["n_hypotheses"]),
                "n_mass_writes": int(score["n_mass_writes"]),
                "n_writes_observed": int(head["n_writes"]),
                "decision_action": str(score["decision_action"]),
                "stop_unwritten": bool(score["stop_unwritten"]),
                "density_align_applied": bool(score["density_align_applied"]),
                "label_energy_kcal": label_energy,
                "label_mass_g": label_mass,
                "label_density_kcal_per_g": label_density,
                "ours_energy_kcal": ours_energy,
                "ours_mass_g": ours_mass,
                "mass_only_energy_kcal": energy,
                "mass_only_mass_g": grams,
                "calorie_head_kcal": float(head["calorie_head_kcal"]),
                "head_span_kcal": float(head["head_span_kcal"]),
                "portion_sum_g": float(head["portion_sum_g"]),
                "oracle_our_grams_kcal": label_density * ours_mass,
                "oracle_head_grams_kcal": label_density * grams,
                "head_density_times_our_grams_kcal": head_density * ours_mass,
                "direct_energy_kcal": float(score["direct_energy_kcal"]),
                "direct_mass_g": float(score["direct_mass_g"]),
            }
        )
    if sum(row["n_writes_observed"] for row in records) != N_WRITES:
        raise EnergyPathError("mass-write count is not the frozen 575")
    if sum(1 for row in records if row["density_align_applied"]) != N_SWAP:
        raise EnergyPathError("score rows do not contain the frozen 106 swaps")
    return records


def _column(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in records], dtype=np.float64)


def _mask_mae(energy: np.ndarray, label: np.ndarray, mask: np.ndarray) -> float:
    return mae(energy[mask] - label[mask])


def slice_row(
    name: str,
    mask: np.ndarray,
    *,
    label_e: np.ndarray,
    label_m: np.ndarray,
    ours_e: np.ndarray,
    ours_m: np.ndarray,
    head_e: np.ndarray,
    head_m: np.ndarray,
    oracle_ours: np.ndarray,
    oracle_head: np.ndarray,
    head_density_on_ours: np.ndarray,
    food_density_mae: np.ndarray,
    head_density_mae: np.ndarray,
    benefit: np.ndarray,
) -> dict[str, Any]:
    n = int(mask.sum())
    if n == 0:
        raise EnergyPathError(f"slice {name} is empty")
    gap = np.abs(oracle_ours - label_e) - np.abs(head_e - label_e)
    return {
        "name": name,
        "n": n,
        "oracle_our_grams_mae_kcal": _mask_mae(oracle_ours, label_e, mask),
        "oracle_head_grams_mae_kcal": _mask_mae(oracle_head, label_e, mask),
        "mass_only_mae_kcal": _mask_mae(head_e, label_e, mask),
        "head_density_times_our_grams_mae_kcal": _mask_mae(head_density_on_ours, label_e, mask),
        "food_table_mae_kcal": _mask_mae(ours_e, label_e, mask),
        "food_table_mass_mae_g": _mask_mae(ours_m, label_m, mask),
        "mass_only_mass_mae_g": _mask_mae(head_m, label_m, mask),
        "food_table_density_mae_kcal_per_g": float(np.mean(food_density_mae[mask])),
        "head_density_mae_kcal_per_g": float(np.mean(head_density_mae[mask])),
        "mean_benefit_vs_true_density_at_head_grams_kcal": float(np.mean(benefit[mask])),
        "n_benefit_positive": int(np.sum(benefit[mask] > 1e-6)),
        "n_benefit_negative": int(np.sum(benefit[mask] < -1e-6)),
        "addition_to_98_minus_75_kcal": float(np.sum(gap[mask]) / N_PAIRED),
    }


def aggregate(records: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]) -> dict[str, Any]:
    label_e = _column(records, "label_energy_kcal")
    label_m = _column(records, "label_mass_g")
    ours_e = _column(records, "ours_energy_kcal")
    ours_m = _column(records, "ours_mass_g")
    head_e = _column(records, "mass_only_energy_kcal")
    head_m = _column(records, "mass_only_mass_g")
    direct_e = _column(records, "direct_energy_kcal")
    direct_m = _column(records, "direct_mass_g")
    oracle_ours = _column(records, "oracle_our_grams_kcal")
    oracle_head = _column(records, "oracle_head_grams_kcal")
    head_on_ours = _column(records, "head_density_times_our_grams_kcal")
    calorie_head = _column(records, "calorie_head_kcal")

    food = error_split(ours_e, ours_m, label_e, label_m)
    head = error_split(head_e, head_m, label_e, label_m)
    oracle = error_split(oracle_ours, ours_m, label_e, label_m)
    oracle_at_head = error_split(oracle_head, head_m, label_e, label_m)

    published = {
        "ours_energy_mae_kcal": mae(ours_e - label_e),
        "ours_mass_mae_g": mae(ours_m - label_m),
        "direct_energy_mae_kcal": mae(direct_e - label_e),
        "direct_mass_mae_g": mae(direct_m - label_m),
        "mass_only_energy_mae_kcal": mae(head_e - label_e),
        "mass_only_mass_mae_g": mae(head_m - label_m),
    }
    published["energy_minus_direct_kcal"] = published["ours_energy_mae_kcal"] - published["direct_energy_mae_kcal"]
    published["mass_minus_direct_g"] = published["ours_mass_mae_g"] - published["direct_mass_mae_g"]
    published["energy_minus_mass_only_kcal"] = published["ours_energy_mae_kcal"] - published["mass_only_energy_mae_kcal"]
    published["mass_minus_mass_only_g"] = published["ours_mass_mae_g"] - published["mass_only_mass_mae_g"]
    for key, expected in PUBLISHED.items():
        assert_close(key, published[key], expected, tol=1e-6)

    gram_gap = np.abs(ours_m - head_m)
    matched = gram_gap < GRAM_MATCH_G
    if int(matched.sum()) != N_MATCHED:
        raise EnergyPathError(f"gram match count is {int(matched.sum())}, expected {N_MATCHED}")
    complete = np.asarray([bool(row["plate_complete"]) for row in records], dtype=bool)
    if not np.array_equal(matched, complete):
        raise EnergyPathError("plate_complete is not the same set as the 0.001 g gram match")
    if np.any(np.asarray([row["decision_action"] for row in records])[~matched] != "RETRIEVE"):
        raise EnergyPathError("a gram-mismatch plate did not end on RETRIEVE")
    if np.any(np.asarray([bool(row["stop_unwritten"]) for row in records])):
        raise EnergyPathError("a frozen score row has stop_unwritten")

    product_gap = np.abs(head_e - head_on_ours)
    if float(np.max(product_gap[matched])) > 0.001:
        raise EnergyPathError("on matched plates the energy head is not implied density times the committed grams")
    if float(np.max(np.abs(head_e - calorie_head))) > 0.0:
        raise EnergyPathError("calorie head column drifted from Mass-only energy")
    if float(np.max(np.abs((head_e / head_m) * head_m - head_e))) > 1e-6:
        raise EnergyPathError("implied density times the mass head does not recover the energy head")

    mae_oracle_ours = mae(oracle_ours - label_e)
    mae_oracle_head = mae(oracle_head - label_e)
    mae_head = mae(head_e - label_e)
    mae_head_on_ours = mae(head_on_ours - label_e)
    mae_food = mae(ours_e - label_e)
    assert_close("oracle_our_grams", mae_oracle_ours, FROZEN_AUDIT["oracle_our_grams_kcal"], tol=1e-9)
    assert_close("oracle_head_grams", mae_oracle_head, FROZEN_AUDIT["oracle_head_grams_kcal"], tol=1e-9)
    assert_close("head density mae", float(np.mean(np.abs(head["density"] - head["label_density"]))), FROZEN_AUDIT["head_density_mae"])
    assert_close("food density mae", float(np.mean(np.abs(food["density"] - food["label_density"]))), FROZEN_AUDIT["food_table_density_mae"])
    assert_close("head cancellation", float(np.mean(head["cancellation"])), FROZEN_AUDIT["head_cancellation_kcal"])
    assert_close("food cancellation", float(np.mean(food["cancellation"])), FROZEN_AUDIT["food_table_cancellation_kcal"])
    assert_close("head |gram|", float(np.mean(np.abs(head["gram"]))), FROZEN_AUDIT["head_abs_gram_term_kcal"])
    assert_close("head |density term|", float(np.mean(np.abs(head["density_term"]))), FROZEN_AUDIT["head_abs_density_term_kcal"])
    assert_close("food |gram|", float(np.mean(np.abs(food["gram"]))), FROZEN_AUDIT["food_table_abs_gram_term_kcal"])
    assert_close("food |density term|", float(np.mean(np.abs(food["density_term"]))), FROZEN_AUDIT["food_table_abs_density_term_kcal"])
    assert_close("head opposite", float(np.mean(head["opposite"])), FROZEN_AUDIT["head_opposite_sign_rate"], tol=1e-12)
    assert_close("food opposite", float(np.mean(food["opposite"])), FROZEN_AUDIT["food_table_opposite_sign_rate"], tol=1e-12)
    if float(np.max(np.abs(oracle["density_term"]))) > 1e-9 or float(np.max(oracle["cancellation"])) > 1e-9:
        raise EnergyPathError("oracle × our grams has a nonzero density term")
    if float(np.max(np.abs(oracle_at_head["density_term"]))) > 1e-9:
        raise EnergyPathError("oracle × Mass-only grams has a nonzero density term")

    audit = payload["audit_summary"]
    assert_close(
        "audit oracle",
        mae_oracle_ours,
        float(audit["oracle"]["ours_label_density_kcal"]),
        tol=1e-9,
    )
    assert_close(
        "audit head channel",
        float(np.mean(head["cancellation"])),
        float(audit["channels"]["mass_only"]["mean_cancellation_kcal"]),
        tol=1e-9,
    )
    assert_close(
        "audit food channel",
        float(np.mean(food["cancellation"])),
        float(audit["channels"]["commit"]["mean_cancellation_kcal"]),
        tol=1e-9,
    )

    # Residual of each predictor against true density at that predictor's own grams.
    residual_head = head_e - oracle_head
    residual_food = ours_e - oracle_ours
    abs_oracle_head = np.abs(oracle_head - label_e)
    abs_head = np.abs(head_e - label_e)
    abs_oracle_ours = np.abs(oracle_ours - label_e)
    abs_food = np.abs(ours_e - label_e)
    cancel_vs_oracle_head = abs_oracle_head + np.abs(residual_head) - abs_head
    cancel_vs_oracle_food = abs_oracle_ours + np.abs(residual_food) - abs_food
    benefit = abs_oracle_head - abs_head
    excess_head = float(np.mean(cancel_vs_oracle_head) - np.mean(np.abs(residual_head)))
    excess_food = float(np.mean(cancel_vs_oracle_food) - np.mean(np.abs(residual_food)))
    gram_set_mae_change = mae_oracle_head - mae_oracle_ours
    density_swap_mae_change = mae_head - mae_oracle_head
    gap = mae_oracle_ours - mae_head
    if abs((mae_oracle_ours + gram_set_mae_change + density_swap_mae_change) - mae_head) > 1e-9:
        raise EnergyPathError("98 vs 75 partition does not add up")
    if abs(excess_head - (mae_oracle_head - mae_head)) > 1e-9:
        raise EnergyPathError("head excess cancellation does not equal the MAE drop at fixed grams")
    if abs(excess_food - (mae_oracle_ours - mae_food)) > 1e-9:
        raise EnergyPathError("food-table excess cancellation does not equal its MAE change")
    if abs(gap - (mae_oracle_ours - mae_head)) > 1e-12:
        raise EnergyPathError("gap identity failed")

    display = {
        "oracle_our_grams": f4(mae_oracle_ours),
        "mass_only": f4(mae_head),
        "food_table": f4(mae_food),
        "head_density": f3(float(np.mean(np.abs(head["density"] - head["label_density"])))),
        "food_density": f3(float(np.mean(np.abs(food["density"] - food["label_density"])))),
        "head_cancellation": f1(float(np.mean(head["cancellation"]))),
        "food_cancellation": f1(float(np.mean(food["cancellation"]))),
        "ours_energy": f4(published["ours_energy_mae_kcal"]),
        "ours_mass": f4(published["ours_mass_mae_g"]),
        "direct_energy": f4(published["direct_energy_mae_kcal"]),
        "direct_mass": f4(published["direct_mass_mae_g"]),
        "mass_only_mass": f4(published["mass_only_mass_mae_g"]),
        "energy_minus_mass_only": f"{published['energy_minus_mass_only_kcal']:+.2f}",
        "mass_minus_mass_only": f"{published['mass_minus_mass_only_g']:+.2f}",
        "energy_minus_direct": f"{published['energy_minus_direct_kcal']:+.2f}",
        "mass_minus_direct": f"{published['mass_minus_direct_g']:+.2f}",
    }
    expected_display = {
        "oracle_our_grams": "98.6201",
        "mass_only": "75.3870",
        "food_table": "116.8494",
        "head_density": "0.753",
        "food_density": "0.631",
        "head_cancellation": "77.8",
        "food_cancellation": "58.2",
        "ours_energy": "116.8494",
        "ours_mass": "50.9483",
        "direct_energy": "105.6157",
        "direct_mass": "73.7456",
        "mass_only_mass": "51.7107",
        "energy_minus_mass_only": "+41.46",
        "mass_minus_mass_only": "-0.76",
        "energy_minus_direct": "+11.23",
        "mass_minus_direct": "-22.80",
    }
    for key, expected in expected_display.items():
        if display[key] != expected:
            raise EnergyPathError(f"display {key} is {display[key]}, expected {expected}")

    food_abs_density_error = np.abs(food["density"] - food["label_density"])
    head_abs_density_error = np.abs(head["density"] - head["label_density"])
    head_closer = head_abs_density_error < food_abs_density_error
    food_closer = food_abs_density_error < head_abs_density_error
    if int(np.sum(head_closer)) + int(np.sum(food_closer)) != N_PAIRED:
        raise EnergyPathError("density-closer counts do not cover 228 plates")

    single = np.asarray([int(row["n_hypotheses"]) == 1 for row in records], dtype=bool)
    benefit_arr = benefit
    slices = [
        slice_row(
            name,
            mask,
            label_e=label_e,
            label_m=label_m,
            ours_e=ours_e,
            ours_m=ours_m,
            head_e=head_e,
            head_m=head_m,
            oracle_ours=oracle_ours,
            oracle_head=oracle_head,
            head_density_on_ours=head_on_ours,
            food_density_mae=food_abs_density_error,
            head_density_mae=head_abs_density_error,
            benefit=benefit_arr,
        )
        for name, mask in (
            ("all", np.ones(N_PAIRED, dtype=bool)),
            ("grams_match", matched),
            ("grams_differ", ~matched),
            ("single_component", single),
            ("multi_component", ~single),
            ("complete_multi", matched & ~single),
        )
    ]
    additions = sum(row["addition_to_98_minus_75_kcal"] for row in slices if row["name"] in {"grams_match", "grams_differ"})
    if abs(additions - gap) > 1e-9:
        raise EnergyPathError("matched and unmatched slices do not add up to the 98 vs 75 gap")

    gap_vector = abs_oracle_ours - abs_head
    bootstrap_gap = bootstrap_mean(gap_vector)
    bootstrap_benefit = bootstrap_mean(benefit_arr)

    def example(index: int, role: str) -> dict[str, Any]:
        row = records[int(index)]
        return {
            "role": role,
            "dish_id": row["dish_id"],
            "single_component": int(row["n_hypotheses"]) == 1,
            "grams_match": bool(matched[index]),
            "n_hypotheses": row["n_hypotheses"],
            "n_mass_writes": row["n_mass_writes"],
            "decision_action": row["decision_action"],
            "label_energy_kcal": row["label_energy_kcal"],
            "label_mass_g": row["label_mass_g"],
            "label_density_kcal_per_g": row["label_density_kcal_per_g"],
            "ours_energy_kcal": row["ours_energy_kcal"],
            "ours_mass_g": row["ours_mass_g"],
            "mass_only_energy_kcal": row["mass_only_energy_kcal"],
            "mass_only_mass_g": row["mass_only_mass_g"],
            "food_table_density_kcal_per_g": float(food["density"][index]),
            "head_density_kcal_per_g": float(head["density"][index]),
            "oracle_our_grams_kcal": float(oracle_ours[index]),
            "oracle_head_grams_kcal": float(oracle_head[index]),
            "abs_oracle_head_grams_kcal": float(abs_oracle_head[index]),
            "abs_mass_only_kcal": float(abs_head[index]),
            "abs_food_table_kcal": float(abs_food[index]),
            "density_residual_kcal": float(residual_head[index]),
            "benefit_kcal": float(benefit_arr[index]),
        }

    single_idx = np.flatnonzero(single)
    help_i = int(single_idx[np.argmax(benefit_arr[single])])
    hurt_i = int(single_idx[np.argmin(benefit_arr[single])])
    examples = [
        example(help_i, "single_component_largest_benefit"),
        example(hurt_i, "single_component_largest_harm"),
    ]

    per_plate = []
    for index, row in enumerate(records):
        per_plate.append(
            {
                "dish_id": row["dish_id"],
                "plate_complete": row["plate_complete"],
                "grams_match_within_0_001g": bool(matched[index]),
                "single_component": bool(single[index]),
                "n_hypotheses": row["n_hypotheses"],
                "n_mass_writes": row["n_mass_writes"],
                "decision_action": row["decision_action"],
                "label_energy_kcal": row["label_energy_kcal"],
                "label_mass_g": row["label_mass_g"],
                "label_density_kcal_per_g": row["label_density_kcal_per_g"],
                "ours_energy_kcal": row["ours_energy_kcal"],
                "ours_mass_g": row["ours_mass_g"],
                "food_table_density_kcal_per_g": float(food["density"][index]),
                "mass_only_energy_kcal": row["mass_only_energy_kcal"],
                "mass_only_mass_g": row["mass_only_mass_g"],
                "calorie_head_kcal": row["calorie_head_kcal"],
                "head_implied_density_kcal_per_g": float(head["density"][index]),
                "oracle_our_grams_kcal": float(oracle_ours[index]),
                "oracle_head_grams_kcal": float(oracle_head[index]),
                "head_density_times_our_grams_kcal": float(head_on_ours[index]),
                "density_residual_at_head_grams_kcal": float(residual_head[index]),
                "abs_oracle_our_grams_kcal": float(abs_oracle_ours[index]),
                "abs_oracle_head_grams_kcal": float(abs_oracle_head[index]),
                "abs_mass_only_kcal": float(abs_head[index]),
                "abs_food_table_kcal": float(abs_food[index]),
                "benefit_vs_true_density_at_head_grams_kcal": float(benefit_arr[index]),
                "gram_error_head_g": float(head_m[index] - label_m[index]),
                "gram_error_ours_g": float(ours_m[index] - label_m[index]),
                "density_error_head_kcal_per_g": float(head["density"][index] - head["label_density"][index]),
                "density_error_food_table_kcal_per_g": float(food["density"][index] - food["label_density"][index]),
                "head_gram_term_kcal": float(head["gram"][index]),
                "head_density_term_kcal": float(head["density_term"][index]),
                "head_cancellation_kcal": float(head["cancellation"][index]),
                "food_table_cancellation_kcal": float(food["cancellation"][index]),
                "opposite_signs_head": bool(head["opposite"][index]),
                "head_span_across_writes_kcal": row["head_span_kcal"],
            }
        )

    mismatched = [
        {
            "dish_id": records[index]["dish_id"],
            "n_hypotheses": records[index]["n_hypotheses"],
            "n_mass_writes": records[index]["n_mass_writes"],
            "decision_action": records[index]["decision_action"],
            "ours_mass_g": float(ours_m[index]),
            "mass_only_mass_g": float(head_m[index]),
            "abs_gram_gap_g": float(gram_gap[index]),
        }
        for index in np.flatnonzero(~matched)
    ]

    paths = [
        {
            "name": "mass_only_energy_head",
            "food_id": "none",
            "density_source": "dish-level energy head / mass head, accounting identity only",
            "grams": "mass head",
            "energy_mae_kcal": mae_head,
            "mass_mae_g": mae(head_m - label_m),
            "density_mae_kcal_per_g": float(np.mean(head_abs_density_error)),
            "scored_commit": True,
        },
        {
            "name": "oracle_density_x_our_grams",
            "food_id": "label density, not a food code",
            "density_source": "ingredient-sum kcal / dish grams",
            "grams": "food-table commit grams",
            "energy_mae_kcal": mae_oracle_ours,
            "mass_mae_g": mae(ours_m - label_m),
            "density_mae_kcal_per_g": 0.0,
            "scored_commit": False,
        },
        {
            "name": "oracle_density_x_mass_only_grams",
            "food_id": "label density, not a food code",
            "density_source": "ingredient-sum kcal / dish grams",
            "grams": "mass head",
            "energy_mae_kcal": mae_oracle_head,
            "mass_mae_g": mae(head_m - label_m),
            "density_mae_kcal_per_g": 0.0,
            "scored_commit": False,
        },
        {
            "name": "head_implied_density_x_our_grams",
            "food_id": "none",
            "density_source": "energy head / mass head",
            "grams": "food-table commit grams",
            "energy_mae_kcal": mae_head_on_ours,
            "mass_mae_g": mae(ours_m - label_m),
            "density_mae_kcal_per_g": float(np.mean(np.abs(head_on_ours / ours_m - label_e / label_m))),
            "scored_commit": False,
        },
        {
            "name": "food_table_commit",
            "food_id": "FNDDS code from retrieval; same-head swap on 106 plates",
            "density_source": "FNDDS kcal per 100 g",
            "grams": "mass-estimator component grams, summed",
            "energy_mae_kcal": mae_food,
            "mass_mae_g": mae(ours_m - label_m),
            "density_mae_kcal_per_g": float(np.mean(food_abs_density_error)),
            "scored_commit": True,
        },
    ]

    return {
        "protocol": "cafe2_massonly_energy_path",
        "diagnostic_only": True,
        "n_paired": N_PAIRED,
        "claim": CLAIM,
        "verdict": "greater_signed_cancellation",
        "density_more_accurate_than_oracle": False,
        "density_more_accurate_than_food_table": False,
        "mass_only_uses_food_id": False,
        "development_505_not_a_row": DEVELOPMENT_505,
        "published": published,
        "display": display,
        "labels": payload["labels"],
        "paths": paths,
        "gram_match": {
            "threshold_g": GRAM_MATCH_G,
            "n_match": int(matched.sum()),
            "n_differ": int((~matched).sum()),
            "max_abs_gap_on_match_g": float(np.max(gram_gap[matched])),
            "max_abs_gap_on_differ_g": float(np.max(gram_gap[~matched])),
            "max_abs_head_minus_implied_times_commit_grams_on_match_kcal": float(np.max(product_gap[matched])),
            "portion_sum_max_abs_gap_on_match_g": float(
                np.max(np.abs(_column(records, "portion_sum_g")[matched] - head_m[matched]))
            ),
            "mismatched_plates": mismatched,
        },
        "path_checks": {
            "n_mass_writes": N_WRITES,
            "max_head_span_kcal": float(max(row["head_span_kcal"] for row in records)),
            "max_abs_calorie_head_minus_mass_only_kcal": float(np.max(np.abs(calorie_head - head_e))),
            "backbone": "vit_b_16",
            "tool": "estimate_mass",
            "mass_only_fields": [
                "dish_id",
                "energy_kcal",
                "mass_g",
                "backbone",
                "tool",
                "rgb_only",
                "depth_used",
            ],
            "nutrients_model_fields": ["energy_kcal", "protein_g", "carb_g", "fat_g"],
            "targets": ["mass_g", "energy_kcal", "protein_g", "carb_g", "fat_g"],
            "checkpoint_n_fit": 2498,
            "heldout_intersection_rgb_train": 0,
            "heldout_intersection_fit": 0,
            "heldout_intersection_dev505": 0,
            "n_swap_from_scores": N_SWAP,
            "mean_abs_food_table_minus_head_kcal": float(np.mean(np.abs(ours_e - head_e))),
            "direct_vlm_placement_max_memory": "8GiB",
            "n_direct_rows_at_8GiB": N_PAIRED,
            "hollow_re_portion": 0,
        },
        "channels": {
            "oracle_our_grams": {
                "energy_mae_kcal": mae_oracle_ours,
                "density_mae_kcal_per_g": 0.0,
                "mean_abs_gram_term_kcal": float(np.mean(np.abs(oracle["gram"]))),
                "mean_abs_density_term_kcal": float(np.mean(np.abs(oracle["density_term"]))),
                "mean_cancellation_kcal": float(np.mean(oracle["cancellation"])),
                "mean_signed_energy_kcal": float(np.mean(oracle["signed"])),
                "opposite_sign_rate": float(np.mean(oracle["opposite"])),
            },
            "food_table": {
                "energy_mae_kcal": mae_food,
                "density_mae_kcal_per_g": float(np.mean(food_abs_density_error)),
                "mean_abs_gram_term_kcal": float(np.mean(np.abs(food["gram"]))),
                "mean_abs_density_term_kcal": float(np.mean(np.abs(food["density_term"]))),
                "mean_cancellation_kcal": float(np.mean(food["cancellation"])),
                "mean_signed_energy_kcal": float(np.mean(food["signed"])),
                "mean_signed_gram_term_kcal": float(np.mean(food["gram"])),
                "mean_signed_density_term_kcal": float(np.mean(food["density_term"])),
                "opposite_sign_rate": float(np.mean(food["opposite"])),
                "mean_signed_density_error_kcal_per_g": float(np.mean(food["density"] - food["label_density"])),
                "median_abs_density_error_kcal_per_g": float(np.median(food_abs_density_error)),
            },
            "mass_only": {
                "energy_mae_kcal": mae_head,
                "density_mae_kcal_per_g": float(np.mean(head_abs_density_error)),
                "mean_abs_gram_term_kcal": float(np.mean(np.abs(head["gram"]))),
                "mean_abs_density_term_kcal": float(np.mean(np.abs(head["density_term"]))),
                "mean_cancellation_kcal": float(np.mean(head["cancellation"])),
                "mean_signed_energy_kcal": float(np.mean(head["signed"])),
                "mean_signed_gram_term_kcal": float(np.mean(head["gram"])),
                "mean_signed_density_term_kcal": float(np.mean(head["density_term"])),
                "opposite_sign_rate": float(np.mean(head["opposite"])),
                "mean_signed_density_error_kcal_per_g": float(np.mean(head["density"] - head["label_density"])),
                "mean_signed_gram_error_g": float(np.mean(head_m - label_m)),
                "median_abs_density_error_kcal_per_g": float(np.median(head_abs_density_error)),
                "mean_abs_implied_density": float(np.mean(np.abs(head["density"]))),
                "mean_abs_label_density": float(np.mean(np.abs(head["label_density"]))),
                "pearson_gram_error_vs_density_error": float(np.corrcoef(head_m - label_m, head["density"] - head["label_density"])[0, 1]),
            },
        },
        "gap_98_vs_75": {
            "oracle_our_grams_mae_kcal": mae_oracle_ours,
            "oracle_head_grams_mae_kcal": mae_oracle_head,
            "mass_only_mae_kcal": mae_head,
            "head_density_times_our_grams_mae_kcal": mae_head_on_ours,
            "gap_kcal": gap,
            "mae_change_our_grams_to_head_grams_at_true_density_kcal": gram_set_mae_change,
            "mae_change_true_density_to_energy_head_at_head_grams_kcal": density_swap_mae_change,
            "mean_abs_density_residual_head_kcal": float(np.mean(np.abs(residual_head))),
            "mean_cancellation_vs_oracle_at_head_grams_kcal": float(np.mean(cancel_vs_oracle_head)),
            "excess_cancellation_head_kcal": excess_head,
            "mean_abs_density_residual_food_kcal": float(np.mean(np.abs(residual_food))),
            "mean_cancellation_vs_oracle_at_our_grams_kcal": float(np.mean(cancel_vs_oracle_food)),
            "excess_cancellation_food_kcal": excess_food,
            "n_benefit_positive": int(np.sum(benefit_arr > 1e-6)),
            "n_benefit_negative": int(np.sum(benefit_arr < -1e-6)),
            "mean_benefit_opposite_signs_kcal": float(np.mean(benefit_arr[head["opposite"]])),
            "n_opposite_signs": int(np.sum(head["opposite"])),
            "mean_benefit_same_sign_kcal": float(np.mean(benefit_arr[~head["opposite"]])),
            "n_same_sign": int(np.sum(~head["opposite"])),
            "n_head_density_closer_than_food_table": int(np.sum(head_closer)),
            "n_food_table_density_closer_than_head": int(np.sum(food_closer)),
            "n_head_density_error_below_1e-4": int(np.sum(head_abs_density_error < 1e-4)),
            "single_component_share_of_gap": float(
                next(row["addition_to_98_minus_75_kcal"] for row in slices if row["name"] == "single_component") / gap
            ),
        },
        "bootstrap": {
            "oracle_our_grams_abs_minus_mass_only_abs": bootstrap_gap,
            "benefit_vs_true_density_at_head_grams": bootstrap_benefit,
        },
        "slices": slices,
        "examples": examples,
        "per_plate": per_plate,
        "input_sha256": payload["hashes"],
        "status_sha256": payload["status_hash"],
        "method_selected": False,
        "early_stopped": False,
        "hyperparameters_tuned": False,
        "tau_retuned": False,
        "codes_or_portions_edited": False,
        "mass_retrained": False,
        "q_retrained": False,
        "rl": False,
        "gates_4_to_7_started": False,
        "p1_unlocked": False,
        "status_edited": False,
    }


def _example_lines(row: Mapping[str, Any]) -> list[str]:
    kind = "one component" if row["single_component"] else "multi-component"
    match = "grams match" if row["grams_match"] else "grams differ"
    return [
        f"- `{row['dish_id']}` ({kind}, {match}, {row['role'].replace('_', ' ')}).",
        (
            f"  Label {row['label_mass_g']:.1f} g / {row['label_energy_kcal']:.1f} kcal "
            f"({row['label_density_kcal_per_g']:.3f} kcal/g)."
        ),
        (
            f"  Committed grams {row['ours_mass_g']:.1f} g. Mass-only grams {row['mass_only_mass_g']:.1f} g. "
            f"Food-table commit {row['ours_energy_kcal']:.1f} kcal "
            f"(absolute error {row['abs_food_table_kcal']:.1f})."
        ),
        (
            f"  Oracle × Mass-only grams {row['oracle_head_grams_kcal']:.1f} kcal "
            f"(absolute error {row['abs_oracle_head_grams_kcal']:.1f}). "
            f"Energy head {row['mass_only_energy_kcal']:.1f} kcal "
            f"(absolute error {row['abs_mass_only_kcal']:.1f})."
        ),
        (
            f"  Implied head density {row['head_density_kcal_per_g']:.3f} kcal/g. "
            f"Food-table density {row['food_table_density_kcal_per_g']:.3f} kcal/g. "
            f"Residual {row['density_residual_kcal']:+.1f} kcal. Benefit {row['benefit_kcal']:+.1f} kcal."
        ),
    ]


def render_report(summary: Mapping[str, Any]) -> str:
    gap = summary["gap_98_vs_75"]
    channels = summary["channels"]
    match = summary["gram_match"]
    head = channels["mass_only"]
    food = channels["food_table"]
    oracle = channels["oracle_our_grams"]
    published = summary["published"]
    boot = summary["bootstrap"]["oracle_our_grams_abs_minus_mass_only_abs"]
    slices = {row["name"]: row for row in summary["slices"]}
    single = slices["single_component"]
    multi = slices["multi_component"]
    lines = [
        "# Mass-only energy path",
        "",
        f"**{summary['claim']}**",
        "",
        (
            f"Oracle label density × our grams has energy MAE **{f4(gap['oracle_our_grams_mae_kcal'])} kcal**. "
            f"Mass-only has energy MAE **{f4(gap['mass_only_mae_kcal'])} kcal**. "
            f"The **{f4(gap['gap_kcal'])} kcal** gap is the excess signed cancellation at the Mass-only grams, "
            f"**{gap['excess_cancellation_head_kcal']:.4f} kcal**, minus the **{gap['mae_change_our_grams_to_head_grams_at_true_density_kcal']:.4f} kcal** "
            "rise that comes from scoring the true density on the Mass-only grams instead of our grams. "
            f"Unweighted kcal/g MAE is **0** for the oracle density, **{f3(food['density_mae_kcal_per_g'])}** for the food-table commit, "
            f"and **{f3(head['density_mae_kcal_per_g'])}** for the calorie head. "
            f"The #65 channel cancellation is **0**, **{f1(food['mean_cancellation_kcal'])}**, and **{f1(head['mean_cancellation_kcal'])} kcal** "
            "on those same three rows. Mass-only energy has no food code."
        ),
        "",
        "The development 505 energy MAE 64.2792 and mass MAE 32.9908 are not rows of this table.",
        "",
        "## Answers",
        "",
        (
            "1. This directory is only the Mass-only energy-path diagnostic on the frozen Cafe2 list, n=228. "
            "No method, hyperparameter, early stop, or model selection was changed."
        ),
        (
            "2. Mass-only energy is the frozen mass estimator's dish-level energy head: "
            "`estimate_mass(...).nutrients.energy_kcal`, backbone `vit_b_16`, targets "
            "`mass_g`, `energy_kcal`, `protein_g`, `carb_g`, `fat_g` in that order. "
            "The scorer does not multiply a food density by grams and does not read a food-id. "
            "The food-table commit is the selected FNDDS kcal/100 g times component grams, summed. "
            "Oracle × our grams is the label kcal/g times those committed grams. "
            f"On **{match['n_match']}** / {summary['n_paired']} plates the mass head matches the committed grams "
            f"within 0.001 g (measured maximum **{match['max_abs_gap_on_match_g']:.4f} g**). "
            "On those plates the energy head equals implied kcal/g times the committed grams within "
            f"**{match['max_abs_head_minus_implied_times_commit_grams_on_match_kcal']:.4f} kcal**."
        ),
        (
            f"3. **{f4(gap['oracle_our_grams_mae_kcal'])}** versus **{f4(gap['mass_only_mae_kcal'])}** is greater signed cancellation. "
            f"The calorie head's unweighted kcal/g MAE **{f3(head['density_mae_kcal_per_g'])}** is larger than the food table's "
            f"**{f3(food['density_mae_kcal_per_g'])}** and larger than the oracle's **0**. "
            f"#65's cancellation figures stand: **{f1(head['mean_cancellation_kcal'])}** kcal for the head and "
            f"**{f1(food['mean_cancellation_kcal'])}** kcal for the food table. "
            "The head is closer than the food table on "
            f"**{gap['n_head_density_closer_than_food_table']}** plates and farther on "
            f"**{gap['n_food_table_density_closer_than_head']}**. "
            f"Replacing the head's density with the label density, at the head's own grams, would raise energy MAE from "
            f"**{f4(gap['mass_only_mae_kcal'])}** to **{f4(gap['oracle_head_grams_mae_kcal'])}**."
        ),
        (
            "4. After this diagnostic, gate ④, gate ⑤, gate ⑥, gate ⑦, and P1 stay locked. "
            "Mass, Q, and τ were not retrained. `docs/STATUS.md` was not edited."
        ),
        "",
        "## How the three energies are scored",
        "",
        (
            "Cafe2 was re-read: **238** rows, dish-level energy column **0** on every row. "
            "The energy target on the 228 is the sum of ingredient calories, and it matches the stored score labels. "
            "Mass is the dish-level gram total. Labels score residuals. "
            "No threshold, code, or gram was chosen from them. No model was trained."
        ),
        "",
        (
            "The frozen plan proof records an empty intersection of these 228 ids with rgb_train, with the fit set "
            "the mass estimator trained on (overhead RGB subset of fit, n=2498), and with the development 505. "
            "This run rechecked those three id lists. The intersection is empty. "
            "The checkpoint meta says the weights are the last train epoch, not a hold-out selection, and that rgb_test labels were not used."
        ),
        "",
        "### Mass-only",
        "",
        (
            "`scripts/heldout_eval.py` calls `estimate_mass` on the overhead RGB and stores "
            "`nutrients.energy_kcal` and `mass_g`. It does not pass food candidates. "
            "The network emits five means in standardized space and denormalizes each one on its own. "
            "The tool then clips each mean at 0 and rounds to 4 decimals. "
            "`nutrients.energy_kcal` is that energy mean. It is not an FNDDS lookup and it is not a food code."
        ),
        "",
        (
            f"The frozen `mass_only.jsonl` rows confirm the path. All **{summary['n_paired']}** have backbone `vit_b_16`, "
            "tool `estimate_mass`, `rgb_only` true, and `depth_used` false. "
            "The fields are dish id, energy, mass, backbone, tool, and the two RGB flags. There is no food-id. "
            f"All **{summary['path_checks']['n_mass_writes']}** mass writes before the scored commit store "
            "`obs_t.nutrients_model.energy_kcal` equal to that energy "
            f"(maximum absolute gap **{summary['path_checks']['max_abs_calorie_head_minus_mass_only_kcal']:.0f} kcal**) "
            "and the same value on every write of a plate "
            f"(maximum span **{summary['path_checks']['max_head_span_kcal']:.0f} kcal**). "
            "`nutrient_refs.model_rgb` repeats the five head outputs. Its mass equals the Mass-only mass. "
            "The nutrient object on the write has energy, protein, carb, and fat only."
        ),
        "",
        (
            "The published Mass-only calorie is this energy head, **75.3870 kcal** MAE, beside the mass head, "
            f"**{f4(published['mass_only_mass_mae_g'])} g** MAE. "
            "An implied kcal/g can be defined afterwards as energy divided by mass. "
            "Multiplying that ratio by the mass head returns the energy head. "
            "That identity is how the #65 channel split accounts the head. It is not a step in the scorer. "
            f"On the **{match['n_match']}** plates whose committed grams already match, implied kcal/g times the committed grams "
            f"differs from the energy head by at most **{match['max_abs_head_minus_implied_times_commit_grams_on_match_kcal']:.4f} kcal**. "
            f"The other **{match['n_differ']}** plates stopped on RETRIEVE with a partial gram total. "
            "Mass-only still scores the full-dish head on those plates. "
            f"Their largest gram gap is **{match['max_abs_gap_on_differ_g']:.1f} g**."
        ),
        "",
        "### Food-table commit",
        "",
        (
            "RE_PORTION takes the component grams from the same mass estimator, then looks up the bound FNDDS code and "
            "scales kcal per 100 g by grams / 100. The dish total is the sum of those component vectors. "
            "On the write that finishes a multi-component plate, a same-head swap may replace codes so that sum moves "
            "toward the energy head. The head is the target of the swap. The committed calorie remains the food-table sum. "
            f"Scores record the swap on **{summary['path_checks']['n_swap_from_scores']}** plates. "
            f"Mean |food-table energy − energy head| is **{summary['path_checks']['mean_abs_food_table_minus_head_kcal']:.2f} kcal**. "
            "#65 already checked that the food-table sum matches the commit within 0.0004 kcal on every plate. "
            "That check was not repeated as a second audit, and those files were not rewritten."
        ),
        "",
        "### Oracle × our grams",
        "",
        (
            "Label kcal/g times the committed grams has energy MAE "
            f"**{f4(oracle['energy_mae_kcal'])}**. "
            "The density term is 0 and cancellation is 0, so this MAE is the mean absolute gram term at the true density. "
            "The same label density times the Mass-only grams has energy MAE "
            f"**{f4(gap['oracle_head_grams_mae_kcal'])}**. "
            "Both totals are arithmetic. They were not commits."
        ),
        "",
        "## 98.6201 versus 75.3870",
        "",
        (
            f"Moving from our grams to the Mass-only grams, while keeping the true density, changes energy MAE from "
            f"**{f4(gap['oracle_our_grams_mae_kcal'])}** to **{f4(gap['oracle_head_grams_mae_kcal'])}** "
            f"({gap['mae_change_our_grams_to_head_grams_at_true_density_kcal']:+.4f} kcal). "
            "The true-density score is higher on the Mass-only grams, so the whole Mass-only advantage is the next step."
        ),
        "",
        (
            f"At the Mass-only grams, replacing the true density with the energy head changes energy MAE from "
            f"**{f4(gap['oracle_head_grams_mae_kcal'])}** to **{f4(gap['mass_only_mae_kcal'])}** "
            f"({gap['mae_change_true_density_to_energy_head_at_head_grams_kcal']:+.4f} kcal). "
            f"{f4(gap['oracle_our_grams_mae_kcal'])} + {gap['mae_change_our_grams_to_head_grams_at_true_density_kcal']:.4f} "
            f"− {abs(gap['mae_change_true_density_to_energy_head_at_head_grams_kcal']):.4f} = "
            f"**{f4(gap['mass_only_mae_kcal'])}**. "
            f"The oracle-minus-head gap is **{gap['gap_kcal']:.4f} kcal** = "
            f"{gap['excess_cancellation_head_kcal']:.4f} − {gap['mae_change_our_grams_to_head_grams_at_true_density_kcal']:.4f}."
        ),
        "",
        (
            "The second change is a density residual, `energy head − true density × Mass-only grams`. "
            f"Its mean absolute size is **{gap['mean_abs_density_residual_head_kcal']:.4f} kcal**. "
            "Cancellation of that residual against the true-density gram error averages "
            f"**{gap['mean_cancellation_vs_oracle_at_head_grams_kcal']:.4f} kcal**. "
            "Cancellation here is |true-density gram error| + |density residual| − |energy-head error|. "
            f"The excess of cancellation over the residual is **{gap['excess_cancellation_head_kcal']:.4f} kcal**, "
            "which is the whole MAE drop at those grams. "
            f"The residual lowers absolute error on **{gap['n_benefit_positive']}** plates and raises it on "
            f"**{gap['n_benefit_negative']}**."
        ),
        "",
        (
            f"Gram error and implied-density error have opposite signs on **{gap['n_opposite_signs']}** / {summary['n_paired']} plates "
            f"({100 * head['opposite_sign_rate']:.1f}%). "
            f"Mean benefit on those plates is **{gap['mean_benefit_opposite_signs_kcal']:+.2f} kcal**. "
            f"On the other **{gap['n_same_sign']}** plates the mean benefit is **{gap['mean_benefit_same_sign_kcal']:+.2f} kcal**. "
            f"Pearson correlation of gram error and implied-density error is **{head['pearson_gram_error_vs_density_error']:.3f}**. "
            f"Mean signed gram error is **{head['mean_signed_gram_error_g']:+.2f} g**. "
            f"Mean signed implied-density error is **{head['mean_signed_density_error_kcal_per_g']:+.3f} kcal/g**. "
            f"Mean |implied density| is **{head['mean_abs_implied_density']:.3f} kcal/g** against label "
            f"**{head['mean_abs_label_density']:.3f} kcal/g**."
        ),
        "",
        (
            "The same residual test on the food table goes the other way. "
            f"Mean |food-table energy − true density × our grams| is **{gap['mean_abs_density_residual_food_kcal']:.4f} kcal**. "
            f"Cancellation against that true-density gram error averages **{gap['mean_cancellation_vs_oracle_at_our_grams_kcal']:.4f} kcal**. "
            f"The excess is **{gap['excess_cancellation_food_kcal']:+.4f} kcal**, and the food-table energy MAE is "
            f"**{f4(food['energy_mae_kcal'])}**, above the oracle's **{f4(oracle['energy_mae_kcal'])}**."
        ),
        "",
        (
            "A 10,000-draw bootstrap that resamples plates (numpy Generator seed 0) puts "
            f"98.6201 − 75.3870 at **[{boot['ci95_low']:+.2f}, {boot['ci95_high']:+.2f}] kcal**. "
            f"{100 * boot['fraction_mean_positive']:.2f}% of draws keep a positive mean. "
            "This interval describes the frozen list. It was not used to choose a method."
        ),
        "",
        "![Energy MAE beside unweighted density error](fig_density_vs_mae.svg)",
        "",
        "![Split of 98.6201 minus 75.3870](fig_gap_split.svg)",
        "",
        "| Path | Food-id | Grams | Energy MAE (kcal) | kcal/g MAE |",
        "|---|---|---|---:|---:|",
        (
            f"| Oracle density × our grams | label density | commit | {f4(gap['oracle_our_grams_mae_kcal'])} | 0.000 |"
        ),
        (
            f"| Oracle density × Mass-only grams | label density | mass head | {f4(gap['oracle_head_grams_mae_kcal'])} | 0.000 |"
        ),
        (
            f"| Energy head × our grams | none | commit | {f4(gap['head_density_times_our_grams_mae_kcal'])} | "
            f"{f3(next(row['density_mae_kcal_per_g'] for row in summary['paths'] if row['name'] == 'head_implied_density_x_our_grams'))} |"
        ),
        (
            f"| Mass-only energy head | none | mass head | {f4(gap['mass_only_mae_kcal'])} | {f3(head['density_mae_kcal_per_g'])} |"
        ),
        (
            f"| Food-table commit | FNDDS, swap on 106 | commit | {f4(food['energy_mae_kcal'])} | {f3(food['density_mae_kcal_per_g'])} |"
        ),
        "",
        (
            "The row `Energy head × our grams` is the implied ratio times the committed grams. "
            f"It equals the published energy head on the {match['n_match']} matched plates. "
            f"On the full 228 its MAE is **{f4(gap['head_density_times_our_grams_mae_kcal'])}**, "
            "because the 6 partial plates use a different gram total. "
            "The published Mass-only figure remains the energy head, **75.3870**."
        ),
        "",
        "| | kcal/g MAE | Plates with the smaller kcal/g error | Median \\|kcal/g error\\| | Mean signed kcal/g error | Mean \\|density term\\| |",
        "|---|---:|---:|---:|---:|---:|",
        "| Oracle density | 0.000 |  | 0.000 | 0.000 | 0.0 |",
        (
            f"| Food-table commit | {f3(food['density_mae_kcal_per_g'])} | "
            f"{gap['n_food_table_density_closer_than_head']} closer than the head | "
            f"{food['median_abs_density_error_kcal_per_g']:.3f} | "
            f"{food['mean_signed_density_error_kcal_per_g']:+.3f} | "
            f"{food['mean_abs_density_term_kcal']:.1f} |"
        ),
        (
            f"| Mass-only energy head | {f3(head['density_mae_kcal_per_g'])} | "
            f"{gap['n_head_density_closer_than_food_table']} closer than the food table | "
            f"{head['median_abs_density_error_kcal_per_g']:.3f} | "
            f"{head['mean_signed_density_error_kcal_per_g']:+.3f} | "
            f"{head['mean_abs_density_term_kcal']:.1f} |"
        ),
        "",
        (
            "No plate has a head density error below 0.0001 kcal/g. "
            f"On the {match['n_match']} matched plates the food-table mass MAE is "
            f"**{slices['grams_match']['food_table_mass_mae_g']:.4f} g** and the mass-head mass MAE is "
            f"**{slices['grams_match']['mass_only_mass_mae_g']:.4f} g**. "
            f"The gram totals differ by at most **{match['max_abs_gap_on_match_g']:.4f} g** there. "
            f"The full-set mass gap versus Mass-only, **{published['mass_minus_mass_only_g']:+.4f} g**, is the "
            f"{match['n_differ']} RETRIEVE plates."
        ),
        "",
        "## Where the 23.2332 kcal sits",
        "",
        "The last column is that slice's addition to the 228-plate mean of oracle×our-grams absolute error minus Mass-only absolute error.",
        "",
        "| Slice | n | Oracle × our grams | Oracle × head grams | Mass-only | Food table | Head kcal/g MAE | Food kcal/g MAE | Addition to 23.2332 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all", "grams_match", "grams_differ", "single_component", "multi_component", "complete_multi"):
        row = slices[name]
        lines.append(
            f"| {name} | {row['n']} | {row['oracle_our_grams_mae_kcal']:.2f} | "
            f"{row['oracle_head_grams_mae_kcal']:.2f} | {row['mass_only_mae_kcal']:.2f} | "
            f"{row['food_table_mae_kcal']:.2f} | {row['head_density_mae_kcal_per_g']:.3f} | "
            f"{row['food_table_density_mae_kcal_per_g']:.3f} | {row['addition_to_98_minus_75_kcal']:+.2f} |"
        )
    lines.extend(
        [
            "",
            (
                f"The 58 one-component plates add **{single['addition_to_98_minus_75_kcal']:+.2f} kcal** "
                f"({100 * gap['single_component_share_of_gap']:.1f}% of the mean gap). "
                f"Their oracle × our grams MAE is **{single['oracle_our_grams_mae_kcal']:.2f}** and their Mass-only MAE is "
                f"**{single['mass_only_mae_kcal']:.2f}**. "
                f"Head kcal/g MAE on that slice is **{single['head_density_mae_kcal_per_g']:.3f}**, "
                f"above the food table's **{single['food_table_density_mae_kcal_per_g']:.3f}**. "
                f"The food-table energy MAE there is **{single['food_table_mae_kcal']:.2f}**. "
                "Every one-component plate is in the 222 whose grams match, so the swap cannot fire and the commit is one FNDDS code "
                "times the shared grams. The head still has no code. Its lower calorie MAE on this slice is the residual cancelling "
                "the gram error, with a worse kcal/g error than that single code."
            ),
            "",
            (
                f"Multi-component plates add **{multi['addition_to_98_minus_75_kcal']:+.2f} kcal**. "
                f"The 6 RETRIEVE plates add **{slices['grams_differ']['addition_to_98_minus_75_kcal']:+.2f} kcal**: "
                "there the full-dish energy head is scored against a partial gram total, and Mass-only absolute error is higher "
                "than oracle × our grams."
            ),
            "",
            "## Channel split already frozen in #65",
            "",
            (
                "The #65 split writes signed calorie error as predicted kcal/g × gram error + label grams × kcal/g error. "
                "Cancellation in that split is |gram term| + |kcal/g term| − |calorie error|. "
                "The gram term uses the predictor's own kcal/g, so a low implied density shrinks it. "
                "That shrinkage is the same density error. The numbers below were recomputed from the frozen scores and match #65."
            ),
            "",
            "| | Energy MAE | kcal/g MAE | Mean \\|gram term\\| | Mean \\|kcal/g term\\| | Mean cancellation | Mean signed calorie error |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| Oracle × our grams | {oracle['energy_mae_kcal']:.4f} | 0.000 | "
                f"{oracle['mean_abs_gram_term_kcal']:.1f} | {oracle['mean_abs_density_term_kcal']:.1f} | "
                f"{oracle['mean_cancellation_kcal']:.1f} | {oracle['mean_signed_energy_kcal']:+.1f} |"
            ),
            (
                f"| Food-table commit | {food['energy_mae_kcal']:.4f} | {food['density_mae_kcal_per_g']:.3f} | "
                f"{food['mean_abs_gram_term_kcal']:.1f} | {food['mean_abs_density_term_kcal']:.1f} | "
                f"{food['mean_cancellation_kcal']:.1f} | {food['mean_signed_energy_kcal']:+.1f} |"
            ),
            (
                f"| Mass-only energy head | {head['energy_mae_kcal']:.4f} | {head['density_mae_kcal_per_g']:.3f} | "
                f"{head['mean_abs_gram_term_kcal']:.1f} | {head['mean_abs_density_term_kcal']:.1f} | "
                f"{head['mean_cancellation_kcal']:.1f} | {head['mean_signed_energy_kcal']:+.1f} |"
            ),
            "",
            (
            f"The channel cancellation above is the frozen #65 pair, head **{f1(head['mean_cancellation_kcal'])}** versus food table "
            f"**{f1(food['mean_cancellation_kcal'])}**. "
            "The residual cancellation earlier in this note uses a different pair: the true-density gram error, plus the "
            f"kcal gap between the energy head and true density × Mass-only grams. That pair cancels "
            f"**{gap['mean_cancellation_vs_oracle_at_head_grams_kcal']:.1f} kcal** and exceeds the residual by "
            f"**{gap['excess_cancellation_head_kcal']:.1f} kcal**. "
            "In both splits the head's kcal/g error is the larger one, and the signed errors oppose."
        ),
            "",
            "## Readings",
            "",
            (
                "These two one-component plates are the largest drop and the largest rise in absolute error when the energy head "
                "replaces the true density at the Mass-only grams. They illustrate the residual. "
                "They were not used to choose a code or a gram."
            ),
            "",
            *_example_lines(summary["examples"][0]),
            *_example_lines(summary["examples"][1]),
            "",
            "## What stays in place",
            "",
            (
                f"The #63 table stands. Coverage-constrained energy MAE **{f4(published['ours_energy_mae_kcal'])} kcal**, "
                f"mass MAE **{f4(published['ours_mass_mae_g'])} g**. "
                f"Direct-VLM **{f4(published['direct_energy_mae_kcal'])}** / **{f4(published['direct_mass_mae_g'])}**. "
                f"Mass-only **{f4(published['mass_only_energy_mae_kcal'])}** / **{f4(published['mass_only_mass_mae_g'])}**. "
                f"Energy versus Direct-VLM **{published['energy_minus_direct_kcal']:+.2f} kcal**, "
                f"versus Mass-only **{published['energy_minus_mass_only_kcal']:+.2f} kcal**. "
                f"Mass versus Direct-VLM **{published['mass_minus_direct_g']:+.2f} g**, "
                f"versus Mass-only **{published['mass_minus_mass_only_g']:+.2f} g**. "
                "Coverage 0.9737 (222/228)."
            ),
            "",
            (
                f"**{summary['claim']}** "
                "This note does not start a mass retrain, a Q retrain, a τ refit, a 505 food-id rewrite, a Cafe2 retest, "
                "gate ④ real tool cost, gate ⑤, gate ⑥, or gate ⑦. There is no RL. "
                "Direct-VLM in the frozen rows used `placement_max_memory=8GiB` on 228 / 228 plates. Those rows were not rerun. "
                "The frozen summary records 0 hollow RE_PORTION writes. "
                "`docs/STATUS.md` was not edited. Later gates stay locked."
            ),
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/heldout_massonly_energy_path.py",
            "```",
            "",
            (
                "Reads the frozen Cafe2 id list, `results/heldout_eval/scores.jsonl`, `mass_only.jsonl`, `rollout.jsonl`, "
                "`direct_vlm.jsonl`, and `summary.json`; `results/heldout_energy_audit/summary.json` for the #65 locks; "
                "the mass-estimator `train_meta.json`; and cafe2 metadata for the label check. "
                "Input sha256 prefixes: "
                + ", ".join(f"{name} `{digest[:16]}`" for name, digest in summary["input_sha256"].items())
                + "."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_plain(summary: Mapping[str, Any]) -> str:
    text = render_report(summary).replace("**", "")
    cleaned: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            cleaned.append(line.lstrip("# ").upper() if line.startswith("# ") else line.lstrip("#"))
        elif line.startswith("!["):
            cleaned.append(line)
        else:
            cleaned.append(line)
    return "\n".join(cleaned).rstrip() + "\n"


def _bar_chart(path: Path, title: str, subtitle: str, groups: Sequence[tuple[str, Sequence[tuple[str, str, float]]]]) -> None:
    assert_output(path)
    width = 880
    height = 420
    left = 64
    top = 72
    bottom = 78
    plot_h = height - top - bottom
    group_w = (width - left - 40) / max(1, len(groups))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="64" y="32" font-family="DejaVu Sans, sans-serif" font-size="16" fill="#111827">{xml(title)}</text>',
        f'<text x="64" y="52" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#4b5563">{xml(subtitle)}</text>',
    ]
    legend: list[tuple[str, str]] = []
    for group_index, (label, series) in enumerate(groups):
        cx = left + (group_index + 0.5) * group_w
        peak = max((value for _name, _color, value in series), default=1.0)
        peak = max(peak, 1e-9)
        bar_w = min(36, (group_w - 24) / max(1, len(series)))
        for series_index, (name, color, value) in enumerate(series):
            offset = (series_index - (len(series) - 1) / 2) * (bar_w + 6)
            bar_h = 0.0 if peak == 0 else value / peak * (plot_h - 24)
            y = top + plot_h - bar_h
            parts.append(
                f'<rect x="{cx + offset - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>'
            )
            shown = f"{value:.1f}" if value >= 10.0 else f"{value:.3f}"
            parts.append(
                f'<text x="{cx + offset:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="11" fill="#111827">{shown}</text>'
            )
            if (name, color) not in legend:
                legend.append((name, color))
        parts.append(
            f'<text x="{cx:.1f}" y="{height - bottom + 28}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">{xml(label)}</text>'
        )
    legend_x = 64
    for name, color in legend:
        parts.append(f'<rect x="{legend_x}" y="{height - 24}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 18}" y="{height - 14}" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">{xml(name)}</text>'
        )
        legend_x += 180
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def gap_svg(path: Path, summary: Mapping[str, Any]) -> None:
    assert_output(path)
    gap = summary["gap_98_vs_75"]
    # Contributions to (oracle × our grams MAE − Mass-only MAE).
    values = [
        ("Gram set\nat true density", -gap["mae_change_our_grams_to_head_grams_at_true_density_kcal"], COLOR_ORACLE),
        ("Head replaces\ntrue density", -gap["mae_change_true_density_to_energy_head_at_head_grams_kcal"], COLOR_HEAD),
        ("Net gap", gap["gap_kcal"], COLOR_FOOD),
    ]
    width = 720
    height = 420
    left = 180
    right = 40
    top = 64
    bottom = 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    peak = max(abs(value) for _label, value, _color in values) * 1.25
    zero_y = top + plot_h / 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="28" font-family="DejaVu Sans, sans-serif" font-size="16" fill="#111827">How 98.6201 becomes 75.3870</text>',
        '<text x="48" y="48" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#4b5563">Contribution to oracle×our-grams MAE minus Mass-only MAE, kcal. Positive means Mass-only is lower.</text>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="1"/>',
    ]
    slot = plot_w / len(values)
    bar_w = 72
    for index, (label, value, color) in enumerate(values):
        cx = left + (index + 0.5) * slot
        bar_h = abs(value) / peak * (plot_h / 2)
        y = zero_y - bar_h if value >= 0 else zero_y
        parts.append(f'<rect x="{cx - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}"/>')
        text_y = y - 8 if value >= 0 else y + bar_h + 16
        parts.append(
            f'<text x="{cx:.1f}" y="{text_y:.1f}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">{value:+.2f}</text>'
        )
        for line_index, line in enumerate(label.split("\n")):
            parts.append(
                f'<text x="{cx:.1f}" y="{height - 28 + line_index * 14:.1f}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">{xml(line)}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_outputs(summary: Mapping[str, Any]) -> None:
    assert_output(OUT / "report.md")
    OUT.mkdir(parents=True, exist_ok=True)
    report = render_report(summary)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    (OUT / "report.txt").write_text(render_plain(summary), encoding="utf-8")
    public = {key: value for key, value in summary.items() if key != "per_plate"}
    (OUT / "summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "dishes.jsonl").open("w", encoding="utf-8") as handle:
        for row in summary["per_plate"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "paths.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "food_id",
                "density_source",
                "grams",
                "energy_mae_kcal",
                "mass_mae_g",
                "density_mae_kcal_per_g",
                "scored_commit",
            ],
        )
        writer.writeheader()
        for row in summary["paths"]:
            writer.writerow(
                {
                    **row,
                    "energy_mae_kcal": f"{row['energy_mae_kcal']:.10f}",
                    "mass_mae_g": f"{row['mass_mae_g']:.10f}",
                    "density_mae_kcal_per_g": f"{row['density_mae_kcal_per_g']:.10f}",
                }
            )
    with (OUT / "slices.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summary["slices"][0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["slices"]:
            writer.writerow({key: (f"{value:.10f}" if isinstance(value, float) else value) for key, value in row.items()})
    gap = summary["gap_98_vs_75"]
    head = summary["channels"]["mass_only"]
    food = summary["channels"]["food_table"]
    _bar_chart(
        OUT / "fig_density_vs_mae.svg",
        "Energy error and density error on the same 228 plates",
        "Each group has its own scale. Oracle density error is 0.",
        [
            (
                "Energy MAE (kcal)",
                [
                    ("Oracle × our grams", COLOR_ORACLE, gap["oracle_our_grams_mae_kcal"]),
                    ("Mass-only energy head", COLOR_HEAD, gap["mass_only_mae_kcal"]),
                    ("Food-table commit", COLOR_FOOD, food["energy_mae_kcal"]),
                ],
            ),
            (
                "kcal/g MAE",
                [
                    ("Oracle × our grams", COLOR_ORACLE, 0.0),
                    ("Mass-only energy head", COLOR_HEAD, head["density_mae_kcal_per_g"]),
                    ("Food-table commit", COLOR_FOOD, food["density_mae_kcal_per_g"]),
                ],
            ),
        ],
    )
    gap_svg(OUT / "fig_gap_split.svg", summary)


def self_check() -> None:
    energy = np.asarray([220.0, 80.0], dtype=np.float64)
    mass = np.asarray([200.0, 50.0], dtype=np.float64)
    label_e = np.asarray([200.0, 100.0], dtype=np.float64)
    label_m = np.asarray([100.0, 100.0], dtype=np.float64)
    parts = error_split(energy, mass, label_e, label_m)
    if abs(float(parts["gram"][0]) - 110.0) > 1e-9 or abs(float(parts["density_term"][0]) + 90.0) > 1e-9:
        raise AssertionError(parts["gram"][0], parts["density_term"][0])
    if abs(float(parts["cancellation"][0]) - 180.0) > 1e-9:
        raise AssertionError(parts["cancellation"][0])
    oracle_energy = (label_e / label_m) * mass
    oracle = error_split(oracle_energy, mass, label_e, label_m)
    if float(np.max(np.abs(oracle["density_term"]))) > 1e-9 or float(np.max(oracle["cancellation"])) > 1e-9:
        raise AssertionError("oracle density residual should be zero")
    # An opposing residual can lower absolute error while increasing density error.
    grams = np.asarray([150.0], dtype=np.float64)
    labels_e = np.asarray([100.0], dtype=np.float64)
    labels_m = np.asarray([100.0], dtype=np.float64)
    true_density_total = (labels_e / labels_m) * grams
    head_total = np.asarray([110.0], dtype=np.float64)
    if mae(true_density_total - labels_e) <= mae(head_total - labels_e):
        raise AssertionError("toy residual should beat the true density on absolute error")
    head_parts = error_split(head_total, grams, labels_e, labels_m)
    if float(head_parts["density"][0]) >= 1.0:
        raise AssertionError("toy head density should be less accurate than the label density 1")
    try:
        error_split(np.asarray([10.0]), np.asarray([0.0]), np.asarray([10.0]), np.asarray([5.0]))
    except EnergyPathError:
        pass
    else:
        raise AssertionError("zero mass should fail the density split")
    try:
        assert_output(ROOT / "results" / "heldout_eval" / "summary.json")
    except EnergyPathError:
        pass
    else:
        raise AssertionError("protected directory was accepted")
    try:
        assert_output(ROOT / "results" / "audit" / "note.md")
    except EnergyPathError:
        pass
    else:
        raise AssertionError("audit directory was accepted")
    sample = bootstrap_mean(np.asarray([1.0, -1.0, 1.0, -1.0]), seed=0)
    if sample["n"] != 4 or sample["draws"] != BOOTSTRAP_DRAWS or sample["seed"] != 0:
        raise AssertionError(sample)
    print("heldout mass-only energy-path self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Mass-only energy-path diagnostic on the frozen Cafe2 held-out")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    assert_sources()
    payload = load_frozen()
    records = build_records(payload)
    summary = aggregate(records, payload)
    write_outputs(summary)
    if sha256_file(STATUS) != payload["status_hash"]:
        raise EnergyPathError("docs/STATUS.md changed during the diagnostic")
    gap = summary["gap_98_vs_75"]
    print(
        f"n={summary['n_paired']} oracle×our-grams {gap['oracle_our_grams_mae_kcal']:.4f} "
        f"mass-only {gap['mass_only_mae_kcal']:.4f} "
        f"gap {gap['gap_kcal']:+.4f} "
        f"excess cancellation {gap['excess_cancellation_head_kcal']:+.4f}"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
