#!/usr/bin/env python3
"""Diagnostic energy audit of the frozen Cafe2 held-out.

Reads ``results/heldout_plan`` and ``results/heldout_eval``. Does not roll out
a model, does not train, and does not write those directories, ``results/audit``,
or ``docs/STATUS.md``.

The three cuts match the #48 error audit:

- hold the bound food codes fixed and compare the 100 g grid with the mass write
- score oracle label density times our grams
- turn the stored #51 same-head density swap off and on, with the 106 swapped plates separate

Direct-VLM and Mass-only stay the frozen #63 rows.

    python scripts/heldout_energy_audit.py --self-check
    python scripts/heldout_energy_audit.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "results" / "heldout_plan"
EVAL = ROOT / "results" / "heldout_eval"
OUT = ROOT / "results" / "heldout_energy_audit"
CAFE2 = ROOT / "data" / "nutrition5k" / "metadata" / "dish_metadata_cafe2.csv"
STATUS = ROOT / "docs" / "STATUS.md"

N_PAIRED = 228
N_SWAP = 106
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
ACCOUNT_TOLERANCE_KCAL = 0.05
ACCOUNT_TOLERANCE_G = 0.05

# Frozen #63 table. The audit recomputes these from scores.jsonl and refuses to continue
# if the file has drifted. Development-set 64.2792 / 32.9908 are intentionally absent.
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

COLOR_COMMIT = "#c2410c"
COLOR_MASS = "#1d4e89"
COLOR_DIRECT = "#4b5563"
COLOR_ORACLE = "#1f7a4d"
COLOR_PRIOR = "#a16207"
COLOR_BOUND = "#9a3412"

PROTECTED_PARTS = frozenset(
    {
        "heldout_plan",
        "heldout_eval",
        "food_id_density",
        "coverage_early_stop",
        "qphi_counterfactual",
        "food_id_semantic_audit",
        "audit",
        "p0d",
        "qphi",
        "mass_estimator",
        "weak_steps",
        "closed_loop",
    }
)


class EnergyAuditError(RuntimeError):
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
        raise EnergyAuditError(f"refusing to write outside {OUT}: {path}")
    if PROTECTED_PARTS.intersection(resolved.parts):
        raise EnergyAuditError(f"refusing to write into a frozen directory: {path}")


def channels(energy: float, mass: float, label_energy: float, label_mass: float) -> dict[str, float]:
    """Signed calorie error = predicted kcal/g × gram error + label grams × kcal/g error."""
    if mass <= 0.0 or label_mass <= 0.0:
        raise EnergyAuditError("density split needs positive predicted and label mass")
    density = energy / mass
    label_density = label_energy / label_mass
    mass_channel = density * (mass - label_mass)
    density_channel = label_mass * (density - label_density)
    signed = energy - label_energy
    if abs((mass_channel + density_channel) - signed) > 1e-6:
        raise EnergyAuditError("energy identity failed")
    return {
        "density_kcal_per_g": density,
        "label_density_kcal_per_g": label_density,
        "mass_channel_kcal": mass_channel,
        "density_channel_kcal": density_channel,
        "signed_energy_kcal": signed,
        "signed_mass_g": mass - label_mass,
        "cancellation_kcal": abs(mass_channel) + abs(density_channel) - abs(signed),
        "opposite_signs": bool((mass - label_mass) * (density - label_density) < 0.0),
    }


def oracle_energy(energy: float, mass: float, label_energy: float, label_mass: float) -> dict[str, float]:
    """Label enters only as a multiplier. Neither total was a deployed commit."""
    parts = channels(energy, mass, label_energy, label_mass)
    return {
        "perfect_grams_kcal": parts["density_kcal_per_g"] * label_mass,
        "perfect_density_kcal": parts["label_density_kcal_per_g"] * mass,
    }


def bootstrap_mean(values: np.ndarray, *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0:
        raise EnergyAuditError("bootstrap needs a non-empty vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, sample.size, size=(BOOTSTRAP_DRAWS, sample.size))
    means = sample[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(sample.size),
        "mean": float(sample.mean()),
        "median": float(np.median(sample)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "fraction_mean_negative": float(np.mean(means < 0.0)),
        "draws": BOOTSTRAP_DRAWS,
        "seed": seed,
        "interval": "percentile",
        "unit": "plate resampled with replacement",
    }


def reconstruct_plate(
    writes: Sequence[Mapping[str, float]],
    committed_energy: float,
    committed_mass: float,
    align: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Hold the bound codes fixed, then separate the gram rewrite from the stored swap.

    ``writes`` are the mass-estimator portions already in the trace. ``old_*`` is the
    gram guess stored before that write, scaled at the same kcal/100 g. ``align`` is
    the density-swap observation on the finishing write, when the rollout stored one.
    """
    if not writes:
        raise EnergyAuditError("a scored plate has no mass write to account")
    new_energy = float(sum(row["new_energy"] for row in writes))
    old_energy = float(sum(row["old_energy"] for row in writes))
    new_mass = float(sum(row["new_g"] for row in writes))
    old_mass = float(sum(row["old_g"] for row in writes))
    before = None if align is None else finite(align.get("before_energy_kcal"))
    after = None if align is None else finite(align.get("after_energy_kcal"))
    applied = bool(align is not None and align.get("applied") is True)
    bound_energy = committed_energy if before is None else before
    energy_gap = abs(bound_energy - new_energy)
    mass_gap = abs(committed_mass - new_mass)
    accounted = energy_gap <= ACCOUNT_TOLERANCE_KCAL and mass_gap <= ACCOUNT_TOLERANCE_G
    prior_energy = bound_energy - new_energy + old_energy
    prior_mass = committed_mass - new_mass + old_mass
    swap_off_energy = bound_energy if applied else committed_energy
    if applied and after is None:
        raise EnergyAuditError("an applied swap has no after_energy_kcal")
    return {
        "accounted": accounted,
        "energy_gap_kcal": energy_gap,
        "mass_gap_g": mass_gap,
        "bound_energy_kcal": bound_energy,
        "prior_energy_kcal": prior_energy,
        "prior_mass_g": prior_mass,
        "swap_off_energy_kcal": swap_off_energy,
        "swap_on_energy_kcal": committed_energy,
        "applied": applied,
        "before_energy_kcal": before,
        "after_energy_kcal": after,
        "new_energy_kcal": new_energy,
        "old_energy_kcal": old_energy,
        "new_mass_g": new_mass,
        "old_mass_g": old_mass,
    }


def group_steps(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dish_id"]), []).append(dict(row))
    for steps in grouped.values():
        steps.sort(key=lambda step: int(step["step_i"]))
    return grouped


def commit_pointer(steps: Sequence[Mapping[str, Any]]) -> tuple[float, float, int]:
    """Same reader as the #63 score: last finite dish total at or before STOP/ABSTAIN."""
    actions = [str(step.get("action_t")) for step in steps]
    decision = next((index for index, action in enumerate(actions) if action in {"STOP", "ABSTAIN"}), len(actions) - 1)
    energy: float | None = None
    mass: float | None = None
    source: int | None = None
    for step in steps[: decision + 1]:
        dish = step.get("nutri_dish") if isinstance(step.get("nutri_dish"), Mapping) else {}
        kcal = finite(dish.get("energy_kcal"))
        if kcal is None:
            continue
        energy = kcal
        mass = finite(dish.get("mass_g"))
        source = int(step["step_i"])
    if energy is None or mass is None or source is None:
        raise EnergyAuditError("trajectory has no finite dish total")
    return energy, mass, source


def _matched_ref(step: Mapping[str, Any]) -> dict[str, Any]:
    obs = step.get("obs_t") if isinstance(step.get("obs_t"), Mapping) else {}
    code = str(step.get("code_i") or "")
    refs = (obs.get("nutrient_refs") or {}).get("fndds") if isinstance(obs.get("nutrient_refs"), Mapping) else None
    if not isinstance(refs, list):
        raise EnergyAuditError(f"step {step.get('step_i')} has no FNDDS nutrient refs")
    matched = [
        ref
        for ref in refs
        if isinstance(ref, Mapping) and str(ref.get("code")) == code and ref.get("available") and ref.get("nutrients")
    ]
    if len(matched) != 1:
        raise EnergyAuditError(f"step {step.get('step_i')} code {code} matched {len(matched)} refs")
    return dict(matched[0])


def plate_writes(steps: Sequence[Mapping[str, Any]], source_step_i: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[float]]:
    writes: list[dict[str, Any]] = []
    align: dict[str, Any] | None = None
    heads: list[float] = []
    for step in steps:
        if int(step["step_i"]) >= int(source_step_i):
            break
        if step.get("action_t") != "RE_PORTION":
            continue
        obs = step.get("obs_t") if isinstance(step.get("obs_t"), Mapping) else {}
        if obs.get("tool_failed") or obs.get("portion_src") != "mass_estimator":
            continue
        ref = _matched_ref(step)
        nutrients = ref.get("nutrients") if isinstance(ref.get("nutrients"), Mapping) else {}
        per100 = ref.get("per_100g") if isinstance(ref.get("per_100g"), Mapping) else {}
        new_g = finite(obs.get("portion_g"))
        old_g = finite(obs.get("from_portion_g"))
        new_energy = finite(nutrients.get("energy_kcal"))
        kcal_per_100g = finite(per100.get("energy_kcal"))
        if new_g is None or old_g is None or new_energy is None or kcal_per_100g is None or new_g <= 0.0 or old_g <= 0.0:
            raise EnergyAuditError(f"step {step.get('step_i')} has a non-finite portion")
        model = obs.get("nutrients_model") if isinstance(obs.get("nutrients_model"), Mapping) else {}
        head = finite(model.get("energy_kcal"))
        if head is None:
            raise EnergyAuditError(f"step {step.get('step_i')} has no calorie head")
        heads.append(head)
        writes.append(
            {
                "code": str(step.get("code_i")),
                "name": "" if ref.get("name") is None else str(ref.get("name")),
                "new_g": new_g,
                "old_g": old_g,
                "new_energy": new_energy,
                "old_energy": new_energy * old_g / new_g,
                "kcal_per_100g": kcal_per_100g,
            }
        )
        info = obs.get("density_align")
        if isinstance(info, Mapping):
            if align is not None:
                raise EnergyAuditError(f"{step.get('dish_id')} stores two density-swap records")
            align = dict(info)
    return writes, align, heads


def cafe2_label_check(scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Re-read cafe2. Dish-level energy is unused; the score target is the ingredient sum."""
    from vdr.data.n5k import parse_dish_metadata_line

    if not CAFE2.is_file():
        raise EnergyAuditError(f"missing cafe2 metadata: {CAFE2}")
    dishes: dict[str, Any] = {}
    n_rows = 0
    column_zero = 0
    ingredient_zero = 0
    with CAFE2.open(encoding="utf-8") as handle:
        for line in handle:
            dish = parse_dish_metadata_line(line)
            if dish is None:
                continue
            n_rows += 1
            if dish.energy_kcal == 0.0:
                column_zero += 1
            ingredient_energy = float(sum(item.energy_kcal for item in dish.ingredients))
            if ingredient_energy == 0.0:
                ingredient_zero += 1
            dishes[dish.dish_id] = (dish, ingredient_energy)
    if n_rows == 0 or column_zero != n_rows:
        raise EnergyAuditError("cafe2 dish-level energy is not 0 on every parsed row")
    max_energy_gap = 0.0
    max_mass_gap = 0.0
    for dish_id, score in scores.items():
        found = dishes.get(dish_id)
        if found is None:
            raise EnergyAuditError(f"{dish_id} is missing from cafe2 metadata")
        dish, ingredient_energy = found
        max_energy_gap = max(max_energy_gap, abs(ingredient_energy - float(score["label_energy_kcal"])))
        max_mass_gap = max(max_mass_gap, abs(float(dish.mass_g) - float(score["label_mass_g"])))
        column = finite(dish.energy_kcal)
        if column != 0.0:
            raise EnergyAuditError(f"{dish_id} dish-level energy column is {column}")
    if max_energy_gap > 1e-6 or max_mass_gap > 1e-6:
        raise EnergyAuditError(
            f"score labels differ from cafe2 ingredient sums (energy {max_energy_gap}, mass {max_mass_gap})"
        )
    return {
        "n_cafe2_rows": n_rows,
        "n_dish_energy_column_zero": column_zero,
        "n_ingredient_energy_zero": ingredient_zero,
        "heldout_label_matches_ingredient_sum": True,
        "max_abs_score_minus_ingredient_kcal": max_energy_gap,
        "max_abs_score_minus_dish_mass_g": max_mass_gap,
        "energy_target": "sum of ingredient calories; the dish-level energy column is 0 on every cafe2 row",
        "mass_target": "dish-level gram total",
    }


def _index(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        dish_id = str(row["dish_id"])
        if dish_id in out:
            raise EnergyAuditError(f"duplicate {key} row for {dish_id}")
        out[dish_id] = dict(row)
    return out


def build_records() -> dict[str, Any]:
    status_hash = sha256_file(STATUS)
    required = {
        "scores": EVAL / "scores.jsonl",
        "rollout": EVAL / "rollout.jsonl",
        "direct": EVAL / "direct_vlm.jsonl",
        "mass": EVAL / "mass_only.jsonl",
        "summary": EVAL / "summary.json",
        "report": EVAL / "report.md",
        "heldout_ids": PLAN / "heldout_dish_ids.txt",
        "dev505_ids": PLAN / "dev505_dish_ids.txt",
        "fit_ids": PLAN / "fit_dish_ids.txt",
        "plan_proof": PLAN / "proof.json",
        "status": STATUS,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise EnergyAuditError("missing frozen input: " + ", ".join(missing))
    hashes = {name: sha256_file(path) for name, path in required.items()}
    heldout_ids = read_ids(required["heldout_ids"])
    dev_ids = set(read_ids(required["dev505_ids"]))
    fit_ids = set(read_ids(required["fit_ids"]))
    if len(heldout_ids) != N_PAIRED or len(set(heldout_ids)) != N_PAIRED:
        raise EnergyAuditError(f"held-out list has {len(heldout_ids)} ids")
    overlap = sorted(set(heldout_ids) & dev_ids)
    if overlap:
        raise EnergyAuditError("held-out overlaps the development 505")
    if set(heldout_ids) & fit_ids:
        raise EnergyAuditError("held-out overlaps the fit set")

    scores = _index(load_jsonl(required["scores"]), "scores")
    direct_rows = _index(load_jsonl(required["direct"]), "direct")
    mass_rows = _index(load_jsonl(required["mass"]), "mass")
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    traces = group_steps(load_jsonl(required["rollout"]))
    if set(scores) != set(heldout_ids) or set(traces) != set(heldout_ids):
        raise EnergyAuditError("scores, trajectories, and the held-out list are different sets")
    if int(summary["n_paired"]) != N_PAIRED or int(summary["density_align_applied_dishes"]) != N_SWAP:
        raise EnergyAuditError("frozen summary is not the n=228 / 106-swap table")

    label_audit = cafe2_label_check(scores)
    records: list[dict[str, Any]] = []
    for dish_id in heldout_ids:
        score = scores[dish_id]
        direct = direct_rows[dish_id]
        mass = mass_rows[dish_id]
        for key, row in (("direct_energy_kcal", direct), ("direct_mass_g", direct)):
            if abs(float(score[key]) - float(row[{"direct_energy_kcal": "energy_kcal", "direct_mass_g": "mass_g"}[key]])) > 1e-6:
                raise EnergyAuditError(f"{dish_id} Direct-VLM score differs from direct_vlm.jsonl")
        if abs(float(score["mass_only_energy_kcal"]) - float(mass["energy_kcal"])) > 1e-6:
            raise EnergyAuditError(f"{dish_id} Mass-only energy differs from mass_only.jsonl")
        if abs(float(score["mass_only_mass_g"]) - float(mass["mass_g"])) > 1e-6:
            raise EnergyAuditError(f"{dish_id} Mass-only mass differs from mass_only.jsonl")
        steps = traces[dish_id]
        commit_energy, commit_mass, source = commit_pointer(steps)
        if abs(commit_energy - float(score["ours_energy_kcal"])) > 1e-3 or abs(commit_mass - float(score["ours_mass_g"])) > 1e-3:
            raise EnergyAuditError(f"{dish_id} rollout commit differs from the frozen score")
        writes, align, heads = plate_writes(steps, source)
        if max(heads) - min(heads) > 1e-3:
            raise EnergyAuditError(f"{dish_id} calorie head changes across mass writes")
        if abs(heads[0] - float(score["mass_only_energy_kcal"])) > 1e-3:
            raise EnergyAuditError(f"{dish_id} stored calorie head differs from Mass-only energy")
        plate = reconstruct_plate(writes, float(score["ours_energy_kcal"]), float(score["ours_mass_g"]), align)
        if not plate["accounted"]:
            raise EnergyAuditError(
                f"{dish_id} food-table sum misses the commit by {plate['energy_gap_kcal']} kcal / {plate['mass_gap_g']} g"
            )
        if plate["applied"] and plate["after_energy_kcal"] is not None:
            if abs(float(plate["after_energy_kcal"]) - float(score["ours_energy_kcal"])) > 0.001:
                raise EnergyAuditError(f"{dish_id} swap-after energy differs from the commit")
        label_energy = float(score["label_energy_kcal"])
        label_mass = float(score["label_mass_g"])
        ours_energy = float(score["ours_energy_kcal"])
        ours_mass = float(score["ours_mass_g"])
        direct_energy = float(score["direct_energy_kcal"])
        direct_mass = float(score["direct_mass_g"])
        mass_energy = float(score["mass_only_energy_kcal"])
        mass_mass = float(score["mass_only_mass_g"])
        ours_parts = channels(ours_energy, ours_mass, label_energy, label_mass)
        bound_parts = channels(float(plate["bound_energy_kcal"]), ours_mass, label_energy, label_mass)
        prior_parts = channels(float(plate["prior_energy_kcal"]), float(plate["prior_mass_g"]), label_energy, label_mass)
        direct_parts = channels(direct_energy, direct_mass, label_energy, label_mass)
        mass_parts = channels(mass_energy, mass_mass, label_energy, label_mass)
        ours_oracle = oracle_energy(ours_energy, ours_mass, label_energy, label_mass)
        changes = list(align.get("changes") or []) if align else []
        records.append(
            {
                "dish_id": dish_id,
                "label_energy_kcal": label_energy,
                "label_mass_g": label_mass,
                "label_density_kcal_per_g": ours_parts["label_density_kcal_per_g"],
                "ours_energy_kcal": ours_energy,
                "ours_mass_g": ours_mass,
                "direct_energy_kcal": direct_energy,
                "direct_mass_g": direct_mass,
                "mass_only_energy_kcal": mass_energy,
                "mass_only_mass_g": mass_mass,
                "calorie_head_kcal": heads[0],
                "bound_energy_kcal": float(plate["bound_energy_kcal"]),
                "prior_energy_kcal": float(plate["prior_energy_kcal"]),
                "prior_mass_g": float(plate["prior_mass_g"]),
                "swap_off_energy_kcal": float(plate["swap_off_energy_kcal"]),
                "oracle_density_kcal": ours_oracle["perfect_density_kcal"],
                "oracle_grams_kcal": ours_oracle["perfect_grams_kcal"],
                "swap_applied": bool(plate["applied"]),
                "n_changed": 0 if align is None else int(align.get("n_changed") or 0),
                "swap_reason": None if align is None else align.get("reason"),
                "changes": changes,
                "n_hypotheses": int(score["n_hypotheses"]),
                "n_mass_writes": len(writes),
                "plate_complete": bool(score["plate_complete"]),
                "decision_action": score["decision_action"],
                "old_grams": [float(row["old_g"]) for row in writes],
                "writes": writes,
                "placement_max_memory": direct.get("placement_max_memory"),
                "ours_parts": ours_parts,
                "bound_parts": bound_parts,
                "prior_parts": prior_parts,
                "direct_parts": direct_parts,
                "mass_parts": mass_parts,
            }
        )
    if sha256_file(STATUS) != status_hash:
        raise EnergyAuditError("docs/STATUS.md changed while the audit was reading")
    return {
        "records": records,
        "hashes": hashes,
        "status_hash": status_hash,
        "label_audit": label_audit,
        "frozen_summary_swaps": int(summary["density_align_applied_dishes"]),
        "frozen_hollow": int((summary.get("hollow_re_portion") or {}).get("hollow", -1)),
    }


def _column(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in records], dtype=np.float64)


def _mae_pair(pred: np.ndarray, label: np.ndarray) -> float:
    return mae(pred - label)


def _channel_block(records: Sequence[Mapping[str, Any]], parts_key: str, energy_key: str, mass_key: str) -> dict[str, float]:
    parts = [row[parts_key] for row in records]
    energy = _column(records, energy_key)
    mass = _column(records, mass_key)
    label_e = _column(records, "label_energy_kcal")
    label_m = _column(records, "label_mass_g")
    density_gap = np.abs(energy / mass - label_e / label_m)
    return {
        "energy_mae_kcal": _mae_pair(energy, label_e),
        "mass_mae_g": _mae_pair(mass, label_m),
        "density_mae_kcal_per_g": float(np.mean(density_gap)),
        "mean_abs_mass_channel_kcal": float(np.mean([abs(row["mass_channel_kcal"]) for row in parts])),
        "mean_abs_density_channel_kcal": float(np.mean([abs(row["density_channel_kcal"]) for row in parts])),
        "mean_mass_channel_kcal": float(np.mean([row["mass_channel_kcal"] for row in parts])),
        "mean_density_channel_kcal": float(np.mean([row["density_channel_kcal"] for row in parts])),
        "mean_cancellation_kcal": float(np.mean([row["cancellation_kcal"] for row in parts])),
        "mean_signed_energy_kcal": float(np.mean([row["signed_energy_kcal"] for row in parts])),
        "opposite_sign_rate": float(np.mean([1.0 if row["opposite_signs"] else 0.0 for row in parts])),
    }


def _slice_row(name: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {"name": name, "n": 0}
    label_e = _column(rows, "label_energy_kcal")
    label_m = _column(rows, "label_mass_g")

    def energy(key: str) -> float:
        return _mae_pair(_column(rows, key), label_e)

    def mass(key: str) -> float:
        return _mae_pair(_column(rows, key), label_m)

    commit_gap = np.abs(_column(rows, "ours_energy_kcal") - label_e) - np.abs(_column(rows, "mass_only_energy_kcal") - label_e)
    mass_gap = np.abs(_column(rows, "ours_mass_g") - label_m) - np.abs(_column(rows, "mass_only_mass_g") - label_m)
    return {
        "name": name,
        "n": n,
        "commit_energy_mae_kcal": energy("ours_energy_kcal"),
        "swap_off_energy_mae_kcal": energy("swap_off_energy_kcal"),
        "bound_energy_mae_kcal": energy("bound_energy_kcal"),
        "prior_energy_mae_kcal": energy("prior_energy_kcal"),
        "oracle_density_energy_mae_kcal": energy("oracle_density_kcal"),
        "mass_only_energy_mae_kcal": energy("mass_only_energy_kcal"),
        "direct_energy_mae_kcal": energy("direct_energy_kcal"),
        "commit_mass_mae_g": mass("ours_mass_g"),
        "prior_mass_mae_g": mass("prior_mass_g"),
        "mass_only_mass_mae_g": mass("mass_only_mass_g"),
        "direct_mass_mae_g": mass("direct_mass_g"),
        "energy_gap_vs_mass_only_kcal": float(commit_gap.mean()),
        "mass_gap_vs_mass_only_g": float(mass_gap.mean()),
        "contribution_to_mean_energy_gap_kcal": float(commit_gap.sum() / N_PAIRED),
        "contribution_to_mean_mass_gap_g": float(mass_gap.sum() / N_PAIRED),
    }


def _example(row: Mapping[str, Any]) -> dict[str, Any]:
    foods = []
    for write in row["writes"]:
        foods.append(
            {
                "name": write["name"],
                "code": write["code"],
                "old_g": write["old_g"],
                "new_g": write["new_g"],
                "kcal_per_100g": write["kcal_per_100g"],
            }
        )
    label_e = float(row["label_energy_kcal"])
    return {
        "dish_id": row["dish_id"],
        "n_hypotheses": row["n_hypotheses"],
        "n_mass_writes": row["n_mass_writes"],
        "plate_complete": row["plate_complete"],
        "swap_applied": row["swap_applied"],
        "decision_action": row["decision_action"],
        "label_energy_kcal": label_e,
        "label_mass_g": row["label_mass_g"],
        "label_density_kcal_per_g": row["label_density_kcal_per_g"],
        "ours_energy_kcal": row["ours_energy_kcal"],
        "ours_mass_g": row["ours_mass_g"],
        "bound_energy_kcal": row["bound_energy_kcal"],
        "prior_energy_kcal": row["prior_energy_kcal"],
        "mass_only_energy_kcal": row["mass_only_energy_kcal"],
        "mass_only_mass_g": row["mass_only_mass_g"],
        "direct_energy_kcal": row["direct_energy_kcal"],
        "direct_mass_g": row["direct_mass_g"],
        "calorie_head_kcal": row["calorie_head_kcal"],
        "energy_ae_commit_minus_mass_only_kcal": abs(row["ours_energy_kcal"] - label_e)
        - abs(row["mass_only_energy_kcal"] - label_e),
        "energy_ae_commit_minus_swap_off_kcal": abs(row["ours_energy_kcal"] - label_e)
        - abs(row["swap_off_energy_kcal"] - label_e),
        "foods": foods,
        "changes": row["changes"],
    }


def aggregate(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = list(payload["records"])
    if len(records) != N_PAIRED:
        raise EnergyAuditError(f"built {len(records)} plates, expected {N_PAIRED}")
    label_e = _column(records, "label_energy_kcal")
    label_m = _column(records, "label_mass_g")
    ours_e = _column(records, "ours_energy_kcal")
    ours_m = _column(records, "ours_mass_g")
    direct_e = _column(records, "direct_energy_kcal")
    direct_m = _column(records, "direct_mass_g")
    mass_e = _column(records, "mass_only_energy_kcal")
    mass_m = _column(records, "mass_only_mass_g")
    energy_vs_mass = np.abs(ours_e - label_e) - np.abs(mass_e - label_e)
    mass_vs_mass = np.abs(ours_m - label_m) - np.abs(mass_m - label_m)
    energy_vs_direct = np.abs(ours_e - label_e) - np.abs(direct_e - label_e)
    mass_vs_direct = np.abs(ours_m - label_m) - np.abs(direct_m - label_m)
    published_energy = {
        "ours": _mae_pair(ours_e, label_e),
        "direct": _mae_pair(direct_e, label_e),
        "mass_only": _mae_pair(mass_e, label_e),
    }
    published_mass = {
        "ours": _mae_pair(ours_m, label_m),
        "direct": _mae_pair(direct_m, label_m),
        "mass_only": _mae_pair(mass_m, label_m),
    }
    checks = {
        "ours_energy_mae_kcal": published_energy["ours"],
        "ours_mass_mae_g": published_mass["ours"],
        "direct_energy_mae_kcal": published_energy["direct"],
        "direct_mass_mae_g": published_mass["direct"],
        "mass_only_energy_mae_kcal": published_energy["mass_only"],
        "mass_only_mass_mae_g": published_mass["mass_only"],
        "energy_minus_direct_kcal": published_energy["ours"] - published_energy["direct"],
        "mass_minus_direct_g": published_mass["ours"] - published_mass["direct"],
        "energy_minus_mass_only_kcal": published_energy["ours"] - published_energy["mass_only"],
        "mass_minus_mass_only_g": published_mass["ours"] - published_mass["mass_only"],
    }
    for key, expected in PUBLISHED.items():
        if abs(checks[key] - expected) > 1e-9:
            raise EnergyAuditError(f"{key} recomputed as {checks[key]} but the frozen table has {expected}")

    swap_mask = np.asarray([bool(row["swap_applied"]) for row in records])
    if int(np.sum(swap_mask)) != N_SWAP:
        raise EnergyAuditError(f"swap count {int(np.sum(swap_mask))} != {N_SWAP}")
    complete = np.asarray([bool(row["plate_complete"]) for row in records])
    single = np.asarray([int(row["n_hypotheses"]) == 1 for row in records])
    old_grams = [gram for row in records for gram in row["old_grams"]]
    if len(old_grams) != 575 or any(abs(gram - 100.0) > 1e-6 for gram in old_grams):
        raise EnergyAuditError("mass writes are not the expected 575 replacements of a 100 g grid")
    if any(row["placement_max_memory"] != "8GiB" for row in records):
        raise EnergyAuditError("Direct-VLM placement is not the frozen 8GiB record on every plate")
    mass_tied = np.abs(ours_m - mass_m) <= 1e-3
    if int(np.sum(mass_tied)) != int(np.sum(complete)) or int(np.sum(complete)) != 222:
        raise EnergyAuditError("complete plates and shared gram totals diverged")
    if payload["frozen_hollow"] != 0:
        raise EnergyAuditError("frozen hollow RE_PORTION count is not 0")

    bound_e = _column(records, "bound_energy_kcal")
    prior_e = _column(records, "prior_energy_kcal")
    prior_m = _column(records, "prior_mass_g")
    swap_off_e = _column(records, "swap_off_energy_kcal")
    oracle_e = _column(records, "oracle_density_kcal")
    mass_down = np.abs(ours_m - label_m) < np.abs(prior_m - label_m)
    energy_down = np.abs(bound_e - label_e) < np.abs(prior_e - label_e)
    energy_up = np.abs(bound_e - label_e) > np.abs(prior_e - label_e)
    rewrite = {
        "n": N_PAIRED,
        "n_writes": len(old_grams),
        "prior_grams_g": 100.0,
        "prior_energy_mae_kcal": _mae_pair(prior_e, label_e),
        "prior_mass_mae_g": _mae_pair(prior_m, label_m),
        "bound_energy_mae_kcal": _mae_pair(bound_e, label_e),
        "bound_mass_mae_g": _mae_pair(ours_m, label_m),
        "energy_mae_change_kcal": _mae_pair(bound_e, label_e) - _mae_pair(prior_e, label_e),
        "mass_mae_change_g": _mae_pair(ours_m, label_m) - _mae_pair(prior_m, label_m),
        "n_mass_ae_down": int(np.sum(mass_down)),
        "n_energy_ae_down": int(np.sum(energy_down)),
        "n_energy_ae_up": int(np.sum(energy_up)),
        "n_mass_down_energy_up": int(np.sum(mass_down & energy_up)),
        "n_mass_down_energy_down": int(np.sum(mass_down & energy_down)),
    }
    account_gaps = []
    for row in records:
        account_gaps.append(abs(float(row["bound_energy_kcal"]) - float(sum(write["new_energy"] for write in row["writes"]))))
    rewrite["max_account_energy_gap_kcal"] = float(max(account_gaps))
    rewrite["max_account_mass_gap_g"] = float(
        max(abs(float(row["ours_mass_g"]) - float(sum(write["new_g"] for write in row["writes"]))) for row in records)
    )

    swapped = [row for row in records if row["swap_applied"]]
    head_gap_on = [abs(row["ours_energy_kcal"] - row["calorie_head_kcal"]) for row in swapped]
    head_gap_off = [abs(row["bound_energy_kcal"] - row["calorie_head_kcal"]) for row in swapped]
    closer_label = sum(
        1
        for row in swapped
        if abs(row["ours_energy_kcal"] - row["label_energy_kcal"]) < abs(row["swap_off_energy_kcal"] - row["label_energy_kcal"])
    )
    farther_label = sum(
        1
        for row in swapped
        if abs(row["ours_energy_kcal"] - row["label_energy_kcal"]) > abs(row["swap_off_energy_kcal"] - row["label_energy_kcal"])
    )
    closer_head = sum(1 for on, off in zip(head_gap_on, head_gap_off) if on < off)
    swap = {
        "n_applied": N_SWAP,
        "n_not_applied": N_PAIRED - N_SWAP,
        "full_swap_off_energy_mae_kcal": _mae_pair(swap_off_e, label_e),
        "full_swap_on_energy_mae_kcal": published_energy["ours"],
        "full_energy_mae_change_kcal": published_energy["ours"] - _mae_pair(swap_off_e, label_e),
        "mass_unchanged": True,
        "swapped_energy_mae_on_kcal": _mae_pair(_column(swapped, "ours_energy_kcal"), _column(swapped, "label_energy_kcal")),
        "swapped_energy_mae_off_kcal": _mae_pair(_column(swapped, "swap_off_energy_kcal"), _column(swapped, "label_energy_kcal")),
        "swapped_mass_only_energy_mae_kcal": _mae_pair(
            _column(swapped, "mass_only_energy_kcal"), _column(swapped, "label_energy_kcal")
        ),
        "swapped_direct_energy_mae_kcal": _mae_pair(_column(swapped, "direct_energy_kcal"), _column(swapped, "label_energy_kcal")),
        "n_closer_to_label": closer_label,
        "n_farther_from_label": farther_label,
        "n_closer_to_calorie_head": closer_head,
        "mean_abs_commit_minus_head_kcal": float(np.mean(head_gap_on)),
        "mean_abs_bound_minus_head_kcal": float(np.mean(head_gap_off)),
        "mean_energy_on_minus_off_kcal": float(np.mean([row["ours_energy_kcal"] - row["bound_energy_kcal"] for row in swapped])),
        "n_changed_hist": {
            str(count): int(sum(1 for row in swapped if int(row["n_changed"]) == count))
            for count in sorted({int(row["n_changed"]) for row in swapped})
        },
    }
    if closer_head != N_SWAP:
        raise EnergyAuditError("an applied swap moved food-table energy away from the calorie head")

    oracle_mass = oracle_energy_vector(mass_e, mass_m, label_e, label_m)
    oracle_direct = oracle_energy_vector(direct_e, direct_m, label_e, label_m)
    oracle = {
        "ours_label_density_kcal": _mae_pair(oracle_e, label_e),
        "mass_only_label_density_kcal": _mae_pair(oracle_mass, label_e),
        "direct_label_density_kcal": _mae_pair(oracle_direct, label_e),
        "ours_label_grams_kcal": _mae_pair(_column(records, "oracle_grams_kcal"), label_e),
        "bound_label_grams_kcal": _mae_pair(
            np.asarray(
                [oracle_energy(row["bound_energy_kcal"], row["ours_mass_g"], row["label_energy_kcal"], row["label_mass_g"])["perfect_grams_kcal"] for row in records]
            ),
            label_e,
        ),
        "mass_only_label_grams_kcal": _mae_pair(
            np.asarray(
                [
                    oracle_energy(row["mass_only_energy_kcal"], row["mass_only_mass_g"], row["label_energy_kcal"], row["label_mass_g"])[
                        "perfect_grams_kcal"
                    ]
                    for row in records
                ]
            ),
            label_e,
        ),
    }

    slices = [
        _slice_row("all", records),
        _slice_row("complete", [row for row in records if row["plate_complete"]]),
        _slice_row("incomplete", [row for row in records if not row["plate_complete"]]),
        _slice_row("single_component", [row for row in records if row["n_hypotheses"] == 1]),
        _slice_row("multi_component", [row for row in records if row["n_hypotheses"] > 1]),
        _slice_row(
            "complete_multi",
            [row for row in records if row["plate_complete"] and row["n_hypotheses"] > 1],
        ),
        _slice_row("swap_applied", swapped),
        _slice_row("swap_not_applied", [row for row in records if not row["swap_applied"]]),
    ]
    contribution_sum = sum(row["contribution_to_mean_energy_gap_kcal"] for row in slices if row["name"] in {"single_component", "multi_component"})
    if abs(contribution_sum - float(energy_vs_mass.mean())) > 1e-8:
        raise EnergyAuditError("single and multi contributions do not add to the energy gap")

    positive = energy_vs_mass[energy_vs_mass > 0]
    order = np.argsort(-energy_vs_mass)
    top10 = float(np.clip(energy_vs_mass[order[:10]], 0, None).sum())
    quantiles = {
        f"p{int(q * 100):02d}": float(np.quantile(energy_vs_mass, q))
        for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    }
    singles = sorted(
        [row for row in records if row["n_hypotheses"] == 1],
        key=lambda row: (
            -(abs(row["ours_energy_kcal"] - row["label_energy_kcal"]) - abs(row["mass_only_energy_kcal"] - row["label_energy_kcal"])),
            row["dish_id"],
        ),
    )
    swap_rows = sorted(
        swapped,
        key=lambda row: (
            abs(row["ours_energy_kcal"] - row["label_energy_kcal"]) - abs(row["swap_off_energy_kcal"] - row["label_energy_kcal"]),
            row["dish_id"],
        ),
    )
    examples = {
        "largest_single_component_gaps_vs_mass_only": [_example(row) for row in singles[:3]],
        "swap_largest_label_ae_drop": _example(swap_rows[0]),
        "swap_largest_label_ae_rise": _example(swap_rows[-1]),
    }
    mean_abs_commit_head = float(np.mean(np.abs(ours_e - mass_e)))
    channels_out = {
        "commit": _channel_block(records, "ours_parts", "ours_energy_kcal", "ours_mass_g"),
        "swap_off": _channel_block(records, "bound_parts", "bound_energy_kcal", "ours_mass_g"),
        "prior_grams": _channel_block(records, "prior_parts", "prior_energy_kcal", "prior_mass_g"),
        "mass_only": _channel_block(records, "mass_parts", "mass_only_energy_kcal", "mass_only_mass_g"),
        "direct": _channel_block(records, "direct_parts", "direct_energy_kcal", "direct_mass_g"),
    }
    # Swap-off channel uses bound energy, which equals swap-off energy except for sub-0.001 kcal dust
    # on plates whose align record was stored and not applied.
    dust = float(np.max(np.abs(bound_e - swap_off_e)))
    if dust > 0.001:
        raise EnergyAuditError(f"bound-code energy and swap-off energy differ by {dust} kcal")

    summary = {
        "protocol": "cafe2_heldout_energy_audit",
        "diagnostic_only": True,
        "n_paired": N_PAIRED,
        "claim": "质量稳赢可外推；能量开发集赢、外测未稳。",
        "development_505_not_a_row": {"energy_mae_kcal": 64.2792, "mass_mae_g": 32.9908},
        "published": checks,
        "bootstrap": {
            "energy_commit_minus_mass_only": bootstrap_mean(energy_vs_mass),
            "mass_commit_minus_mass_only": bootstrap_mean(mass_vs_mass),
            "energy_commit_minus_direct": bootstrap_mean(energy_vs_direct),
            "mass_commit_minus_direct": bootstrap_mean(mass_vs_direct),
        },
        "plates_vs_mass_only": {
            "energy_closer": int(np.sum(energy_vs_mass < 0)),
            "energy_farther": int(np.sum(energy_vs_mass > 0)),
            "energy_tie": int(np.sum(energy_vs_mass == 0)),
            "mass_closer": int(np.sum(mass_vs_mass < -1e-3)),
            "mass_farther": int(np.sum(mass_vs_mass > 1e-3)),
            "mass_tied_within_0_001g": int(np.sum(np.abs(mass_vs_mass) <= 1e-3)),
        },
        "plates_vs_direct": {
            "energy_closer": int(np.sum(energy_vs_direct < 0)),
            "energy_farther": int(np.sum(energy_vs_direct > 0)),
            "mass_closer": int(np.sum(mass_vs_direct < 0)),
            "mass_farther": int(np.sum(mass_vs_direct > 0)),
        },
        "median_abs": {
            "commit_energy_kcal": float(np.median(np.abs(ours_e - label_e))),
            "mass_only_energy_kcal": float(np.median(np.abs(mass_e - label_e))),
            "direct_energy_kcal": float(np.median(np.abs(direct_e - label_e))),
            "commit_mass_g": float(np.median(np.abs(ours_m - label_m))),
            "mass_only_mass_g": float(np.median(np.abs(mass_m - label_m))),
            "direct_mass_g": float(np.median(np.abs(direct_m - label_m))),
        },
        "energy_gap_vs_mass_only_quantiles_kcal": quantiles,
        "positive_energy_gap_vs_mass_only_sum_kcal": float(positive.sum()) if positive.size else 0.0,
        "top10_share_of_positive_energy_gap": 0.0 if positive.size == 0 else top10 / float(positive.sum()),
        "mean_abs_commit_minus_calorie_head_kcal": mean_abs_commit_head,
        "rewrite_codes_fixed": rewrite,
        "swap": swap,
        "bound_vs_swap_off_max_abs_kcal": dust,
        "oracle": oracle,
        "channels": channels_out,
        "slices": slices,
        "examples": examples,
        "labels": payload["label_audit"],
        "direct_vlm_placement_max_memory": "8GiB",
        "n_direct_rows_at_8GiB": int(sum(1 for row in records if row["placement_max_memory"] == "8GiB")),
        "hollow_re_portion_from_frozen_summary": payload["frozen_hollow"],
        "input_sha256": payload["hashes"],
        "status_sha256": payload["status_hash"],
        "method_selected": False,
        "early_stopped": False,
        "tau_retuned": False,
        "codes_or_portions_edited": False,
        "mass_retrained": False,
        "q_retrained": False,
        "rl": False,
        "gates_4_to_7_started": False,
        "p1_unlocked": False,
    }
    return summary


def oracle_energy_vector(energy: np.ndarray, mass: np.ndarray, label_energy: np.ndarray, label_mass: np.ndarray) -> np.ndarray:
    return (label_energy / label_mass) * mass


def _food_phrase(foods: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for food in foods:
        parts.append(
            f"{food['name']} ({food['new_g']:.1f} g, was {food['old_g']:.1f} g, {food['kcal_per_100g']:.1f} kcal/100 g, code {food['code']})"
        )
    return "; ".join(parts)


def _change_phrase(changes: Sequence[Mapping[str, Any]]) -> str:
    if not changes:
        return "none"
    parts = []
    for change in changes:
        parts.append(f"{change.get('query')}: {change.get('from_name')} → {change.get('to_name')}")
    return "; ".join(parts)


def render_report(summary: Mapping[str, Any]) -> str:
    published = summary["published"]
    boot_em = summary["bootstrap"]["energy_commit_minus_mass_only"]
    boot_mm = summary["bootstrap"]["mass_commit_minus_mass_only"]
    boot_ed = summary["bootstrap"]["energy_commit_minus_direct"]
    boot_md = summary["bootstrap"]["mass_commit_minus_direct"]
    rewrite = summary["rewrite_codes_fixed"]
    swap = summary["swap"]
    oracle = summary["oracle"]
    channels = summary["channels"]
    slices = {row["name"]: row for row in summary["slices"]}
    single = slices["single_component"]
    multi = slices["multi_component"]
    complete = slices["complete"]
    incomplete = slices["incomplete"]
    swapped = slices["swap_applied"]
    plain = slices["swap_not_applied"]
    versus = summary["plates_vs_mass_only"]
    labels = summary["labels"]
    examples = summary["examples"]
    quant = summary["energy_gap_vs_mass_only_quantiles_kcal"]

    def channel_line(name: str, block: Mapping[str, Any]) -> str:
        return (
            f"| {name} | {block['mass_mae_g']:.2f} | {block['energy_mae_kcal']:.2f} | {block['density_mae_kcal_per_g']:.3f} | "
            f"{block['mean_abs_mass_channel_kcal']:.1f} | {block['mean_abs_density_channel_kcal']:.1f} | "
            f"{block['mean_cancellation_kcal']:.1f} | {block['mean_signed_energy_kcal']:+.1f} |"
        )

    def energy_line(row: Mapping[str, Any]) -> str:
        return (
            f"| {row['name']} | {row['n']} | {row['commit_energy_mae_kcal']:.2f} | {row['swap_off_energy_mae_kcal']:.2f} | "
            f"{row['prior_energy_mae_kcal']:.2f} | {row['oracle_density_energy_mae_kcal']:.2f} | "
            f"{row['mass_only_energy_mae_kcal']:.2f} | {row['direct_energy_mae_kcal']:.2f} | "
            f"{row['contribution_to_mean_energy_gap_kcal']:+.2f} |"
        )

    worst = examples["largest_single_component_gaps_vs_mass_only"]
    helped = examples["swap_largest_label_ae_drop"]
    hurt = examples["swap_largest_label_ae_rise"]

    def plate_block(title: str, row: Mapping[str, Any]) -> list[str]:
        return [
            f"- `{row['dish_id']}` ({title}).",
            f"  Label {row['label_mass_g']:.1f} g / {row['label_energy_kcal']:.1f} kcal ({row['label_density_kcal_per_g'] * 100:.1f} kcal/100 g).",
            f"  Commit {row['ours_mass_g']:.1f} g / {row['ours_energy_kcal']:.1f} kcal. "
            f"Bound codes at the same grams {row['bound_energy_kcal']:.1f} kcal. "
            f"Prior grams {row['prior_energy_kcal']:.1f} kcal.",
            f"  Mass-only {row['mass_only_mass_g']:.1f} g / {row['mass_only_energy_kcal']:.1f} kcal. "
            f"Direct-VLM {row['direct_mass_g']:.1f} g / {row['direct_energy_kcal']:.1f} kcal.",
            f"  Bound foods at the mass write: {_food_phrase(row['foods'])}.",
            f"  Swap changes: {_change_phrase(row['changes'])}.",
        ]

    lines = [
        "# Cafe2 energy audit",
        "",
        f"**{summary['claim']}**",
        "",
        "On the frozen Cafe2 held-out, n=228, the coverage-constrained commit has energy MAE "
        f"**{f4(published['ours_energy_mae_kcal'])} kcal** and mass MAE **{f4(published['ours_mass_mae_g'])} g**. "
        f"Direct-VLM has **{f4(published['direct_energy_mae_kcal'])} kcal** and **{f4(published['direct_mass_mae_g'])} g**. "
        f"Mass-only has **{f4(published['mass_only_energy_mae_kcal'])} kcal** and **{f4(published['mass_only_mass_mae_g'])} g**. "
        f"Versus Mass-only the commit is **{published['energy_minus_mass_only_kcal']:+.4f} kcal** and **{published['mass_minus_mass_only_g']:+.4f} g**. "
        f"Versus Direct-VLM it is **{published['energy_minus_direct_kcal']:+.4f} kcal** and **{published['mass_minus_direct_g']:+.4f} g**.",
        "",
        "The +41.4624 kcal is the food-table density on grams that match Mass-only. "
        f"On **{int(complete['n'])}** of the 228 plates the committed gram total and the Mass-only gram total differ by less than 0.001 g, "
        f"and the energy MAE gap on those plates is **{complete['energy_gap_vs_mass_only_kcal']:+.2f} kcal**. "
        "The commit multiplies those grams by the selected FNDDS kcal per 100 g. "
        "Mass-only scores the frozen mass estimator's calorie head on the same grams. "
        f"The **{single['n']}** one-component plates, where a same-head swap is ineligible, contribute "
        f"**{single['contribution_to_mean_energy_gap_kcal']:+.2f} kcal** of the +41.46 mean. "
        f"Their energy MAE is {single['commit_energy_mae_kcal']:.2f} versus Mass-only {single['mass_only_energy_mae_kcal']:.2f}, "
        f"at the same mass MAE {single['commit_mass_mae_g']:.2f} g.",
        "",
        f"A 10,000-draw bootstrap that resamples plates (numpy Generator seed 0) puts the energy gap versus Mass-only at "
        f"**[{boot_em['ci95_low']:+.2f}, {boot_em['ci95_high']:+.2f}] kcal**. "
        f"{100 * (1.0 - boot_em['fraction_mean_negative']):.1f}% of draws keep a positive mean. "
        f"The mass gap versus Mass-only is **[{boot_mm['ci95_low']:+.2f}, {boot_mm['ci95_high']:+.2f}] g**, and that interval contains both signs. "
        f"Versus Direct-VLM, the mass interval is **[{boot_md['ci95_low']:+.2f}, {boot_md['ci95_high']:+.2f}] g** with every draw negative, "
        f"and the energy interval is **[{boot_ed['ci95_low']:+.2f}, {boot_ed['ci95_high']:+.2f}] kcal**, which contains both signs. "
        "Mass versus Direct-VLM stays ahead on this external set. The development-set energy win does not reappear here.",
        "",
        "The development 505 energy MAE 64.2792 and mass MAE 32.9908 are not rows of this table.",
        "",
        "## What was measured",
        "",
        "The plate list, the trajectories, and the Direct-VLM and Mass-only scores are the frozen #63 artifacts. "
        f"Cafe2 was re-read: **{labels['n_cafe2_rows']}** rows, dish-level energy column **0** on every row. "
        "The energy target on the 228 is the sum of ingredient calories, and it matches the stored score labels. "
        "Mass is the dish-level gram total. "
        "Labels score residuals. No threshold, code, or gram was chosen from them. No model was trained. `docs/STATUS.md` was not edited.",
        "",
        "A calorie is kcal/g times grams. With both masses positive the signed calorie error splits exactly into",
        "",
        "`predicted kcal/g × (predicted grams − label grams) + label grams × (predicted kcal/g − label kcal/g)`.",
        "",
        "The first term is the gram term. The second is the food-density term. "
        "Cancellation is `|gram term| + |density term| − |calorie error|`. "
        "Two further calorie totals keep one saved factor and replace the other with the label. Those totals are arithmetic. They were not commits.",
        "",
        "The bound-code total is the food-table sum at the mass-estimator grams before the same-head swap. "
        "When the rollout stored an align record, that total is `before_energy_kcal`. On a one-component plate the rollout stores no align record, and the commit is already that total. "
        f"Bound-code energy and the swap-off energy differ by at most **{summary['bound_vs_swap_off_max_abs_kcal']:.4f} kcal** on a plate. "
        f"The largest food-table accounting gap against the summed component energies is **{rewrite['max_account_energy_gap_kcal']:.4f} kcal** "
        f"and **{rewrite['max_account_mass_gap_g']:.4f} g**, inside the 0.05 tolerance on all 228 plates.",
        "",
        "## Codes held fixed",
        "",
        f"Every one of the **{rewrite['n_writes']}** mass writes replaced a **{rewrite['prior_grams_g']:.0f} g** grid portion. "
        "The food code on that write stays the bound code. "
        f"At the grid grams, energy MAE is **{rewrite['prior_energy_mae_kcal']:.4f} kcal** and mass MAE is **{rewrite['prior_mass_mae_g']:.4f} g**. "
        f"At the mass-estimator grams, with those same codes, energy MAE is **{rewrite['bound_energy_mae_kcal']:.4f} kcal** and mass MAE is **{rewrite['bound_mass_mae_g']:.4f} g**. "
        f"The rewrite changes energy MAE by **{rewrite['energy_mae_change_kcal']:+.4f} kcal** and mass MAE by **{rewrite['mass_mae_change_g']:+.4f} g**.",
        "",
        f"Mass absolute error falls on **{rewrite['n_mass_ae_down']}** plates. "
        f"Energy absolute error falls on **{rewrite['n_energy_ae_down']}** and rises on **{rewrite['n_energy_ae_up']}**. "
        f"**{rewrite['n_mass_down_energy_down']}** plates lose mass error and energy error together. "
        f"**{rewrite['n_mass_down_energy_up']}** lose mass error while energy error rises. "
        "On the full 228, holding the bound codes fixed, the mass rewrite lowers energy MAE and mass MAE together.",
        "",
        f"The pooled drop comes from the multi-component plates. With those codes held fixed, energy MAE moves from "
        f"**{multi['prior_energy_mae_kcal']:.2f}** to **{multi['bound_energy_mae_kcal']:.2f} kcal** and mass MAE from "
        f"**{multi['prior_mass_mae_g']:.2f}** to **{multi['commit_mass_mae_g']:.2f} g**. "
        f"On the **{single['n']}** one-component plates the same rewrite moves energy MAE from "
        f"**{single['prior_energy_mae_kcal']:.2f}** to **{single['commit_energy_mae_kcal']:.2f} kcal** and mass MAE from "
        f"**{single['prior_mass_mae_g']:.2f}** to **{single['commit_mass_mae_g']:.2f} g**. "
        "A same-head swap cannot fire on a one-component plate, so the commit and the bound-code total are the same number there. "
        "Those plates are also the largest share of the energy gap versus Mass-only.",
        "",
        "Components that never received a mass write stay out of both totals. "
        f"The **{incomplete['n']}** plates that ended on RETRIEVE are in this comparison: their commit is exactly the components that were written, and the rollout does not invent the missing portions. "
        f"Those six plates are the whole **{published['mass_minus_mass_only_g']:+.4f} g** mass gap versus Mass-only "
        f"(contribution {incomplete['contribution_to_mean_mass_gap_g']:+.4f} g). "
        f"Their energy contribution to the +41.46 mean is {incomplete['contribution_to_mean_energy_gap_kcal']:+.2f} kcal.",
        "",
        "## Oracle label density",
        "",
        f"Label density times our grams has energy MAE **{oracle['ours_label_density_kcal']:.4f} kcal**. "
        f"The same label density times Mass-only grams has **{oracle['mass_only_label_density_kcal']:.4f} kcal**, and times Direct-VLM grams has **{oracle['direct_label_density_kcal']:.4f} kcal**. "
        f"Our saved kcal/g times the label grams has energy MAE **{oracle['ours_label_grams_kcal']:.4f} kcal**. "
        f"The bound-code kcal/g times the label grams has **{oracle['bound_label_grams_kcal']:.4f} kcal**. "
        f"The calorie head times the label grams has **{oracle['mass_only_label_grams_kcal']:.4f} kcal**.",
        "",
        f"At the cafe kcal/g, our grams land at {oracle['ours_label_density_kcal']:.2f} kcal MAE, "
        f"which is below the food-table commit ({published['ours_energy_mae_kcal']:.2f}) and below Direct-VLM ({published['direct_energy_mae_kcal']:.2f}). "
        f"Mass-only's scored energy MAE remains lower, at {published['mass_only_energy_mae_kcal']:.2f}. "
        f"Unweighted kcal/g absolute error is **{channels['commit']['density_mae_kcal_per_g']:.3f} kcal/g** for the commit and "
        f"**{channels['mass_only']['density_mae_kcal_per_g']:.3f} kcal/g** for the calorie head. "
        f"Mean cancellation is {channels['commit']['mean_cancellation_kcal']:.1f} kcal for the commit and "
        f"{channels['mass_only']['mean_cancellation_kcal']:.1f} kcal for the head. "
        f"The head's mean signed calorie error is {channels['mass_only']['mean_signed_energy_kcal']:+.1f} kcal "
        f"({channels['mass_only']['mean_mass_channel_kcal']:+.1f} from grams, {channels['mass_only']['mean_density_channel_kcal']:+.1f} from kcal/g). "
        f"The commit's mean signed error is {channels['commit']['mean_signed_energy_kcal']:+.1f} kcal "
        f"({channels['commit']['mean_mass_channel_kcal']:+.1f} from grams, {channels['commit']['mean_density_channel_kcal']:+.1f} from kcal/g).",
        "",
        "## Density swap on and off",
        "",
        f"The frozen same-head swap fired on **{swap['n_applied']}** plates and left **{swap['n_not_applied']}** unchanged. "
        f"With those swapped codes put back, full-set energy MAE is **{swap['full_swap_off_energy_mae_kcal']:.4f} kcal**. "
        f"With the swap left on, it is **{swap['full_swap_on_energy_mae_kcal']:.4f} kcal**. "
        f"The swap changes full-set energy MAE by **{swap['full_energy_mae_change_kcal']:+.4f} kcal**. Mass is unchanged. "
        f"The gap versus Mass-only is **{swap['full_swap_off_energy_mae_kcal'] - published['mass_only_energy_mae_kcal']:+.4f} kcal** with the swap off and "
        f"**{published['energy_minus_mass_only_kcal']:+.4f} kcal** with the swap on.",
        "",
        f"On the 106 plates, energy MAE is {swap['swapped_energy_mae_off_kcal']:.4f} with the swap off and "
        f"**{swap['swapped_energy_mae_on_kcal']:.4f}** with it on. "
        f"Mass-only on the same 106 is {swap['swapped_mass_only_energy_mae_kcal']:.4f}. Direct-VLM is {swap['swapped_direct_energy_mae_kcal']:.4f}. "
        f"The swap moves all **{swap['n_closer_to_calorie_head']}** food-table totals closer to the calorie head "
        f"(mean absolute gap {swap['mean_abs_bound_minus_head_kcal']:.2f} kcal to {swap['mean_abs_commit_minus_head_kcal']:.2f} kcal). "
        f"It moves **{swap['n_closer_to_label']}** closer to the cafe label and **{swap['n_farther_from_label']}** farther. "
        f"Mean signed energy change on those plates is {swap['mean_energy_on_minus_off_kcal']:+.2f} kcal. "
        "Changed-code counts on the 106: "
        + ", ".join(
            f"{count} {'code' if str(count) == '1' else 'codes'} on {n} plates"
            for count, n in swap["n_changed_hist"].items()
        )
        + ".",
        "",
        "This swap is the #51 density/kcal heuristic. It chooses the candidate whose food-table energy is closer to the calorie head. "
        "On the development food-code audit the automatic Exact count after a swap was 0. "
        "A lower calorie error here is a score against the ingredient-calorie sum. It is not an identity correction.",
        "",
        f"The 122 plates with no swap contribute **{plain['contribution_to_mean_energy_gap_kcal']:+.2f} kcal** of the +41.46 mean. "
        f"The 106 swapped plates contribute **{swapped['contribution_to_mean_energy_gap_kcal']:+.2f} kcal**.",
        "",
        "## Where the energy gap sits",
        "",
        f"Energy absolute error is lower than Mass-only on **{versus['energy_closer']}** plates and higher on **{versus['energy_farther']}**. "
        f"The median plate-level gap is **{boot_em['median']:+.2f} kcal**. "
        f"Quantiles of commit minus Mass-only energy absolute error (kcal): "
        + ", ".join(f"{name} {value:+.1f}" for name, value in quant.items())
        + ". "
        f"The positive gaps sum to {summary['positive_energy_gap_vs_mass_only_sum_kcal']:.0f} kcal, and the 10 largest are "
        f"{100 * summary['top10_share_of_positive_energy_gap']:.1f}% of that sum.",
        "",
        f"Median absolute errors are {summary['median_abs']['commit_energy_kcal']:.2f} versus "
        f"{summary['median_abs']['mass_only_energy_kcal']:.2f} kcal for energy, and "
        f"{summary['median_abs']['commit_mass_g']:.2f} versus {summary['median_abs']['mass_only_mass_g']:.2f} g for mass. "
        f"Mean |commit − calorie head| is {summary['mean_abs_commit_minus_calorie_head_kcal']:.2f} kcal.",
        "",
        "Energy MAE by slice. The last column is that slice's addition to the 228-plate mean energy gap versus Mass-only.",
        "",
        "| Slice | n | Commit | Swap off | Prior grams | Oracle density × our grams | Mass-only | Direct-VLM | Addition to +41.46 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        energy_line(slices["all"]),
        energy_line(complete),
        energy_line(incomplete),
        energy_line(single),
        energy_line(multi),
        energy_line(slices["complete_multi"]),
        energy_line(swapped),
        energy_line(plain),
        "",
        f"One-component plates add {single['contribution_to_mean_energy_gap_kcal']:+.2f} kcal. "
        f"Multi-component plates add {multi['contribution_to_mean_energy_gap_kcal']:+.2f} kcal. "
        f"On complete multi-component plates the commit energy MAE is {slices['complete_multi']['commit_energy_mae_kcal']:.2f}, "
        f"swap-off is {slices['complete_multi']['swap_off_energy_mae_kcal']:.2f}, "
        f"Mass-only is {slices['complete_multi']['mass_only_energy_mae_kcal']:.2f}, and "
        f"Direct-VLM is {slices['complete_multi']['direct_energy_mae_kcal']:.2f}, with mass MAE "
        f"{slices['complete_multi']['commit_mass_mae_g']:.2f} g against Direct-VLM {slices['complete_multi']['direct_mass_mae_g']:.2f} g.",
        "",
        "![Energy MAE under the three accounting cuts and the two frozen baselines](fig_energy_accounting.svg)",
        "",
        "## Gram term and food-density term",
        "",
        "| | Mass MAE (g) | Energy MAE (kcal) | kcal/g MAE | Mean \\|gram term\\| | Mean \\|kcal/g term\\| | Mean cancellation | Mean signed calorie error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        channel_line("Commit, swap on", channels["commit"]),
        channel_line("Bound codes, swap off", channels["swap_off"]),
        channel_line("100 g grid, bound codes", channels["prior_grams"]),
        channel_line("Mass-only calorie head", channels["mass_only"]),
        channel_line("Direct-VLM", channels["direct"]),
        "",
        f"Gram error and kcal/g error have opposite signs on {100 * channels['commit']['opposite_sign_rate']:.1f}% of plates for the commit, "
        f"{100 * channels['mass_only']['opposite_sign_rate']:.1f}% for Mass-only, and "
        f"{100 * channels['direct']['opposite_sign_rate']:.1f}% for Direct-VLM.",
        "",
        "![Gram term, density term, calorie MAE, and cancellation](fig_error_channels.svg)",
        "",
        "## Readings",
        "",
        "These plates are the largest one-component energy-absolute-error gaps versus Mass-only, plus the swap that most reduces label absolute error and the swap that most increases it. They illustrate the density assignment. They were not used to choose a code or a gram.",
        "",
        *plate_block("one component, largest energy gap versus Mass-only", worst[0]),
        *plate_block("one component", worst[1]),
        *plate_block("one component", worst[2]),
        *plate_block("swap, largest drop in label absolute error", helped),
        *plate_block("swap, largest rise in label absolute error", hurt),
        "",
        "## What stays in place",
        "",
        "The #63 table stands. Mass MAE is lower than Direct-VLM by 22.7973 g, and that gap's bootstrap interval lies below zero. "
        "Energy MAE is higher than Direct-VLM by 11.2337 kcal, and that gap's interval contains both signs. "
        "Energy MAE is higher than Mass-only by 41.4624 kcal, and that gap's interval lies above zero. "
        "On 222 plates the gram totals match, so that energy gap is the FNDDS product versus the calorie head. "
        "On the full 228, the mass rewrite with codes held fixed lowers both mass MAE and energy MAE relative to the 100 g grid. "
        f"On the {single['n']} one-component plates that rewrite raises energy MAE from {single['prior_energy_mae_kcal']:.2f} to {single['commit_energy_mae_kcal']:.2f}. "
        "The stored swap lowers full-set energy MAE by a further 8.3566 kcal. "
        "The remaining gap versus Mass-only is concentrated on the one-component plates, which the swap does not touch.",
        "",
        "This note does not start a mass retrain, a Q retrain, a τ refit, a retrieval re-rank, gate ④ real tool cost, gate ⑤, gate ⑥, or gate ⑦. "
        "There is no RL. The label-density and label-gram totals are arithmetic. They are not proposed commits. "
        f"Direct-VLM in the frozen rows used `placement_max_memory={summary['direct_vlm_placement_max_memory']}` on "
        f"{summary['n_direct_rows_at_8GiB']} / {N_PAIRED} plates. Those rows were not rerun. "
        "The frozen summary records 0 hollow RE_PORTION writes. "
        "#51 remains a density/kcal heuristic.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/heldout_energy_audit.py",
        "```",
        "",
        "Reads `results/heldout_plan/heldout_dish_ids.txt`, `results/heldout_eval/scores.jsonl`, `rollout.jsonl`, `direct_vlm.jsonl`, `mass_only.jsonl`, and `summary.json`, plus the cafe2 metadata for the label check. "
        "Input sha256 prefixes: "
        + ", ".join(f"{name} `{digest[:16]}`" for name, digest in summary["input_sha256"].items())
        + ".",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_plain(summary: Mapping[str, Any]) -> str:
    text = render_report(summary)
    kept = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if line.startswith("!["):
            continue
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n"


def accounting_svg(path: Path, summary: Mapping[str, Any]) -> None:
    oracle = summary["oracle"]
    rewrite = summary["rewrite_codes_fixed"]
    published = summary["published"]
    labels = [
        "100 g grid\nbound codes",
        "Mass grams\nbound codes",
        "Swap on\ncommit",
        "Oracle density\nx our grams",
        "Mass-only",
        "Direct-VLM",
    ]
    values = [
        rewrite["prior_energy_mae_kcal"],
        rewrite["bound_energy_mae_kcal"],
        published["ours_energy_mae_kcal"],
        oracle["ours_label_density_kcal"],
        published["mass_only_energy_mae_kcal"],
        published["direct_energy_mae_kcal"],
    ]
    colors = [COLOR_PRIOR, COLOR_BOUND, COLOR_COMMIT, COLOR_ORACLE, COLOR_MASS, COLOR_DIRECT]
    width, height = 860, 460
    left, right, top, bottom = 72, 24, 56, 88
    peak = max(values) * 1.18
    plot_w = width - left - right
    plot_h = height - top - bottom
    gap = 18
    bar_w = (plot_w - gap * (len(values) + 1)) / len(values)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="72" y="28" font-family="DejaVu Sans, sans-serif" font-size="16" fill="#111827">Cafe2 energy MAE, n=228</text>',
        '<text x="72" y="46" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#4b5563">Accounting rows use the stored grams and codes. Oracle density is arithmetic.</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
    ]
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + gap + index * (bar_w + gap)
        bar_h = value / peak * plot_h
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">{value:.1f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{height - bottom + 22}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">'
        )
        for line_index, line in enumerate(label.split("\n")):
            dy = 0 if line_index == 0 else 15
            parts.append(f'<tspan x="{x + bar_w / 2:.1f}" dy="{dy}">{xml(line)}</tspan>')
        parts.append("</text>")
    parts.append(
        f'<text x="22" y="{(top + top + plot_h) / 2:.0f}" text-anchor="middle" transform="rotate(-90 22 {(top + top + plot_h) / 2:.0f})" font-family="DejaVu Sans, sans-serif" font-size="13" fill="#111827">kcal</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def channel_svg(path: Path, summary: Mapping[str, Any]) -> None:
    labels = ["|gram term|", "|kcal/g term|", "calorie MAE", "cancellation"]
    series = [
        ("Commit", COLOR_COMMIT, summary["channels"]["commit"]),
        ("Mass-only", COLOR_MASS, summary["channels"]["mass_only"]),
        ("Direct-VLM", COLOR_DIRECT, summary["channels"]["direct"]),
    ]
    keys = (
        "mean_abs_mass_channel_kcal",
        "mean_abs_density_channel_kcal",
        "energy_mae_kcal",
        "mean_cancellation_kcal",
    )
    width, height = 820, 480
    left, right, top, bottom = 64, 24, 64, 88
    peak = max(block[key] for _name, _color, block in series for key in keys) * 1.22
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / len(labels)
    bar_w = 22
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="64" y="28" font-family="DejaVu Sans, sans-serif" font-size="16" fill="#111827">Where the calorie error comes from</text>',
        '<text x="64" y="46" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#4b5563">Means over 228 plates. Cancellation is |gram term| + |kcal/g term| − calorie absolute error.</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
    ]
    for index, label in enumerate(labels):
        cx = left + (index + 0.5) * group_w
        for series_index, (_name, color, block) in enumerate(series):
            value = float(block[keys[index]])
            offset = (series_index - 1) * (bar_w + 4)
            bar_h = value / peak * plot_h
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{cx + offset - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}"/>')
            parts.append(
                f'<text x="{cx + offset:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="11" fill="#111827">{value:.0f}</text>'
            )
        parts.append(
            f'<text x="{cx:.1f}" y="{height - bottom + 24}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">{xml(label)}</text>'
        )
    legend_y = height - 28
    legend_x = 64
    for name, color, _block in series:
        parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 18}" y="{legend_y + 11}" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">{xml(name)}</text>'
        )
        legend_x += 180
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def public_record(row: Mapping[str, Any]) -> dict[str, Any]:
    label_e = float(row["label_energy_kcal"])
    return {
        "dish_id": row["dish_id"],
        "label_energy_kcal": row["label_energy_kcal"],
        "label_mass_g": row["label_mass_g"],
        "label_density_kcal_per_g": row["label_density_kcal_per_g"],
        "ours_energy_kcal": row["ours_energy_kcal"],
        "ours_mass_g": row["ours_mass_g"],
        "direct_energy_kcal": row["direct_energy_kcal"],
        "direct_mass_g": row["direct_mass_g"],
        "mass_only_energy_kcal": row["mass_only_energy_kcal"],
        "mass_only_mass_g": row["mass_only_mass_g"],
        "calorie_head_kcal": row["calorie_head_kcal"],
        "bound_energy_kcal": row["bound_energy_kcal"],
        "prior_energy_kcal": row["prior_energy_kcal"],
        "prior_mass_g": row["prior_mass_g"],
        "swap_off_energy_kcal": row["swap_off_energy_kcal"],
        "oracle_label_density_kcal": row["oracle_density_kcal"],
        "oracle_label_grams_kcal": row["oracle_grams_kcal"],
        "swap_applied": row["swap_applied"],
        "n_changed": row["n_changed"],
        "swap_reason": row["swap_reason"],
        "changes": row["changes"],
        "n_hypotheses": row["n_hypotheses"],
        "n_mass_writes": row["n_mass_writes"],
        "plate_complete": row["plate_complete"],
        "decision_action": row["decision_action"],
        "energy_ae_commit_kcal": abs(row["ours_energy_kcal"] - label_e),
        "energy_ae_swap_off_kcal": abs(row["swap_off_energy_kcal"] - label_e),
        "energy_ae_bound_kcal": abs(row["bound_energy_kcal"] - label_e),
        "energy_ae_prior_kcal": abs(row["prior_energy_kcal"] - label_e),
        "energy_ae_oracle_density_kcal": abs(row["oracle_density_kcal"] - label_e),
        "energy_ae_mass_only_kcal": abs(row["mass_only_energy_kcal"] - label_e),
        "energy_ae_direct_kcal": abs(row["direct_energy_kcal"] - label_e),
        "mass_ae_commit_g": abs(row["ours_mass_g"] - row["label_mass_g"]),
        "mass_ae_prior_g": abs(row["prior_mass_g"] - row["label_mass_g"]),
        "mass_ae_mass_only_g": abs(row["mass_only_mass_g"] - row["label_mass_g"]),
        "writes": [
            {
                "code": write["code"],
                "name": write["name"],
                "old_g": write["old_g"],
                "new_g": write["new_g"],
                "kcal_per_100g": write["kcal_per_100g"],
                "old_energy_kcal": write["old_energy"],
                "new_energy_kcal": write["new_energy"],
            }
            for write in row["writes"]
        ],
    }


def write_outputs(payload: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    assert_output(OUT / "report.md")
    OUT.mkdir(parents=True, exist_ok=True)
    report = render_report(summary)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    (OUT / "report.txt").write_text(render_plain(summary), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "dishes.jsonl").open("w", encoding="utf-8") as handle:
        for row in payload["records"]:
            handle.write(json.dumps(public_record(row), ensure_ascii=False) + "\n")
    with (OUT / "cuts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slice",
                "n",
                "commit_energy_mae_kcal",
                "swap_off_energy_mae_kcal",
                "prior_energy_mae_kcal",
                "oracle_density_energy_mae_kcal",
                "mass_only_energy_mae_kcal",
                "direct_energy_mae_kcal",
                "commit_mass_mae_g",
                "prior_mass_mae_g",
                "mass_only_mass_mae_g",
                "direct_mass_mae_g",
                "energy_gap_vs_mass_only_kcal",
                "contribution_to_mean_energy_gap_kcal",
            ],
        )
        writer.writeheader()
        for row in summary["slices"]:
            writer.writerow(
                {
                    "slice": row["name"],
                    "n": row["n"],
                    "commit_energy_mae_kcal": f"{row['commit_energy_mae_kcal']:.6f}",
                    "swap_off_energy_mae_kcal": f"{row['swap_off_energy_mae_kcal']:.6f}",
                    "prior_energy_mae_kcal": f"{row['prior_energy_mae_kcal']:.6f}",
                    "oracle_density_energy_mae_kcal": f"{row['oracle_density_energy_mae_kcal']:.6f}",
                    "mass_only_energy_mae_kcal": f"{row['mass_only_energy_mae_kcal']:.6f}",
                    "direct_energy_mae_kcal": f"{row['direct_energy_mae_kcal']:.6f}",
                    "commit_mass_mae_g": f"{row['commit_mass_mae_g']:.6f}",
                    "prior_mass_mae_g": f"{row['prior_mass_mae_g']:.6f}",
                    "mass_only_mass_mae_g": f"{row['mass_only_mass_mae_g']:.6f}",
                    "direct_mass_mae_g": f"{row['direct_mass_mae_g']:.6f}",
                    "energy_gap_vs_mass_only_kcal": f"{row['energy_gap_vs_mass_only_kcal']:.6f}",
                    "contribution_to_mean_energy_gap_kcal": f"{row['contribution_to_mean_energy_gap_kcal']:.6f}",
                }
            )
    accounting_svg(OUT / "fig_energy_accounting.svg", summary)
    channel_svg(OUT / "fig_error_channels.svg", summary)
    if sha256_file(STATUS) != payload["status_hash"]:
        raise EnergyAuditError("docs/STATUS.md changed during the audit")


def self_check() -> None:
    parts = channels(220.0, 200.0, 200.0, 100.0)
    if abs(parts["mass_channel_kcal"] - 110.0) > 1e-9 or abs(parts["density_channel_kcal"] + 90.0) > 1e-9:
        raise AssertionError(parts)
    if abs(parts["cancellation_kcal"] - 180.0) > 1e-9:
        raise AssertionError(parts)
    oracle = oracle_energy(220.0, 200.0, 200.0, 100.0)
    if abs(oracle["perfect_grams_kcal"] - 110.0) > 1e-9 or abs(oracle["perfect_density_kcal"] - 400.0) > 1e-9:
        raise AssertionError(oracle)
    try:
        channels(10.0, 0.0, 10.0, 5.0)
    except EnergyAuditError:
        pass
    else:
        raise AssertionError("zero mass should fail the density split")

    writes = [
        {"new_energy": 50.0, "old_energy": 100.0, "new_g": 50.0, "old_g": 100.0},
        {"new_energy": 100.0, "old_energy": 200.0, "new_g": 50.0, "old_g": 100.0},
    ]
    align = {"applied": True, "before_energy_kcal": 150.0, "after_energy_kcal": 80.0, "n_changed": 1}
    plate = reconstruct_plate(writes, 80.0, 100.0, align)
    if not plate["accounted"]:
        raise AssertionError(plate)
    if abs(plate["bound_energy_kcal"] - 150.0) > 1e-9 or abs(plate["prior_energy_kcal"] - 300.0) > 1e-9:
        raise AssertionError(plate)
    if abs(plate["prior_mass_g"] - 200.0) > 1e-9 or abs(plate["swap_off_energy_kcal"] - 150.0) > 1e-9:
        raise AssertionError(plate)
    if abs(plate["swap_on_energy_kcal"] - 80.0) > 1e-9:
        raise AssertionError(plate)
    held = reconstruct_plate(writes[:1], 50.0, 50.0, None)
    if abs(held["bound_energy_kcal"] - 50.0) > 1e-9 or abs(held["prior_energy_kcal"] - 100.0) > 1e-9:
        raise AssertionError(held)
    if held["applied"]:
        raise AssertionError(held)
    sample = bootstrap_mean(np.asarray([1.0, -1.0, 1.0, -1.0]), seed=0)
    if sample["n"] != 4 or sample["draws"] != BOOTSTRAP_DRAWS:
        raise AssertionError(sample)
    print("heldout energy-audit self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Cafe2 energy audit of the frozen held-out")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    payload = build_records()
    summary = aggregate(payload)
    write_outputs(payload, summary)
    published = summary["published"]
    print(
        f"paired n={summary['n_paired']} "
        f"ΔE vs mass-only {published['energy_minus_mass_only_kcal']:+.4f} kcal "
        f"ΔM vs mass-only {published['mass_minus_mass_only_g']:+.4f} g"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
