#!/usr/bin/env python3
"""Read-only audit: why P0d cut mass error but not energy error.

The paired full set is already in ``results/p0d``. This script does not train,
does not roll out the vision model, does not fit a threshold, and does not
write ``docs/STATUS.md``, ``results/p0d``, ``results/weak_steps``, ``results/qphi``,
``results/n5k_*``, or ``results/mass_estimator``.

Cafe labels score residuals. Two accounting rows replace either the saved
grams or the saved kcal/g with the label inside an identity. They are not
extra systems and they do not choose an action.

    python scripts/p0d_error_audit.py --self-check
    python scripts/p0d_error_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P0D = ROOT / "results" / "p0d"
OUT = ROOT / "results" / "audit"
DISHES = P0D / "same_budget_dishes.jsonl"
TABLE = P0D / "same_budget.json"
TRACE = P0D / "ours_new_rgb_test.jsonl"
TRACE_SUMMARY = P0D / "ours_new_rgb_test_summary.json"
REPORT_P0D = P0D / "report_same_budget.txt"
DIRECT_JSONL = ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl"
Q_WEIGHTS = ROOT / "results" / "qphi" / "q_phi.npz"
Q_CALIBRATION = ROOT / "results" / "qphi" / "calibration.json"

PUBLISHED_ENERGY_DELTA = 1.420968867326735
PUBLISHED_MASS_DELTA = -17.351335643564354
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
N_PAIRED = 505

# Quadrant colors: mass absolute error down is to the left.
COLOR_BOTH_BETTER = "#1f7a4d"
COLOR_MASS_ONLY = "#c2410c"
COLOR_ENERGY_ONLY = "#1d4e89"
COLOR_BOTH_WORSE = "#6b7280"
COLOR_DIRECT = "#1d4e89"
COLOR_OURS = "#c2410c"

MACRO_KEYS = ("protein_g", "carb_g", "fat_g")


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


def f4(value: float) -> str:
    return f"{value:.4f}"


def f2(value: float) -> str:
    return f"{value:.2f}"


def f1(value: float) -> str:
    return f"{value:.1f}"


def pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "n/a"
    return f"{100.0 * part / whole:.1f}%"


def xml(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


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


def median(values: np.ndarray) -> float:
    return float(np.median(values))


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_mean(values: np.ndarray, *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Percentile interval of the paired mean. Plates are drawn with replacement."""
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError("bootstrap needs a non-empty vector")
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


def channels(energy: float, mass: float, label_energy: float, label_mass: float) -> dict[str, float] | None:
    """Signed calorie error = predicted kcal/g × gram error + label grams × kcal/g error.

    The two terms add to ``energy - label_energy`` when both masses are positive.
    """
    if mass <= 0.0 or label_mass <= 0.0:
        return None
    density = energy / mass
    label_density = label_energy / label_mass
    mass_channel = density * (mass - label_mass)
    density_channel = label_mass * (density - label_density)
    signed = energy - label_energy
    if abs((mass_channel + density_channel) - signed) > 1e-6:
        raise RuntimeError("energy identity failed")
    return {
        "density_kcal_per_g": density,
        "label_density_kcal_per_g": label_density,
        "mass_channel_kcal": mass_channel,
        "density_channel_kcal": density_channel,
        "signed_energy_kcal": signed,
        "signed_mass_g": mass - label_mass,
        "cancellation_kcal": abs(mass_channel) + abs(density_channel) - abs(signed),
        "opposite_signs": float((mass - label_mass) * (density - label_density) < 0.0),
    }


def oracle_energy(energy: float, mass: float, label_energy: float, label_mass: float) -> dict[str, float] | None:
    """Label enters only as a multiplier. Neither row was a deployed commit."""
    parts = channels(energy, mass, label_energy, label_mass)
    if parts is None:
        return None
    return {
        "perfect_grams_kcal": parts["density_kcal_per_g"] * label_mass,
        "perfect_density_kcal": parts["label_density_kcal_per_g"] * mass,
    }


def _removed_tail(records: Sequence[Mapping[str, Any]], deltas: np.ndarray, order: np.ndarray) -> dict[str, Any]:
    """Plates whose removal brings the mean energy gap to ≤ 0. Descriptive only."""
    k = drop_until_mean_nonpositive(deltas)
    removed = []
    for index in order[:k]:
        row = records[int(index)]
        removed.append(
            {
                "dish_id": row["dish_id"],
                "energy_delta_ae_kcal": float(row["energy_delta_ae"]),
                "mass_delta_ae_g": float(row["mass_delta_ae"]),
                "label_energy_kcal": row["label_energy_kcal"],
                "label_mass_g": row["label_mass_g"],
                "ours_energy_kcal": row["ours_energy_kcal"],
                "ours_mass_g": row["ours_mass_g"],
                "direct_energy_kcal": row["direct_energy_kcal"],
                "direct_mass_g": row["direct_mass_g"],
                "foods": row["foods"],
                "plate_complete": row["plate_complete"],
                "decision_action": row["decision_action"],
            }
        )
    keep = np.ones(deltas.size, dtype=bool)
    if k:
        keep[order[:k]] = False
    return {
        "n_removed": k,
        "plates": removed,
        "remaining_n": int(np.sum(keep)),
        "remaining_mean_energy_delta_kcal": float(deltas[keep].mean()) if np.any(keep) else None,
        "removed_energy_delta_sum_kcal": float(deltas[order[:k]].sum()) if k else 0.0,
    }


def drop_until_mean_nonpositive(deltas: np.ndarray) -> int:
    """How many largest positive gaps must be removed before the mean is ≤ 0."""
    values = np.asarray(deltas, dtype=np.float64)
    if float(values.mean()) <= 0.0:
        return 0
    order = np.argsort(-values)
    running = float(values.sum())
    n = int(values.size)
    for removed, index in enumerate(order, start=1):
        running -= float(values[index])
        left = n - removed
        if left == 0 or running / left <= 0.0:
            return removed
    return n


def symmetric_trim_mean(values: np.ndarray, each_tail: float = 0.10) -> dict[str, float]:
    sample = np.sort(np.asarray(values, dtype=np.float64))
    cut = int(math.floor(sample.size * each_tail))
    kept = sample[cut : sample.size - cut] if cut else sample
    return {"n": float(kept.size), "mean": float(kept.mean()), "each_tail": each_tail}


def index_dishes(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), {})[str(row["dish_id"])] = dict(row)
    return by_method


def group_trace(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dish_id"]), []).append(dict(row))
    for steps in grouped.values():
        steps.sort(key=lambda step: int(step["step_i"]))
    return grouped


def component_write(step: Mapping[str, Any]) -> dict[str, Any] | None:
    if step.get("action_t") != "RE_PORTION":
        return None
    obs = step.get("obs_t") if isinstance(step.get("obs_t"), Mapping) else {}
    if obs.get("tool_failed") or obs.get("portion_src") != "mass_estimator":
        return None
    code = step.get("code_i")
    refs = (obs.get("nutrient_refs") or {}).get("fndds") if isinstance(obs.get("nutrient_refs"), Mapping) else None
    if not isinstance(refs, list):
        return {"ok": False, "reason": "no_fndds_refs", "step_i": int(step["step_i"])}
    matched = None
    for ref in refs:
        if isinstance(ref, Mapping) and str(ref.get("code")) == str(code) and ref.get("available") and ref.get("nutrients"):
            matched = ref
            break
    if matched is None:
        return {"ok": False, "reason": "code_not_in_refs", "step_i": int(step["step_i"]), "code": code}
    new_g = finite(obs.get("portion_g"))
    old_g = finite(obs.get("from_portion_g"))
    nutrients = matched.get("nutrients")
    if new_g is None or old_g is None or not isinstance(nutrients, Mapping) or new_g <= 0.0:
        return {"ok": False, "reason": "bad_portion", "step_i": int(step["step_i"])}
    scale = old_g / new_g
    per100 = matched.get("per_100g") if isinstance(matched.get("per_100g"), Mapping) else {}
    model = obs.get("nutrients_model") if isinstance(obs.get("nutrients_model"), Mapping) else {}
    return {
        "ok": True,
        "step_i": int(step["step_i"]),
        "code": str(code),
        "name": matched.get("name"),
        "new_g": new_g,
        "old_g": old_g,
        "kcal_per_100g": finite(per100.get("energy_kcal")),
        "new_energy": finite(nutrients.get("energy_kcal")),
        "new_mass": new_g,
        "old_energy": None if finite(nutrients.get("energy_kcal")) is None else finite(nutrients.get("energy_kcal")) * scale,
        "old_mass": old_g,
        "model_energy": finite(model.get("energy_kcal")),
        "model_mass": finite(obs.get("mass_g")),
        "clipped": bool(new_g <= 1.0 + 1e-6 or new_g >= 1500.0 - 1e-6),
    }


def trace_facts(steps: Sequence[Mapping[str, Any]], source_step_i: int) -> dict[str, Any]:
    source = next((step for step in steps if int(step["step_i"]) == int(source_step_i)), None)
    dish = source.get("nutri_dish") if source is not None else None
    writes = [component_write(step) for step in steps if int(step["step_i"]) < int(source_step_i)]
    writes = [row for row in writes if row is not None]
    ok_writes = [row for row in writes if row.get("ok")]
    n_hyp = finite((steps[0].get("obs_t") or {}).get("vlm_n_hypotheses")) if steps else None
    actions = [str(step.get("action_t")) for step in steps if int(step["step_i"]) <= int(source_step_i)]
    terminal = next((step for step in steps if step.get("action_t") in {"STOP", "ABSTAIN"}), steps[-1] if steps else None)
    terminal_obs = terminal.get("obs_t") if terminal is not None and isinstance(terminal.get("obs_t"), Mapping) else {}
    model_energy = [row["model_energy"] for row in ok_writes if row.get("model_energy") is not None]
    model_mass = [row["model_mass"] for row in ok_writes if row.get("model_mass") is not None]
    return {
        "source_energy": None if not isinstance(dish, Mapping) else finite(dish.get("energy_kcal")),
        "source_mass": None if not isinstance(dish, Mapping) else finite(dish.get("mass_g")),
        "source_macros": {
            key: (None if not isinstance(dish, Mapping) else finite(dish.get(key))) for key in MACRO_KEYS
        },
        "n_hypotheses": None if n_hyp is None else int(n_hyp),
        "n_mass_writes": len(ok_writes),
        "n_mass_write_failures": sum(1 for row in writes if not row.get("ok")),
        "actions_through_source": actions,
        "n_correct_code": sum(1 for action in actions if action == "CORRECT_CODE"),
        "decision_action_trace": None if terminal is None else terminal.get("action_t"),
        "n_kept": terminal_obs.get("n_kept"),
        "writes": ok_writes,
        "model_energy": None if not model_energy else float(model_energy[0]),
        "model_mass": None if not model_mass else float(model_mass[0]),
        "model_energy_span": None if len(model_energy) < 2 else float(max(model_energy) - min(model_energy)),
        "model_mass_span": None if len(model_mass) < 2 else float(max(model_mass) - min(model_mass)),
        "portion_sum_g": float(sum(row["new_g"] for row in ok_writes)),
        "n_clipped_portions": sum(1 for row in ok_writes if row["clipped"]),
    }


def load_direct_macros(path: Path) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for row in load_jsonl(path):
        label = row.get("label") if isinstance(row.get("label"), Mapping) else {}
        guess = (row.get("b0") or {}).get("n") if isinstance(row.get("b0"), Mapping) else None
        guess = guess if isinstance(guess, Mapping) else {}
        out[str(row["dish_id"])] = {
            "label_protein_g": finite(label.get("protein_g")),
            "label_carb_g": finite(label.get("carb_g")),
            "label_fat_g": finite(label.get("fat_g")),
            "direct_protein_g": finite(guess.get("protein_g")),
            "direct_carb_g": finite(guess.get("carb_g")),
            "direct_fat_g": finite(guess.get("fat_g")),
            "direct_energy_kcal": finite(guess.get("energy_kcal")),
            "direct_mass_g": finite(guess.get("mass_g")),
        }
    return out


def replay_selector(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Re-score the saved steps with the frozen Q weights and the saved fence.

    The trace does not store whether a step came from Q/Cost or from the rule
    utility. A step counts only when the frozen selector reproduces ``action_t``.
    """
    from vdr.controller.closed_loop_data import _belief_at_step
    from vdr.controller.q_calibration import calibrated_decision
    from vdr.controller.q_calibration import load_calibrated_policy
    from vdr.controller.q_state import build_z

    policy = load_calibrated_policy(Q_WEIGHTS, Q_CALIBRATION)
    margin = float(policy.terminal_margin)
    n_steps = 0
    n_match = 0
    mismatch_actions: dict[str, int] = {}
    by_dish_terminal: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, int] = {}
    rule_steps_by_dish: dict[str, int] = {}
    failures = 0
    for dish_id, steps in grouped.items():
        prev = None
        rule_steps_by_dish[dish_id] = 0
        for step in steps:
            n_steps += 1
            try:
                vector = build_z(step, prev, _belief_at_step(int(step["step_i"])))
                row_q = policy.model.predict_q(vector.reshape(1, -1))[0]
                values = {
                    action_name: float(row_q[index])
                    for index, action_name in enumerate(policy.model.actions)
                }
                action, source = calibrated_decision(values, vector, margin)
            except Exception:
                failures += 1
                prev = step
                continue
            logged = str(step.get("action_t"))
            matched = action == logged
            n_match += int(matched)
            if not matched:
                mismatch_actions[logged] = mismatch_actions.get(logged, 0) + 1
            if step.get("action_t") in {"STOP", "ABSTAIN"} or step is steps[-1]:
                if dish_id not in by_dish_terminal or step.get("action_t") in {"STOP", "ABSTAIN"}:
                    by_dish_terminal[dish_id] = {
                        "source": source if matched else "replay_mismatch",
                        "replay_action": action,
                        "logged_action": logged,
                        "matched": matched,
                    }
            source_counts[source] = source_counts.get(source, 0) + int(matched)
            if matched and source == "rule_utility":
                rule_steps_by_dish[dish_id] += 1
            prev = step
    return {
        "n_steps": n_steps,
        "n_match": n_match,
        "n_failures": failures,
        "match_rate": None if n_steps == 0 else n_match / n_steps,
        "mismatch_logged_actions": mismatch_actions,
        "matched_step_sources": source_counts,
        "terminal": by_dish_terminal,
        "rule_steps_by_dish": rule_steps_by_dish,
        "tau": margin,
        "weights_sha256": sha256_file(Q_WEIGHTS),
    }


def macro_block(
    ours: np.ndarray,
    direct: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    keep = np.isfinite(ours) & np.isfinite(direct) & np.isfinite(labels)
    if not np.any(keep):
        return {"n": 0}
    return {
        "n": int(np.sum(keep)),
        "ours_mae": mae(ours[keep] - labels[keep]),
        "direct_mae": mae(direct[keep] - labels[keep]),
        "delta_ours_minus_direct": mae(ours[keep] - labels[keep]) - mae(direct[keep] - labels[keep]),
        "ours_bias": float(np.mean(ours[keep] - labels[keep])),
        "direct_bias": float(np.mean(direct[keep] - labels[keep])),
    }


def quadrant_name(mass_delta: float, energy_delta: float) -> str:
    mass_down = mass_delta < 0.0
    energy_down = energy_delta < 0.0
    if mass_delta == 0.0 or energy_delta == 0.0:
        return "tie"
    if mass_down and energy_down:
        return "mass_down_energy_down"
    if mass_down and not energy_down:
        return "mass_down_energy_up"
    if energy_down:
        return "mass_up_energy_down"
    return "mass_up_energy_up"


QUADRANT_LABELS = {
    "mass_down_energy_down": "Mass and energy both closer",
    "mass_down_energy_up": "Mass closer, energy farther",
    "mass_up_energy_down": "Mass farther, energy closer",
    "mass_up_energy_up": "Mass and energy both farther",
    "tie": "Tie on mass or energy",
}


def summarize_mask(name: str, mask: np.ndarray, energy_delta: np.ndarray, mass_delta: np.ndarray) -> dict[str, Any]:
    n_all = int(energy_delta.size)
    chosen = np.asarray(mask, dtype=bool)
    n = int(np.sum(chosen))
    if n == 0:
        return {"name": name, "n": 0, "share_of_plates": 0.0, "contribution_to_mean_energy_kcal": 0.0}
    return {
        "name": name,
        "label": QUADRANT_LABELS.get(name, name),
        "n": n,
        "share_of_plates": n / n_all,
        "mean_energy_delta_kcal": float(energy_delta[chosen].mean()),
        "mean_mass_delta_g": float(mass_delta[chosen].mean()),
        "contribution_to_mean_energy_kcal": float(energy_delta[chosen].sum() / n_all),
        "contribution_to_mean_mass_g": float(mass_delta[chosen].sum() / n_all),
    }


def slice_metrics(mask: np.ndarray, energy_ae: dict[str, np.ndarray], mass_ae: dict[str, np.ndarray]) -> dict[str, Any]:
    chosen = np.asarray(mask, dtype=bool)
    n = int(np.sum(chosen))
    if n == 0:
        return {"n": 0}
    out: dict[str, Any] = {"n": n}
    for key in energy_ae:
        out[f"{key}_energy_mae"] = mae(energy_ae[key][chosen])
        out[f"{key}_mass_mae"] = mae(mass_ae[key][chosen])
    out["energy_delta_ours_minus_direct"] = out["ours_energy_mae"] - out["direct_energy_mae"]
    out["mass_delta_ours_minus_direct"] = out["ours_mass_mae"] - out["direct_mass_mae"]
    return out


def food_phrase(writes: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for row in writes:
        name = row.get("name") or row.get("code")
        kcal = row.get("kcal_per_100g")
        kcal_text = "unknown kcal/100 g" if kcal is None else f"{kcal:.1f} kcal/100 g"
        parts.append(f"{name} ({row['new_g']:.1f} g, was {row['old_g']:.1f} g, {kcal_text})")
    return "; ".join(parts)


def pick_examples(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pool = [
        row
        for row in rows
        if row["quadrant"] == "mass_down_energy_up" and row["mass_delta_ae"] <= -10.0
    ]
    pool.sort(key=lambda row: row["energy_delta_ae"], reverse=True)
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()

    def take(predicate: Any, fallback: bool = False) -> None:
        for row in pool:
            if row["dish_id"] in seen:
                continue
            if predicate(row):
                chosen.append(row)
                seen.add(row["dish_id"])
                return
        if fallback:
            for row in pool:
                if row["dish_id"] not in seen:
                    chosen.append(row)
                    seen.add(row["dish_id"])
                    return

    take(lambda row: row.get("density_gap_ours", 0.0) > row.get("density_gap_direct", 0.0) and row["ours_density"] < row["label_density"])
    take(lambda row: row.get("density_gap_ours", 0.0) > row.get("density_gap_direct", 0.0) and row["ours_density"] > row["label_density"])
    take(lambda row: row.get("n_hypotheses") == 1 and row.get("plate_complete"))
    take(lambda row: (row.get("n_hypotheses") or 0) > 1)
    while len(chosen) < 4:
        before = len(chosen)
        take(lambda row: True, fallback=True)
        if len(chosen) == before:
            break
    return chosen[:4]


def head_slice(records: Sequence[Mapping[str, Any]], mask: np.ndarray) -> dict[str, Any]:
    """Score the stored dish-level head on plates where a mass write saved it."""
    model_e = optional_column(records, "model_energy_kcal")
    model_m = optional_column(records, "model_mass_g")
    keep = np.asarray(mask, dtype=bool) & np.isfinite(model_e) & np.isfinite(model_m)
    if not np.any(keep):
        return {"n": 0}
    label_e = column(records, "label_energy_kcal")
    label_m = column(records, "label_mass_g")
    return {
        "n": int(np.sum(keep)),
        "head_energy_mae": mae(model_e[keep] - label_e[keep]),
        "committed_energy_mae": mae(column(records, "ours_energy_kcal")[keep] - label_e[keep]),
        "direct_energy_mae": mae(column(records, "direct_energy_kcal")[keep] - label_e[keep]),
        "head_mass_mae": mae(model_m[keep] - label_m[keep]),
        "committed_mass_mae": mae(column(records, "ours_mass_g")[keep] - label_m[keep]),
        "direct_mass_mae": mae(column(records, "direct_mass_g")[keep] - label_m[keep]),
    }


def channel_summary(prefix: str, mass_ch: np.ndarray, dens_ch: np.ndarray, signed: np.ndarray, cancel: np.ndarray, opposite: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean_signed_energy_kcal": float(np.mean(signed)),
        f"{prefix}_mean_mass_channel_kcal": float(np.mean(mass_ch)),
        f"{prefix}_mean_density_channel_kcal": float(np.mean(dens_ch)),
        f"{prefix}_mean_abs_mass_channel_kcal": float(np.mean(np.abs(mass_ch))),
        f"{prefix}_mean_abs_density_channel_kcal": float(np.mean(np.abs(dens_ch))),
        f"{prefix}_mean_cancellation_kcal": float(np.mean(cancel)),
        f"{prefix}_opposite_sign_rate": float(np.mean(opposite)),
        f"{prefix}_energy_mae_kcal": float(np.mean(np.abs(signed))),
    }


def build_records() -> dict[str, Any]:
    dish_rows = load_jsonl(DISHES)
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    by_method = index_dishes(dish_rows)
    published_ids = [str(dish_id) for dish_id in table["comparison"]["paired_dish_ids"]]
    if len(published_ids) != N_PAIRED or len(set(published_ids)) != N_PAIRED:
        raise RuntimeError(f"paired id list has {len(published_ids)} rows, expected {N_PAIRED}")
    for method in ("ours_new", "direct_vlm", "ours_old"):
        missing = [dish_id for dish_id in published_ids if dish_id not in by_method[method]]
        if missing:
            raise RuntimeError(f"{method} is missing {len(missing)} paired dishes")
    traces = group_trace(load_jsonl(TRACE))
    macros = load_direct_macros(DIRECT_JSONL)
    replay = replay_selector(traces)

    records: list[dict[str, Any]] = []
    for dish_id in published_ids:
        ours = by_method["ours_new"][dish_id]
        direct = by_method["direct_vlm"][dish_id]
        closed = by_method["ours_old"][dish_id]
        if not (ours["paired"] and direct["paired"] and closed["paired"]):
            raise RuntimeError(f"{dish_id} is not paired in every scored method")
        facts = trace_facts(traces[dish_id], int(ours["source_step_i"]))
        label_e = float(ours["label_energy_kcal"])
        label_m = float(ours["label_mass_g"])
        if abs(label_e - float(direct["label_energy_kcal"])) > 1e-6 or abs(label_m - float(direct["label_mass_g"])) > 1e-6:
            raise RuntimeError(f"{dish_id} labels disagree across methods")
        ours_e = float(ours["energy_kcal"])
        ours_m = float(ours["mass_g"])
        direct_e = float(direct["energy_kcal"])
        direct_m = float(direct["mass_g"])
        closed_e = float(closed["energy_kcal"])
        closed_m = float(closed["mass_g"])
        ours_parts = channels(ours_e, ours_m, label_e, label_m)
        direct_parts = channels(direct_e, direct_m, label_e, label_m)
        closed_parts = channels(closed_e, closed_m, label_e, label_m)
        ours_oracle = oracle_energy(ours_e, ours_m, label_e, label_m)
        direct_oracle = oracle_energy(direct_e, direct_m, label_e, label_m)
        aligned = (
            facts["source_energy"] is not None
            and abs(facts["source_energy"] - ours_e) <= 0.02
            and facts["source_mass"] is not None
            and abs(facts["source_mass"] - ours_m) <= 0.02
        )
        new_energy = sum(row["new_energy"] for row in facts["writes"] if row["new_energy"] is not None)
        new_mass = sum(row["new_mass"] for row in facts["writes"])
        old_energy = sum(row["old_energy"] for row in facts["writes"] if row["old_energy"] is not None)
        old_mass = sum(row["old_mass"] for row in facts["writes"])
        rewrite_gap_e = None if not aligned else ours_e - new_energy
        rewrite_gap_m = None if not aligned else ours_m - new_mass
        rewrite_ok = (
            aligned
            and facts["n_mass_write_failures"] == 0
            and rewrite_gap_e is not None
            and abs(rewrite_gap_e) <= 0.05
            and rewrite_gap_m is not None
            and abs(rewrite_gap_m) <= 0.05
        )
        grid_energy = None if not rewrite_ok else ours_e - new_energy + old_energy
        grid_mass = None if not rewrite_ok else ours_m - new_mass + old_mass
        macro = macros.get(dish_id, {})
        if macro.get("direct_energy_kcal") is not None and abs(float(macro["direct_energy_kcal"]) - direct_e) > 1e-3:
            raise RuntimeError(f"{dish_id} direct energy in the frozen file disagrees with the paired table")
        terminal = replay["terminal"].get(dish_id, {})
        n_hyp = facts["n_hypotheses"]
        plate_complete = bool(n_hyp is not None and n_hyp > 0 and facts["n_mass_writes"] == n_hyp)
        energy_delta = abs(ours_e - label_e) - abs(direct_e - label_e)
        mass_delta = abs(ours_m - label_m) - abs(direct_m - label_m)
        record = {
            "dish_id": dish_id,
            "label_energy_kcal": label_e,
            "label_mass_g": label_m,
            "label_density": None if ours_parts is None else ours_parts["label_density_kcal_per_g"],
            "ours_energy_kcal": ours_e,
            "ours_mass_g": ours_m,
            "direct_energy_kcal": direct_e,
            "direct_mass_g": direct_m,
            "closed_energy_kcal": closed_e,
            "closed_mass_g": closed_m,
            "ours_energy_ae": abs(ours_e - label_e),
            "direct_energy_ae": abs(direct_e - label_e),
            "closed_energy_ae": abs(closed_e - label_e),
            "ours_mass_ae": abs(ours_m - label_m),
            "direct_mass_ae": abs(direct_m - label_m),
            "closed_mass_ae": abs(closed_m - label_m),
            "energy_delta_ae": energy_delta,
            "mass_delta_ae": mass_delta,
            "quadrant": quadrant_name(mass_delta, energy_delta),
            "decision_action": ours["decision_action"],
            "tool_count": int(ours["tool_count"]),
            "source_step_i": int(ours["source_step_i"]),
            "n_hypotheses": n_hyp,
            "n_mass_writes": facts["n_mass_writes"],
            "plate_complete": plate_complete,
            "n_correct_code": facts["n_correct_code"],
            "trace_aligned": aligned,
            "rewrite_accounted": rewrite_ok,
            "grid_energy_kcal": grid_energy,
            "grid_mass_g": grid_mass,
            "foods": food_phrase(facts["writes"]),
            "model_energy_kcal": facts["model_energy"],
            "model_mass_g": facts["model_mass"],
            "model_energy_span": facts["model_energy_span"],
            "model_mass_span": facts["model_mass_span"],
            "portion_sum_g": facts["portion_sum_g"],
            "n_clipped_portions": facts["n_clipped_portions"],
            "selector_source": terminal.get("source"),
            "selector_replay_matched": bool(terminal.get("matched")),
            "n_rule_utility_steps": int(replay["rule_steps_by_dish"].get(dish_id, 0)),
            "label_protein_g": macro.get("label_protein_g"),
            "label_carb_g": macro.get("label_carb_g"),
            "label_fat_g": macro.get("label_fat_g"),
            "direct_protein_g": macro.get("direct_protein_g"),
            "direct_carb_g": macro.get("direct_carb_g"),
            "direct_fat_g": macro.get("direct_fat_g"),
            "ours_protein_g": facts["source_macros"]["protein_g"],
            "ours_carb_g": facts["source_macros"]["carb_g"],
            "ours_fat_g": facts["source_macros"]["fat_g"],
        }
        for prefix, parts, oracle in (
            ("ours", ours_parts, ours_oracle),
            ("direct", direct_parts, direct_oracle),
            ("closed", closed_parts, None),
        ):
            if parts is None:
                record[f"{prefix}_density"] = None
                record[f"{prefix}_mass_channel"] = None
                record[f"{prefix}_density_channel"] = None
                record[f"{prefix}_cancellation"] = None
                record[f"{prefix}_opposite"] = None
                record[f"{prefix}_signed_energy"] = None
                record[f"{prefix}_signed_mass"] = None
            else:
                record[f"{prefix}_density"] = parts["density_kcal_per_g"]
                record[f"{prefix}_mass_channel"] = parts["mass_channel_kcal"]
                record[f"{prefix}_density_channel"] = parts["density_channel_kcal"]
                record[f"{prefix}_cancellation"] = parts["cancellation_kcal"]
                record[f"{prefix}_opposite"] = parts["opposite_signs"]
                record[f"{prefix}_signed_energy"] = parts["signed_energy_kcal"]
                record[f"{prefix}_signed_mass"] = parts["signed_mass_g"]
            if oracle is not None:
                record[f"{prefix}_perfect_grams_kcal"] = oracle["perfect_grams_kcal"]
                record[f"{prefix}_perfect_density_kcal"] = oracle["perfect_density_kcal"]
        if record["ours_density"] is not None and record["label_density"] is not None:
            record["density_gap_ours"] = abs(record["ours_density"] - record["label_density"])
            record["density_gap_direct"] = abs(record["direct_density"] - record["label_density"])
        records.append(record)
    return {"records": records, "replay": replay, "table": table}


def column(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in records], dtype=np.float64)


def optional_column(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in records:
        number = row.get(key)
        values.append(np.nan if number is None else float(number))
    return np.asarray(values, dtype=np.float64)


def aggregate(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = list(payload["records"])
    n = len(records)
    if n != N_PAIRED:
        raise RuntimeError(f"built {n} paired rows, expected {N_PAIRED}")
    energy_delta = column(records, "energy_delta_ae")
    mass_delta = column(records, "mass_delta_ae")
    if abs(float(energy_delta.mean()) - PUBLISHED_ENERGY_DELTA) > 1e-9:
        raise RuntimeError(f"energy delta {energy_delta.mean()} != published {PUBLISHED_ENERGY_DELTA}")
    if abs(float(mass_delta.mean()) - PUBLISHED_MASS_DELTA) > 1e-9:
        raise RuntimeError(f"mass delta {mass_delta.mean()} != published {PUBLISHED_MASS_DELTA}")

    quadrants = []
    names = []
    for row in records:
        if row["quadrant"] not in names:
            names.append(row["quadrant"])
    for name in (
        "mass_down_energy_down",
        "mass_down_energy_up",
        "mass_up_energy_down",
        "mass_up_energy_up",
        "tie",
    ):
        mask = np.asarray([row["quadrant"] == name for row in records])
        if np.any(mask) or name != "tie":
            quadrants.append(summarize_mask(name, mask, energy_delta, mass_delta))
    contribution_sum = sum(block["contribution_to_mean_energy_kcal"] for block in quadrants)
    if abs(contribution_sum - float(energy_delta.mean())) > 1e-8:
        raise RuntimeError("quadrant contributions do not add to the mean energy gap")

    defined = np.asarray([row["ours_density"] is not None and row["direct_density"] is not None for row in records])
    if int(np.sum(defined)) != n:
        raise RuntimeError(f"density undefined on {n - int(np.sum(defined))} paired plates")
    ours_mass_ch = column(records, "ours_mass_channel")
    ours_dens_ch = column(records, "ours_density_channel")
    direct_mass_ch = column(records, "direct_mass_channel")
    direct_dens_ch = column(records, "direct_density_channel")
    closed_mass_ch = column(records, "closed_mass_channel")
    closed_dens_ch = column(records, "closed_density_channel")

    def abs_error_from(pred_key: str) -> float:
        pred = column(records, pred_key)
        label = column(records, "label_energy_kcal")
        return mae(pred - label)

    paradox = np.asarray([row["quadrant"] == "mass_down_energy_up" for row in records])
    both = np.asarray([row["quadrant"] == "mass_down_energy_down" for row in records])
    density_worse = column(records, "density_gap_ours") > column(records, "density_gap_direct")
    direct_opposite = column(records, "direct_opposite") > 0.5
    ours_opposite = column(records, "ours_opposite") > 0.5

    def subset_phrase(mask: np.ndarray) -> dict[str, Any]:
        block = summarize_mask("subset", mask, energy_delta, mass_delta)
        block["mean_density_gap_ours"] = float(column(records, "density_gap_ours")[mask].mean()) if np.any(mask) else None
        block["mean_density_gap_direct"] = float(column(records, "density_gap_direct")[mask].mean()) if np.any(mask) else None
        block["direct_opposite_sign_rate"] = float(direct_opposite[mask].mean()) if np.any(mask) else None
        block["ours_opposite_sign_rate"] = float(ours_opposite[mask].mean()) if np.any(mask) else None
        block["density_worse_than_direct_rate"] = float(density_worse[mask].mean()) if np.any(mask) else None
        block["incomplete_rate"] = float(np.mean([not row["plate_complete"] for row, keep in zip(records, mask) if keep])) if np.any(mask) else None
        return block

    energy_ae = {
        "ours": column(records, "ours_energy_ae"),
        "direct": column(records, "direct_energy_ae"),
        "closed": column(records, "closed_energy_ae"),
    }
    mass_ae = {
        "ours": column(records, "ours_mass_ae"),
        "direct": column(records, "direct_mass_ae"),
        "closed": column(records, "closed_mass_ae"),
    }
    slices = {
        "all": slice_metrics(np.ones(n, dtype=bool), energy_ae, mass_ae),
        "stop": slice_metrics(np.asarray([row["decision_action"] == "STOP" for row in records]), energy_ae, mass_ae),
        "abstain": slice_metrics(np.asarray([row["decision_action"] == "ABSTAIN" for row in records]), energy_ae, mass_ae),
        "correct_code_terminal": slice_metrics(
            np.asarray([row["decision_action"] == "CORRECT_CODE" for row in records]), energy_ae, mass_ae
        ),
        "plate_complete": slice_metrics(np.asarray([row["plate_complete"] for row in records]), energy_ae, mass_ae),
        "plate_incomplete": slice_metrics(np.asarray([not row["plate_complete"] for row in records]), energy_ae, mass_ae),
        "any_correct_code": slice_metrics(np.asarray([row["n_correct_code"] > 0 for row in records]), energy_ae, mass_ae),
        "no_correct_code": slice_metrics(np.asarray([row["n_correct_code"] == 0 for row in records]), energy_ae, mass_ae),
        "single_component_complete": slice_metrics(
            np.asarray([row["plate_complete"] and row["n_hypotheses"] == 1 for row in records]), energy_ae, mass_ae
        ),
        "multi_component_complete": slice_metrics(
            np.asarray([row["plate_complete"] and (row["n_hypotheses"] or 0) > 1 for row in records]), energy_ae, mass_ae
        ),
        "any_rule_defer": slice_metrics(
            np.asarray([row["n_rule_utility_steps"] > 0 for row in records]), energy_ae, mass_ae
        ),
        "no_rule_defer": slice_metrics(
            np.asarray([row["n_rule_utility_steps"] == 0 for row in records]), energy_ae, mass_ae
        ),
    }
    replay = payload["replay"]
    source_masks = {}
    for source in ("q_over_cost", "rule_utility"):
        source_masks[source] = slice_metrics(
            np.asarray([row["selector_replay_matched"] and row["selector_source"] == source for row in records]),
            energy_ae,
            mass_ae,
        )
    matched_terminals = sum(1 for row in records if row["selector_replay_matched"])

    rewrite_mask = np.asarray([row["rewrite_accounted"] and row["n_mass_writes"] > 0 for row in records])
    rewrite = {"n": int(np.sum(rewrite_mask)), "n_accounted": int(sum(1 for row in records if row["rewrite_accounted"]))}
    if np.any(rewrite_mask):
        grid_e = optional_column(records, "grid_energy_kcal")
        grid_m = optional_column(records, "grid_mass_g")
        label_e = column(records, "label_energy_kcal")
        label_m = column(records, "label_mass_g")
        ours_e = column(records, "ours_energy_kcal")
        ours_m = column(records, "ours_mass_g")
        keep = rewrite_mask & np.isfinite(grid_e) & np.isfinite(grid_m)
        rewrite.update(
            {
                "n_with_mass_write": int(np.sum(keep)),
                "grid_energy_mae": mae(grid_e[keep] - label_e[keep]),
                "rewritten_energy_mae": mae(ours_e[keep] - label_e[keep]),
                "grid_mass_mae": mae(grid_m[keep] - label_m[keep]),
                "rewritten_mass_mae": mae(ours_m[keep] - label_m[keep]),
                "energy_mae_change": mae(ours_e[keep] - label_e[keep]) - mae(grid_e[keep] - label_e[keep]),
                "mass_mae_change": mae(ours_m[keep] - label_m[keep]) - mae(grid_m[keep] - label_m[keep]),
                "mean_plate_energy_ae_change": float(np.mean(np.abs(ours_e[keep] - label_e[keep]) - np.abs(grid_e[keep] - label_e[keep]))),
                "mean_plate_mass_ae_change": float(np.mean(np.abs(ours_m[keep] - label_m[keep]) - np.abs(grid_m[keep] - label_m[keep]))),
                "n_mass_ae_down": int(np.sum(np.abs(ours_m[keep] - label_m[keep]) < np.abs(grid_m[keep] - label_m[keep]))),
                "n_energy_ae_up": int(np.sum(np.abs(ours_e[keep] - label_e[keep]) > np.abs(grid_e[keep] - label_e[keep]))),
                "n_mass_down_energy_up": int(
                    np.sum(
                        (np.abs(ours_m[keep] - label_m[keep]) < np.abs(grid_m[keep] - label_m[keep]))
                        & (np.abs(ours_e[keep] - label_e[keep]) > np.abs(grid_e[keep] - label_e[keep]))
                    )
                ),
            }
        )

    complete = np.asarray([row["plate_complete"] and row["model_energy_kcal"] is not None for row in records])
    model = {"n_with_head": int(sum(1 for row in records if row["model_energy_kcal"] is not None))}
    spans = [row["model_energy_span"] for row in records if row["model_energy_span"] is not None]
    model["max_repeat_energy_span"] = None if not spans else float(max(spans))
    if np.any(complete):
        model_e = optional_column(records, "model_energy_kcal")
        model_m = optional_column(records, "model_mass_g")
        label_e = column(records, "label_energy_kcal")
        label_m = column(records, "label_mass_g")
        ours_e = column(records, "ours_energy_kcal")
        ours_m = column(records, "ours_mass_g")
        direct_e = column(records, "direct_energy_kcal")
        direct_m = column(records, "direct_mass_g")
        portion_gap = np.abs(optional_column(records, "portion_sum_g") - model_m)
        model.update(
            {
                "n_complete": int(np.sum(complete)),
                "head_energy_mae": mae(model_e[complete] - label_e[complete]),
                "committed_energy_mae": mae(ours_e[complete] - label_e[complete]),
                "direct_energy_mae": mae(direct_e[complete] - label_e[complete]),
                "head_mass_mae": mae(model_m[complete] - label_m[complete]),
                "committed_mass_mae": mae(ours_m[complete] - label_m[complete]),
                "direct_mass_mae": mae(direct_m[complete] - label_m[complete]),
                "mean_abs_portion_sum_minus_head_mass_g": float(np.nanmean(portion_gap[complete])),
                "mean_abs_committed_minus_head_energy_kcal": float(np.mean(np.abs(ours_e[complete] - model_e[complete]))),
            }
        )

    single = np.asarray([row["plate_complete"] and row["n_hypotheses"] == 1 for row in records])
    single_density = {"n": int(np.sum(single))}
    if np.any(single):
        # Committed kcal/g is the selected code's kcal/g on a one-component plate.
        single_density.update(
            {
                "ours_density_mae_kcal_per_g": float(column(records, "density_gap_ours")[single].mean()),
                "direct_density_mae_kcal_per_g": float(column(records, "density_gap_direct")[single].mean()),
                "ours_energy_mae": mae(column(records, "ours_energy_ae")[single]),
                "direct_energy_mae": mae(column(records, "direct_energy_ae")[single]),
                "ours_mass_mae": mae(column(records, "ours_mass_ae")[single]),
                "direct_mass_mae": mae(column(records, "direct_mass_ae")[single]),
            }
        )

    if any(row["n_hypotheses"] is None for row in records):
        raise RuntimeError("a paired plate is missing the hypothesis count")
    signed_energy = column(records, "ours_signed_energy")
    direct_signed = column(records, "direct_signed_energy")
    completeness_bias = {}
    for label, mask in (
        ("complete", np.asarray([row["plate_complete"] for row in records])),
        ("incomplete", np.asarray([not row["plate_complete"] for row in records])),
    ):
        completeness_bias[label] = {
            "n": int(np.sum(mask)),
            "ours_mean_signed_energy_kcal": float(np.mean(signed_energy[mask])),
            "direct_mean_signed_energy_kcal": float(np.mean(direct_signed[mask])),
            "ours_mean_signed_mass_g": float(np.mean(column(records, "ours_signed_mass")[mask])),
            "direct_mean_signed_mass_g": float(np.mean(column(records, "direct_signed_mass")[mask])),
        }
    cross_tab = []
    for action in ("STOP", "ABSTAIN", "CORRECT_CODE"):
        for complete_flag in (True, False):
            mask = np.asarray(
                [row["decision_action"] == action and bool(row["plate_complete"]) is complete_flag for row in records]
            )
            if not np.any(mask):
                cross_tab.append({"decision_action": action, "plate_complete": complete_flag, "n": 0})
                continue
            cross_tab.append(
                {
                    "decision_action": action,
                    "plate_complete": complete_flag,
                    "n": int(np.sum(mask)),
                    "energy_delta_kcal": float(energy_delta[mask].mean()),
                    "mass_delta_g": float(mass_delta[mask].mean()),
                    "contribution_to_mean_energy_kcal": float(energy_delta[mask].sum() / n),
                    "ours_mean_signed_energy_kcal": float(np.mean(signed_energy[mask])),
                    "direct_mean_signed_energy_kcal": float(np.mean(direct_signed[mask])),
                }
            )
    order = np.argsort(-energy_delta)
    positive = energy_delta[energy_delta > 0]
    positive_sum = float(positive.sum()) if positive.size else 0.0
    top10 = energy_delta[order[:10]]
    top10_positive = float(np.clip(top10, 0, None).sum())

    examples = []
    for row in pick_examples(records):
        examples.append(
            {
                "dish_id": row["dish_id"],
                "decision_action": row["decision_action"],
                "n_hypotheses": row["n_hypotheses"],
                "n_mass_writes": row["n_mass_writes"],
                "plate_complete": row["plate_complete"],
                "label_energy_kcal": row["label_energy_kcal"],
                "label_mass_g": row["label_mass_g"],
                "label_density_kcal_per_g": row["label_density"],
                "ours_energy_kcal": row["ours_energy_kcal"],
                "ours_mass_g": row["ours_mass_g"],
                "ours_density_kcal_per_g": row["ours_density"],
                "direct_energy_kcal": row["direct_energy_kcal"],
                "direct_mass_g": row["direct_mass_g"],
                "direct_density_kcal_per_g": row["direct_density"],
                "energy_delta_ae_kcal": row["energy_delta_ae"],
                "mass_delta_ae_g": row["mass_delta_ae"],
                "foods": row["foods"],
                "direct_opposite_signs": bool(row["direct_opposite"]),
                "ours_opposite_signs": bool(row["ours_opposite"]),
            }
        )

    macros_summary = {
        "protein_g": macro_block(
            optional_column(records, "ours_protein_g"),
            optional_column(records, "direct_protein_g"),
            optional_column(records, "label_protein_g"),
        ),
        "carb_g": macro_block(
            optional_column(records, "ours_carb_g"),
            optional_column(records, "direct_carb_g"),
            optional_column(records, "label_carb_g"),
        ),
        "fat_g": macro_block(
            optional_column(records, "ours_fat_g"),
            optional_column(records, "direct_fat_g"),
            optional_column(records, "label_fat_g"),
        ),
    }

    mass_down = mass_delta < 0
    energy_on_mass_down = {
        "n_mass_down": int(np.sum(mass_down)),
        "density_worse_and_energy_worse": int(np.sum(mass_down & density_worse & (energy_delta > 0))),
        "density_worse_and_energy_better": int(np.sum(mass_down & density_worse & (energy_delta < 0))),
        "density_better_and_energy_worse": int(np.sum(mass_down & ~density_worse & (energy_delta > 0))),
        "density_better_and_energy_better": int(np.sum(mass_down & ~density_worse & (energy_delta < 0))),
    }

    summary = {
        "definition": "p0d_same_budget_overhead_force_answer",
        "n_paired": n,
        "population": "Official rgb_test plates with an overhead photo that every scored method could commit. Two overhead plates stay unscored.",
        "methods": {
            "ours": "Mass Estimator + calibrated Q/Cost selection",
            "direct": "Direct vision-language model",
            "closed": "Closed-loop control",
        },
        "energy_mae": {
            "ours": mae(energy_ae["ours"]),
            "direct": mae(energy_ae["direct"]),
            "closed": mae(energy_ae["closed"]),
            "delta_ours_minus_direct": float(energy_delta.mean()),
            "delta_ours_minus_closed": mae(energy_ae["ours"]) - mae(energy_ae["closed"]),
        },
        "mass_mae": {
            "ours": mae(mass_ae["ours"]),
            "direct": mae(mass_ae["direct"]),
            "closed": mae(mass_ae["closed"]),
            "delta_ours_minus_direct": float(mass_delta.mean()),
            "delta_ours_minus_closed": mae(mass_ae["ours"]) - mae(mass_ae["closed"]),
        },
        "median_abs": {
            "ours_energy": median(energy_ae["ours"]),
            "direct_energy": median(energy_ae["direct"]),
            "ours_mass": median(mass_ae["ours"]),
            "direct_mass": median(mass_ae["direct"]),
        },
        "relative": {
            "mass_mae_change_vs_direct": float(mass_delta.mean()) / mae(mass_ae["direct"]),
            "energy_mae_change_vs_direct": float(energy_delta.mean()) / mae(energy_ae["direct"]),
        },
        "bootstrap": {
            "energy_ours_minus_direct": bootstrap_mean(energy_delta, seed=BOOTSTRAP_SEED),
            "mass_ours_minus_direct": bootstrap_mean(mass_delta, seed=BOOTSTRAP_SEED),
            "energy_ours_minus_closed": bootstrap_mean(energy_ae["ours"] - energy_ae["closed"], seed=BOOTSTRAP_SEED),
            "mass_ours_minus_closed": bootstrap_mean(mass_ae["ours"] - mass_ae["closed"], seed=BOOTSTRAP_SEED),
        },
        "plates": {
            "energy_closer": int(np.sum(energy_delta < 0)),
            "energy_farther": int(np.sum(energy_delta > 0)),
            "energy_tie": int(np.sum(energy_delta == 0)),
            "mass_closer": int(np.sum(mass_delta < 0)),
            "mass_farther": int(np.sum(mass_delta > 0)),
            "mass_tie": int(np.sum(mass_delta == 0)),
        },
        "median_paired_delta": {
            "energy_kcal": float(np.median(energy_delta)),
            "mass_g": float(np.median(mass_delta)),
        },
        "trim_10": {
            "energy": symmetric_trim_mean(energy_delta, 0.10),
            "mass": symmetric_trim_mean(mass_delta, 0.10),
        },
        "tail": {
            "largest_positive_energy_gaps_removed_until_mean_le_0": drop_until_mean_nonpositive(energy_delta),
            "positive_energy_gap_sum_kcal": positive_sum,
            "top10_positive_energy_gap_sum_kcal": top10_positive,
            "top10_share_of_positive_sum": None if positive_sum == 0 else top10_positive / positive_sum,
            "energy_delta_quantiles": {
                str(q): float(np.quantile(energy_delta, q)) for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
            },
            "removed_until_mean_le_0": _removed_tail(records, energy_delta, order),
        },
        "completeness_bias": completeness_bias,
        "terminal_by_completeness": cross_tab,
        "quadrants": quadrants,
        "channels": {
            **channel_summary("ours", ours_mass_ch, ours_dens_ch, column(records, "ours_signed_energy"), column(records, "ours_cancellation"), column(records, "ours_opposite")),
            **channel_summary("direct", direct_mass_ch, direct_dens_ch, column(records, "direct_signed_energy"), column(records, "direct_cancellation"), column(records, "direct_opposite")),
            **channel_summary("closed", closed_mass_ch, closed_dens_ch, column(records, "closed_signed_energy"), column(records, "closed_cancellation"), column(records, "closed_opposite")),
            "pearson_signed_mass_vs_density_gap_ours": pearson(
                column(records, "ours_signed_mass"),
                column(records, "ours_density") - column(records, "label_density"),
            ),
            "pearson_signed_mass_vs_density_gap_direct": pearson(
                column(records, "direct_signed_mass"),
                column(records, "direct_density") - column(records, "label_density"),
            ),
            "density_mae_kcal_per_g": {
                "ours": float(column(records, "density_gap_ours").mean()),
                "direct": float(column(records, "density_gap_direct").mean()),
                "closed": float(np.mean(np.abs(column(records, "closed_density") - column(records, "label_density")))),
            },
        },
        "oracle_energy_mae_kcal": {
            "note": "Identity on the saved grams and kcal/g. The label is a multiplier, not a tuning target.",
            "ours_perfect_grams": abs_error_from("ours_perfect_grams_kcal"),
            "ours_perfect_density": abs_error_from("ours_perfect_density_kcal"),
            "direct_perfect_grams": abs_error_from("direct_perfect_grams_kcal"),
            "direct_perfect_density": abs_error_from("direct_perfect_density_kcal"),
            "ours_actual": mae(energy_ae["ours"]),
            "direct_actual": mae(energy_ae["direct"]),
        },
        "mass_down_energy_up": subset_phrase(paradox),
        "mass_down_energy_down": subset_phrase(both),
        "mass_down_by_density": energy_on_mass_down,
        "slices": slices,
        "selector": {
            "match_rate": replay["match_rate"],
            "n_steps": replay["n_steps"],
            "n_match": replay["n_match"],
            "n_failures": replay["n_failures"],
            "mismatch_logged_actions": replay["mismatch_logged_actions"],
            "matched_step_sources": replay["matched_step_sources"],
            "matched_terminals": matched_terminals,
            "tau": replay["tau"],
            "by_terminal_source": source_masks,
        },
        "rewrite_same_codes": rewrite,
        "specialist_head_on_complete_plates": model,
        "head_by_slice": {
            "single_complete": head_slice(
                records, np.asarray([row["plate_complete"] and row["n_hypotheses"] == 1 for row in records])
            ),
            "multi_complete": head_slice(
                records, np.asarray([row["plate_complete"] and (row["n_hypotheses"] or 0) > 1 for row in records])
            ),
            "unfinished": head_slice(records, np.asarray([not row["plate_complete"] for row in records])),
        },
        "n_never_called_mass": int(sum(1 for row in records if row["model_energy_kcal"] is None)),
        "never_called_mass_actions": {
            action: sum(1 for row in records if row["model_energy_kcal"] is None and row["decision_action"] == action)
            for action in ("STOP", "ABSTAIN", "CORRECT_CODE", "RETRIEVE")
            if any(row["model_energy_kcal"] is None and row["decision_action"] == action for row in records)
        },
        "single_component_complete": single_density,
        "macros": macros_summary,
        "examples": examples,
        "n_trace_aligned": int(sum(1 for row in records if row["trace_aligned"])),
        "n_clipped_portions": int(sum(row["n_clipped_portions"] for row in records)),
        "inputs": {
            "same_budget_dishes_sha256": sha256_file(DISHES),
            "same_budget_sha256": sha256_file(TABLE),
            "ours_new_rgb_test_sha256": sha256_file(TRACE),
            "direct_jsonl_sha256": sha256_file(DIRECT_JSONL),
            "q_weights_sha256_prefix": sha256_file(Q_WEIGHTS)[:16],
        },
    }
    return summary


def _limits(values: np.ndarray, low_q: float = 0.01, high_q: float = 0.99) -> tuple[float, float, int]:
    sample = np.asarray(values, dtype=np.float64)
    lo = float(np.quantile(sample, low_q))
    hi = float(np.quantile(sample, high_q))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    span = hi - lo
    lo -= 0.06 * span
    hi += 0.06 * span
    outside = int(np.sum((sample < lo) | (sample > hi)))
    return lo, hi, outside


def scatter_svg(path: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    xs = column(records, "mass_delta_ae")
    ys = column(records, "energy_delta_ae")
    colors = []
    for row in records:
        colors.append(
            {
                "mass_down_energy_down": COLOR_BOTH_BETTER,
                "mass_down_energy_up": COLOR_MASS_ONLY,
                "mass_up_energy_down": COLOR_ENERGY_ONLY,
                "mass_up_energy_up": COLOR_BOTH_WORSE,
            }.get(row["quadrant"], "#111111")
        )
    x0, x1, x_out = _limits(xs)
    y0, y1, y_out = _limits(ys)
    width, height = 760, 640
    left, right, top, bottom = 86, 28, 52, 120

    def sx(value: float) -> float:
        clipped = min(max(value, x0), x1)
        return left + (clipped - x0) / (x1 - x0) * (width - left - right)

    def sy(value: float) -> float:
        clipped = min(max(value, y0), y1)
        return top + (y1 - clipped) / (y1 - y0) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="86" y="28" font-family="DejaVu Sans, sans-serif" font-size="16" fill="#111827">Paired change versus the direct vision-language model</text>',
        f'<text x="86" y="46" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#4b5563">n={len(records)}. Left of zero: lower mass absolute error. Below zero: lower energy absolute error.</text>',
    ]
    # grid
    for fraction in (0.25, 0.5, 0.75):
        gx = left + fraction * (width - left - right)
        gy = top + fraction * (height - top - bottom)
        parts.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{height - bottom}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{width - right}" y2="{gy:.1f}" stroke="#e5e7eb"/>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{top}" x2="{sx(0):.1f}" y2="{height - bottom}" stroke="#111827" stroke-dasharray="4 3"/>')
    parts.append(f'<line x1="{left}" y1="{sy(0):.1f}" x2="{width - right}" y2="{sy(0):.1f}" stroke="#111827" stroke-dasharray="4 3"/>')
    parts.append(f'<rect x="{left}" y="{top}" width="{width - left - right}" height="{height - top - bottom}" fill="none" stroke="#111827"/>')
    for x, y, color in zip(xs, ys, colors):
        parts.append(f'<circle cx="{sx(float(x)):.2f}" cy="{sy(float(y)):.2f}" r="3.3" fill="{color}" fill-opacity="0.78"/>')
    parts.append(f'<text x="{(left + width - right) / 2:.0f}" y="{height - bottom + 28}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="13" fill="#111827">Mass absolute error change (g): this system minus direct model</text>')
    parts.append(f'<text x="22" y="{(top + height - bottom) / 2:.0f}" text-anchor="middle" transform="rotate(-90 22 {(top + height - bottom) / 2:.0f})" font-family="DejaVu Sans, sans-serif" font-size="13" fill="#111827">Energy absolute error change (kcal)</text>')
    legend = [
        (COLOR_BOTH_BETTER, "Both closer"),
        (COLOR_MASS_ONLY, "Mass closer, energy farther"),
        (COLOR_ENERGY_ONLY, "Energy closer, mass farther"),
        (COLOR_BOTH_WORSE, "Both farther"),
    ]
    for index, (color, label) in enumerate(legend):
        col = index % 2
        row = index // 2
        x = 110 + col * 320
        y = height - 58 + row * 18
        parts.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}"/>')
        parts.append(f'<text x="{x + 12}" y="{y + 4}" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">{xml(label)}</text>')
    note = f"Axes show the central 98% of each change. {max(x_out, y_out)} points sit on the frame because they fall outside it."
    parts.append(f'<text x="86" y="{height - 16}" font-family="DejaVu Sans, sans-serif" font-size="11" fill="#6b7280">{xml(note)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)), "x_outside": x_out, "y_outside": y_out}


def bar_svg(path: Path, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str, colors: Sequence[str]) -> None:
    width, height = 760, 460
    left, right, top, bottom = 84, 24, 48, 110
    series = [float(value) for value in values]
    peak = max(max(abs(value) for value in series), 1.0)
    y0, y1 = -peak * 1.15, peak * 1.15
    plot_w = width - left - right
    plot_h = height - top - bottom
    gap = 18
    bar_w = (plot_w - gap * (len(series) + 1)) / len(series)

    def sy(value: float) -> float:
        return top + (y1 - value) / (y1 - y0) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="84" y="28" font-family="DejaVu Sans, sans-serif" font-size="16" fill="#111827">{xml(title)}</text>',
        f'<line x1="{left}" y1="{sy(0):.1f}" x2="{width - right}" y2="{sy(0):.1f}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827"/>',
    ]
    for index, (label, value, color) in enumerate(zip(labels, series, colors)):
        x = left + gap + index * (bar_w + gap)
        y = sy(value)
        base = sy(0)
        y_rect = min(y, base)
        parts.append(
            f'<rect x="{x:.1f}" y="{y_rect:.1f}" width="{bar_w:.1f}" height="{abs(base - y):.1f}" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y_rect - 6:.1f}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">{value:+.2f}</text>'
        )
        label_lines = str(label).split("\n")
        text_x = x + bar_w / 2
        parts.append(
            f'<text x="{text_x:.1f}" y="{height - bottom + 22}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">'
        )
        for line_index, line in enumerate(label_lines):
            dy = 0 if line_index == 0 else 15
            parts.append(f'<tspan x="{text_x:.1f}" dy="{dy}">{xml(line)}</tspan>')
        parts.append("</text>")
    parts.append(
        f'<text x="22" y="{(top + height - bottom) / 2:.0f}" text-anchor="middle" transform="rotate(-90 22 {(top + height - bottom) / 2:.0f})" font-family="DejaVu Sans, sans-serif" font-size="13" fill="#111827">{xml(ylabel)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def channel_svg(path: Path, summary: Mapping[str, Any]) -> None:
    channels = summary["channels"]
    labels = ["|gram term|", "|kcal/g term|", "calorie MAE", "cancellation"]
    ours = [
        channels["ours_mean_abs_mass_channel_kcal"],
        channels["ours_mean_abs_density_channel_kcal"],
        channels["ours_energy_mae_kcal"],
        channels["ours_mean_cancellation_kcal"],
    ]
    direct = [
        channels["direct_mean_abs_mass_channel_kcal"],
        channels["direct_mean_abs_density_channel_kcal"],
        channels["direct_energy_mae_kcal"],
        channels["direct_mean_cancellation_kcal"],
    ]
    width, height = 760, 520
    left, right, top, bottom = 72, 24, 56, 120
    peak = max(max(ours), max(direct)) * 1.2
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / len(labels)
    bar_w = 28

    def sy(value: float) -> float:
        return top + (peak - value) / peak * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="72" y="24" font-family="DejaVu Sans, sans-serif" font-size="16" fill="#111827">Where the calorie error comes from</text>',
        '<text x="72" y="42" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#4b5563">Means over 505 plates. Cancellation is |gram term| + |kcal/g term| − calorie absolute error.</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
    ]
    for index, (label, ours_value, direct_value) in enumerate(zip(labels, ours, direct)):
        cx = left + (index + 0.5) * group_w
        for value, dx, color in ((direct_value, -bar_w - 4, COLOR_DIRECT), (ours_value, 4, COLOR_OURS)):
            y = sy(value)
            parts.append(f'<rect x="{cx + dx:.1f}" y="{y:.1f}" width="{bar_w}" height="{top + plot_h - y:.1f}" fill="{color}"/>')
            parts.append(
                f'<text x="{cx + dx + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="11" fill="#111827">{value:.1f}</text>'
            )
        parts.append(
            f'<text x="{cx:.1f}" y="{height - bottom + 28}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">{xml(label)}</text>'
        )
    parts.append(f'<rect x="86" y="{height - 58}" width="12" height="12" fill="{COLOR_OURS}"/>')
    parts.append(f'<text x="104" y="{height - 48}" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">Mass estimator + calibrated selection</text>')
    parts.append(f'<rect x="400" y="{height - 58}" width="12" height="12" fill="{COLOR_DIRECT}"/>')
    parts.append(f'<text x="418" y="{height - 48}" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111827">Direct vision-language model</text>')
    parts.append(
        f'<text x="18" y="{(top + top + plot_h) / 2:.0f}" text-anchor="middle" transform="rotate(-90 18 {(top + top + plot_h) / 2:.0f})" font-family="DejaVu Sans, sans-serif" font-size="13" fill="#111827">kcal</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def render_report(summary: Mapping[str, Any]) -> str:
    energy = summary["energy_mae"]
    mass = summary["mass_mae"]
    boot_e = summary["bootstrap"]["energy_ours_minus_direct"]
    boot_m = summary["bootstrap"]["mass_ours_minus_direct"]
    boot_closed_e = summary["bootstrap"]["energy_ours_minus_closed"]
    boot_closed_m = summary["bootstrap"]["mass_ours_minus_closed"]
    channels = summary["channels"]
    oracle = summary["oracle_energy_mae_kcal"]
    plates = summary["plates"]
    tail = summary["tail"]
    removed = tail["removed_until_mean_le_0"]
    rewrite = summary["rewrite_same_codes"]
    head = summary["specialist_head_on_complete_plates"]
    single_density = summary["single_component_complete"]
    selector = summary["selector"]
    slices = summary["slices"]
    macros = summary["macros"]
    bias = summary["completeness_bias"]
    quads = {row["name"]: row for row in summary["quadrants"]}
    n = summary["n_paired"]

    def contribution(block: Mapping[str, Any]) -> float:
        return float(block["energy_delta_ours_minus_direct"]) * int(block["n"]) / n

    def quad_line(name: str) -> str:
        block = quads[name]
        return (
            f"| {block['label']} | {block['n']} | {block['mean_energy_delta_kcal']:+.2f} | "
            f"{block['mean_mass_delta_g']:+.2f} | {block['contribution_to_mean_energy_kcal']:+.2f} |"
        )

    def slice_line(key: str, label: str) -> str:
        block = slices[key]
        return (
            f"| {label} | {block['n']} | {block['ours_energy_mae']:.2f} | {block['direct_energy_mae']:.2f} | "
            f"{block['energy_delta_ours_minus_direct']:+.2f} | {block['ours_mass_mae']:.2f} | "
            f"{block['direct_mass_mae']:.2f} | {block['mass_delta_ours_minus_direct']:+.2f} |"
        )

    finished = slices["plate_complete"]
    unfinished = slices["plate_incomplete"]
    one = slices["single_component_complete"]
    multi = slices["multi_component_complete"]
    deferred = slices["any_rule_defer"]
    held = slices["no_rule_defer"]

    removed_ids = ", ".join(
        f"`{plate['dish_id']}` ({plate['energy_delta_ae_kcal']:+.1f} kcal)" for plate in removed["plates"]
    )
    removed_sentence = (
        f"Removing the {removed['n_removed']} largest energy-absolute-error increases ({removed_ids}) "
        f"leaves a mean energy change of {removed['remaining_mean_energy_delta_kcal']:+.2f} kcal "
        f"on the other {removed['remaining_n']} plates. "
        "That removal is a sensitivity check. The published MAE still includes those plates."
    )

    example_lines = []
    shown = {plate["dish_id"] for plate in removed["plates"]}
    for row in summary["examples"]:
        if row["dish_id"] in shown:
            continue
        example_lines.append(
            "\n".join(
                [
                    f"- `{row['dish_id']}` ({row['decision_action']}, {row['n_mass_writes']} of {row['n_hypotheses']} named components mass-written).",
                    f"  Label {row['label_mass_g']:.1f} g / {row['label_energy_kcal']:.1f} kcal ({row['label_density_kcal_per_g'] * 100:.1f} kcal/100 g).",
                    f"  This system {row['ours_mass_g']:.1f} g / {row['ours_energy_kcal']:.1f} kcal ({row['ours_density_kcal_per_g'] * 100:.1f} kcal/100 g).",
                    f"  Direct model {row['direct_mass_g']:.1f} g / {row['direct_energy_kcal']:.1f} kcal ({row['direct_density_kcal_per_g'] * 100:.1f} kcal/100 g).",
                    f"  Absolute-error change: mass {row['mass_delta_ae_g']:+.1f} g, energy {row['energy_delta_ae_kcal']:+.1f} kcal.",
                    f"  Foods written: {row['foods'] or 'none'}.",
                ]
            )
        )
        if len(example_lines) == 2:
            break

    def plate_bullet(plate: Mapping[str, Any]) -> str:
        foods = plate.get("foods") or "none"
        return "\n".join(
            [
                f"- `{plate['dish_id']}` (energy absolute-error change {plate['energy_delta_ae_kcal']:+.1f} kcal, mass {plate['mass_delta_ae_g']:+.1f} g).",
                f"  Label {plate['label_mass_g']:.1f} g / {plate['label_energy_kcal']:.1f} kcal.",
                f"  This system {plate['ours_mass_g']:.1f} g / {plate['ours_energy_kcal']:.1f} kcal.",
                f"  Direct model {plate['direct_mass_g']:.1f} g / {plate['direct_energy_kcal']:.1f} kcal.",
                f"  Foods written: {foods}.",
            ]
        )

    sensitivity_lines = [plate_bullet(plate) for plate in removed["plates"]]
    head_single = summary["head_by_slice"]["single_complete"]
    head_multi = summary["head_by_slice"]["multi_complete"]
    stop_done = next(
        row for row in summary["terminal_by_completeness"] if row["decision_action"] == "STOP" and row["plate_complete"]
    )
    stop_open = next(
        row for row in summary["terminal_by_completeness"] if row["decision_action"] == "STOP" and not row["plate_complete"]
    )
    never_bits = list(summary["never_called_mass_actions"].items())
    if len(never_bits) == 1 and never_bits[0][1] == summary["n_never_called_mass"]:
        never_called = f"all {never_bits[0][1]} end on {never_bits[0][0]}"
    else:
        never_called = ", ".join(f"{count} end on {name}" for name, count in never_bits)

    cross_lines = []
    for row in summary["terminal_by_completeness"]:
        if row["n"] == 0:
            continue
        state = "every component mass-written" if row["plate_complete"] else "a named component left unwritten"
        cross_lines.append(
            f"| {row['decision_action']} | {state} | {row['n']} | {row['energy_delta_kcal']:+.2f} | "
            f"{row['mass_delta_g']:+.2f} | {row['contribution_to_mean_energy_kcal']:+.2f} | "
            f"{row['ours_mean_signed_energy_kcal']:+.1f} |"
        )

    rule_steps = int(selector["matched_step_sources"].get("rule_utility", 0))
    q_steps = int(selector["matched_step_sources"].get("q_over_cost", 0))
    q_terminals = selector["by_terminal_source"]["q_over_cost"]["n"]
    rule_terminals = selector["by_terminal_source"]["rule_utility"]["n"]

    lines = [
        "# Error audit: mass fell, full-set energy stayed above the direct model",
        "",
        "On the 505 paired overhead plates, the mass estimator with calibrated Q/Cost selection has "
        f"mass MAE **{mass['ours']:.4f} g** and energy MAE **{energy['ours']:.4f} kcal**. "
        f"The direct vision-language model has mass MAE **{mass['direct']:.4f} g** and energy MAE **{energy['direct']:.4f} kcal**. "
        f"The paired changes are **ΔM {mass['delta_ours_minus_direct']:+.4f} g** and **ΔE {energy['delta_ours_minus_direct']:+.4f} kcal** "
        "(this system minus the direct model). "
        f"Mass MAE is {100 * abs(summary['relative']['mass_mae_change_vs_direct']):.1f}% lower. "
        f"Energy MAE is {100 * summary['relative']['energy_mae_change_vs_direct']:.1f}% higher.",
        "",
        "The mass gap is stable and the energy gap is a small remainder. "
        f"A 10,000-draw bootstrap that resamples plates (numpy Generator seed {boot_m['seed']}) gives a 95% percentile interval of "
        f"**[{boot_m['ci95_low']:+.2f}, {boot_m['ci95_high']:+.2f}] g** for mass. "
        f"Every draw has a negative mass mean. "
        f"The same interval for energy is **[{boot_e['ci95_low']:+.2f}, {boot_e['ci95_high']:+.2f}] kcal**, "
        f"and {100 * boot_e['fraction_mean_negative']:.1f}% of draws have a negative energy mean. "
        f"The median paired energy change is **{summary['median_paired_delta']['energy_kcal']:+.2f} kcal**. "
        f"Energy absolute error is lower on **{plates['energy_closer']}** plates and higher on **{plates['energy_farther']}**. "
        f"A 10% trim of each tail leaves an energy-change mean of {summary['trim_10']['energy']['mean']:+.2f} kcal. "
        + removed_sentence,
        "",
        "The pooled energy mean is the mixture of two plate sets. "
        f"On **{finished['n']}** plates where every named component received a mass write, energy MAE is "
        f"{finished['ours_energy_mae']:.2f} versus {finished['direct_energy_mae']:.2f} "
        f"(Δ {finished['energy_delta_ours_minus_direct']:+.2f} kcal) and mass MAE is "
        f"{finished['ours_mass_mae']:.2f} versus {finished['direct_mass_mae']:.2f} "
        f"(Δ {finished['mass_delta_ours_minus_direct']:+.2f} g). "
        f"That set contributes {contribution(finished):+.2f} kcal to the 505-plate energy mean. "
        f"On the other **{unfinished['n']}** plates, at least one named component has no mass write. "
        f"Energy MAE is {unfinished['ours_energy_mae']:.2f} versus {unfinished['direct_energy_mae']:.2f} "
        f"(Δ {unfinished['energy_delta_ours_minus_direct']:+.2f} kcal) and mass MAE is "
        f"{unfinished['ours_mass_mae']:.2f} versus {unfinished['direct_mass_mae']:.2f} "
        f"(Δ {unfinished['mass_delta_ours_minus_direct']:+.2f} g). "
        f"That set contributes {contribution(unfinished):+.2f} kcal. "
        "The unfinished set is larger, and its higher energy error leaves the pooled mean at +1.42 kcal.",
        "",
        f"Inside the finished set, the {one['n']} one-component plates have ΔE {one['energy_delta_ours_minus_direct']:+.2f} kcal "
        f"and ΔM {one['mass_delta_ours_minus_direct']:+.2f} g. "
        f"The {multi['n']} multi-component plates have ΔE {multi['energy_delta_ours_minus_direct']:+.2f} kcal "
        f"and ΔM {multi['mass_delta_ours_minus_direct']:+.2f} g. "
        "The pooled energy win on finished plates is the one-component group. "
        "The multi-component group is where a large mass win sits next to a higher energy MAE.",
        "",
        "The committed calorie is the selected food code's FNDDS kcal per 100 g times the written grams. "
        f"On the finished plates the written grams match the specialist's dish mass to "
        f"{head['mean_abs_portion_sum_minus_head_mass_g']:.3f} g on average, while the committed calories differ from "
        f"the specialist's own calorie head by {head['mean_abs_committed_minus_head_energy_kcal']:.2f} kcal on average. "
        f"That stored head has energy MAE {head['head_energy_mae']:.2f} kcal on these {head['n_complete']} plates. "
        f"The food-table commit on the same plates has {head['committed_energy_mae']:.2f} kcal, and the direct model has "
        f"{head['direct_energy_mae']:.2f} kcal. "
        "The full-set score uses the food-table product. The head is a stored forward pass, scored here only as a reading of what the commit did not use.",
        "",
        "Closed-loop control, the earlier controller without this mass estimator, has energy MAE "
        f"{energy['closed']:.4f} kcal and mass MAE {mass['closed']:.4f} g on the same 505 plates. "
        f"The mass estimator changes that by ΔE {energy['delta_ours_minus_closed']:+.4f} kcal "
        f"(95% interval [{boot_closed_e['ci95_low']:+.2f}, {boot_closed_e['ci95_high']:+.2f}]) and "
        f"ΔM {mass['delta_ours_minus_closed']:+.4f} g "
        f"(95% interval [{boot_closed_m['ci95_low']:+.2f}, {boot_closed_m['ci95_high']:+.2f}]). "
        "Both of those intervals contain negative and positive values. "
        "The point estimates are lower than closed-loop control, and the energy point estimate stays above the direct model.",
        "",
        "## What was measured",
        "",
        "The paired table, the per-plate rows, and the mass-estimator trace are the P0d artifacts. "
        "Protein, carbohydrate, and fat come from the frozen direct-model file for the same plates. "
        "Labels score residuals. No threshold, code, or gram weight was chosen from these labels. "
        "No model was trained. `docs/STATUS.md` was not edited.",
        "",
        "A committed calorie is kcal/g times grams. With both masses positive the signed calorie error splits exactly into",
        "",
        "`predicted kcal/g × (predicted grams − label grams) + label grams × (predicted kcal/g − label kcal/g)`.",
        "",
        "The first term is the gram term. The second is the food-density term. "
        "Cancellation is `|gram term| + |density term| − |calorie error|`, the part of the two terms that offsets inside the absolute error. "
        "Two further calorie totals keep one saved factor and replace the other with the label. "
        "Those totals are arithmetic on the saved predictions. They were not commits.",
        "",
        "## How the plates split",
        "",
        f"Mass absolute error is lower on **{plates['mass_closer']}** plates, higher on **{plates['mass_farther']}**, and tied on **{plates['mass_tie']}**. "
        f"The median paired mass change is {summary['median_paired_delta']['mass_g']:+.2f} g. "
        f"Median absolute errors are {summary['median_abs']['ours_mass']:.2f} g versus {summary['median_abs']['direct_mass']:.2f} g for mass, and "
        f"{summary['median_abs']['ours_energy']:.2f} kcal versus {summary['median_abs']['direct_energy']:.2f} kcal for energy. "
        f"The positive energy-absolute-error gaps sum to {tail['positive_energy_gap_sum_kcal']:.0f} kcal of plate-level difference. "
        f"The 10 largest are {100 * tail['top10_share_of_positive_sum']:.1f}% of that positive sum, so the upper side is a broad set plus a short extreme tail. "
        f"Energy-change quantiles (kcal): "
        + ", ".join(f"p{int(float(name) * 100):02d} {value:+.1f}" for name, value in tail["energy_delta_quantiles"].items())
        + ".",
        "",
        "| Group | Plates | Mean energy change (kcal) | Mean mass change (g) | Addition to the 505-plate energy mean (kcal) |",
        "|---|---:|---:|---:|---:|",
        quad_line("mass_down_energy_down"),
        quad_line("mass_down_energy_up"),
        quad_line("mass_up_energy_down"),
        quad_line("mass_up_energy_up"),
        "",
        f"These four groups add "
        f"{sum(quads[name]['contribution_to_mean_energy_kcal'] for name in ('mass_down_energy_down', 'mass_down_energy_up', 'mass_up_energy_down', 'mass_up_energy_up')):+.2f} kcal. "
        f"The one tied plate adds {quads['tie']['contribution_to_mean_energy_kcal']:+.2f} kcal, and the total is {energy['delta_ours_minus_direct']:+.2f} kcal. "
        f"Plates where both errors got closer contribute {quads['mass_down_energy_down']['contribution_to_mean_energy_kcal']:+.2f} kcal. "
        f"Plates where mass got closer and energy got farther contribute {quads['mass_down_energy_up']['contribution_to_mean_energy_kcal']:+.2f} kcal. "
        f"Plates where both got farther contribute {quads['mass_up_energy_up']['contribution_to_mean_energy_kcal']:+.2f} kcal. "
        "The mean is what remains after those large groups offset.",
        "",
        "![Paired mass and energy changes. Axes show the central 98% of each change. The two largest energy increases are off this frame and are listed with the food codes.](fig_mass_energy_delta.svg)",
        "",
        "![Each group's addition to the mean energy gap. These four bars sum to +1.40 kcal; the tied plate adds the rest.](fig_energy_gap_by_quadrant.svg)",
        "",
        "On the 112 plates where mass got closer and energy got farther, "
        f"the mean mass change is {summary['mass_down_energy_up']['mean_mass_delta_g']:+.1f} g and the mean energy change is "
        f"{summary['mass_down_energy_up']['mean_energy_delta_kcal']:+.1f} kcal. "
        f"The direct model's gram error and kcal/g error have opposite signs on {100 * summary['mass_down_energy_up']['direct_opposite_sign_rate']:.1f}% of them, "
        f"and this system's gram and kcal/g errors have opposite signs on {100 * summary['mass_down_energy_up']['ours_opposite_sign_rate']:.1f}%. "
        f"kcal/g absolute error is higher than the direct model's on {100 * summary['mass_down_energy_up']['density_worse_than_direct_rate']:.1f}% of them. "
        "Moving the grams toward the label removes a cancellation that had been holding the direct model's calorie error down.",
        "",
        "Among the plates whose mass absolute error fell:",
        "",
        "| kcal/g absolute error versus the direct model | Energy closer | Energy farther |",
        "|---|---:|---:|",
        f"| Lower | {summary['mass_down_by_density']['density_better_and_energy_better']} | {summary['mass_down_by_density']['density_better_and_energy_worse']} |",
        f"| Higher | {summary['mass_down_by_density']['density_worse_and_energy_better']} | {summary['mass_down_by_density']['density_worse_and_energy_worse']} |",
        "",
        "## Gram term and food-density term",
        "",
        "| | Mass MAE (g) | Energy MAE (kcal) | kcal/g MAE | Mean \\|gram term\\| (kcal) | Mean \\|kcal/g term\\| (kcal) | Mean cancellation (kcal) | Mean signed calorie error (kcal) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Mass estimator + calibrated selection | {mass['ours']:.2f} | {energy['ours']:.2f} | "
            f"{channels['density_mae_kcal_per_g']['ours']:.3f} | {channels['ours_mean_abs_mass_channel_kcal']:.1f} | "
            f"{channels['ours_mean_abs_density_channel_kcal']:.1f} | {channels['ours_mean_cancellation_kcal']:.1f} | "
            f"{channels['ours_mean_signed_energy_kcal']:+.1f} |"
        ),
        (
            f"| Direct vision-language model | {mass['direct']:.2f} | {energy['direct']:.2f} | "
            f"{channels['density_mae_kcal_per_g']['direct']:.3f} | {channels['direct_mean_abs_mass_channel_kcal']:.1f} | "
            f"{channels['direct_mean_abs_density_channel_kcal']:.1f} | {channels['direct_mean_cancellation_kcal']:.1f} | "
            f"{channels['direct_mean_signed_energy_kcal']:+.1f} |"
        ),
        (
            f"| Closed-loop control | {mass['closed']:.2f} | {energy['closed']:.2f} | "
            f"{channels['density_mae_kcal_per_g']['closed']:.3f} | {channels['closed_mean_abs_mass_channel_kcal']:.1f} | "
            f"{channels['closed_mean_abs_density_channel_kcal']:.1f} | {channels['closed_mean_cancellation_kcal']:.1f} | "
            f"{channels['closed_mean_signed_energy_kcal']:+.1f} |"
        ),
        "",
        f"Mean signed calorie error for this system is {channels['ours_mean_signed_energy_kcal']:+.1f} kcal, "
        f"from a gram term of {channels['ours_mean_mass_channel_kcal']:+.1f} kcal and a density term of "
        f"{channels['ours_mean_density_channel_kcal']:+.1f} kcal. "
        f"The direct model's signed error is {channels['direct_mean_signed_energy_kcal']:+.1f} kcal "
        f"({channels['direct_mean_mass_channel_kcal']:+.1f} from grams, {channels['direct_mean_density_channel_kcal']:+.1f} from kcal/g). "
        f"Gram error and kcal/g error have opposite signs on {100 * channels['ours_opposite_sign_rate']:.1f}% of plates here and on "
        f"{100 * channels['direct_opposite_sign_rate']:.1f}% for the direct model. "
        f"Their correlation is {channels['pearson_signed_mass_vs_density_gap_ours']:.3f} here and "
        f"{channels['pearson_signed_mass_vs_density_gap_direct']:.3f} for the direct model.",
        "",
        "Unweighted kcal/g absolute error is slightly lower for this system "
        f"({channels['density_mae_kcal_per_g']['ours']:.3f} versus {channels['density_mae_kcal_per_g']['direct']:.3f} kcal/g). "
        "The calorie-valued density term, label grams times that kcal/g error, is higher "
        f"({channels['ours_mean_abs_density_channel_kcal']:.1f} versus {channels['direct_mean_abs_density_channel_kcal']:.1f} kcal). "
        "The direct model also cancels more of the gram term against the density term "
        f"({channels['direct_mean_cancellation_kcal']:.1f} versus {channels['ours_mean_cancellation_kcal']:.1f} kcal).",
        "",
        "Replacing the saved grams with the label grams, and keeping the saved kcal/g, gives energy MAE "
        f"**{oracle['ours_perfect_grams']:.2f} kcal** here and **{oracle['direct_perfect_grams']:.2f} kcal** for the direct model. "
        "Replacing the saved kcal/g with the label kcal/g, and keeping the saved grams, gives energy MAE "
        f"**{oracle['ours_perfect_density']:.2f} kcal** here and **{oracle['direct_perfect_density']:.2f} kcal** for the direct model. "
        f"The scored energy MAEs are {oracle['ours_actual']:.2f} and {oracle['direct_actual']:.2f}. "
        "At the label kcal/g, this system's grams would produce the lower calorie error. "
        "At the saved kcal/g, even label grams would leave this system's calorie error above the direct model's scored MAE.",
        "",
        "![Gram term, density term, calorie MAE, and cancellation](fig_error_channels.svg)",
        "",
        "Signed calorie error by whether every named component was mass-written:",
        "",
        f"- Finished plates: this system {bias['complete']['ours_mean_signed_energy_kcal']:+.1f} kcal, direct model {bias['complete']['direct_mean_signed_energy_kcal']:+.1f} kcal. "
        f"Signed mass error {bias['complete']['ours_mean_signed_mass_g']:+.1f} g versus {bias['complete']['direct_mean_signed_mass_g']:+.1f} g.",
        f"- Unfinished plates: this system {bias['incomplete']['ours_mean_signed_energy_kcal']:+.1f} kcal, direct model {bias['incomplete']['direct_mean_signed_energy_kcal']:+.1f} kcal. "
        f"Signed mass error {bias['incomplete']['ours_mean_signed_mass_g']:+.1f} g versus {bias['incomplete']['direct_mean_signed_mass_g']:+.1f} g.",
        "",
        "## Food codes after the gram write",
        "",
        "The two plates whose removal brings the mean energy change to zero or below:",
        "",
        *sensitivity_lines,
        "",
        (
            "Both of those plates mass-wrote every named component"
            + (
                ", and both commits include a bacon code at 494 kcal/100 g."
                if removed["plates"]
                and all("494.0 kcal/100 g" in (plate.get("foods") or "") and "Bacon" in (plate.get("foods") or "") for plate in removed["plates"])
                else "."
            )
        ),
        "",
        "Two more plates where mass got closer and energy got farther:",
        "",
        *example_lines,
        "",
        (
            f"Holding the written food codes fixed, the earlier gram guess and the mass-estimator grams can be compared on "
            f"{rewrite['n_with_mass_write']} plates whose committed total matches the sum of the written components within 0.05 kcal and 0.05 g. "
            f"Mass MAE moves from {rewrite['grid_mass_mae']:.2f} g to {rewrite['rewritten_mass_mae']:.2f} g "
            f"({rewrite['mass_mae_change']:+.2f} g). "
            f"Energy MAE moves from {rewrite['grid_energy_mae']:.2f} kcal to {rewrite['rewritten_energy_mae']:.2f} kcal "
            f"({rewrite['energy_mae_change']:+.2f} kcal). "
            f"On {rewrite['n_mass_down_energy_up']} of these plates the rewrite lowers mass absolute error and raises energy absolute error, "
            f"and the set-level energy MAE still falls. "
            "The earlier grams are the grams already stored on that component before the mass write, rather than the direct model's plate mass."
        ),
        "",
        (
            f"On the {single_density['n']} finished one-component plates, committed kcal/g is that code's kcal/g. "
            f"kcal/g absolute error is {single_density['ours_density_mae_kcal_per_g']:.3f} kcal/g here and "
            f"{single_density['direct_density_mae_kcal_per_g']:.3f} kcal/g for the direct model. "
            f"Energy MAE is {single_density['ours_energy_mae']:.2f} versus {single_density['direct_energy_mae']:.2f} kcal. "
            f"Mass MAE is {single_density['ours_mass_mae']:.2f} versus {single_density['direct_mass_mae']:.2f} g. "
            f"The stored calorie head on these plates has energy MAE {head_single['head_energy_mae']:.2f} kcal, "
            f"with mass MAE {head_single['head_mass_mae']:.2f} g."
        ),
        "",
        (
            f"On the {head_multi['n']} finished multi-component plates the stored head has energy MAE "
            f"{head_multi['head_energy_mae']:.2f} kcal and mass MAE {head_multi['head_mass_mae']:.2f} g. "
            f"The food-table commit has energy MAE {head_multi['committed_energy_mae']:.2f} kcal and mass MAE "
            f"{head_multi['committed_mass_mae']:.2f} g. "
            f"The direct model has energy MAE {head_multi['direct_energy_mae']:.2f} kcal and mass MAE "
            f"{head_multi['direct_mass_mae']:.2f} g. "
            "The grams match the head. The calorie gap between the head and the commit is the summed food-code table."
        ),
        "",
        "Protein, carbohydrate, and fat use the committed vectors. Bias is prediction minus label.",
        "",
        "| Nutrient | n | This system MAE | Direct model MAE | Δ MAE | This system bias | Direct model bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Protein (g) | {macros['protein_g']['n']} | {macros['protein_g']['ours_mae']:.2f} | {macros['protein_g']['direct_mae']:.2f} | "
            f"{macros['protein_g']['delta_ours_minus_direct']:+.2f} | {macros['protein_g']['ours_bias']:+.2f} | {macros['protein_g']['direct_bias']:+.2f} |"
        ),
        (
            f"| Carbohydrate (g) | {macros['carb_g']['n']} | {macros['carb_g']['ours_mae']:.2f} | {macros['carb_g']['direct_mae']:.2f} | "
            f"{macros['carb_g']['delta_ours_minus_direct']:+.2f} | {macros['carb_g']['ours_bias']:+.2f} | {macros['carb_g']['direct_bias']:+.2f} |"
        ),
        (
            f"| Fat (g) | {macros['fat_g']['n']} | {macros['fat_g']['ours_mae']:.2f} | {macros['fat_g']['direct_mae']:.2f} | "
            f"{macros['fat_g']['delta_ours_minus_direct']:+.2f} | {macros['fat_g']['ours_bias']:+.2f} | {macros['fat_g']['direct_bias']:+.2f} |"
        ),
        "",
        f"Fat MAE is higher by {macros['fat_g']['delta_ours_minus_direct']:+.2f} g. "
        f"The signed fat error is {macros['fat_g']['ours_bias']:+.2f} g here and {macros['fat_g']['direct_bias']:+.2f} g for the direct model, "
        "so the higher fat MAE is a wider scatter. Protein and carbohydrate MAEs are lower, and both are signed low.",
        "",
        "## STOP, ABSTAIN, and unfinished components",
        "",
        "The full-set MAE scores the vector at the first STOP or ABSTAIN, including abstentions. "
        f"Trace totals match the paired commit on {summary['n_trace_aligned']}/{n} plates. "
        f"Portions clipped to the 1 g or 1500 g bound: {summary['n_clipped_portions']}.",
        "",
        "| Slice | n | Energy MAE, this system | Energy MAE, direct | ΔE | Mass MAE, this system | Mass MAE, direct | ΔM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        slice_line("stop", "STOP"),
        slice_line("abstain", "ABSTAIN"),
        slice_line("correct_code_terminal", "Ended on CORRECT_CODE"),
        slice_line("plate_complete", "Every named component mass-written"),
        slice_line("plate_incomplete", "A named component left unwritten"),
        slice_line("single_component_complete", "One component, mass-written"),
        slice_line("multi_component_complete", "Several components, all mass-written"),
        slice_line("no_correct_code", "No CORRECT_CODE through the commit"),
        slice_line("any_correct_code", "At least one CORRECT_CODE"),
        "",
        "| Decision | Components | n | Mean ΔE (kcal) | Mean ΔM (g) | Addition to the energy mean (kcal) | Mean signed calorie error, this system |",
        "|---|---|---:|---:|---:|---:|---:|",
        *cross_lines,
        "",
        "ABSTAIN is included in the full-set MAE. On this pool its energy MAE is lower than the direct model's. "
        "The cross-tab shows where the pooled mean comes from: "
        f"{stop_done['n']} finished STOP plates add {stop_done['contribution_to_mean_energy_kcal']:+.2f} kcal, "
        f"and {stop_open['n']} STOP plates with a component left unwritten add {stop_open['contribution_to_mean_energy_kcal']:+.2f} kcal. "
        f"The {slices['correct_code_terminal']['n']} CORRECT_CODE endings add {contribution(slices['correct_code_terminal']):+.2f} kcal. "
        f"{summary['n_never_called_mass']} paired plates never call the mass estimator ({never_called}).",
        "",
        "## Q/Cost and the rule-utility fence",
        "",
        "The trace does not store the fence decision. "
        f"Replaying the frozen Q weights and the saved fence (τ = {selector['tau']:.6f}) reproduces "
        f"**{selector['n_match']}/{selector['n_steps']}** logged actions, with {selector['n_failures']} scoring failures. "
        f"{q_steps} matched steps are the Q/Cost choice and {rule_steps} are rule-utility steps. "
        "A rule-utility step is a case where Q/Cost preferred STOP or ABSTAIN by less than the fence, and the rule chose a tool instead. "
        f"The committing action is Q/Cost on {q_terminals} paired plates and rule utility on {rule_terminals}. "
        + (
            f"Every paired plate contains at least one rule-utility step (n={deferred['n']}), and every committing action is Q/Cost. "
            if held["n"] == 0
            else (
                f"Plates with at least one earlier rule-utility step: n={deferred['n']}, "
                f"ΔE {deferred['energy_delta_ours_minus_direct']:+.2f} kcal, ΔM {deferred['mass_delta_ours_minus_direct']:+.2f} g. "
                f"Plates with none: n={held['n']}, "
                f"ΔE {held['energy_delta_ours_minus_direct']:+.2f} kcal, ΔM {held['mass_delta_ours_minus_direct']:+.2f} g. "
            )
        )
        + "This audit does not resimulate the trajectories with the fence removed.",
        "",
        "## What stays in place",
        "",
        "The published full-set comparison stands. Mass MAE is lower by 17.3513 g, and that gap's bootstrap interval lies below zero. "
        "Energy MAE is higher by 1.4210 kcal, and that gap's interval contains both signs. "
        "The median plate has a lower energy absolute error. "
        "Finished one-component plates have lower mass and energy MAE. "
        "Finished multi-component plates have a much lower mass MAE and a higher energy MAE, and the stored calorie head on those plates is closer than the summed food codes. "
        f"The {stop_open['n']} STOP plates that leave a named component unwritten have a mean signed calorie error of {stop_open['ours_mean_signed_energy_kcal']:+.1f} kcal "
        f"and add {stop_open['contribution_to_mean_energy_kcal']:+.2f} kcal to the pooled energy mean. "
        "Versus the grams already on the component, the mass write lowers both mass MAE and energy MAE. "
        "Versus closed-loop control, the point estimates are lower and the paired intervals contain both signs.",
        "",
        "This note does not start a mass retrain, a Q retrain, a retrieval re-rank, or another full-set run. "
        "The label-gram and label-density totals are arithmetic. They are not proposed commits.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/p0d_error_audit.py",
        "```",
        "",
        "Reads `results/p0d/same_budget.json`, `same_budget_dishes.jsonl`, `ours_new_rgb_test.jsonl`, "
        "`ours_new_rgb_test_summary.json`, `report_same_budget.txt`, the frozen direct-model jsonl for macros, "
        "and the frozen Q weights only to replay actions. "
        f"Input sha256 prefixes: dishes `{summary['inputs']['same_budget_dishes_sha256'][:16]}`, "
        f"table `{summary['inputs']['same_budget_sha256'][:16]}`, "
        f"trace `{summary['inputs']['ours_new_rgb_test_sha256'][:16]}`, "
        f"direct `{summary['inputs']['direct_jsonl_sha256'][:16]}`, "
        f"Q `{summary['inputs']['q_weights_sha256_prefix']}`.",
        "",
    ]
    return "\n".join(lines) + "\n"

def render_txt(summary: Mapping[str, Any]) -> str:
    """Plain companion. The markdown report is the one with figures."""
    text = render_report(summary)
    kept = []
    for line in text.splitlines():
        if line.startswith("![") or line.startswith("```") or line.startswith("#"):
            if line.startswith("#"):
                kept.append(line.lstrip("# ").upper())
            continue
        kept.append(line)
    return "\n".join(kept).strip() + "\n"


def public_record(row: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "dish_id",
        "quadrant",
        "decision_action",
        "tool_count",
        "n_hypotheses",
        "n_mass_writes",
        "plate_complete",
        "n_correct_code",
        "label_energy_kcal",
        "label_mass_g",
        "label_density",
        "ours_energy_kcal",
        "ours_mass_g",
        "ours_density",
        "direct_energy_kcal",
        "direct_mass_g",
        "direct_density",
        "closed_energy_kcal",
        "closed_mass_g",
        "closed_density",
        "energy_delta_ae",
        "mass_delta_ae",
        "ours_mass_channel",
        "ours_density_channel",
        "ours_cancellation",
        "direct_mass_channel",
        "direct_density_channel",
        "direct_cancellation",
        "ours_opposite",
        "direct_opposite",
        "ours_perfect_grams_kcal",
        "ours_perfect_density_kcal",
        "direct_perfect_grams_kcal",
        "direct_perfect_density_kcal",
        "rewrite_accounted",
        "grid_energy_kcal",
        "grid_mass_g",
        "model_energy_kcal",
        "model_mass_g",
        "selector_source",
        "selector_replay_matched",
        "n_rule_utility_steps",
        "foods",
    )
    return {key: row.get(key) for key in keep}


def write_outputs(payload: Mapping[str, Any], summary: Mapping[str, Any], out_dir: Path) -> None:
    resolved = out_dir.resolve()
    if resolved != OUT.resolve():
        raise RuntimeError(f"refusing to write outside {OUT}")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(payload["records"])
    scatter_svg(out_dir / "fig_mass_energy_delta.svg", records)
    quads = [row for row in summary["quadrants"] if row.get("n", 0) and row["name"] != "tie"]
    bar_svg(
        out_dir / "fig_energy_gap_by_quadrant.svg",
        [
            {
                "mass_down_energy_down": "Both\ncloser",
                "mass_down_energy_up": "Mass closer,\nenergy farther",
                "mass_up_energy_down": "Mass farther,\nenergy closer",
                "mass_up_energy_up": "Both\nfarther",
            }.get(row["name"], row["label"])
            for row in quads
        ],
        [row["contribution_to_mean_energy_kcal"] for row in quads],
        "Each group's addition to the mean energy gap (kcal)",
        "kcal added to the 505-plate mean",
        [COLOR_BOTH_BETTER, COLOR_MASS_ONLY, COLOR_ENERGY_ONLY, COLOR_BOTH_WORSE][: len(quads)],
    )
    channel_svg(out_dir / "fig_error_channels.svg", summary)
    (out_dir / "error_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "paired_residuals.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(public_record(row), ensure_ascii=False) + "\n")
    report = render_report(summary)
    (out_dir / "report_error_audit.md").write_text(report, encoding="utf-8")
    (out_dir / "report_error_audit.txt").write_text(render_txt(summary), encoding="utf-8")


def self_check() -> None:
    parts = channels(220.0, 200.0, 200.0, 100.0)
    if parts is None:
        raise AssertionError("channels dropped a positive mass")
    if abs(parts["mass_channel_kcal"] - 110.0) > 1e-9:
        raise AssertionError(parts)
    if abs(parts["density_channel_kcal"] + 90.0) > 1e-9:
        raise AssertionError(parts)
    if abs(parts["mass_channel_kcal"] + parts["density_channel_kcal"] - 20.0) > 1e-9:
        raise AssertionError(parts)
    if abs(parts["cancellation_kcal"] - 180.0) > 1e-9:
        raise AssertionError(parts)
    oracle = oracle_energy(220.0, 200.0, 200.0, 100.0)
    if oracle is None or abs(oracle["perfect_grams_kcal"] - 110.0) > 1e-9 or abs(oracle["perfect_density_kcal"] - 400.0) > 1e-9:
        raise AssertionError(oracle)
    if channels(10.0, 0.0, 10.0, 5.0) is not None:
        raise AssertionError("zero mass should skip the density split")
    dropped = drop_until_mean_nonpositive(np.asarray([10.0, 10.0, -1.0, -1.0]))
    if dropped != 2:
        raise AssertionError(dropped)
    if quadrant_name(-1.0, 2.0) != "mass_down_energy_up":
        raise AssertionError("quadrant")
    sample = bootstrap_mean(np.asarray([1.0, -1.0, 1.0, -1.0]), seed=0)
    if sample["n"] != 4 or sample["draws"] != BOOTSTRAP_DRAWS:
        raise AssertionError(sample)
    print("p0d error-audit self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only P0d energy-versus-mass audit")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    payload = build_records()
    summary = aggregate(payload)
    write_outputs(payload, summary, OUT)
    energy = summary["energy_mae"]["delta_ours_minus_direct"]
    mass = summary["mass_mae"]["delta_ours_minus_direct"]
    print(f"paired n={summary['n_paired']} ΔE {energy:+.4f} kcal ΔM {mass:+.4f} g")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
