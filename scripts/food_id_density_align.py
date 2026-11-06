#!/usr/bin/env python3
"""Replay the same-head food-code swap on the saved paired set.

The mass estimator and the calibrated selector are not run again. Grams and
actions stay as recorded. On each finished multi-component plate the food-table
energy is recomputed from a same-head code that was already in that component's
retrieve window. Cafe labels are used only after that choice, to score.

    python scripts/food_id_density_align.py --self-check
    python scripts/food_id_density_align.py

Writes ``results/food_id_density/``. Does not write ``docs/STATUS.md``, the
P0d trajectories, Q weights, the mass checkpoint, or the frozen weak-step logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

import vdr.controller.features  # noqa: E402,F401
from vdr.eval.same_budget import audit_mass_portions, group_steps  # noqa: E402
from vdr.tools.density_align import align_multicomponent_codes, run_self_check  # noqa: E402

P0D = ROOT / "results" / "p0d"
OUT = ROOT / "results" / "food_id_density"
DISHES = P0D / "same_budget_dishes.jsonl"
TRACE = P0D / "ours_new_rgb_test.jsonl"
STATUS = ROOT / "docs" / "STATUS.md"
FROZEN = (
    STATUS,
    ROOT / "results" / "qphi" / "q_phi.npz",
    ROOT / "results" / "qphi" / "calibration.json",
    ROOT / "results" / "mass_estimator" / "train_meta.json",
    ROOT / "results" / "weak_steps" / "rgb_train_fit_all.jsonl",
    ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.jsonl",
    TRACE,
    DISHES,
)

PAIRED_N = 505
MULTI_N = 50
# Published P0d / #48 point estimates. The replay must start from these.
PAIRED_ENERGY_BEFORE = 98.5521
PAIRED_MASS_BEFORE = 58.9952
MULTI_ENERGY_BEFORE = 190.79
MULTI_MASS_BEFORE = 44.30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in FROZEN if path.is_file()}


def load_dishes(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_method.setdefault(str(row["method"]), {})[str(row["dish_id"])] = row
    return by_method


def finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def mae(errors: Sequence[float]) -> float:
    if not errors:
        raise RuntimeError("empty MAE")
    return sum(abs(value) for value in errors) / len(errors)


def mass_writes(steps: Sequence[Mapping[str, Any]], source_step_i: int) -> tuple[int | None, list[dict[str, Any]]]:
    """Successful mass writes before the committed step, plus the hypothesis count."""
    n_hyp = None
    writes: list[dict[str, Any]] = []
    for step in steps:
        obs = step.get("obs_t") if isinstance(step.get("obs_t"), Mapping) else {}
        if n_hyp is None and obs.get("vlm_n_hypotheses") is not None:
            try:
                n_hyp = int(obs["vlm_n_hypotheses"])
            except (TypeError, ValueError):
                n_hyp = None
        if int(step.get("step_i") or -1) >= int(source_step_i):
            continue
        if step.get("action_t") != "RE_PORTION":
            continue
        if obs.get("tool_failed") or obs.get("portion_src") != "mass_estimator":
            continue
        refs = (obs.get("nutrient_refs") or {}).get("fndds") if isinstance(obs.get("nutrient_refs"), Mapping) else None
        if not isinstance(refs, list):
            continue
        code = str(step.get("code_i") or "")
        candidates = []
        bound = None
        for ref in refs:
            if not isinstance(ref, Mapping) or not ref.get("available"):
                continue
            per = ref.get("per_100g") if isinstance(ref.get("per_100g"), Mapping) else {}
            energy = finite(per.get("energy_kcal"))
            if energy is None:
                continue
            score = finite(ref.get("score"))
            item = {
                "code": str(ref.get("code") or ""),
                "name": "" if ref.get("name") is None else str(ref.get("name")),
                "score": 0.0 if score is None else score,
                "per_100g": {
                    key: float(per[key])
                    for key in ("energy_kcal", "protein_g", "carb_g", "fat_g")
                    if finite(per.get(key)) is not None
                },
            }
            if not item["code"]:
                continue
            candidates.append(item)
            if item["code"] == code:
                bound = item
        model = obs.get("nutrients_model") if isinstance(obs.get("nutrients_model"), Mapping) else {}
        writes.append(
            {
                "step_i": int(step["step_i"]),
                "query": "" if step.get("hyp_i") is None else str(step.get("hyp_i")),
                "code": code,
                "name": "" if bound is None else bound["name"],
                "grams": finite(obs.get("portion_g")),
                "candidates": candidates,
                "bound_in_refs": bound is not None,
                "model_energy": finite(model.get("energy_kcal")),
                "model_mass": finite(obs.get("mass_g")),
            }
        )
    return n_hyp, writes


def replay_plate(row: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = row.get("source_step_i")
    base = {
        "dish_id": row["dish_id"],
        "energy_kcal": float(row["energy_kcal"]),
        "mass_g": float(row["mass_g"]),
        "label_energy_kcal": float(row["label_energy_kcal"]),
        "label_mass_g": float(row["label_mass_g"]),
        "applied": False,
        "reason": "not_replayed",
        "n_changed": 0,
        "changes": [],
        "head_energy_kcal": None,
        "head_mass_g": None,
        "table_before_kcal": None,
        "table_after_kcal": None,
    }
    if source is None:
        base["reason"] = "no_source_step"
        return base
    n_hyp, writes = mass_writes(steps, int(source))
    base["n_hypotheses"] = n_hyp
    base["n_mass_writes"] = len(writes)
    if not n_hyp or n_hyp < 2:
        base["reason"] = "single_or_unknown"
        return base
    if len(writes) != n_hyp:
        base["reason"] = "unfinished"
        return base
    if any(not write["bound_in_refs"] or write["grams"] is None or write["grams"] <= 0 for write in writes):
        base["reason"] = "missing_code"
        return base
    if any(write["model_energy"] is None for write in writes):
        base["reason"] = "no_head"
        return base
    head = float(writes[-1]["model_energy"])
    first = float(writes[0]["model_energy"])
    base["head_energy_kcal"] = head
    base["head_mass_g"] = writes[-1]["model_mass"]
    base["head_repeat_span_kcal"] = abs(head - first)
    result = align_multicomponent_codes(
        [
            {
                "query": write["query"],
                "code": write["code"],
                "name": write["name"],
                "grams": write["grams"],
                "candidates": write["candidates"],
            }
            for write in writes
        ],
        dish_energy_kcal=head,
    )
    base["reason"] = result["reason"]
    base["table_before_kcal"] = result.get("before_energy_kcal")
    base["table_after_kcal"] = result.get("after_energy_kcal")
    base["applied"] = bool(result["applied"])
    base["n_changed"] = int(result.get("n_changed") or 0)
    if result["applied"] and result["before_energy_kcal"] is not None and result["after_energy_kcal"] is not None:
        base["energy_kcal"] = float(row["energy_kcal"]) - float(result["before_energy_kcal"]) + float(result["after_energy_kcal"])
        base["changes"] = [
            {
                "query": item.get("query"),
                "from_name": item.get("from_name"),
                "to_name": item.get("name"),
                "from_code": item.get("from_code"),
                "to_code": item.get("code"),
                "grams": item.get("grams"),
            }
            for item in result["components"]
            if item.get("changed")
        ]
    return base


def slice_mae(rows: Sequence[Mapping[str, Any]], energy_key: str, mass_key: str = "mass_g") -> dict[str, float]:
    energy = [abs(float(row[energy_key]) - float(row["label_energy_kcal"])) for row in rows]
    mass = [abs(float(row[mass_key]) - float(row["label_mass_g"])) for row in rows]
    return {"n": float(len(rows)), "energy_mae_kcal": mae(energy), "mass_mae_g": mae(mass)}


def other_mae(rows: Sequence[Mapping[str, Any]], table: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    energy = []
    mass = []
    for row in rows:
        other = table[str(row["dish_id"])]
        energy.append(abs(float(other["energy_kcal"]) - float(other["label_energy_kcal"])))
        mass.append(abs(float(other["mass_g"]) - float(other["label_mass_g"])))
    return {"n": float(len(rows)), "energy_mae_kcal": mae(energy), "mass_mae_g": mae(mass)}


def f4(value: float) -> str:
    return f"{value:.4f}"


def f2(value: float) -> str:
    return f"{value:.2f}"


def write_reports(payload: Mapping[str, Any]) -> None:
    paired = payload["paired"]
    multi = payload["multi_component_finished"]
    hollow = payload["hollow_re_portion"]
    lines = [
        "# Food-code density alignment on finished multi-component plates",
        "",
        "The mass estimator's grams stay. On a plate whose every named component already has a mass write, the committed calorie is still the bound food code's FNDDS kcal per 100 g times those grams. This pass may replace that code with another code from the same retrieve window when three things are true: the new code's FNDDS head does not add a food noun, it does not introduce a dish form or a prepared form the query did not ask for, and the plate's food-table energy moves at least 15% closer to the mass estimator's own calorie total. The 15% figure is the existing density-gain cutoff. It was not chosen on this test set.",
        "",
        "Actions and grams are the ones already stored for the paired comparison. The selector is not asked again, and the mass network is not trained again. Labels are used only to score the calorie and the gram error after the code choice.",
        "",
        "## Paired plates",
        "",
        f"n={int(paired['n'])}. Mass mean absolute error is unchanged because the grams are unchanged.",
        "",
        "| | Energy MAE (kcal) | Mass MAE (g) |",
        "|---|---:|---:|",
        f"| Mass estimator + calibrated selection, before | {f4(paired['before_energy_mae_kcal'])} | {f4(paired['before_mass_mae_g'])} |",
        f"| Same plates after the food-code swap | {f4(paired['after_energy_mae_kcal'])} | {f4(paired['after_mass_mae_g'])} |",
        f"| Direct vision-language model | {f4(paired['direct_energy_mae_kcal'])} | {f4(paired['direct_mass_mae_g'])} |",
        f"| Closed-loop control | {f4(paired['closed_energy_mae_kcal'])} | {f4(paired['closed_mass_mae_g'])} |",
        "",
        (
            f"Energy MAE changes by {paired['after_energy_mae_kcal'] - paired['before_energy_mae_kcal']:+.4f} kcal "
            f"against the previous food-table commit and by "
            f"{paired['after_energy_mae_kcal'] - paired['direct_energy_mae_kcal']:+.4f} kcal against the direct vision-language model. "
            f"Mass MAE stays at {f4(paired['after_mass_mae_g'])} g."
        ),
        "",
        "## Finished multi-component plates",
        "",
        (
            f"n={int(multi['n'])}. These are the plates where every named component received a mass write and the plate named more than one component. "
            f"The swap ran on {multi['n_applied']} of them and changed {multi['n_components_changed']} component codes. "
            f"Absolute calorie error fell on {multi['n_energy_better']} plates and rose on {multi['n_energy_worse']}."
        ),
        "",
        "| | Energy MAE (kcal) | Mass MAE (g) |",
        "|---|---:|---:|",
        f"| Food-table commit, before | {f2(multi['before_energy_mae_kcal'])} | {f2(multi['before_mass_mae_g'])} |",
        f"| After the same-head swap | {f2(multi['after_energy_mae_kcal'])} | {f2(multi['after_mass_mae_g'])} |",
        f"| Direct vision-language model | {f2(multi['direct_energy_mae_kcal'])} | {f2(multi['direct_mass_mae_g'])} |",
        f"| Stored calorie head (not a commit) | {f2(multi['head_energy_mae_kcal'])} | {f2(multi['head_mass_mae_g'])} |",
        "",
        (
            f"On this slice the food-table energy MAE was {multi['before_energy_mae_kcal'] - multi['direct_energy_mae_kcal']:+.2f} kcal "
            f"versus the direct model before the swap and "
            f"{multi['after_energy_mae_kcal'] - multi['direct_energy_mae_kcal']:+.2f} kcal after it. "
            "The stored calorie head is listed only as a reading of the forward pass the swap uses as its target. The scored commit is still the food-table product."
        ),
        "",
        "## Portion checks",
        "",
        (
            f"Successful mass portions in the saved trajectories: {hollow['re_portion_success']}. "
            f"Hollow portions (a success that is not the mass estimator): {hollow['hollow']}. "
            f"Portions whose dish energy and mass were unchanged at the next step: {hollow['re_portion_unchanged_dish_nutrition']}. "
            "The swap does not add a portion step and does not put the old grams back, so a portion that already changed the dish total still changes it."
        ),
        "",
        "## Examples",
        "",
        "Absolute calorie error on the finished multi-component slice. A negative change is a smaller error.",
        "",
    ]
    for example in payload["examples"]:
        change = example["energy_ae_after_kcal"] - example["energy_ae_before_kcal"]
        lines.append(
            f"- `{example['dish_id']}`: label {example['label_energy_kcal']:.0f} kcal, "
            f"mass-estimator head {example['head_energy_kcal']:.0f} kcal, "
            f"food table {example['table_before_kcal']:.0f} → {example['table_after_kcal']:.0f} kcal, "
            f"absolute error {example['energy_ae_before_kcal']:.0f} → {example['energy_ae_after_kcal']:.0f} "
            f"({change:+.0f})."
        )
        for swap in example["changes"]:
            lines.append(
                f"  - {swap['query']}: {swap['from_name']} → {swap['to_name']} ({swap['grams']:.0f} g)"
            )
    lines.extend(
        [
            "",
            "## What this does not do",
            "",
            "The mass network is not trained again. The Q weights and the terminal-margin fence are not changed and are not refit. Retrieval is not reranked and no new candidate list is fetched. Coverage and early stopping are untouched. `docs/STATUS.md` is untouched. Labels were not used to pick a code, a cutoff, or a gram weight.",
            "",
            "The food noun is the FNDDS text before the first comma. A preparation written after that comma, such as `Apple, dried` against `Apple, raw`, can still be selected when it pulls the plate total toward the calorie head. Seven of the fifty finished multi-component plates have a larger calorie error after the swap. The head itself remains closer than the swapped table (77.75 kcal versus 131.65 kcal on that slice).",
            "",
            "A later full pass would let the selector see the swapped calorie total before it stops. This table holds the recorded actions fixed, which is the comparison that isolates the food-code change.",
            "",
            "```bash",
            "python scripts/food_id_density_align.py",
            "```",
            "",
        ]
    )
    text = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    (OUT / "report.txt").write_text(text, encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pick_examples(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    changed = [row for row in rows if row.get("applied")]
    changed.sort(key=lambda row: (row["energy_ae_after"] - row["energy_ae_before"], row["dish_id"]))
    chosen = []
    if changed:
        chosen.append(changed[0])
        if len(changed) > 2:
            chosen.append(changed[len(changed) // 2])
        if changed[-1]["dish_id"] != chosen[0]["dish_id"]:
            chosen.append(changed[-1])
    examples = []
    for row in chosen:
        examples.append(
            {
                "dish_id": row["dish_id"],
                "label_energy_kcal": row["label_energy_kcal"],
                "head_energy_kcal": row["head_energy_kcal"],
                "table_before_kcal": row["table_before_kcal"],
                "table_after_kcal": row["table_after_kcal"],
                "energy_ae_before_kcal": row["energy_ae_before"],
                "energy_ae_after_kcal": row["energy_ae_after"],
                "changes": row["changes"],
            }
        )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the same-head food-code swap on the paired set")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        print("food-id density self-check ok")
        return 0
    run_self_check()
    before = frozen_hashes()
    missing = [path.name for path in FROZEN if not path.is_file()]
    if missing:
        raise SystemExit("missing frozen inputs: " + ", ".join(missing))
    dishes = load_dishes(DISHES)
    ours = dishes["ours_new"]
    direct = dishes["direct_vlm"]
    closed = dishes["ours_old"]
    grouped = group_steps(TRACE)
    hollow = audit_mass_portions(grouped)
    if hollow["hollow"] != 0 or hollow["re_portion_unchanged_dish_nutrition"] != 0:
        raise SystemExit(f"portion audit is not clean: {hollow}")
    paired_rows = []
    multi_rows = []
    reasons: dict[str, int] = {}
    sum_gap = 0.0
    n_gap = 0
    max_gap = 0.0
    for dish_id, row in sorted(ours.items()):
        if not row.get("paired") or not row.get("scored"):
            continue
        if finite(row.get("energy_kcal")) is None or finite(row.get("mass_g")) is None:
            continue
        replay = replay_plate(row, grouped.get(dish_id, []))
        replay["energy_ae_before"] = abs(float(row["energy_kcal"]) - float(row["label_energy_kcal"]))
        replay["energy_ae_after"] = abs(float(replay["energy_kcal"]) - float(row["label_energy_kcal"]))
        replay["mass_ae"] = abs(float(replay["mass_g"]) - float(row["label_mass_g"]))
        reasons[replay["reason"]] = reasons.get(replay["reason"], 0) + 1
        if replay.get("table_before_kcal") is not None and replay["reason"] in {"aligned", "gap_below_gain"}:
            gap = abs(float(row["energy_kcal"]) - float(replay["table_before_kcal"]))
            sum_gap += gap
            n_gap += 1
            max_gap = max(max_gap, gap)
        paired_rows.append(replay)
        if replay["reason"] in {"aligned", "gap_below_gain"}:
            multi_rows.append(replay)
    if len(paired_rows) != PAIRED_N:
        raise SystemExit(f"paired n={len(paired_rows)}, expected {PAIRED_N}")
    if len(multi_rows) != MULTI_N:
        raise SystemExit(f"finished multi-component n={len(multi_rows)}, expected {MULTI_N}; reasons={reasons}")
    before_energy = mae([row["energy_ae_before"] for row in paired_rows])
    before_mass = mae([row["mass_ae"] for row in paired_rows])
    after_energy = mae([row["energy_ae_after"] for row in paired_rows])
    after_mass = mae([abs(float(row["mass_g"]) - float(row["label_mass_g"])) for row in paired_rows])
    if abs(before_energy - PAIRED_ENERGY_BEFORE) > 1e-3 or abs(before_mass - PAIRED_MASS_BEFORE) > 1e-3:
        raise SystemExit(f"paired baseline drifted: E {before_energy} M {before_mass}")
    if abs(after_mass - before_mass) > 1e-6:
        raise SystemExit("mass MAE moved; the swap was supposed to keep the grams")
    multi_before_e = mae([row["energy_ae_before"] for row in multi_rows])
    multi_before_m = mae([row["mass_ae"] for row in multi_rows])
    multi_after_e = mae([row["energy_ae_after"] for row in multi_rows])
    multi_after_m = mae([row["mass_ae"] for row in multi_rows])
    if abs(multi_before_e - MULTI_ENERGY_BEFORE) > 0.02 or abs(multi_before_m - MULTI_MASS_BEFORE) > 0.02:
        raise SystemExit(f"multi baseline drifted: E {multi_before_e} M {multi_before_m}")
    direct_paired = other_mae(paired_rows, direct)
    closed_paired = other_mae(paired_rows, closed)
    direct_multi = other_mae(multi_rows, direct)
    head_energy = mae([abs(float(row["head_energy_kcal"]) - float(row["label_energy_kcal"])) for row in multi_rows])
    head_mass = mae([abs(float(row["head_mass_g"]) - float(row["label_mass_g"])) for row in multi_rows])
    applied = [row for row in multi_rows if row["applied"]]
    better = sum(1 for row in multi_rows if row["energy_ae_after"] < row["energy_ae_before"] - 1e-6)
    worse = sum(1 for row in multi_rows if row["energy_ae_after"] > row["energy_ae_before"] + 1e-6)
    payload = {
        "definition": (
            "Same recorded actions and grams. On a finished multi-component plate, "
            "replace a bound FNDDS code with a same-head code from its existing retrieve "
            "window when the food-table energy moves at least 15% closer to the mass "
            "estimator calorie head. Labels score the result and do not choose the code."
        ),
        "paired": {
            "n": len(paired_rows),
            "before_energy_mae_kcal": before_energy,
            "after_energy_mae_kcal": after_energy,
            "before_mass_mae_g": before_mass,
            "after_mass_mae_g": after_mass,
            "direct_energy_mae_kcal": direct_paired["energy_mae_kcal"],
            "direct_mass_mae_g": direct_paired["mass_mae_g"],
            "closed_energy_mae_kcal": closed_paired["energy_mae_kcal"],
            "closed_mass_mae_g": closed_paired["mass_mae_g"],
            "energy_mae_change_kcal": after_energy - before_energy,
            "energy_minus_direct_before_kcal": before_energy - direct_paired["energy_mae_kcal"],
            "energy_minus_direct_after_kcal": after_energy - direct_paired["energy_mae_kcal"],
        },
        "multi_component_finished": {
            "n": len(multi_rows),
            "n_applied": len(applied),
            "n_components_changed": int(sum(row["n_changed"] for row in applied)),
            "n_energy_better": better,
            "n_energy_worse": worse,
            "before_energy_mae_kcal": multi_before_e,
            "after_energy_mae_kcal": multi_after_e,
            "before_mass_mae_g": multi_before_m,
            "after_mass_mae_g": multi_after_m,
            "direct_energy_mae_kcal": direct_multi["energy_mae_kcal"],
            "direct_mass_mae_g": direct_multi["mass_mae_g"],
            "head_energy_mae_kcal": head_energy,
            "head_mass_mae_g": head_mass,
            "mean_abs_commit_minus_component_sum_kcal": None if n_gap == 0 else sum_gap / n_gap,
            "max_abs_commit_minus_component_sum_kcal": max_gap,
        },
        "hollow_re_portion": hollow,
        "reasons_on_paired": reasons,
        "examples": pick_examples(multi_rows),
        "plates": [
            {
                "dish_id": row["dish_id"],
                "applied": row["applied"],
                "reason": row["reason"],
                "n_changed": row["n_changed"],
                "energy_before_kcal": float(ours[row["dish_id"]]["energy_kcal"]),
                "energy_after_kcal": row["energy_kcal"],
                "mass_g": row["mass_g"],
                "label_energy_kcal": row["label_energy_kcal"],
                "label_mass_g": row["label_mass_g"],
            }
            for row in paired_rows
        ],
        "q_weights_retrained": False,
        "mass_retrained": False,
        "retrieval_rerank": False,
        "uses_rgb_test_labels_to_choose_codes": False,
        "status_file": "docs/STATUS.md was not modified.",
    }
    write_reports(payload)
    after = frozen_hashes()
    drifted = [name for name, digest in before.items() if after.get(name) != digest]
    if drifted:
        raise SystemExit("frozen files changed: " + ", ".join(drifted))
    print(
        f"paired n={len(paired_rows)} energy {before_energy:.4f} -> {after_energy:.4f} "
        f"mass {before_mass:.4f} -> {after_mass:.4f}"
    )
    print(
        f"multi n={len(multi_rows)} energy {multi_before_e:.2f} -> {multi_after_e:.2f} "
        f"applied {len(applied)} better {better} worse {worse} hollow {hollow['hollow']}"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
