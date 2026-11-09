#!/usr/bin/env python3
"""Reproducible external held-out plan for gate ③.

The paired 505 development dishes stay on the diagnostic side. Fit dishes are
the official rgb_train ids whose sha256 cut is ``fit`` (the Mass / Q training
dishes). Held-out is every cafe2 dish that has an overhead RGB image. Membership
uses the source file and the image path only. Nutrition numbers are not read,
and they are not a filter.

    python scripts/heldout_plan.py --self-check
    python scripts/heldout_plan.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.data.n5k import overhead_rgb, parse_dish_metadata_line, read_ids  # noqa: E402
from vdr.weak.teacher import pool_of_dish  # noqa: E402

N5K = ROOT / "data" / "nutrition5k"
CAFE1 = N5K / "metadata" / "dish_metadata_cafe1.csv"
CAFE2 = N5K / "metadata" / "dish_metadata_cafe2.csv"
PAIRED_ROWS = ROOT / "results" / "p0d" / "same_budget_dishes.jsonl"
COVERAGE_SCORES = ROOT / "results" / "coverage_early_stop" / "paired_scores.jsonl"
COVERAGE_SUMMARY = ROOT / "results" / "coverage_early_stop" / "summary.json"
FIT_STEPS = ROOT / "results" / "weak_steps" / "rgb_train_fit_all.jsonl"
OUT = ROOT / "results" / "heldout_plan"
EXPECTED_DEV = 505
EXPECTED_FIT = 3679


class HeldoutPlanError(RuntimeError):
    """The split is not disjoint, or a frozen artifact does not match."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_ids(path: Path) -> list[str]:
    """Dish ids from a cafe metadata file. Numeric fields are not parsed."""
    ids: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            dish_id = line.split(",", 1)[0].strip()
            if not dish_id.startswith("dish_"):
                raise HeldoutPlanError(f"{path.name}:{line_no} does not start with a dish id")
            if dish_id in seen:
                raise HeldoutPlanError(f"{path.name} repeats {dish_id}")
            seen.add(dish_id)
            ids.append(dish_id)
    return ids


def jsonl_ids(path: Path, *, paired_method: str | None = None) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if paired_method is not None:
                if row.get("method") != paired_method or row.get("paired") is not True:
                    continue
            dish_id = row.get("dish_id")
            if not isinstance(dish_id, str) or not dish_id.startswith("dish_"):
                raise HeldoutPlanError(f"{path.name} has a row without a dish id")
            ids.add(dish_id)
    return ids


def paired_by_method(path: Path) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("paired") is not True:
                continue
            method = str(row.get("method"))
            dish_id = row.get("dish_id")
            if not isinstance(dish_id, str):
                raise HeldoutPlanError(f"{path.name} paired row has no dish id")
            grouped.setdefault(method, set()).add(dish_id)
    return grouped


def cafe2_label_audit() -> dict[str, Any]:
    """Describe cafe2 scoring labels. Does not drop a dish.

    Every cafe2 dish-level energy is 0 in the official file, while the
    ingredient blocks carry the calories and the dish-level mass matches the
    ingredient grams. Cafe1 is the opposite: its dish-level energy matches the
    ingredient sum, which is why the paired 505 keeps ``load_labels``.
    """
    dishes = []
    with CAFE2.open(encoding="utf-8") as handle:
        for line in handle:
            dish = parse_dish_metadata_line(line)
            if dish is not None:
                dishes.append(dish)
    if len(dishes) != len(metadata_ids(CAFE2)):
        raise HeldoutPlanError("cafe2 label parse dropped a dish id")
    max_mass_gap = 0.0
    energy_column_zero = 0
    ingredient_energy_zero = 0
    for dish in dishes:
        ingredient_energy = sum(item.energy_kcal for item in dish.ingredients)
        ingredient_mass = sum(item.mass_g for item in dish.ingredients)
        max_mass_gap = max(max_mass_gap, abs(dish.mass_g - ingredient_mass))
        if dish.energy_kcal == 0.0:
            energy_column_zero += 1
        if ingredient_energy == 0.0:
            ingredient_energy_zero += 1
    if energy_column_zero != len(dishes):
        raise HeldoutPlanError(
            "cafe2 dish-level energy is not uniformly zero; refusing a mixed label rule"
        )
    # 1e-3 g is float dust on the summed ingredient grams, not a dish filter.
    if max_mass_gap > 1e-3:
        raise HeldoutPlanError(f"cafe2 dish mass does not match ingredient grams (max {max_mass_gap})")
    return {
        "mass": "dish-level mass_g. It equals the sum of ingredient grams on every cafe2 row.",
        "energy": (
            "sum of ingredient energy_kcal on the same cafe2 row. "
            "The dish-level energy column is 0 on every cafe2 row, so that column is not used."
        ),
        "n_cafe2": len(dishes),
        "n_dish_energy_column_zero": energy_column_zero,
        "n_ingredient_energy_zero": ingredient_energy_zero,
        "max_abs_dish_mass_minus_ingredient_g": max_mass_gap,
        "membership_uses_these_numbers": False,
        "development_505_label": "unchanged cafe1 dish-level load_labels; not recomputed here",
    }


def cafe2_score_labels(dish_ids: Iterable[str]) -> dict[str, dict[str, float]]:
    """Score targets for the already chosen held-out ids. Not a membership filter."""
    wanted = list(dish_ids)
    by_id: dict[str, dict[str, float]] = {}
    with CAFE2.open(encoding="utf-8") as handle:
        for line in handle:
            dish = parse_dish_metadata_line(line)
            if dish is None:
                continue
            by_id[dish.dish_id] = {
                "energy_kcal": sum(item.energy_kcal for item in dish.ingredients),
                "mass_g": float(dish.mass_g),
            }
    missing = [dish_id for dish_id in wanted if dish_id not in by_id]
    if missing:
        raise HeldoutPlanError("held-out id missing from cafe2 metadata: " + ", ".join(missing[:5]))
    audit = cafe2_label_audit()
    if audit["n_dish_energy_column_zero"] != audit["n_cafe2"]:
        raise HeldoutPlanError("cafe2 energy column is no longer uniformly zero")
    return {dish_id: by_id[dish_id] for dish_id in wanted}


def write_ids(path: Path, ids: Iterable[str]) -> None:
    text = "".join(f"{dish_id}\n" for dish_id in ids)
    path.write_text(text, encoding="utf-8")


def read_id_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def identify() -> dict[str, Any]:
    """Build the split from frozen files. Does not write and does not score."""
    for path in (CAFE1, CAFE2, PAIRED_ROWS, COVERAGE_SCORES, COVERAGE_SUMMARY, FIT_STEPS):
        if not path.is_file():
            raise HeldoutPlanError(f"missing {path.relative_to(ROOT)}")

    cafe1 = metadata_ids(CAFE1)
    cafe2 = metadata_ids(CAFE2)
    cafe1_set = set(cafe1)
    cafe2_set = set(cafe2)
    if cafe1_set & cafe2_set:
        raise HeldoutPlanError("cafe1 and cafe2 share dish ids")

    rgb_train = read_ids("rgb_train")
    rgb_test = read_ids("rgb_test")
    rgb_train_set = set(rgb_train)
    rgb_test_set = set(rgb_test)
    if rgb_train_set & rgb_test_set:
        raise HeldoutPlanError("official rgb_train and rgb_test overlap")

    fit = sorted(dish_id for dish_id in rgb_train if pool_of_dish(dish_id) == "fit")
    internal_holdout = sorted(dish_id for dish_id in rgb_train if pool_of_dish(dish_id) == "holdout")
    fit_set = set(fit)
    internal_set = set(internal_holdout)
    if fit_set & internal_set:
        raise HeldoutPlanError("fit and the rgb_train internal hold-out overlap")
    if len(fit_set) + len(internal_set) != len(rgb_train_set):
        raise HeldoutPlanError("the hash cut does not cover rgb_train")

    logged_fit = jsonl_ids(FIT_STEPS)
    if logged_fit != fit_set:
        raise HeldoutPlanError(
            "weak-step fit dishes differ from the hash cut: "
            f"only_logged={len(logged_fit - fit_set)} only_hash={len(fit_set - logged_fit)}"
        )
    if len(fit_set) != EXPECTED_FIT:
        raise HeldoutPlanError(f"fit n={len(fit_set)}, frozen record is {EXPECTED_FIT}")

    paired = paired_by_method(PAIRED_ROWS)
    required = {"ours_new", "direct_vlm", "ours_old", "open_loop_bc"}
    if set(paired) != required:
        raise HeldoutPlanError(f"paired methods are {sorted(paired)}")
    dev = paired["ours_new"]
    if any(paired[method] != dev for method in required):
        raise HeldoutPlanError("paired dish ids differ across the four P0d methods")
    coverage_ids = jsonl_ids(COVERAGE_SCORES)
    if coverage_ids != dev:
        raise HeldoutPlanError(
            "coverage paired_scores dish ids differ from the P0d paired set: "
            f"only_coverage={len(coverage_ids - dev)} only_p0d={len(dev - coverage_ids)}"
        )
    summary = json.loads(COVERAGE_SUMMARY.read_text(encoding="utf-8"))
    summary_n = int(summary["paired"]["n"])
    if summary_n != len(dev):
        raise HeldoutPlanError(f"coverage summary n={summary_n} but the id set has {len(dev)}")
    if len(dev) != EXPECTED_DEV:
        raise HeldoutPlanError(f"development set n={len(dev)}, expected {EXPECTED_DEV}")
    if not dev <= rgb_test_set:
        raise HeldoutPlanError("a paired development dish is outside official rgb_test")
    if dev & rgb_train_set:
        raise HeldoutPlanError("a paired development dish is also in rgb_train")
    if not dev <= cafe1_set:
        raise HeldoutPlanError("a paired development dish is outside cafe1")

    heldout: list[str] = []
    missing_image: list[str] = []
    for dish_id in cafe2:
        if overhead_rgb(dish_id).is_file():
            heldout.append(dish_id)
        else:
            missing_image.append(dish_id)
    heldout.sort()
    missing_image.sort()
    held_set = set(heldout)
    intersections = {
        "fit": sorted(held_set & fit_set),
        "dev505": sorted(held_set & dev),
        "rgb_train": sorted(held_set & rgb_train_set),
        "rgb_train_internal_holdout": sorted(held_set & internal_set),
        "rgb_test": sorted(held_set & rgb_test_set),
        "cafe1": sorted(held_set & cafe1_set),
    }
    leaked = {name: ids for name, ids in intersections.items() if ids}
    if leaked:
        preview = {name: ids[:5] for name, ids in leaked.items()}
        raise HeldoutPlanError(f"held-out is not disjoint: {preview}")

    # Labels are described after membership is fixed. The audit refuses a
    # mixed cafe2 energy column; it does not remove dishes.
    label_audit = cafe2_label_audit()
    fit_with_image = [dish_id for dish_id in fit if overhead_rgb(dish_id).is_file()]
    dev_sorted = sorted(dev)
    return {
        "rule": {
            "dev505": (
                "A development dish is a dish_id in results/p0d/same_budget_dishes.jsonl "
                "with method=='ours_new' and paired==true. That set equals the paired "
                "ids of direct_vlm, ours_old, and open_loop_bc, and equals "
                "results/coverage_early_stop/paired_scores.jsonl."
            ),
            "fit": (
                "Fit is an official rgb_train dish_id with pool_of_dish=='fit': "
                "int(sha256(dish_id).hexdigest()[:8], 16) % 10 != 0. "
                "This is vdr.weak.teacher.pool_of_dish. The same ids are the dish "
                "set of results/weak_steps/rgb_train_fit_all.jsonl. Mass training "
                "used the overhead-RGB subset of this set. Q and tau used the weak "
                "steps of this set. The rgb_train internal hold-out (% 10 == 0) is "
                "not fit and is also excluded from this external list."
            ),
            "heldout": (
                "Held-out is every dish_id in data/nutrition5k/metadata/dish_metadata_cafe2.csv "
                "whose overhead rgb.png exists. The list is sorted by dish_id. Cafe2 is the source. The first CSV "
                "field is the only field read. No nutrition number, error, or "
                "component count is a filter. Ids are kept whole; decision steps "
                "are not split. Missing images are listed and are not a label drop."
            ),
            "not_used": (
                "Held-out membership is fixed before evaluation. It is not a method "
                "choice, an early stop, a tau choice, or a food-code or portion edit."
            ),
        },
        "counts": {
            "cafe1": len(cafe1_set),
            "cafe2": len(cafe2_set),
            "cafe2_with_overhead": len(held_set),
            "cafe2_missing_overhead": len(missing_image),
            "rgb_train": len(rgb_train_set),
            "rgb_test": len(rgb_test_set),
            "fit": len(fit_set),
            "fit_with_overhead": len(fit_with_image),
            "rgb_train_internal_holdout": len(internal_set),
            "dev505": len(dev),
            "heldout": len(held_set),
        },
        "intersections_with_heldout": {name: len(ids) for name, ids in intersections.items()},
        "fit_intersection_heldout_empty": True,
        "dev505_intersection_heldout_empty": True,
        "source_isolation": "Held-out is cafe2 only. The paired 505 and rgb_train are cafe1",
        "score_labels": label_audit,
        "dev505_ids": dev_sorted,
        "fit_ids": fit,
        "heldout_ids": heldout,
        "cafe2_missing_overhead_ids": missing_image,
        "sources": {
            "p0d_same_budget_dishes_sha256": sha256_file(PAIRED_ROWS),
            "coverage_paired_scores_sha256": sha256_file(COVERAGE_SCORES),
            "coverage_summary_paired_n": summary_n,
            "weak_steps_fit_sha256": sha256_file(FIT_STEPS),
            "cafe2_metadata_sha256": sha256_file(CAFE2),
            "rgb_train_ids_sha256": sha256_file(N5K / "dish_ids" / "splits" / "rgb_train_ids.txt"),
            "rgb_test_ids_sha256": sha256_file(N5K / "dish_ids" / "splits" / "rgb_test_ids.txt"),
        },
    }


def render_report(payload: dict[str, Any]) -> str:
    counts = payload["counts"]
    intersections = payload["intersections_with_heldout"]
    lines = [
        "# External held-out plan",
        "",
        "This file is the split only. It is not an evaluation and it does not select a method.",
        "",
        "## How the paired 505 is identified",
        "",
        payload["rule"]["dev505"],
        "",
        f"Count: **{counts['dev505']}**. Every id is an official rgb_test id and a cafe1 id. "
        f"None are in rgb_train. `results/coverage_early_stop/summary.json` records paired n="
        f"{payload['sources']['coverage_summary_paired_n']}.",
        "",
        "## How fit is identified",
        "",
        payload["rule"]["fit"],
        "",
        f"Count: **{counts['fit']}** dishes, of which **{counts['fit_with_overhead']}** have an overhead RGB image. "
        f"The rgb_train internal hold-out is **{counts['rgb_train_internal_holdout']}** dishes and is not part of fit.",
        "",
        "## Held-out",
        "",
        payload["rule"]["heldout"],
        "",
        f"Cafe2 has **{counts['cafe2']}** dishes. **{counts['heldout']}** have `realsense_overhead/<dish_id>/rgb.png` "
        f"and are the held-out list. **{counts['cafe2_missing_overhead']}** cafe2 dishes have no overhead image and "
        "are listed in `cafe2_missing_overhead.txt`. They were not dropped because of a label.",
        "",
        "Each held-out id appears once. The whole dish is on one side of the split.",
        "",
        "## Disjointness",
        "",
        "| Set | Intersection with held-out |",
        "|---|---:|",
    ]
    for name, count in intersections.items():
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            "`fit ∩ held-out dish_id = ∅` and `dev505 ∩ held-out dish_id = ∅`.",
            "",
            payload["source_isolation"] + ".",
            "",
            payload["rule"]["not_used"],
            "",
            "## Scoring labels",
            "",
            "Membership above does not read nutrition numbers. Scoring does, and only after predictions exist.",
            "",
            payload["score_labels"]["energy"],
            "",
            payload["score_labels"]["mass"],
            "",
            f"Checked on all {payload['score_labels']['n_cafe2']} cafe2 rows: "
            f"dish-level energy column is zero on {payload['score_labels']['n_dish_energy_column_zero']}, "
            f"and the largest dish-mass gap versus ingredient grams is "
            f"{payload['score_labels']['max_abs_dish_mass_minus_ingredient_g']:.6f} g. "
            f"Ingredient-energy sum is zero on {payload['score_labels']['n_ingredient_energy_zero']} rows; "
            "those rows stay in the list. The paired 505 keeps cafe1 dish-level labels.",
            "",
            "The #53 development numbers are not a row of a later held-out table. "
            "#53 does not prove the method. #51 is a density / kcal heuristic "
            "(automatic Exact on the development audit was 0). On the separate one-step "
            "audit, Q ≈ rule ≪ oracle. Those statements are framing; this script did not recompute them.",
            "",
        ]
    )
    return "\n".join(lines)


def write_plan(out_dir: Path = OUT) -> dict[str, Any]:
    if out_dir.resolve() != OUT.resolve():
        raise HeldoutPlanError(f"refusing to write outside {OUT}")
    payload = identify()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ids(out_dir / "dev505_dish_ids.txt", payload["dev505_ids"])
    write_ids(out_dir / "fit_dish_ids.txt", payload["fit_ids"])
    write_ids(out_dir / "heldout_dish_ids.txt", payload["heldout_ids"])
    write_ids(out_dir / "cafe2_missing_overhead.txt", payload["cafe2_missing_overhead_ids"])
    public = {
        key: value
        for key, value in payload.items()
        if key not in {"dev505_ids", "fit_ids", "heldout_ids", "cafe2_missing_overhead_ids"}
    }
    public["files"] = {
        "dev505_dish_ids.txt": sha256_file(out_dir / "dev505_dish_ids.txt"),
        "fit_dish_ids.txt": sha256_file(out_dir / "fit_dish_ids.txt"),
        "heldout_dish_ids.txt": sha256_file(out_dir / "heldout_dish_ids.txt"),
        "cafe2_missing_overhead.txt": sha256_file(out_dir / "cafe2_missing_overhead.txt"),
    }
    (out_dir / "proof.json").write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(payload), encoding="utf-8")
    return payload


def load_heldout_ids(plan_dir: Path = OUT) -> list[str]:
    """Return the written held-out list after checking it still matches the rule."""
    written = read_id_file(plan_dir / "heldout_dish_ids.txt")
    if written != sorted(written) or len(written) != len(set(written)):
        raise HeldoutPlanError("heldout_dish_ids.txt is not a sorted unique list")
    fresh = identify()
    if written != fresh["heldout_ids"]:
        raise HeldoutPlanError("heldout_dish_ids.txt does not match a fresh identification")
    if read_id_file(plan_dir / "dev505_dish_ids.txt") != fresh["dev505_ids"]:
        raise HeldoutPlanError("dev505_dish_ids.txt does not match a fresh identification")
    if read_id_file(plan_dir / "fit_dish_ids.txt") != fresh["fit_ids"]:
        raise HeldoutPlanError("fit_dish_ids.txt does not match a fresh identification")
    proof = json.loads((plan_dir / "proof.json").read_text(encoding="utf-8"))
    if proof.get("fit_intersection_heldout_empty") is not True:
        raise HeldoutPlanError("proof does not record an empty fit intersection")
    if proof.get("dev505_intersection_heldout_empty") is not True:
        raise HeldoutPlanError("proof does not record an empty 505 intersection")
    if any(proof["intersections_with_heldout"].values()):
        raise HeldoutPlanError("proof intersections are not all zero")
    return written


def run_self_check() -> None:
    sample = "dish_1556575327"
    digest = hashlib.sha256(sample.encode("utf-8")).hexdigest()
    expected = "holdout" if int(digest[:8], 16) % 10 == 0 else "fit"
    if pool_of_dish(sample) != expected:
        raise HeldoutPlanError("pool_of_dish drifted from sha256(dish_id)[:8] % 10")
    payload = identify()
    if payload["counts"]["dev505"] != EXPECTED_DEV or payload["counts"]["fit"] != EXPECTED_FIT:
        raise HeldoutPlanError("self-check counts drifted")
    if payload["counts"]["heldout"] < 1:
        raise HeldoutPlanError("held-out is empty")
    held = set(payload["heldout_ids"])
    if held & set(payload["fit_ids"]) or held & set(payload["dev505_ids"]):
        raise HeldoutPlanError("self-check found an overlap")
    print(
        f"heldout plan self-check ok dev505={payload['counts']['dev505']} "
        f"fit={payload['counts']['fit']} heldout={payload['counts']['heldout']}",
        flush=True,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Write the external held-out dish lists")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return 0
    payload = write_plan()
    print(
        f"wrote {OUT.relative_to(ROOT)} dev505={payload['counts']['dev505']} "
        f"fit={payload['counts']['fit']} heldout={payload['counts']['heldout']} "
        f"intersections={payload['intersections_with_heldout']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
