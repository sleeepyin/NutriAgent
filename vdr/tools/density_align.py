"""Same-head FNDDS code choice for a finished multi-component plate.

The mass write already fixed the grams. The committed calorie is still
``kcal/100g × grams`` from the bound food code. On a plate with more than one
named component, those codes are often a different energy density from the
dish the mass estimator just scored. This module keeps every gram and stays
inside the retrieve window already in hand. It does not call the retriever
again, does not read a cafe label, and does not train anything.

A candidate may replace the bound code only when all of the following hold.
The cutoffs are the ones the train teacher already uses. They are not fit on
rgb_test.

- The plate has at least two components, and every one of them is being written.
- The candidate's score is inside ``MIN_RETRIEVE_SCORE`` and ``MAX_SCORE_DROP``.
- The candidate's FNDDS head (the text before the first comma) does not add a
  noun the bound code's head does not already have. ``Apple, dried`` shares
  the head ``Apple``. ``Cantaloupe nectar`` adds ``nectar`` and is refused.
- The candidate does not introduce a prepared-form token the query and the
  bound name both lack, and it does not introduce a dish form the query did
  not ask for.
- Among those candidates, the assignment whose food-table energy is closest
  to the mass estimator's dish energy is kept only when that gap shrinks by
  at least ``DENSITY_GAIN`` (15%). Otherwise the bound codes stay.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

from vdr.tools.fndds_retrieve import (
    PREPARED_FORM_TOKENS,
    _head_lemmas,
    _lemmas,
    name_is_unrequested_faux_meat,
    query_dish_forms,
)
from vdr.weak.teacher import DENSITY_GAIN, MAX_SCORE_DROP, MIN_RETRIEVE_SCORE

# Function words dropped from an FNDDS head. Not a threshold and not fit on a split.
_HEAD_STOP = frozenset(
    {
        "with",
        "and",
        "or",
        "the",
        "a",
        "of",
        "ns",
        "nfs",
        "no",
        "for",
        "use",
        "on",
        "to",
        "from",
        "as",
        "in",
        "added",
        "fat",
        "type",
        "made",
        "fresh",
        "cooked",
    }
)

_MACRO_KEYS = ("energy_kcal", "protein_g", "carb_g", "fat_g")


class DensityAlignError(ValueError):
    """The plate handed to the aligner is not a component table."""


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _head(name: str) -> set[str]:
    return _head_lemmas(name) - _HEAD_STOP


def _unrequested_dish_form(query: str, name: str) -> bool:
    return bool(set(query_dish_forms(name)) - set(query_dish_forms(query)))


def same_bound_food(query: str, bound_name: str, cand_name: str) -> bool:
    """True when ``cand_name`` is a prep of the bound food, not a new food."""
    bound_head = _head(bound_name)
    cand_head = _head(cand_name)
    if not cand_head or not cand_head <= bound_head:
        return False
    introduced = (_lemmas(cand_name) & PREPARED_FORM_TOKENS) - _lemmas(query) - _lemmas(bound_name)
    if introduced:
        return False
    if _unrequested_dish_form(query, cand_name):
        return False
    if name_is_unrequested_faux_meat(query, cand_name):
        return False
    return True


def _kcal(per_100g: Mapping[str, Any] | None, grams: float) -> float | None:
    if not isinstance(per_100g, Mapping):
        return None
    kcal = _finite(per_100g.get("energy_kcal"))
    if kcal is None:
        return None
    return kcal * float(grams) / 100.0


def _scale(per_100g: Mapping[str, Any], grams: float) -> dict[str, float]:
    factor = float(grams) / 100.0
    out = {"mass_g": float(grams)}
    for key in _MACRO_KEYS:
        value = _finite(per_100g.get(key))
        if value is None:
            continue
        out[key] = value * factor
    return out


def _bound_row(component: Mapping[str, Any]) -> dict[str, Any] | None:
    code = str(component.get("code") or "")
    if not code:
        return None
    for cand in component.get("candidates") or []:
        if isinstance(cand, Mapping) and str(cand.get("code") or "") == code:
            return dict(cand)
    return None


def _pool(component: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Score-window prep variants of the bound food. The bound code always remains."""
    bound = _bound_row(component)
    if bound is None:
        raise DensityAlignError("bound code is missing from the candidate list")
    query = str(component.get("query") or "")
    bound_name = str(bound.get("name") or component.get("name") or "")
    cands = [cand for cand in (component.get("candidates") or []) if isinstance(cand, Mapping)]
    top = 0.0
    for cand in cands:
        score = _finite(cand.get("score"))
        if score is not None and score > top:
            top = score
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cand in cands:
        code = str(cand.get("code") or "")
        if not code or code in seen:
            continue
        if _kcal(cand.get("per_100g"), 1.0) is None:
            continue
        score = _finite(cand.get("score"))
        if score is None:
            score = 0.0
        in_window = score >= MIN_RETRIEVE_SCORE and top - score <= MAX_SCORE_DROP
        if code != str(bound.get("code")) and not in_window:
            continue
        name = str(cand.get("name") or "")
        if code != str(bound.get("code")) and not same_bound_food(query, bound_name, name):
            continue
        seen.add(code)
        row = dict(cand)
        row["code"] = code
        row["name"] = name
        row["score"] = float(score)
        pool.append(row)
    if str(bound.get("code")) not in seen:
        score = _finite(bound.get("score"))
        bound = dict(bound)
        bound["score"] = 0.0 if score is None else float(score)
        bound["name"] = str(bound.get("name") or "")
        pool.append(bound)
    return pool


def align_multicomponent_codes(
    components: Sequence[Mapping[str, Any]],
    *,
    dish_energy_kcal: float,
) -> dict[str, Any]:
    """Return the bound codes, or a same-head swap closer to ``dish_energy_kcal``.

    ``components`` are the foods already retrieved for this plate. Each one
    needs ``query``, ``code``, ``grams``, and ``candidates`` of
    ``{code, name, score, per_100g}``. Cafe labels are not an argument.
    """
    head = _finite(dish_energy_kcal)
    if head is None:
        raise DensityAlignError("dish energy is not finite")
    rows = [dict(component) for component in components]
    if len(rows) < 2:
        return _unchanged(rows, head, "single_component")
    grams = [_finite(row.get("grams")) for row in rows]
    if any(gram is None or gram <= 0.0 for gram in grams):
        return _unchanged(rows, head, "bad_grams")
    try:
        pools = [_pool(row) for row in rows]
    except DensityAlignError:
        return _unchanged(rows, head, "missing_bound_code")
    bound_energy = 0.0
    for row, pool, gram in zip(rows, pools, grams):
        bound = next(cand for cand in pool if cand["code"] == str(row.get("code")))
        kcal = _kcal(bound.get("per_100g"), float(gram))
        if kcal is None:
            return _unchanged(rows, head, "missing_density")
        bound_energy += kcal
    current_gap = abs(bound_energy - head)
    best_item: tuple[float, int, float, float] | None = None
    best_combo: tuple[dict[str, Any], ...] | None = None
    for combo in itertools.product(*pools):
        energy = 0.0
        feasible = True
        for cand, gram in zip(combo, grams):
            kcal = _kcal(cand.get("per_100g"), float(gram))
            if kcal is None:
                feasible = False
                break
            energy += kcal
        if not feasible:
            continue
        changes = sum(cand["code"] != str(row.get("code")) for cand, row in zip(combo, rows))
        score = sum(float(cand.get("score") or 0.0) for cand in combo)
        item = (abs(energy - head), changes, -score, energy)
        if best_item is None or item < best_item:
            best_item = item
            best_combo = tuple(dict(cand) for cand in combo)
    if best_item is None or best_combo is None:
        return _unchanged(rows, head, "empty_window")
    new_gap, changes, _score, new_energy = best_item
    shrunk = current_gap - new_gap
    if changes == 0 or current_gap <= 0.0 or shrunk < DENSITY_GAIN * current_gap:
        return _unchanged(rows, head, "gap_below_gain", before_energy=bound_energy)
    chosen = []
    for row, cand, gram in zip(rows, best_combo, grams):
        nutrients = _scale(cand.get("per_100g") or {}, float(gram))
        chosen.append(
            {
                "query": str(row.get("query") or ""),
                "from_code": str(row.get("code") or ""),
                "from_name": str(row.get("name") or ""),
                "code": cand["code"],
                "name": cand.get("name") or "",
                "grams": float(gram),
                "score": cand.get("score"),
                "changed": cand["code"] != str(row.get("code") or ""),
                "per_100g": dict(cand.get("per_100g") or {}),
                "nutrients": nutrients,
            }
        )
    return {
        "applied": True,
        "reason": "aligned",
        "n_components": len(rows),
        "n_changed": int(changes),
        "dish_energy_kcal": head,
        "before_energy_kcal": bound_energy,
        "after_energy_kcal": new_energy,
        "before_gap_kcal": current_gap,
        "after_gap_kcal": new_gap,
        "components": chosen,
    }


def _unchanged(
    rows: Sequence[Mapping[str, Any]],
    head: float,
    reason: str,
    *,
    before_energy: float | None = None,
) -> dict[str, Any]:
    return {
        "applied": False,
        "reason": reason,
        "n_components": len(rows),
        "n_changed": 0,
        "dish_energy_kcal": head,
        "before_energy_kcal": before_energy,
        "after_energy_kcal": before_energy,
        "before_gap_kcal": None if before_energy is None else abs(before_energy - head),
        "after_gap_kcal": None if before_energy is None else abs(before_energy - head),
        "components": [],
    }


def run_self_check() -> None:
    """CPU cases. No catalog file, no cafe label, no rgb_test dish."""
    spinach_raw = {"code": "raw", "name": "Spinach, raw", "score": 1.1, "per_100g": {"energy_kcal": 27.0, "protein_g": 2.0, "carb_g": 3.0, "fat_g": 0.0}}
    spinach_cooked = {
        "code": "cooked",
        "name": "Spinach, fresh, cooked, no added fat",
        "score": 1.1,
        "per_100g": {"energy_kcal": 55.0, "protein_g": 3.0, "carb_g": 4.0, "fat_g": 1.0},
    }
    potato = {"code": "potato", "name": "Potato, NFS", "score": 0.9, "per_100g": {"energy_kcal": 130.0, "protein_g": 2.0, "carb_g": 30.0, "fat_g": 0.0}}
    plate = [
        {"query": "spinach", "code": "raw", "name": "Spinach, raw", "grams": 100.0, "candidates": [spinach_raw, spinach_cooked]},
        {"query": "potato", "code": "potato", "name": "Potato, NFS", "grams": 100.0, "candidates": [potato]},
    ]
    # 27+130=157 versus head 200. Cooked spinach makes 185. The gap shrinks by more than 15%.
    swapped = align_multicomponent_codes(plate, dish_energy_kcal=200.0)
    if not swapped["applied"] or swapped["components"][0]["code"] != "cooked":
        raise AssertionError(swapped)
    if abs(swapped["after_energy_kcal"] - 185.0) > 1e-6:
        raise AssertionError(swapped["after_energy_kcal"])
    if swapped["components"][1]["changed"]:
        raise AssertionError("potato code moved")

    single = align_multicomponent_codes(plate[:1], dish_energy_kcal=200.0)
    if single["applied"] or single["reason"] != "single_component":
        raise AssertionError(single)

    # A 10% gap cut is below DENSITY_GAIN. The bound codes stay.
    mild = align_multicomponent_codes(plate, dish_energy_kcal=147.0)
    # bound 157, gap 10. Cooked is 185, gap 38, which is worse, so the bound stays anyway.
    if mild["applied"]:
        raise AssertionError(mild)
    near = [
        {
            "query": "spinach",
            "code": "raw",
            "name": "Spinach, raw",
            "grams": 100.0,
            "candidates": [
                spinach_raw,
                {**spinach_cooked, "per_100g": {"energy_kcal": 36.0, "protein_g": 2.0, "carb_g": 3.0, "fat_g": 0.0}},
            ],
        },
        plate[1],
    ]
    # bound 157, head 167, gap 10. Cooked-at-36 makes 166, gap 1. Shrinkage is 9, which is 90% > 15%.
    moved = align_multicomponent_codes(near, dish_energy_kcal=167.0)
    if not moved["applied"]:
        raise AssertionError(moved)
    tiny = align_multicomponent_codes(near, dish_energy_kcal=160.0)
    # bound gap |157-160|=3. Alternative |166-160|=6, worse. Stay.
    if tiny["applied"]:
        raise AssertionError(tiny)
    # Alternative gap shrinks by only 10%: bound 100 away, alternative 90 away.
    far = [
        {
            "query": "spinach",
            "code": "raw",
            "name": "Spinach, raw",
            "grams": 100.0,
            "candidates": [
                {"code": "raw", "name": "Spinach, raw", "score": 1.0, "per_100g": {"energy_kcal": 0.0}},
                {"code": "cooked", "name": "Spinach, cooked", "score": 1.0, "per_100g": {"energy_kcal": 10.0}},
            ],
        },
        {
            "query": "potato",
            "code": "potato",
            "name": "Potato, NFS",
            "grams": 100.0,
            "candidates": [{"code": "potato", "name": "Potato, NFS", "score": 1.0, "per_100g": {"energy_kcal": 0.0}}],
        },
    ]
    # energies 0 vs 10. Head 100. Gaps 100 and 90. 10 < 0.15*100.
    held = align_multicomponent_codes(far, dish_energy_kcal=100.0)
    if held["applied"] or held["reason"] != "gap_below_gain":
        raise AssertionError(held)

    nectar = [
        {
            "query": "cantaloupe",
            "code": "raw",
            "name": "Cantaloupe, raw",
            "grams": 100.0,
            "candidates": [
                {"code": "raw", "name": "Cantaloupe, raw", "score": 1.0, "per_100g": {"energy_kcal": 38.0}},
                {"code": "nectar", "name": "Cantaloupe nectar", "score": 1.0, "per_100g": {"energy_kcal": 65.0}},
            ],
        },
        plate[1],
    ]
    # Nectar would cut the gap and is refused because the head noun changes.
    refused = align_multicomponent_codes(nectar, dish_energy_kcal=195.0)
    if refused["applied"]:
        raise AssertionError(refused)
    if not same_bound_food("cantaloupe", "Cantaloupe, raw", "Cantaloupe, raw"):
        raise AssertionError("raw cantaloupe should match itself")
    if same_bound_food("cantaloupe", "Cantaloupe, raw", "Cantaloupe nectar"):
        raise AssertionError("nectar added a head noun")

    sandwich = [
        {
            "query": "bacon",
            "code": "bacon",
            "name": "Bacon, NS as to type of meat, cooked",
            "grams": 100.0,
            "candidates": [
                {
                    "code": "bacon",
                    "name": "Bacon, NS as to type of meat, cooked",
                    "score": 1.0,
                    "per_100g": {"energy_kcal": 484.0},
                },
                {
                    "code": "blt",
                    "name": "Bacon, lettuce, tomato sandwich on white",
                    "score": 1.0,
                    "per_100g": {"energy_kcal": 231.0},
                },
            ],
        },
        plate[1],
    ]
    # 484+130=614. Sandwich would make 361, closer to head 300, and is a dish form the query did not use.
    kept = align_multicomponent_codes(sandwich, dish_energy_kcal=300.0)
    if kept["applied"]:
        raise AssertionError(kept)

    dried = [
        {
            "query": "apple",
            "code": "raw",
            "name": "Apple, raw",
            "grams": 100.0,
            "candidates": [
                {"code": "raw", "name": "Apple, raw", "score": 1.0, "per_100g": {"energy_kcal": 61.0}},
                {"code": "dried", "name": "Apple, dried", "score": 1.0, "per_100g": {"energy_kcal": 243.0}},
            ],
        },
        plate[1],
    ]
    # "Apple, dried" keeps the head noun Apple. It is eligible. Head 400: raw sum 191 gap 209, dried sum 373 gap 27.
    dried_swap = align_multicomponent_codes(dried, dish_energy_kcal=400.0)
    if not dried_swap["applied"] or dried_swap["components"][0]["code"] != "dried":
        raise AssertionError(dried_swap)

    outside = [
        {
            "query": "spinach",
            "code": "raw",
            "name": "Spinach, raw",
            "grams": 100.0,
            "candidates": [
                spinach_raw,
                {**spinach_cooked, "score": 0.5},
            ],
        },
        plate[1],
    ]
    # Top score is 1.1. 0.5 is more than MAX_SCORE_DROP (0.20) below it, so cooked is outside the window.
    window = align_multicomponent_codes(outside, dish_energy_kcal=200.0)
    if window["applied"]:
        raise AssertionError(window)


if __name__ == "__main__":
    run_self_check()
    print("density align self-check ok")
