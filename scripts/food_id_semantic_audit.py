#!/usr/bin/env python3
"""Semantic audit of the frozen #51 same-head food-code swaps.

Reads ``results/food_id_density/`` and the frozen P0d traces those swaps were
scored from. Writes only ``results/food_id_semantic_audit/``. Does not re-run
the closed-loop evaluators, does not retrain Mass / Q / τ, and does not edit
``docs/STATUS.md``.

``summary.json`` stores dish totals plus three example swaps. The other
component codes are recovered by the same deterministic aligner on the frozen
trace, then checked against those examples and the published 37 / 68 counts.

Ground-truth food ids are Nutrition5K ``ingr_*`` codes. The FNDDS catalog has
no food-group field and its aliases are only the description plus the head
clause, so Exact / Near / Wrong is a documented noun heuristic, not a
string match across those two code systems.

    python scripts/food_id_semantic_audit.py --self-check
    python scripts/food_id_semantic_audit.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.tools.fndds_retrieve import (  # noqa: E402
    MEAT_ALTERNATIVE_TOKENS,
    _TOKEN_RE,
    _has_meat_alternative_qualifier,
    _lemma,
)

OUT = ROOT / "results" / "food_id_semantic_audit"
FROZEN_SUMMARY = ROOT / "results" / "food_id_density" / "summary.json"
FROZEN_REPORT = ROOT / "results" / "food_id_density" / "report.md"
CATALOG = ROOT / "data" / "fndds" / "2019-2020" / "foods.json"
INGREDIENT_META = ROOT / "data" / "nutrition5k" / "metadata" / "ingredients_metadata.csv"

# Same keep floor as vdr.weak.teacher.MASS_KEEP_G / ENERGY_KEEP_KCAL.
# Not refit on this audit.
MASS_KEEP_G = 8.0
ENERGY_KEEP_KCAL = 8.0

MACRO_KEYS = ("energy_kcal", "protein_g", "carb_g", "fat_g")
LABEL_RANK = {"Wrong": 0, "Near": 1, "Exact": 2}

# Tokens that are not the food. Listed in full in the report.
FUNCTION = frozenset(
    {
        "a",
        "about",
        "added",
        "an",
        "and",
        "another",
        "approximately",
        "approx",
        "as",
        "at",
        "by",
        "during",
        "eaten",
        "except",
        "fat",
        "for",
        "form",
        "from",
        "in",
        "include",
        "includes",
        "including",
        "into",
        "made",
        "not",
        "of",
        "on",
        "or",
        "other",
        "over",
        "per",
        "plus",
        "quantity",
        "specified",
        "than",
        "the",
        "to",
        "type",
        "under",
        "versus",
        "vs",
        "with",
        "without",
    }
)
GENERIC = frozenset(
    {
        "dinner",
        "dish",
        "entree",
        "food",
        "homemade",
        "meal",
        "product",
        "regular",
        "style",
    }
)
# Unspecified or raw produce. Dropped when comparing preparation sets.
DEFAULT_PREP = frozenset({"fresh", "nfs", "ns", "plain", "raw"})
# Preparation that changes density. A difference blocks Exact.
DENSITY_PREP = frozenset(
    {
        "baked",
        "barbecued",
        "bbq",
        "blanched",
        "boiled",
        "breaded",
        "broiled",
        "canned",
        "cooked",
        "creamed",
        "dehydrated",
        "drained",
        "dried",
        "fried",
        "frozen",
        "grilled",
        "mashed",
        "microwaved",
        "pickled",
        "poached",
        "refried",
        "roasted",
        "salted",
        "saute",
        "sauteed",
        "scrambled",
        "smoked",
        "steamed",
        "stewed",
        "toasted",
        "unroasted",
        "unsalted",
    }
)
# Cut or part. A difference blocks Exact and yields Near when the food nouns match.
CUT = frozenset(
    {
        "back",
        "boneless",
        "bone",
        "breast",
        "brisket",
        "chuck",
        "chunk",
        "chopped",
        "diced",
        "drumstick",
        "drumsticks",
        "fillet",
        "filet",
        "ground",
        "leg",
        "legs",
        "loin",
        "meat",
        "neck",
        "only",
        "oxtail",
        "oxtails",
        "peel",
        "piece",
        "rib",
        "ribs",
        "round",
        "shank",
        "shredded",
        "skin",
        "skinless",
        "slice",
        "sliced",
        "strip",
        "strips",
        "thigh",
        "wing",
        "wings",
    }
)
# Style words that are not a separate commodity. Difference → Near, not Exact.
STYLE = frozenset(
    {
        "crust",
        "curd",
        "dry",
        "extra",
        "farmer",
        "lowfat",
        "nonfat",
        "reduced",
        "skim",
        "stuffed",
        "thick",
        "thin",
    }
)
# Variety or color. Kept inside the food-noun set so they block Exact.
# Near when the two sides differ only by these, or when one noun set contains the other.
QUALIFIER = frozenset(
    {
        "baby",
        "bell",
        "black",
        "brown",
        "cherry",
        "chile",
        "chili",
        "chinese",
        "dark",
        "garbanzo",
        "glutinous",
        "golden",
        "green",
        "hot",
        "kidney",
        "light",
        "navy",
        "new",
        "pinto",
        "purple",
        "red",
        "roma",
        "spanish",
        "sweet",
        "vidalia",
        "white",
        "whole",
        "wild",
        "yellow",
        "yolk",
    }
)


_RULE_WORDS = {
    "function_tokens": sorted(FUNCTION),
    "generic_tokens": sorted(GENERIC),
    "default_prep_tokens": sorted(DEFAULT_PREP),
    "density_prep_tokens": sorted(DENSITY_PREP),
    "cut_tokens": sorted(CUT),
    "style_tokens": sorted(STYLE),
    "qualifier_tokens": sorted(QUALIFIER),
}


def _through_lemma(words: frozenset[str]) -> frozenset[str]:
    """Store tokens as the retriever lemma emits them.

    ``_lemma`` strips a trailing s that is not part of ``ss``. ``glutinous``
    therefore becomes ``glutinou``. The lists above stay readable; the sets
    used at match time follow the tokenizer.
    """
    return frozenset(_lemma(word) for word in words)


FUNCTION = _through_lemma(FUNCTION)
GENERIC = _through_lemma(GENERIC)
DEFAULT_PREP = _through_lemma(DEFAULT_PREP)
DENSITY_PREP = _through_lemma(DENSITY_PREP)
CUT = _through_lemma(CUT)
STYLE = _through_lemma(STYLE)
QUALIFIER = _through_lemma(QUALIFIER)
_LEMMA_DISPLAY: dict[str, str] = {}
for _words in _RULE_WORDS.values():
    for _word in _words:
        _shown = _LEMMA_DISPLAY.get(_lemma(_word))
        if _shown is None or len(_word) < len(_shown):
            _LEMMA_DISPLAY[_lemma(_word)] = _word

_ACCENTS = str.maketrans(
    {
        "á": "a",
        "à": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ñ": "n",
        "ó": "o",
        "ú": "u",
        "ü": "u",
    }
)

HUMAN_MUST_CONFIRM = (
    "Automatic Exact / Near / Wrong labels are suggestions only. "
    "They do not close the audit on the 37/50 code-swapped multi-component dishes. "
    "A person has to confirm every review row.",
    "Confirm which cafe ingredient, if any, the vision query was naming. "
    "The match uses food-noun overlap with the query. "
    "It has no synonym table, so steak against beef, yam against sweet potato, "
    "and salmon against fish stay unmatched or Wrong until a person links them.",
    "Confirm Exact: the food nouns match, and the code does not add a preparation, "
    "cut, or style the cafe name lacks. Raw, fresh, NFS, NS, and plain are treated "
    "as unspecified.",
    "Confirm Near: same food nouns with a different preparation, cut, or style; "
    "one name's food nouns contained in the other; or a variety/color difference "
    "such as sweet versus plain potato, white versus brown rice, or chinese broccoli. "
    "Near is not an identity success. A qualifier can name a different commodity.",
    "Confirm Wrong: the query shares no food noun with any cafe ingredient, the code "
    "negates a cafe noun such as no cheese, the code is a meat analog the cafe "
    "ingredient is not, or the leftover nouns are not a subset or a variety qualifier.",
    "A smaller calorie error is not evidence that the swapped code is the right food. "
    "Rows with energy_improved_but_identity_not_all_exact are the disagreement set "
    "and have to be read one by one.",
    "Ambiguous ties, and matches under the existing 8 g / 8 kcal ingredient floor, "
    "need a person to pick the referent. Deprecated cafe rows are ignored.",
)


@dataclass(frozen=True)
class NameParts:
    core: frozenset[str]
    density: frozenset[str]
    cut: frozenset[str]
    style: frozenset[str]
    negated: frozenset[str]
    faux: bool


def normalize_text(text: str) -> str:
    return (text or "").translate(_ACCENTS)


def token_lemmas(text: str) -> list[str]:
    lemmas = []
    for token in _TOKEN_RE.findall(normalize_text(text)):
        lemma = _lemma(token)
        if len(lemma) <= 1:
            continue
        lemmas.append(lemma)
    return lemmas


def parse_name(text: str) -> NameParts:
    """Split an FNDDS or cafe name into food nouns, prep, cut, and style."""
    core: set[str] = set()
    density: set[str] = set()
    cut: set[str] = set()
    style: set[str] = set()
    negated: set[str] = set()
    for clause in normalize_text(text).split(","):
        lemmas = token_lemmas(clause)
        if not lemmas:
            continue
        if "use" in lemmas:
            for lemma in lemmas:
                if lemma in FUNCTION or lemma in {"use", "no", "without"} or lemma in MEAT_ALTERNATIVE_TOKENS:
                    continue
                if lemma in DEFAULT_PREP or lemma in DENSITY_PREP:
                    density.add(lemma)
                elif lemma not in GENERIC:
                    style.add(lemma)
            continue
        negated_clause = lemmas[0] in {"no", "without"}
        for lemma in lemmas:
            if lemma in {"no", "without"} or lemma in MEAT_ALTERNATIVE_TOKENS:
                continue
            if lemma in FUNCTION or lemma in GENERIC:
                continue
            if lemma in DEFAULT_PREP or lemma in DENSITY_PREP:
                density.add(lemma)
                continue
            if lemma in CUT:
                cut.add(lemma)
                continue
            if lemma in STYLE:
                style.add(lemma)
                continue
            if negated_clause:
                negated.add(lemma)
            else:
                core.add(lemma)
    return NameParts(
        core=frozenset(core),
        density=frozenset(density),
        cut=frozenset(cut),
        style=frozenset(style),
        negated=frozenset(negated),
        faux=_has_meat_alternative_qualifier(text),
    )


def _display_noun(noun: str) -> str:
    return _LEMMA_DISPLAY.get(noun, noun)


def _fmt(nouns: frozenset[str] | set[str]) -> str:
    if not nouns:
        return "{}"
    return "{" + ", ".join(sorted(_display_noun(noun) for noun in nouns)) + "}"


def judge_names(pred_name: str, gt_name: str) -> tuple[str, str]:
    """Exact / Near / Wrong for one swapped name against one cafe ingredient name."""
    pred = parse_name(pred_name)
    gt = parse_name(gt_name)
    if pred.faux and not gt.faux:
        return (
            "Wrong",
            "code is marked meatless, vegetarian, vegan, or imitation and the cafe ingredient is not",
        )
    if not gt.core:
        return ("Wrong", "cafe ingredient name has no food noun after form tokens are removed")
    if not pred.core:
        return ("Wrong", "code name has no food noun after form tokens are removed")
    hit = pred.negated & gt.core
    if hit:
        return ("Wrong", f"code negates {_fmt(hit)}, which is in the cafe name")
    pred_prep = pred.density - DEFAULT_PREP
    gt_prep = gt.density - DEFAULT_PREP
    form_equal = pred_prep == gt_prep and pred.cut == gt.cut and pred.style == gt.style and not pred.negated
    if pred.core == gt.core and form_equal:
        return (
            "Exact",
            f"food nouns {_fmt(pred.core)} match, with no extra prep, cut, or style",
        )
    if pred.core == gt.core:
        bits = []
        extra_prep = pred_prep - gt_prep
        missing_prep = gt_prep - pred_prep
        if extra_prep or missing_prep:
            bits.append(f"prep code-only {_fmt(extra_prep)} cafe-only {_fmt(missing_prep)}")
        if pred.cut != gt.cut:
            bits.append(f"cut code {_fmt(pred.cut)} cafe {_fmt(gt.cut)}")
        if pred.style != gt.style:
            bits.append(f"style code {_fmt(pred.style)} cafe {_fmt(gt.style)}")
        if pred.negated:
            bits.append(f"code negates {_fmt(pred.negated)}")
        return ("Near", "same food nouns; " + "; ".join(bits))
    shared = pred.core & gt.core
    only_pred = pred.core - gt.core
    only_gt = gt.core - pred.core
    if shared and (pred.core <= gt.core or gt.core <= pred.core):
        return (
            "Near",
            f"shared {_fmt(shared)}; extra code {_fmt(only_pred)}; extra cafe {_fmt(only_gt)}",
        )
    if shared and only_pred <= QUALIFIER and only_gt <= QUALIFIER:
        return (
            "Near",
            f"shared {_fmt(shared)}; qualifier code {_fmt(only_pred)} cafe {_fmt(only_gt)}",
        )
    if shared:
        return (
            "Wrong",
            f"shared {_fmt(shared)} but the other nouns are not a subset or a variety qualifier: "
            f"code {_fmt(only_pred)} cafe {_fmt(only_gt)}",
        )
    return ("Wrong", f"no shared food noun: code {_fmt(pred.core)} cafe {_fmt(gt.core)}")


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def match_ingredient(
    query: str,
    ingredients: Sequence[Any],
) -> dict[str, Any]:
    """Pick a cafe ingredient by food-noun overlap with the vision query.

    The pre-swap and post-swap FNDDS names do not vote. A swapped name must
    not be able to select the ground-truth row it is graded against.
    """
    query_core = parse_name(query).core
    if not query_core:
        head = (query or "").split(",")[0]
        query_core = parse_name(head).core
    usable = []
    for ingredient in ingredients:
        name = (ingredient.name or "").strip()
        if not name or name.lower() == "deprecated":
            continue
        usable.append(ingredient)
    scored = []
    for ingredient in usable:
        ing_core = parse_name(ingredient.name).core
        shared = query_core & ing_core
        if not shared:
            continue
        scored.append(
            (
                len(shared),
                _jaccard(query_core, ing_core),
                float(ingredient.mass_g),
                str(ingredient.ingr_id),
                ingredient,
                shared,
            )
        )
    if not scored:
        return {
            "ingredient": None,
            "ambiguous": False,
            "below_keep_floor": False,
            "others": [],
            "query_core": query_core,
        }
    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    best = scored[0]
    ambiguous = any(
        item[4].ingr_id != best[4].ingr_id and item[0] == best[0] and abs(item[1] - best[1]) < 1e-12
        for item in scored[1:]
    )
    others = [item[4].name for item in scored[1:]]
    below = float(best[4].mass_g) < MASS_KEEP_G and float(best[4].energy_kcal) < ENERGY_KEEP_KCAL
    return {
        "ingredient": best[4],
        "ambiguous": ambiguous,
        "below_keep_floor": below,
        "others": others,
        "query_core": query_core,
        "shared": best[5],
    }


def identity_change(before: str, after: str) -> str:
    delta = LABEL_RANK[after] - LABEL_RANK[before]
    if delta > 0:
        return "improved"
    if delta < 0:
        return "worsened"
    return "same"


def human_prompt(row: Mapping[str, Any]) -> str:
    parts = [
        "Suggestion only. Confirm the referent, the Exact/Near/Wrong call, and that the calorie change is not evidence of identity."
    ]
    if row.get("match_ambiguous"):
        parts.append("The noun overlap tied another cafe ingredient.")
    if row.get("match_below_keep_floor"):
        parts.append("The matched cafe ingredient is under the 8 g / 8 kcal keep floor.")
    if not row.get("gt_food_code"):
        parts.append("No cafe ingredient shared a food noun with the query.")
    if row.get("energy_improved_but_identity_not_all_exact") and row.get("auto_label") != "Exact":
        parts.append("Plate calorie error fell and this code is not auto-Exact.")
    return " ".join(parts)


def run_self_check() -> None:
    """CPU cases. No trace, no cafe file, no rgb_test dish."""
    cases = [
        ("Almonds, NFS", "almonds", "Exact"),
        ("Almonds, salted", "almonds", "Near"),
        ("Almonds, unroasted", "almonds", "Near"),
        ("Apple, raw", "apple", "Exact"),
        ("Apple, dried", "apple", "Near"),
        ("Apple, baked", "apple", "Near"),
        ("Spinach, raw", "spinach (raw)", "Exact"),
        ("Spinach, fresh, cooked, no added fat", "spinach (raw)", "Near"),
        ("Spinach, fresh, cooked, fat added, NS as to fat type", "spinach (raw)", "Near"),
        ("Green beans, raw", "green beans", "Exact"),
        ("Green beans, fresh, cooked, fat added, NS as to fat type", "green beans", "Near"),
        ("Green beans, NS as to form, cooked", "green beans", "Near"),
        ("Chicken leg, drumstick and thigh, sauteed, skin not eaten", "chicken thighs", "Near"),
        ("Potato, NFS", "potatoes", "Exact"),
        ("Potato, baked, NFS", "potatoes", "Near"),
        ("Potato, baked, NFS", "sweet potato", "Near"),
        ("Sweet potato, NFS", "sweet potato", "Exact"),
        ("Sweet potato, canned, fat added", "sweet potato", "Near"),
        ("Sweet potato, NFS", "yam", "Wrong"),
        ("Cheese, cottage, dry curd", "cottage cheese", "Near"),
        ("Cottage cheese, farmer's", "cottage cheese", "Near"),
        ("Cheese, cottage, dry curd", "egg whites", "Wrong"),
        ("Chicken, canned, meat only", "chicken", "Near"),
        ("Chicken, neck or ribs", "chicken", "Near"),
        ("Chicken, back", "pork", "Wrong"),
        ("Beef, NFS", "steak", "Wrong"),
        ("Fish, pike", "salmon", "Wrong"),
        ("Rice, white, cooked, glutinous", "brown rice", "Near"),
        ("Broccoli, chinese, raw", "broccoli", "Near"),
        ("Pizza, no cheese, thick crust", "cheese pizza", "Wrong"),
        ("Pizza, no cheese, thick crust", "pizza", "Near"),
        ("Pizza, extra cheese, thin crust", "cheese pizza", "Near"),
        ("Pizza, cheese, stuffed crust", "pizza", "Near"),
        ("Bacon, for use with vegetables", "bacon", "Near"),
        ("Bacon strip, meatless", "bacon", "Wrong"),
        ("Carrots, fresh, cooked with butter or margarine", "carrot", "Near"),
        ("Strawberries, canned", "strawberries", "Near"),
        ("Pineapple, canned, in syrup", "pineapple", "Near"),
        ("Egg, whole, raw", "egg whites", "Near"),
    ]
    for pred, gt, expected in cases:
        label, reason = judge_names(pred, gt)
        if label != expected or not reason:
            raise AssertionError(f"{pred!r} vs {gt!r}: got {label} ({reason}), expected {expected}")
    form_label, form_reason = judge_names("Green beans, NS as to form, cooked", "green beans")
    if form_label != "Near" or "form" in form_reason or "cooked" not in form_reason:
        raise AssertionError(f"form token leaked: {form_label} {form_reason}")
    leg_label, leg_reason = judge_names(
        "Chicken leg, drumstick and thigh, sauteed, skin not eaten",
        "chicken thighs",
    )
    if leg_label != "Near" or "{and" in leg_reason or ", and" in leg_reason:
        raise AssertionError(f"and/leg token leaked: {leg_label} {leg_reason}")
    rice_label, rice_reason = judge_names("Rice, white, cooked, glutinous", "brown rice")
    if rice_label != "Near" or "glutinou" in rice_reason.replace("glutinous", ""):
        raise AssertionError(f"glutinous lemma leaked: {rice_label} {rice_reason}")

    class _Ing:
        def __init__(self, ingr_id: str, name: str, mass_g: float, energy_kcal: float) -> None:
            self.ingr_id = ingr_id
            self.name = name
            self.mass_g = mass_g
            self.energy_kcal = energy_kcal

    missed = match_ingredient(
        "cottage cheese",
        [_Ing("ingr_w", "egg whites", 472, 245), _Ing("ingr_c", "cantaloupe", 136, 46)],
    )
    if missed["ingredient"] is not None:
        raise AssertionError("cottage cheese matched a cafe ingredient")
    pepper = match_ingredient(
        "pepper",
        [_Ing("ingr_p", "pepper", 0.4, 0.0), _Ing("ingr_b", "bell peppers", 15, 5)],
    )
    if pepper["ingredient"] is None or pepper["ingredient"].name != "pepper" or not pepper["below_keep_floor"]:
        raise AssertionError(f"pepper match failed: {pepper}")
    if "bell peppers" not in pepper["others"]:
        raise AssertionError("bell peppers should stay visible as another overlap")
    sweet = match_ingredient(
        "roasted sweet potato",
        [_Ing("ingr_d", "deprecated", 40, 20), _Ing("ingr_p", "potatoes", 74, 50), _Ing("ingr_y", "yam", 110, 120)],
    )
    if sweet["ingredient"] is None or sweet["ingredient"].name != "potatoes":
        raise AssertionError(f"sweet potato query should noun-match potatoes, not yam: {sweet}")
    tie = match_ingredient(
        "almonds",
        [_Ing("ingr_1", "almonds", 10, 60), _Ing("ingr_2", "almonds", 50, 300)],
    )
    if not tie["ambiguous"] or tie["ingredient"].ingr_id != "ingr_2":
        raise AssertionError(f"tie break failed: {tie}")
    if identity_change("Exact", "Near") != "worsened" or identity_change("Wrong", "Wrong") != "same":
        raise AssertionError("identity change ranks drifted")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_align_module() -> Any:
    path = ROOT / "scripts" / "food_id_density_align.py"
    spec = importlib.util.spec_from_file_location("food_id_density_align", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def catalog_alias_audit() -> dict[str, Any]:
    """Confirm foods.json aliases add no cross-food synonyms."""
    if not CATALOG.is_file():
        return {"checked": False, "reason": "catalog file absent"}
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    foods = data.get("foods") or []
    extra = 0
    keys: set[str] = set()
    for food in foods:
        keys |= set(food.keys())
        aliases = {str(alias).lower() for alias in food.get("aliases") or []}
        name = str(food.get("name") or "").strip().lower()
        head = name.split(",")[0].strip()
        if aliases != {name, head}:
            extra += 1
    ingredient_columns: list[str] = []
    if INGREDIENT_META.is_file():
        with INGREDIENT_META.open(encoding="utf-8") as handle:
            ingredient_columns = list(csv.DictReader(handle).fieldnames or [])
    return {
        "checked": True,
        "n_foods": len(foods),
        "alias_rows_that_are_not_name_plus_head": extra,
        "food_record_keys": sorted(keys),
        "ingredient_metadata_columns": ingredient_columns,
        "food_group_field_present": bool(keys & {"food_group", "wweia", "category", "taxonomy"}),
    }


def _bound(write: Mapping[str, Any]) -> Mapping[str, Any]:
    code = str(write.get("code") or "")
    for cand in write.get("candidates") or []:
        if str(cand.get("code") or "") == code:
            return cand
    raise RuntimeError(f"bound code {code} missing from the frozen candidate list")


def _scaled(per: Mapping[str, Any], grams: float) -> dict[str, float]:
    factor = float(grams) / 100.0
    out = {}
    for key in MACRO_KEYS:
        value = per.get(key) if isinstance(per, Mapping) else None
        if value is None:
            raise RuntimeError(f"missing {key} on a frozen per-100g record")
        out[key] = float(value) * factor
    return out


def _sum_macros(parts: Sequence[Mapping[str, float]]) -> dict[str, float]:
    total = {key: 0.0 for key in MACRO_KEYS}
    for part in parts:
        for key in MACRO_KEYS:
            total[key] += float(part[key])
    return total


def _mae(errors: Sequence[float]) -> float:
    if not errors:
        raise RuntimeError("empty MAE")
    return sum(abs(value) for value in errors) / len(errors)


def _counts(labels: Sequence[str]) -> dict[str, int]:
    return {name: sum(1 for label in labels if label == name) for name in ("Exact", "Near", "Wrong")}


def _rate_block(labels: Sequence[str]) -> dict[str, Any]:
    counts = _counts(labels)
    total = len(labels)
    return {
        "n": total,
        "n_exact": counts["Exact"],
        "n_near": counts["Near"],
        "n_wrong": counts["Wrong"],
        "exact_rate": counts["Exact"] / total,
        "near_rate": counts["Near"] / total,
        "wrong_rate": counts["Wrong"] / total,
    }


def _macro_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    block: dict[str, Any] = {"n": len(rows)}
    for key, label_key in (
        ("energy_kcal", "label_energy_kcal"),
        ("protein_g", "label_protein_g"),
        ("carb_g", "label_carb_g"),
        ("fat_g", "label_fat_g"),
    ):
        before = [abs(float(row["before"][key]) - float(row[label_key])) for row in rows]
        after = [abs(float(row["after"][key]) - float(row[label_key])) for row in rows]
        block[f"{key}_mae_before"] = _mae(before)
        block[f"{key}_mae_after"] = _mae(after)
        block[f"{key}_mae_delta"] = block[f"{key}_mae_after"] - block[f"{key}_mae_before"]
    return block


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "/").replace("\n", " ")


def _yn(value: bool) -> str:
    return "yes" if value else "no"


def _f4(value: float) -> str:
    return f"{value:.4f}"


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{100.0 * part / total:.2f}%"


def build_rows(align_mod: Any, summary: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from vdr.data.n5k import load_dish_metadata
    from vdr.eval.same_budget import group_steps

    dishes = align_mod.load_dishes(align_mod.DISHES)
    ours = dishes["ours_new"]
    grouped = group_steps(align_mod.TRACE)
    metadata = load_dish_metadata()
    examples = {row["dish_id"]: row for row in summary["examples"]}
    multi = [row for row in summary["plates"] if row.get("reason") in {"aligned", "gap_below_gain"}]
    if len(multi) != 50:
        raise SystemExit(f"finished multi-component plates in the frozen summary: {len(multi)}, expected 50")

    component_rows: list[dict[str, Any]] = []
    dish_rows: list[dict[str, Any]] = []
    for plate in multi:
        dish_id = str(plate["dish_id"])
        if dish_id not in metadata:
            raise SystemExit(f"{dish_id} has no cafe metadata")
        dish = metadata[dish_id]
        source = ours[dish_id].get("source_step_i")
        if source is None:
            raise SystemExit(f"{dish_id} has no source step on the frozen trace")
        _n_hyp, writes = align_mod.mass_writes(grouped.get(dish_id, []), int(source))
        if not writes or any(write.get("model_energy") is None for write in writes):
            raise SystemExit(f"{dish_id} is missing a frozen mass write")
        head = float(writes[-1]["model_energy"])
        aligned = align_mod.align_multicomponent_codes(
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
        if bool(aligned["applied"]) != bool(plate["applied"]) or aligned["reason"] != plate["reason"]:
            raise SystemExit(
                f"{dish_id} aligner disagreed with the frozen summary: "
                f"{aligned['reason']}/{aligned['applied']} vs {plate['reason']}/{plate['applied']}"
            )
        if int(aligned.get("n_changed") or 0) != int(plate["n_changed"]):
            raise SystemExit(f"{dish_id} changed-code count disagreed with the frozen summary")

        before_parts = []
        after_parts = []
        changed_records = []
        if aligned["applied"]:
            chosen_rows = aligned["components"]
            if len(chosen_rows) != len(writes):
                raise SystemExit(f"{dish_id} component count drifted")
        else:
            chosen_rows = [None] * len(writes)
        for index, (write, chosen) in enumerate(zip(writes, chosen_rows)):
            bound = _bound(write)
            grams = float(write["grams"])
            before = _scaled(bound.get("per_100g") or {}, grams)
            if chosen is None:
                after = dict(before)
                changed = False
            else:
                if str(chosen.get("from_code") or "") != str(write["code"]):
                    raise SystemExit(f"{dish_id} component order drifted at {index}")
                after = _scaled(chosen.get("per_100g") or {}, grams)
                changed = bool(chosen.get("changed"))
            before_parts.append(before)
            after_parts.append(after)
            if not changed:
                continue
            changed_records.append(
                {
                    "component_index": index,
                    "query": str(write["query"]),
                    "grams": grams,
                    "from_code": str(write["code"]),
                    "from_name": str(write["name"]),
                    "to_code": str(chosen["code"]),
                    "to_name": str(chosen.get("name") or ""),
                    "before_macros": before,
                    "after_macros": after,
                }
            )
        before_total = _sum_macros(before_parts)
        after_total = _sum_macros(after_parts)
        for key, summary_key in (("energy_kcal", "energy_before_kcal"),):
            residual = abs(before_total[key] - float(plate[summary_key]))
            if residual > 0.05:
                raise SystemExit(f"{dish_id} food-table energy residual {residual:.4f} kcal")
        after_residual = abs(after_total["energy_kcal"] - float(plate["energy_after_kcal"]))
        if after_residual > 0.05:
            raise SystemExit(f"{dish_id} post-swap energy residual {after_residual:.4f} kcal")
        if dish_id in examples:
            got = [(row["from_code"], row["to_code"], row["query"]) for row in changed_records]
            exp = [(row["from_code"], row["to_code"], row["query"]) for row in examples[dish_id]["changes"]]
            if got != exp:
                raise SystemExit(f"{dish_id} example swaps disagreed: {got} vs {exp}")

        label_energy = float(plate["label_energy_kcal"])
        cafe_energy_gap = abs(float(dish.energy_kcal) - label_energy)
        if cafe_energy_gap > 0.05:
            raise SystemExit(f"{dish_id} cafe energy disagrees with the frozen label by {cafe_energy_gap:.4f}")
        ae_before = abs(float(plate["energy_before_kcal"]) - label_energy)
        ae_after = abs(float(plate["energy_after_kcal"]) - label_energy)
        energy_improved = ae_after < ae_before - 1e-6
        energy_worsened = ae_after > ae_before + 1e-6
        cafe_names = [ingredient.name for ingredient in dish.ingredients if (ingredient.name or "").strip()]

        judged = []
        for record in changed_records:
            matched = match_ingredient(record["query"], dish.ingredients)
            ingredient = matched["ingredient"]
            if ingredient is None:
                label, reason = (
                    "Wrong",
                    f"no cafe ingredient shares a food noun with the query {record['query']!r}",
                )
                before_label, before_reason = (
                    "Wrong",
                    f"no cafe ingredient shares a food noun with the query {record['query']!r}",
                )
                gt_code = ""
                gt_name = ""
                gt_mass = None
            else:
                label, reason = judge_names(record["to_name"], ingredient.name)
                before_label, before_reason = judge_names(record["from_name"], ingredient.name)
                gt_code = str(ingredient.ingr_id)
                gt_name = ingredient.name
                gt_mass = float(ingredient.mass_g)
                if matched["ambiguous"]:
                    reason = "Ambiguous cafe match. " + reason
                if matched["below_keep_floor"]:
                    reason = "Matched ingredient is under the 8 g / 8 kcal floor. " + reason
            judged.append(
                {
                    **record,
                    "auto_label": label,
                    "rationale": reason,
                    "auto_label_before": before_label,
                    "rationale_before": before_reason,
                    "gt_food_code": gt_code,
                    "gt_food_name": gt_name,
                    "gt_mass_g": gt_mass,
                    "match_ambiguous": bool(matched["ambiguous"]),
                    "match_below_keep_floor": bool(matched["below_keep_floor"]),
                    "other_overlapping_ingredients": matched["others"],
                }
            )
        labels = [row["auto_label"] for row in judged]
        worst = "Exact"
        if any(label == "Wrong" for label in labels):
            worst = "Wrong"
        elif any(label == "Near" for label in labels):
            worst = "Near"
        all_exact = bool(labels) and all(label == "Exact" for label in labels)
        disagreement = bool(plate["applied"]) and energy_improved and not all_exact
        dish_row = {
            "dish_id": dish_id,
            "applied": bool(plate["applied"]),
            "reason": plate["reason"],
            "n_changed": int(plate["n_changed"]),
            "n_exact": sum(label == "Exact" for label in labels),
            "n_near": sum(label == "Near" for label in labels),
            "n_wrong": sum(label == "Wrong" for label in labels),
            "worst_auto_label": worst if plate["applied"] else "",
            "all_exact": all_exact,
            "energy_before_kcal": float(plate["energy_before_kcal"]),
            "energy_after_kcal": float(plate["energy_after_kcal"]),
            "label_energy_kcal": label_energy,
            "energy_ae_before": ae_before,
            "energy_ae_after": ae_after,
            "energy_ae_change": ae_after - ae_before,
            "energy_improved": energy_improved,
            "energy_worsened": energy_worsened,
            "energy_improved_but_identity_not_all_exact": disagreement,
            "before": before_total,
            "after": after_total,
            "label_protein_g": float(dish.protein_g),
            "label_carb_g": float(dish.carb_g),
            "label_fat_g": float(dish.fat_g),
            "cafe_ingredients": cafe_names,
            "needs_human_review": bool(plate["applied"]),
        }
        dish_rows.append(dish_row)
        for record in judged:
            component_rows.append(
                {
                    "needs_human_review": True,
                    "auto_label_is_suggestion_only": True,
                    "dish_id": dish_id,
                    "component_index": record["component_index"],
                    "query": record["query"],
                    "grams": record["grams"],
                    "from_code": record["from_code"],
                    "from_name": record["from_name"],
                    "to_code": record["to_code"],
                    "to_name": record["to_name"],
                    "gt_food_code": record["gt_food_code"],
                    "gt_food_code_system": "nutrition5k_ingr_id" if record["gt_food_code"] else "",
                    "gt_food_name": record["gt_food_name"],
                    "gt_mass_g": record["gt_mass_g"],
                    "auto_label": record["auto_label"],
                    "auto_label_before": record["auto_label_before"],
                    "identity_change": identity_change(record["auto_label_before"], record["auto_label"]),
                    "rationale": record["rationale"],
                    "rationale_before": record["rationale_before"],
                    "match_ambiguous": record["match_ambiguous"],
                    "match_below_keep_floor": record["match_below_keep_floor"],
                    "other_overlapping_ingredients": "; ".join(record["other_overlapping_ingredients"]),
                    "cafe_ingredients": "; ".join(cafe_names),
                    "plate_energy_ae_before": ae_before,
                    "plate_energy_ae_after": ae_after,
                    "plate_energy_ae_change": ae_after - ae_before,
                    "energy_improved": energy_improved,
                    "energy_improved_but_identity_not_all_exact": disagreement,
                    "this_code_not_exact_on_improved_plate": energy_improved and record["auto_label"] != "Exact",
                    "component_energy_before_kcal": record["before_macros"]["energy_kcal"],
                    "component_energy_after_kcal": record["after_macros"]["energy_kcal"],
                    "component_protein_before_g": record["before_macros"]["protein_g"],
                    "component_protein_after_g": record["after_macros"]["protein_g"],
                    "component_carb_before_g": record["before_macros"]["carb_g"],
                    "component_carb_after_g": record["after_macros"]["carb_g"],
                    "component_fat_before_g": record["before_macros"]["fat_g"],
                    "component_fat_after_g": record["after_macros"]["fat_g"],
                    "human_must_confirm": "",
                }
            )
    for row in component_rows:
        row["human_must_confirm"] = human_prompt(row)
    return dish_rows, component_rows


def _transition(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    matrix = {}
    for before in ("Exact", "Near", "Wrong"):
        for after in ("Exact", "Near", "Wrong"):
            matrix[f"{before}_to_{after}"] = sum(
                1 for row in rows if row["auto_label_before"] == before and row["auto_label"] == after
            )
    return matrix


def _dish_identity_split(dishes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    applied = [row for row in dishes if row["applied"]]
    improved = [row for row in applied if row["energy_improved"]]
    worsened = [row for row in applied if row["energy_worsened"]]

    def bucket(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "n": len(rows),
            "all_exact": sum(1 for row in rows if row["all_exact"]),
            "any_wrong": sum(1 for row in rows if row["n_wrong"] > 0),
            "near_without_wrong": sum(1 for row in rows if row["n_wrong"] == 0 and row["n_near"] > 0),
            "identity_not_all_exact": sum(1 for row in rows if not row["all_exact"]),
        }

    return {"swapped": bucket(applied), "energy_improved": bucket(improved), "energy_worsened": bucket(worsened)}


def _rule_dump() -> dict[str, list[str]]:
    dumped = {key: list(value) for key, value in _RULE_WORDS.items()}
    dumped["faux_tokens"] = sorted(MEAT_ALTERNATIVE_TOKENS)
    return dumped


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return lines


def write_reports(payload: Mapping[str, Any], dishes: Sequence[Mapping[str, Any]], components: Sequence[Mapping[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    identity = payload["identity_post_swap"]
    before = payload["identity_pre_swap"]
    macros_50 = payload["macros_finished_multi"]
    macros_37 = payload["macros_swapped_plates"]
    macros_13 = payload["macros_unswapped_plates"]
    split = payload["dish_identity_split"]
    improved = split["energy_improved"]
    transition = payload["identity_transition"]
    rules = payload["rules"]
    alias = payload["catalog_alias_audit"]

    def macro_line(title: str, block: Mapping[str, Any]) -> str:
        return (
            f"| {title} | {block['n']} "
            f"| {_f4(block['protein_g_mae_before'])} | {_f4(block['protein_g_mae_after'])} "
            f"| {_f4(block['carb_g_mae_before'])} | {_f4(block['carb_g_mae_after'])} "
            f"| {_f4(block['fat_g_mae_before'])} | {_f4(block['fat_g_mae_after'])} "
            f"| {_f4(block['energy_kcal_mae_before'])} | {_f4(block['energy_kcal_mae_after'])} |"
        )

    lines = [
        "# Food-id semantic audit of the #51 same-head swaps",
        "",
        HUMAN_MUST_CONFIRM[0],
        "",
        "This folder is a read of the frozen #51 artifacts. The mass estimator, the calibrated selector, and the #51 / #53 / #57 evaluators are not run. Grams and actions stay the ones already stored. Labels are used only to score identity and macros after the code choice that #51 already recorded.",
        "",
        "## What was compared",
        "",
        f"Finished multi-component plates in the frozen summary: **{payload['n_finished_multi']}**. Plates where the same-head swap changed a code: **{payload['n_swapped_plates']}**. Component codes changed: **{payload['n_swapped_codes']}**. Absolute calorie error fell on **{payload['n_energy_better']}** of the 50 and rose on **{payload['n_energy_worse']}**.",
        "",
        "The frozen `results/food_id_density/summary.json` stores those dish totals and three example swaps. The rest of the component codes are recovered by `align_multicomponent_codes` on the frozen P0d trace (`results/p0d/ours_new_rgb_test.jsonl` plus the paired dish file). Recovery is accepted only when the applied flag, the reason, the changed-code count, the dish energy, and the three published examples match the frozen summary.",
        "",
        "Ground-truth food codes are Nutrition5K `ingr_*` ids on the cafe ingredient rows. FNDDS codes and `ingr_*` ids are different systems, so Exact is not string equality of the two codes. It is the noun rule below, applied to the swapped FNDDS name and the matched cafe ingredient name.",
        "",
        "## Taxonomy that is not in the repo",
        "",
        (
            f"FNDDS `foods.json` was checked ({alias.get('n_foods', 0)} foods). "
            f"Alias rows that are anything other than the lowercased description plus the head clause: "
            f"**{alias.get('alias_rows_that_are_not_name_plus_head', 'not checked')}**. "
            f"Record keys: {', '.join(alias.get('food_record_keys') or ['not checked'])}. "
            "There is no food-group, WWEIA, or taxonomy field. "
            f"Nutrition5K `ingredients_metadata.csv` columns: {', '.join(alias.get('ingredient_metadata_columns') or ['not checked'])}. "
            "Those columns are id, name, and per-gram macros. No family and no synonym column."
        ),
        "",
        "The fallback is the token rule in `scripts/food_id_semantic_audit.py`. It uses the repository lemma function from FNDDS retrieval (`_lemma` / `_TOKEN_RE`). The token lists were not fit on these 50 plates. A hit on a variety word is Near, never Exact. Commodity synonyms are not rewritten into Near.",
        "",
        "### Exact",
        "",
        "Food-noun sets are equal, the code does not negate a noun, and preparation beyond raw / fresh / NFS / NS / plain, cut, and style are the same as the cafe name.",
        "",
        "### Near",
        "",
        "Any of the following, and not Exact:",
        "",
        "- The food nouns match, and preparation, cut, or style differs (raw apple versus dried apple, chicken versus canned chicken meat, cottage cheese versus dry-curd cottage cheese).",
        "- One food-noun set contains the other (potato versus sweet potato, pizza versus cheese pizza, broccoli versus chinese broccoli).",
        "- The food nouns overlap and every unshared noun is in the variety/color list (white rice versus brown rice).",
        "",
        "### Wrong",
        "",
        "No cafe ingredient shares a food noun with the vision query; the code negates a cafe noun; the code is a meat analog and the cafe ingredient is not; or the leftover nouns are not a subset and not variety qualifiers (pike versus salmon, beef versus steak, sweet potato versus yam).",
        "",
        "The matched cafe row is the ingredient whose food nouns overlap the vision query the most (Jaccard, then grams). Ties are flagged ambiguous and the heavier row is kept only as a placeholder. Ingredients named `deprecated` are skipped. The swapped FNDDS name is not used to choose the cafe row.",
        "",
        "### Token lists",
        "",
        f"- Default prep, ignored when it is the only prep: {', '.join(rules['default_prep_tokens'])}.",
        f"- Density-changing prep: {', '.join(rules['density_prep_tokens'])}.",
        f"- Cut: {', '.join(rules['cut_tokens'])}.",
        f"- Style: {', '.join(rules['style_tokens'])}.",
        f"- Variety/color qualifiers: {', '.join(rules['qualifier_tokens'])}.",
        f"- Faux-meat tokens: {', '.join(rules['faux_tokens'])}.",
        "",
        "## Identity of the swapped codes",
        "",
        "Denominator is the **68** swapped component codes, not the 50 plates. The label is the post-swap code against the matched cafe ingredient. Pre-swap labels use the same cafe row.",
        "",
        "| | n | Exact | Near | Wrong |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Post-swap code | {identity['n']} | {identity['n_exact']} ({_pct(identity['n_exact'], identity['n'])}) "
            f"| {identity['n_near']} ({_pct(identity['n_near'], identity['n'])}) "
            f"| {identity['n_wrong']} ({_pct(identity['n_wrong'], identity['n'])}) |"
        ),
        (
            f"| Pre-swap code, same cafe row | {before['n']} | {before['n_exact']} ({_pct(before['n_exact'], before['n'])}) "
            f"| {before['n_near']} ({_pct(before['n_near'], before['n'])}) "
            f"| {before['n_wrong']} ({_pct(before['n_wrong'], before['n'])}) |"
        ),
        "",
        (
            f"Across the 68 codes the auto label improved on **{payload['n_identity_improved']}**, "
            f"stayed the same on **{payload['n_identity_same']}**, and worsened on **{payload['n_identity_worsened']}**."
        ),
        "",
        "| Pre \\ Post | Exact | Near | Wrong |",
        "|---|---:|---:|---:|",
        f"| Exact | {transition['Exact_to_Exact']} | {transition['Exact_to_Near']} | {transition['Exact_to_Wrong']} |",
        f"| Near | {transition['Near_to_Exact']} | {transition['Near_to_Near']} | {transition['Near_to_Wrong']} |",
        f"| Wrong | {transition['Wrong_to_Exact']} | {transition['Wrong_to_Near']} | {transition['Wrong_to_Wrong']} |",
        "",
    ]
    if payload["identity_improved_examples"]:
        lines.append(
            "Codes whose auto label moved up the order Exact > Near > Wrong. "
            "The move is still a suggestion:"
        )
        lines.append("")
        for example in payload["identity_improved_examples"]:
            lines.append(
                f"- `{example['dish_id']}` query `{example['query']}`: "
                f"{example['from_name']} → {example['to_name']} against cafe "
                f"{example['gt_food_name'] or '(unmatched)'} "
                f"({example['auto_label_before']} → {example['auto_label']}). {example['rationale']}"
            )
        lines.append("")
    lines.extend(
        [
        (
            f"On the 37 swapped plates the worst auto label is Exact on **{split['swapped']['all_exact']}**, "
            f"Near with no Wrong code on **{split['swapped']['near_without_wrong']}**, "
            f"and Wrong on at least one code on **{split['swapped']['any_wrong']}**."
        ),
        "",
        "## Macro error",
        "",
        "Protein, carbohydrate, fat, and energy are the food-table sums: each component's frozen FNDDS per-100 g value times the stored grams. The ground truth is the cafe dish total. Grams are the mass estimator's grams, and the vision list is not the cafe ingredient list, so this MAE mixes identity, preparation, missing foods, and portion error. The pre-swap to post-swap change is only the code swap.",
        "",
        "Energy on the 50-plate slice is the #51 food-table energy. Recomputed MAE is "
        f"**{_f4(macros_50['energy_kcal_mae_before'])} → {_f4(macros_50['energy_kcal_mae_after'])}** kcal. "
        f"The frozen summary reports **{_f4(payload['published_energy_mae_before'])} → {_f4(payload['published_energy_mae_after'])}**.",
        "",
        "| Slice | n | Protein MAE before | after | Carb MAE before | after | Fat MAE before | after | Energy MAE before | after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        macro_line("Finished multi-component", macros_50),
        macro_line("Code-swapped plates", macros_37),
        macro_line("Finished plates with no code change", macros_13),
        "",
        "Units are grams for protein, carbohydrate, and fat, and kcal for energy. MAE after minus MAE before: "
        f"swapped plates protein **{macros_37['protein_g_mae_delta']:+.4f}**, "
        f"carb **{macros_37['carb_g_mae_delta']:+.4f}**, "
        f"fat **{macros_37['fat_g_mae_delta']:+.4f}**, "
        f"energy **{macros_37['energy_kcal_mae_delta']:+.4f}**.",
        "",
        "## kcal getting better ≠ food identity becoming correct",
        "",
        (
            f"Of the **{improved['n']}** swapped plates whose absolute energy error fell, "
            f"**{improved['all_exact']}** have every swapped code auto-labeled Exact, and "
            f"**{improved['identity_not_all_exact']}** do not. "
            f"Among those {improved['identity_not_all_exact']}, "
            f"**{improved['any_wrong']}** have at least one Wrong code and "
            f"**{improved['near_without_wrong']}** are Near with no Wrong code."
        ),
        "",
        (
            f"Of the **{payload['n_codes_on_improved_plates']}** swapped codes that sit on an energy-improved plate, "
            f"**{payload['codes_on_improved_not_exact']}** are Near or Wrong. "
            f"Post-swap counts on those codes: Exact **{payload['improved_code_labels']['n_exact']}**, "
            f"Near **{payload['improved_code_labels']['n_near']}**, "
            f"Wrong **{payload['improved_code_labels']['n_wrong']}**."
        ),
        "",
        (
            f"Energy error rose on **{split['energy_worsened']['n']}** swapped plates. "
            f"Of those, **{split['energy_worsened']['all_exact']}** are all Exact and "
            f"**{split['energy_worsened']['identity_not_all_exact']}** are not."
        ),
        "",
        "The disagreement count used in this audit is the plate count "
        f"**{improved['identity_not_all_exact']} / {improved['n']}**: calorie error fell, and the auto identity of the swapped codes is not all Exact. "
        "That count is a suggestion until the review pack is confirmed by a person.",
        "",
        (
            f"On the 37 swapped plates, energy MAE changed by **{macros_37['energy_kcal_mae_delta']:+.4f} kcal**, "
            f"protein by **{macros_37['protein_g_mae_delta']:+.4f} g**, "
            f"carbohydrate by **{macros_37['carb_g_mae_delta']:+.4f} g**, "
            f"and fat by **{macros_37['fat_g_mae_delta']:+.4f} g**. "
            f"**{transition['Exact_to_Near']}** codes moved from auto-Exact to auto-Near, and "
            f"**{identity['n_exact']}** swapped codes are auto-Exact after the swap."
        ),
        "",
        "## Human review",
        "",
        (
            f"Review rows: **{payload['n_human_review_component_rows']}** component rows "
            f"covering **{payload['n_human_review_dishes']}** dishes. "
            "Every one of those rows has `needs_human_review=yes` and `auto_label_is_suggestion_only=yes`. "
            f"The other **{payload['n_unswapped_plates']}** finished multi-component plates did not change a code and are not in the review pack."
        ),
        "",
        "A person has to confirm:",
        "",
        ]
    )
    for index, item in enumerate(HUMAN_MUST_CONFIRM, start=1):
        lines.append(f"{index}. {item}")
    lines.extend(
        [
            "",
            "Tables: `human_review_components.csv` (one row per swapped code), `human_review_dishes.csv` (one row per swapped dish), and `human_review.md`.",
            "",
            "## What this audit does not do",
            "",
            "It does not retrain the mass estimator, Q, or τ. It does not rerank retrieval, run reinforcement learning, or retune rgb_test thresholds. It does not rerun #51, #53, or #57. It does not edit `docs/STATUS.md`, so gates ③–⑦ and P1 stay locked. It does not overwrite `results/food_id_density/`, `results/coverage_early_stop/`, or `results/qphi_counterfactual/`.",
            "",
            "```bash",
            "python scripts/food_id_semantic_audit.py --self-check",
            "python scripts/food_id_semantic_audit.py",
            "```",
            "",
        ]
    )
    text = "\n".join(lines)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    (OUT / "report.txt").write_text(text, encoding="utf-8")

    review_lines = [
        "# Human review pack: 37/50 code-swapped multi-component dishes",
        "",
        HUMAN_MUST_CONFIRM[0],
        "",
        f"Component rows that need a person: **{len(components)}**. Dishes: **{len({row['dish_id'] for row in components})}**. "
        "The auto label column is a suggestion. `needs_human_review` is yes on every row below.",
        "",
        "What a person confirms:",
        "",
    ]
    for index, item in enumerate(HUMAN_MUST_CONFIRM, start=1):
        review_lines.append(f"{index}. {item}")
    review_lines.extend(
        [
            "",
            f"Disagreement plates (calorie error fell, swapped codes are not all auto-Exact): **{improved['identity_not_all_exact']}**.",
            "",
            "## Dishes",
            "",
        ]
    )
    dish_table = []
    for row in sorted(dishes, key=lambda item: (not item["energy_improved_but_identity_not_all_exact"], item["dish_id"])):
        if not row["applied"]:
            continue
        dish_table.append(
            [
                row["dish_id"],
                row["n_changed"],
                f"{row['n_exact']}/{row['n_near']}/{row['n_wrong']}",
                row["worst_auto_label"],
                _yn(row["all_exact"]),
                f"{row['energy_ae_change']:+.1f}",
                _yn(row["energy_improved_but_identity_not_all_exact"]),
                "yes",
            ]
        )
    review_lines.extend(
        _md_table(
            ["dish", "codes", "E/N/W", "worst", "all exact", "kcal AE change", "disagreement", "needs human"],
            dish_table,
        )
    )
    review_lines.extend(["", "## Swapped codes", ""])
    code_table = []
    ordered = sorted(
        components,
        key=lambda item: (
            not item["energy_improved_but_identity_not_all_exact"],
            item["dish_id"],
            item["component_index"],
        ),
    )
    for row in ordered:
        code_table.append(
            [
                row["dish_id"],
                row["query"],
                f"{row['from_name']} → {row['to_name']}",
                row["gt_food_code"] or "(none)",
                row["gt_food_name"] or "(unmatched)",
                row["auto_label"],
                row["rationale"],
                _yn(row["energy_improved_but_identity_not_all_exact"]),
                "yes",
            ]
        )
    review_lines.extend(
        _md_table(
            ["dish", "query", "swap", "gt food code", "gt name", "auto", "rationale", "disagreement", "needs human"],
            code_table,
        )
    )
    review_lines.append("")
    (OUT / "human_review.md").write_text("\n".join(review_lines), encoding="utf-8")

    component_fields = [
        "needs_human_review",
        "auto_label_is_suggestion_only",
        "dish_id",
        "component_index",
        "query",
        "grams",
        "from_code",
        "from_name",
        "to_code",
        "to_name",
        "gt_food_code",
        "gt_food_code_system",
        "gt_food_name",
        "gt_mass_g",
        "auto_label",
        "auto_label_before",
        "identity_change",
        "rationale",
        "match_ambiguous",
        "match_below_keep_floor",
        "other_overlapping_ingredients",
        "cafe_ingredients",
        "plate_energy_ae_before",
        "plate_energy_ae_after",
        "plate_energy_ae_change",
        "energy_improved",
        "energy_improved_but_identity_not_all_exact",
        "this_code_not_exact_on_improved_plate",
        "component_energy_before_kcal",
        "component_energy_after_kcal",
        "component_protein_before_g",
        "component_protein_after_g",
        "component_carb_before_g",
        "component_carb_after_g",
        "component_fat_before_g",
        "component_fat_after_g",
        "human_must_confirm",
    ]
    with (OUT / "human_review_components.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=component_fields, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    **row,
                    "needs_human_review": _yn(True),
                    "auto_label_is_suggestion_only": _yn(True),
                    "match_ambiguous": _yn(bool(row["match_ambiguous"])),
                    "match_below_keep_floor": _yn(bool(row["match_below_keep_floor"])),
                    "energy_improved": _yn(bool(row["energy_improved"])),
                    "energy_improved_but_identity_not_all_exact": _yn(
                        bool(row["energy_improved_but_identity_not_all_exact"])
                    ),
                    "this_code_not_exact_on_improved_plate": _yn(bool(row["this_code_not_exact_on_improved_plate"])),
                }
            )

    dish_fields = [
        "needs_human_review",
        "auto_label_is_suggestion_only",
        "dish_id",
        "n_changed",
        "n_exact",
        "n_near",
        "n_wrong",
        "worst_auto_label",
        "all_exact",
        "energy_before_kcal",
        "energy_after_kcal",
        "label_energy_kcal",
        "energy_ae_before",
        "energy_ae_after",
        "energy_ae_change",
        "energy_improved",
        "energy_improved_but_identity_not_all_exact",
        "protein_before_g",
        "protein_after_g",
        "protein_gt_g",
        "protein_ae_before",
        "protein_ae_after",
        "carb_before_g",
        "carb_after_g",
        "carb_gt_g",
        "carb_ae_before",
        "carb_ae_after",
        "fat_before_g",
        "fat_after_g",
        "fat_gt_g",
        "fat_ae_before",
        "fat_ae_after",
        "cafe_ingredients",
        "human_must_confirm",
    ]
    swapped_dishes = [row for row in dishes if row["applied"]]
    with (OUT / "human_review_dishes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dish_fields)
        writer.writeheader()
        for row in sorted(swapped_dishes, key=lambda item: item["dish_id"]):
            writer.writerow(
                {
                    "needs_human_review": "yes",
                    "auto_label_is_suggestion_only": "yes",
                    "dish_id": row["dish_id"],
                    "n_changed": row["n_changed"],
                    "n_exact": row["n_exact"],
                    "n_near": row["n_near"],
                    "n_wrong": row["n_wrong"],
                    "worst_auto_label": row["worst_auto_label"],
                    "all_exact": _yn(row["all_exact"]),
                    "energy_before_kcal": row["energy_before_kcal"],
                    "energy_after_kcal": row["energy_after_kcal"],
                    "label_energy_kcal": row["label_energy_kcal"],
                    "energy_ae_before": row["energy_ae_before"],
                    "energy_ae_after": row["energy_ae_after"],
                    "energy_ae_change": row["energy_ae_change"],
                    "energy_improved": _yn(row["energy_improved"]),
                    "energy_improved_but_identity_not_all_exact": _yn(
                        row["energy_improved_but_identity_not_all_exact"]
                    ),
                    "protein_before_g": row["before"]["protein_g"],
                    "protein_after_g": row["after"]["protein_g"],
                    "protein_gt_g": row["label_protein_g"],
                    "protein_ae_before": abs(row["before"]["protein_g"] - row["label_protein_g"]),
                    "protein_ae_after": abs(row["after"]["protein_g"] - row["label_protein_g"]),
                    "carb_before_g": row["before"]["carb_g"],
                    "carb_after_g": row["after"]["carb_g"],
                    "carb_gt_g": row["label_carb_g"],
                    "carb_ae_before": abs(row["before"]["carb_g"] - row["label_carb_g"]),
                    "carb_ae_after": abs(row["after"]["carb_g"] - row["label_carb_g"]),
                    "fat_before_g": row["before"]["fat_g"],
                    "fat_after_g": row["after"]["fat_g"],
                    "fat_gt_g": row["label_fat_g"],
                    "fat_ae_before": abs(row["before"]["fat_g"] - row["label_fat_g"]),
                    "fat_ae_after": abs(row["after"]["fat_g"] - row["label_fat_g"]),
                    "cafe_ingredients": "; ".join(row["cafe_ingredients"]),
                    "human_must_confirm": HUMAN_MUST_CONFIRM[0]
                    + " Confirm the plate-level worst label and the calorie-versus-identity flag.",
                }
            )

    readme = "\n".join(
        [
            "# Food-id semantic audit",
            "",
            "Read-only audit of the #51 same-head FNDDS swaps in `results/food_id_density/` (squash `6c8fda02`).",
            "",
            "Regenerate:",
            "",
            "```bash",
            "python scripts/food_id_semantic_audit.py --self-check",
            "python scripts/food_id_semantic_audit.py",
            "```",
            "",
            "Outputs in this folder:",
            "",
            "- `report.md` — Exact / Near / Wrong, protein / carb / fat / energy MAE, and the calorie-versus-identity count.",
            "- `human_review_components.csv`, `human_review_dishes.csv`, `human_review.md` — the 37-dish review pack.",
            "- `summary.json` — the same numbers in machine-readable form.",
            "",
            "Automatic labels are suggestions only. Every swapped component row is marked for human review. This folder does not unlock gates ③–⑦, P1, or a Mass / Q / τ retrain.",
            "",
        ]
    )
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_audit() -> None:
    run_self_check()
    if not FROZEN_SUMMARY.is_file():
        raise SystemExit(f"missing frozen summary: {FROZEN_SUMMARY}")
    frozen_paths = [
        FROZEN_SUMMARY,
        FROZEN_REPORT,
        ROOT / "results" / "p0d" / "ours_new_rgb_test.jsonl",
        ROOT / "results" / "p0d" / "same_budget_dishes.jsonl",
        ROOT / "docs" / "STATUS.md",
    ]
    before_hash = {str(path.relative_to(ROOT)): sha256_file(path) for path in frozen_paths if path.is_file()}
    summary = json.loads(FROZEN_SUMMARY.read_text(encoding="utf-8"))
    multi = summary["multi_component_finished"]
    if int(multi["n"]) != 50 or int(multi["n_applied"]) != 37 or int(multi["n_components_changed"]) != 68:
        raise SystemExit(f"frozen multi-component counts are not 50/37/68: {multi}")
    if int(multi["n_energy_better"]) != 30 or int(multi["n_energy_worse"]) != 7:
        raise SystemExit(f"frozen energy better/worse is not 30/7: {multi}")
    align_mod = load_align_module()
    from vdr.tools.density_align import align_multicomponent_codes

    align_mod.align_multicomponent_codes = align_multicomponent_codes
    dishes, components = build_rows(align_mod, summary)
    if len(components) != 68:
        raise SystemExit(f"recovered {len(components)} swapped codes, expected 68")
    swapped = [row for row in dishes if row["applied"]]
    unswapped = [row for row in dishes if not row["applied"]]
    if len(swapped) != 37 or len(unswapped) != 13:
        raise SystemExit(f"plate split {len(swapped)}/{len(unswapped)}, expected 37/13")
    better = sum(1 for row in dishes if row["energy_improved"])
    worse = sum(1 for row in dishes if row["energy_worsened"])
    if better != 30 or worse != 7:
        raise SystemExit(f"energy better/worse recomputed as {better}/{worse}, expected 30/7")
    macros_50 = _macro_block(dishes)
    if abs(macros_50["energy_kcal_mae_before"] - float(multi["before_energy_mae_kcal"])) > 0.02:
        raise SystemExit("recomputed pre-swap energy MAE drifted from the frozen summary")
    if abs(macros_50["energy_kcal_mae_after"] - float(multi["after_energy_mae_kcal"])) > 0.02:
        raise SystemExit("recomputed post-swap energy MAE drifted from the frozen summary")

    improved_codes = [row for row in components if row["energy_improved"]]
    payload: dict[str, Any] = {
        "definition": (
            "Read-only semantic audit of the #51 same-head swaps. "
            "Exact / Near / Wrong compares each swapped FNDDS name with the Nutrition5K "
            "cafe ingredient whose food nouns overlap the vision query. "
            "Auto labels are suggestions. Every swapped row needs a person."
        ),
        "source_squash": "6c8fda02",
        "status_commit_required": "8a0e9233",
        "n_finished_multi": 50,
        "n_swapped_plates": 37,
        "n_unswapped_plates": 13,
        "n_swapped_codes": 68,
        "n_energy_better": 30,
        "n_energy_worse": 7,
        "n_human_review_component_rows": len(components),
        "n_human_review_dishes": len(swapped),
        "published_energy_mae_before": float(multi["before_energy_mae_kcal"]),
        "published_energy_mae_after": float(multi["after_energy_mae_kcal"]),
        "identity_post_swap": _rate_block([row["auto_label"] for row in components]),
        "identity_pre_swap": _rate_block([row["auto_label_before"] for row in components]),
        "identity_transition": _transition(components),
        "n_identity_improved": sum(1 for row in components if row["identity_change"] == "improved"),
        "n_identity_same": sum(1 for row in components if row["identity_change"] == "same"),
        "n_identity_worsened": sum(1 for row in components if row["identity_change"] == "worsened"),
        "identity_improved_examples": [
            {
                "dish_id": row["dish_id"],
                "query": row["query"],
                "from_name": row["from_name"],
                "to_name": row["to_name"],
                "gt_food_name": row["gt_food_name"],
                "auto_label_before": row["auto_label_before"],
                "auto_label": row["auto_label"],
                "rationale": row["rationale"],
            }
            for row in components
            if row["identity_change"] == "improved"
        ],
        "dish_identity_split": _dish_identity_split(dishes),
        "n_codes_on_improved_plates": len(improved_codes),
        "codes_on_improved_not_exact": sum(1 for row in improved_codes if row["auto_label"] != "Exact"),
        "improved_code_labels": _rate_block([row["auto_label"] for row in improved_codes]),
        "macros_finished_multi": macros_50,
        "macros_swapped_plates": _macro_block(swapped),
        "macros_unswapped_plates": _macro_block(unswapped),
        "catalog_alias_audit": catalog_alias_audit(),
        "rules": _rule_dump(),
        "human_must_confirm": list(HUMAN_MUST_CONFIRM),
        "auto_labels_are_suggestions_only": True,
        "human_review_required_for_all_swapped_rows": True,
        "q_weights_retrained": False,
        "mass_retrained": False,
        "tau_refit": False,
        "closed_loop_rerun": False,
        "input_sha256": before_hash,
    }
    # Dish rows in the JSON omit nothing required to recompute the MAE.
    payload["dishes"] = [
        {
            "dish_id": row["dish_id"],
            "applied": row["applied"],
            "n_changed": row["n_changed"],
            "n_exact": row["n_exact"],
            "n_near": row["n_near"],
            "n_wrong": row["n_wrong"],
            "worst_auto_label": row["worst_auto_label"],
            "all_exact": row["all_exact"],
            "energy_ae_before": row["energy_ae_before"],
            "energy_ae_after": row["energy_ae_after"],
            "energy_improved": row["energy_improved"],
            "energy_improved_but_identity_not_all_exact": row["energy_improved_but_identity_not_all_exact"],
            "needs_human_review": row["needs_human_review"],
            "before": row["before"],
            "after": row["after"],
            "label_energy_kcal": row["label_energy_kcal"],
            "label_protein_g": row["label_protein_g"],
            "label_carb_g": row["label_carb_g"],
            "label_fat_g": row["label_fat_g"],
        }
        for row in dishes
    ]
    write_reports(payload, dishes, components)
    after_hash = {str(path.relative_to(ROOT)): sha256_file(path) for path in frozen_paths if path.is_file()}
    drifted = [name for name, digest in before_hash.items() if after_hash.get(name) != digest]
    if drifted:
        raise SystemExit("frozen files changed: " + ", ".join(drifted))
    post = payload["identity_post_swap"]
    improved = payload["dish_identity_split"]["energy_improved"]
    print(
        f"codes {post['n']} exact {post['n_exact']} near {post['n_near']} wrong {post['n_wrong']} "
        f"review_rows {payload['n_human_review_component_rows']} dishes {payload['n_human_review_dishes']}"
    )
    print(
        f"energy MAE {macros_50['energy_kcal_mae_before']:.4f} -> {macros_50['energy_kcal_mae_after']:.4f} "
        f"protein {macros_50['protein_g_mae_before']:.4f} -> {macros_50['protein_g_mae_after']:.4f} "
        f"carb {macros_50['carb_g_mae_before']:.4f} -> {macros_50['carb_g_mae_after']:.4f} "
        f"fat {macros_50['fat_g_mae_before']:.4f} -> {macros_50['fat_g_mae_after']:.4f}"
    )
    print(
        f"disagreement plates {improved['identity_not_all_exact']}/{improved['n']} "
        f"all_exact_and_improved {improved['all_exact']}"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic audit of the frozen #51 food-code swaps")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        print("food-id semantic audit self-check ok")
        return 0
    run_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
