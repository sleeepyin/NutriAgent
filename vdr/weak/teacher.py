"""Rule teacher for train-only weak controller steps.

For each official ``rgb_train`` dish the teacher:

1. Reads ingredient names from Nutrition5K cafe metadata and treats them as
   weak perception hypotheses (``hypothesis_source=n5k_ingredient_metadata``).
   This path does not call a VLM and does not need a GPU.
2. Retrieves FNDDS with the existing lexical ranker and places grams with the
   existing portion grid, using a fixed 100 g hint. Neither prior is changed
   here.
3. Compares that draw to the train ingredient's own kcal/g and grams, which
   are privileged. The comparison chooses ``CORRECT_CODE``, a second
   ``RETRIEVE`` (plain-produce or dish-form rewrite), ``RE_PORTION``,
   ``ABSTAIN``, or ``STOP``. ``RE_DECOMPOSE`` is not emitted.
   ``VOLUME_ESTIMATE`` is never emitted.

``rgb_test`` and ``depth_test`` are refused before any dish is retrieved.
A stable 10% slice of ``rgb_train`` (see ``pool_of_dish``) is the hold-out
for any later threshold check. The cutoffs in this file are fixed a priori;
they were not fit on the 507-dish test pool.

Behavior-cloning features must use the pre-action fields
(``hyp_i``, ``code_i``, ``cands_i``, ``portion_g_i``, ``verify_codes``,
``step_i``) and the discrete ``action_t`` label. ``obs_t`` is the tool
return. On ``CORRECT_CODE`` and ``RE_PORTION`` it contains teacher-chosen
code or grams, so it is privileged and must not be copied into the policy
input.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Callable, Sequence

from vdr.data.n5k import DishMeta, Ingredient, load_dish_metadata, read_ids, test_split_ids
from vdr.schemas.loader import IR_STEP_SCHEMA_PATH, load_schema, validate_instance
from vdr.tools.fndds_retrieve import (
    RetrieveResult,
    query_dish_forms,
    retrieve,
    rewrite_dish_form_query,
    rewrite_plain_query,
)
from vdr.tools.nutrition_lookup import (
    freeze_nutrition_vector,
    lookup_per_100g,
    scale_by_portion,
)
from vdr.tools.portion_grid import DEFAULT_PORTION_GRID_G, search_portion
from vdr.tools.rule_verify import verify

TEACHER_VERSION = "0.1.0"
TRAIN_SPLIT = "rgb_train"

# Fixed a priori. Not fit on rgb_test. Inspect changes on the train hold-out.
MASS_KEEP_G = 8.0
ENERGY_KEEP_KCAL = 8.0
BLIND_PORTION_HINT_G = 100.0
TOP_K = 5
MIN_RETRIEVE_SCORE = 0.45
DENSITY_REL_OK = 0.40
DENSITY_GAIN = 0.15
DENSITY_REL_FLOOR_KCAL = 25.0
MAX_SCORE_DROP = 0.20
PORTION_REL_OK = 0.25
UNRESOLVED_ENERGY_SHARE = 0.40
HOLDOUT_MOD = 10

TEACHER_ACTIONS = frozenset(
    {"RETRIEVE", "CORRECT_CODE", "RE_PORTION", "ABSTAIN", "STOP"}
)
NUTRI_KEYS = ("energy_kcal", "protein_g", "carb_g", "fat_g", "mass_g")

PROTECTED_RESULT_NAMES = frozenset(
    {
        "n5k_rgb_test_overhead_b0_b1.json",
        "n5k_rgb_test_overhead_b0_b1.jsonl",
        "n5k_subset_b0_b1.json",
        "n5k_subset_b0_b1.jsonl",
        "n5k_subset_b0_b1_n50_seed42.json",
        "n5k_subset_b0_b1_n50_seed42.jsonl",
        "n5k_overhead_rgb.json",
        "b0_b1_image_smoke.json",
    }
)

THRESHOLDS: dict[str, float | int] = {
    "mass_keep_g": MASS_KEEP_G,
    "energy_keep_kcal": ENERGY_KEEP_KCAL,
    "blind_portion_hint_g": BLIND_PORTION_HINT_G,
    "top_k": TOP_K,
    "min_retrieve_score": MIN_RETRIEVE_SCORE,
    "density_rel_ok": DENSITY_REL_OK,
    "density_gain": DENSITY_GAIN,
    "density_rel_floor_kcal": DENSITY_REL_FLOOR_KCAL,
    "max_score_drop": MAX_SCORE_DROP,
    "portion_rel_ok": PORTION_REL_OK,
    "unresolved_energy_share": UNRESOLVED_ENERGY_SHARE,
    "holdout_mod": HOLDOUT_MOD,
}


class TestSplitRefused(ValueError):
    """The caller asked the teacher to read a test split or a test dish."""


class UnsafeOutputPath(ValueError):
    """Refusing to write over a frozen evaluation artifact."""


RetrieveFn = Callable[..., RetrieveResult]
LookupFn = Callable[..., dict[str, float]]
PortionFn = Callable[..., Any]


def refuse_test_split(split: str) -> None:
    """Reject every split other than official rgb_train."""
    if split != TRAIN_SPLIT:
        raise TestSplitRefused(
            f"weak-step teacher only accepts split {TRAIN_SPLIT!r}; "
            f"refused {split!r}. Do not use rgb_test or depth_test GT."
        )


def pool_of_dish(dish_id: str) -> str:
    """Stable 10% hold-out inside rgb_train.

    First 8 hex digits of ``sha256(dish_id)``, modulo 10. Zero is
    ``holdout``. This is not the Nutrition5K test split. Later threshold
    checks belong on ``holdout`` only.
    """
    digest = hashlib.sha256(dish_id.encode("utf-8")).hexdigest()
    if int(digest[:8], 16) % HOLDOUT_MOD == 0:
        return "holdout"
    return "fit"


def assert_safe_output(path: Path) -> None:
    """Do not overwrite the frozen 507-dish B0/B1 artifacts."""
    if path.name in PROTECTED_RESULT_NAMES:
        raise UnsafeOutputPath(
            f"refusing to write {path.name}: that file is a frozen eval artifact"
        )


def keep_ingredient(ing: Ingredient) -> bool:
    """Drop spice-scale rows. 8 g / 8 kcal is a fixed floor, not a fit."""
    if not (ing.name or "").strip():
        return False
    return float(ing.mass_g) >= MASS_KEEP_G or float(ing.energy_kcal) >= ENERGY_KEEP_KCAL


def ingredient_kcal_per_100g(ing: Ingredient) -> float | None:
    if float(ing.mass_g) <= 1e-3:
        return None
    return 100.0 * float(ing.energy_kcal) / float(ing.mass_g)


def snap_to_grid(grams: float, grid: Sequence[float] = DEFAULT_PORTION_GRID_G) -> float:
    """Nearest existing grid point. Does not rescore the portion prior."""
    return float(min(grid, key=lambda point: (abs(float(point) - float(grams)), float(point))))


def candidate_rewrite(query: str) -> str | None:
    """One alternate query, or None when the rewrite would repeat itself."""
    plain = rewrite_plain_query(query)
    if plain and plain.strip().lower() != query.strip().lower():
        return plain
    if query_dish_forms(query):
        dish = rewrite_dish_form_query(query)
        if dish and dish.strip().lower() != query.strip().lower():
            return dish
    return None


def _density(code: str, lookup_fn: LookupFn, catalog_path: str | Path | None) -> float | None:
    try:
        per = lookup_fn(code, catalog_path=catalog_path)
    except KeyError:
        return None
    if per.get("energy_kcal") is None:
        return None
    return float(per["energy_kcal"])


def _rel_density(pred: float | None, target: float) -> float | None:
    if pred is None:
        return None
    return abs(float(pred) - float(target)) / max(float(target), DENSITY_REL_FLOOR_KCAL)


def choose_code(
    cands: Sequence[dict[str, Any]],
    ing: Ingredient,
    lookup_fn: LookupFn,
    catalog_path: str | Path | None,
) -> tuple[str, str]:
    """Pick a top-k code. Returns ``(code, ok|correct|none)``.

    Top-1 stands when its kcal/100g is close to the train ingredient.
    Another candidate replaces it only when it is inside the score window,
    clears the density band, and improves on top-1. An unrelated low-score
    food that happens to match kcal/g is not selected.
    """
    target = ingredient_kcal_per_100g(ing)
    if target is None or not cands:
        return "", "none"
    top = cands[0]
    top_score = float(top.get("score", 0.0))
    top_rel = _rel_density(_density(str(top["code"]), lookup_fn, catalog_path), target)
    if (
        top_rel is not None
        and top_rel <= DENSITY_REL_OK
        and top_score >= MIN_RETRIEVE_SCORE
    ):
        return str(top["code"]), "ok"

    best: tuple[float, float, str] | None = None
    for cand in cands:
        score = float(cand.get("score", 0.0))
        if score < MIN_RETRIEVE_SCORE or top_score - score > MAX_SCORE_DROP:
            continue
        rel = _rel_density(_density(str(cand["code"]), lookup_fn, catalog_path), target)
        if rel is None or rel > DENSITY_REL_OK:
            continue
        if top_rel is not None and top_rel - rel < DENSITY_GAIN:
            continue
        item = (rel, -score, str(cand["code"]))
        if best is None or item < best:
            best = item
    if best is None:
        return "", "none"
    return best[2], "correct"


def _public_cands(cands: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cand in cands:
        item: dict[str, Any] = {
            "code": str(cand["code"]),
            "score": round(float(cand.get("score", 0.0)), 4),
        }
        if cand.get("name") is not None:
            item["name"] = str(cand["name"])
        out.append(item)
    return out


def _round_nutri(nutri: dict[str, Any] | None) -> dict[str, float | None] | None:
    if nutri is None:
        return None
    out: dict[str, float | None] = {}
    for key in NUTRI_KEYS:
        value = nutri.get(key)
        if value is None:
            out[key] = None
        else:
            out[key] = round(max(0.0, float(value)), 4)
    return out


def _sum_nutri(parts: Sequence[dict[str, Any] | None]) -> dict[str, float | None] | None:
    acc = {key: 0.0 for key in NUTRI_KEYS}
    seen = False
    for part in parts:
        if not part:
            continue
        seen = True
        for key in NUTRI_KEYS:
            value = part.get(key)
            if value is not None:
                acc[key] += float(value)
    if not seen:
        return None
    return _round_nutri(acc)


def _null_nutri() -> dict[str, None]:
    return {key: None for key in NUTRI_KEYS}


def _scaled(
    code: str,
    portion_g: float,
    lookup_fn: LookupFn,
    catalog_path: str | Path | None,
) -> dict[str, float | None] | None:
    try:
        per = lookup_fn(code, catalog_path=catalog_path)
    except KeyError:
        return None
    scaled = scale_by_portion(per, portion_g)
    scaled["mass_g"] = float(portion_g)
    frozen = freeze_nutrition_vector(scaled)
    return _round_nutri(frozen)


def _blind_portion(
    code: str,
    portion_fn: PortionFn,
    catalog_path: str | Path | None,
) -> float | None:
    try:
        result = portion_fn(
            code,
            hint_g=BLIND_PORTION_HINT_G,
            catalog_path=catalog_path,
        )
    except KeyError:
        return None
    return float(result.portion_g)


def _binding_edges(
    comp_i: int,
    code: str,
    query: str,
    portion: float | None,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = [
        {
            "src": f"c{comp_i}",
            "dst": f"q{comp_i}",
            "type": "code<-query",
            "attrs": {"query": query, "code": code},
        },
        {
            "src": f"c{comp_i}",
            "dst": f"h{comp_i}",
            "type": "code<-hyp",
            "attrs": {"code": code},
        },
    ]
    if portion is not None:
        edges.append(
            {
                "src": f"p{comp_i}",
                "dst": f"c{comp_i}",
                "type": "portion<-code",
                "attrs": {"portion_g": round(float(portion), 4), "code": code},
            }
        )
        edges.append(
            {
                "src": f"n{comp_i}",
                "dst": f"c{comp_i}",
                "type": "n<-code×portion",
                "attrs": {"code": code},
            }
        )
    return edges


def _obs(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "label_quality": "weak",
        "hypothesis_source": "n5k_ingredient_metadata",
        "teacher_version": TEACHER_VERSION,
        "privileged": True,
        "volume_tool": False,
    }
    base.update(extra)
    return base


def _verify(
    nutri: dict[str, Any] | None,
    cands: Sequence[dict[str, Any]],
    portion: float | None,
    edges: Sequence[dict[str, Any]],
) -> list[str]:
    return verify(
        {
            "n": nutri,
            "cands_i": list(cands),
            "portion_g_i": portion,
            "evidence_edges": list(edges),
        }
    )


_IR_SCHEMA: dict[str, Any] | None = None


def _schema() -> dict[str, Any]:
    global _IR_SCHEMA
    if _IR_SCHEMA is None:
        _IR_SCHEMA = load_schema(IR_STEP_SCHEMA_PATH)
    return _IR_SCHEMA


def _step(
    *,
    dish_id: str,
    step_i: int,
    component_i: int | None,
    hyp: str | None,
    code: str | None,
    cands: list[dict[str, Any]],
    portion: float | None,
    portion_src: str | None,
    nutri: dict[str, Any] | None,
    nutri_dish: dict[str, Any] | None,
    edges: list[dict[str, Any]],
    action: str,
    obs: dict[str, Any],
    status: str | None,
    abstain_reason: str | None,
) -> dict[str, Any]:
    if action not in TEACHER_ACTIONS:
        raise RuntimeError(f"teacher emitted an action outside its set: {action}")
    record = {
        "dish_id": dish_id,
        "step_i": step_i,
        "component_i": component_i,
        "hyp_i": hyp,
        "code_i": code,
        "cands_i": cands,
        "portion_g_i": None if portion is None else round(float(portion), 4),
        "portion_src_i": portion_src,
        "nutri_i": _round_nutri(nutri),
        "nutri_dish": _round_nutri(nutri_dish) if nutri_dish is not None else None,
        "verify_codes": _verify(nutri, cands, portion, edges),
        "evidence_edges": edges,
        "action_t": action,
        "obs_t": obs,
        "s": status,
        "abstain_reason": abstain_reason,
    }
    validate_instance(record, _schema())
    return record


def generate_dish_steps(
    dish: DishMeta,
    *,
    catalog_path: str | Path | None = None,
    retrieve_fn: RetrieveFn = retrieve,
    lookup_fn: LookupFn = lookup_per_100g,
    portion_fn: PortionFn = search_portion,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Weak IR steps for one dish, plus a train-only audit dict.

    The audit dict carries train GT and is for the summary file. It is not
    an IR step and must not be used as a policy feature.
    """
    kept = [ing for ing in dish.ingredients if keep_ingredient(ing)]
    steps: list[dict[str, Any]] = []
    finalized: list[dict[str, Any]] = []
    unresolved_kcal = 0.0
    n_unresolved = 0
    correction = False
    step_i = 0

    def emit(record: dict[str, Any]) -> None:
        nonlocal step_i
        steps.append(record)
        step_i += 1

    if not kept:
        emit(
            _step(
                dish_id=dish.dish_id,
                step_i=0,
                component_i=None,
                hyp=None,
                code=None,
                cands=[],
                portion=None,
                portion_src=None,
                nutri=None,
                nutri_dish=_null_nutri(),
                edges=[
                    {
                        "src": "dish",
                        "dst": "empty",
                        "type": "abstain_from",
                        "attrs": {"reason": "no_substantial_ingredient"},
                    }
                ],
                action="ABSTAIN",
                obs=_obs(s="abstain", n_kept=0, n_unresolved=0),
                status="abstain",
                abstain_reason="no_substantial_ingredient",
            )
        )
        return steps, _audit(dish, steps, grounded_energy=None, n_kept=0, n_unresolved=0)

    for comp_i, ing in enumerate(kept):
        queries = [ing.name.strip()]
        rewrite = candidate_rewrite(queries[0])
        if rewrite is not None:
            queries.append(rewrite)

        accepted_code = ""
        accepted_decision = "none"
        accepted_cands: list[dict[str, Any]] = []
        accepted_query = queries[0]
        accepted_top: str | None = None
        failed_top: str | None = None
        failed_cands: list[dict[str, Any]] = []
        failed_query = queries[0]

        def realize(code: str, query: str) -> tuple[float | None, dict[str, Any] | None, list[dict[str, Any]]]:
            portion = _blind_portion(code, portion_fn, catalog_path)
            if portion is None:
                return None, None, _binding_edges(comp_i, code, query, None)
            nutri = _scaled(code, portion, lookup_fn, catalog_path)
            if nutri is None:
                return None, None, _binding_edges(comp_i, code, query, None)
            return portion, nutri, _binding_edges(comp_i, code, query, portion)

        for query_i, query in enumerate(queries):
            found = retrieve_fn(query, top_k=TOP_K, catalog_path=catalog_path)
            cands = _public_cands(found.cands)
            top_code = str(cands[0]["code"]) if cands else None
            if query_i == 0:
                pre_code = None
                pre_cands: list[dict[str, Any]] = []
                pre_portion = None
                pre_src = None
                pre_nutri = None
                pre_edges: list[dict[str, Any]] = []
            else:
                correction = True
                pre_code = failed_top
                pre_cands = failed_cands
                pre_portion = None
                pre_src = None
                pre_nutri = None
                pre_edges = []
                if pre_code:
                    pre_portion, pre_nutri, pre_edges = realize(pre_code, failed_query)
                    if pre_portion is not None:
                        pre_src = "grid_search"
            visible = list(finalized)
            if pre_nutri is not None:
                visible.append(pre_nutri)
            emit(
                _step(
                    dish_id=dish.dish_id,
                    step_i=step_i,
                    component_i=comp_i,
                    hyp=ing.name,
                    code=pre_code,
                    cands=pre_cands,
                    portion=pre_portion,
                    portion_src=pre_src,
                    nutri=pre_nutri,
                    nutri_dish=_sum_nutri(visible),
                    edges=pre_edges,
                    action="RETRIEVE",
                    obs=_obs(
                        query=query,
                        cands=cands,
                        selected_code=top_code,
                    ),
                    status=None,
                    abstain_reason=None,
                )
            )
            code, decision = choose_code(cands, ing, lookup_fn, catalog_path)
            if decision != "none":
                accepted_code = code
                accepted_decision = decision
                accepted_cands = cands
                accepted_query = query
                accepted_top = top_code
                break
            failed_top = top_code
            failed_cands = cands
            failed_query = query
        else:
            n_unresolved += 1
            unresolved_kcal += max(float(ing.energy_kcal), 0.0)
            continue

        code_now = accepted_top
        if not code_now:
            n_unresolved += 1
            unresolved_kcal += max(float(ing.energy_kcal), 0.0)
            continue
        portion_now, nutri_now, edges_now = realize(code_now, accepted_query)
        if portion_now is None or nutri_now is None:
            n_unresolved += 1
            unresolved_kcal += max(float(ing.energy_kcal), 0.0)
            continue
        if accepted_decision == "correct" and accepted_code != code_now:
            correction = True
            emit(
                _step(
                    dish_id=dish.dish_id,
                    step_i=step_i,
                    component_i=comp_i,
                    hyp=ing.name,
                    code=code_now,
                    cands=accepted_cands,
                    portion=portion_now,
                    portion_src="grid_search",
                    nutri=nutri_now,
                    nutri_dish=_sum_nutri([*finalized, nutri_now]),
                    edges=edges_now,
                    action="CORRECT_CODE",
                    obs=_obs(**{"from": code_now, "to": accepted_code}),
                    status=None,
                    abstain_reason=None,
                )
            )
            code_now = accepted_code
            portion_now, nutri_now, edges_now = realize(code_now, accepted_query)
            if portion_now is None or nutri_now is None:
                n_unresolved += 1
                unresolved_kcal += max(float(ing.energy_kcal), 0.0)
                continue

        oracle = snap_to_grid(float(ing.mass_g))
        if abs(portion_now - oracle) / max(oracle, 1.0) > PORTION_REL_OK:
            correction = True
            emit(
                _step(
                    dish_id=dish.dish_id,
                    step_i=step_i,
                    component_i=comp_i,
                    hyp=ing.name,
                    code=code_now,
                    cands=accepted_cands,
                    portion=portion_now,
                    portion_src="grid_search",
                    nutri=nutri_now,
                    nutri_dish=_sum_nutri([*finalized, nutri_now]),
                    edges=edges_now,
                    action="RE_PORTION",
                    obs=_obs(
                        portion_g=round(oracle, 4),
                        from_portion_g=round(portion_now, 4),
                        portion_src="other",
                        portion_note="n5k_ingredient_mass_snapped_to_grid",
                    ),
                    status=None,
                    abstain_reason=None,
                )
            )
            portion_now = oracle
            nutri_now = _scaled(code_now, portion_now, lookup_fn, catalog_path)
        if nutri_now is None:
            n_unresolved += 1
            unresolved_kcal += max(float(ing.energy_kcal), 0.0)
            continue
        finalized.append(nutri_now)

    share = unresolved_kcal / max(float(dish.energy_kcal), 1.0)
    grounded = _sum_nutri(finalized)
    if not finalized or share > UNRESOLVED_ENERGY_SHARE:
        reason = "no_grounded_component" if not finalized else "unresolved_ingredients"
        emit(
            _step(
                dish_id=dish.dish_id,
                step_i=step_i,
                component_i=None,
                hyp=None,
                code=None,
                cands=[],
                portion=None,
                portion_src=None,
                nutri=None,
                nutri_dish=_null_nutri(),
                edges=[
                    {
                        "src": "dish",
                        "dst": "fail",
                        "type": "abstain_from",
                        "attrs": {"reason": reason},
                    }
                ],
                action="ABSTAIN",
                obs=_obs(s="abstain", n_kept=len(kept), n_unresolved=n_unresolved),
                status="abstain",
                abstain_reason=reason,
            )
        )
    else:
        status = "corrected" if correction else "pass"
        emit(
            _step(
                dish_id=dish.dish_id,
                step_i=step_i,
                component_i=None,
                hyp=None,
                code=None,
                cands=[],
                portion=None,
                portion_src=None,
                nutri=None,
                nutri_dish=grounded,
                edges=[{"src": "dish", "dst": "n_dish", "type": "n<-aggregate"}],
                action="STOP",
                obs=_obs(s=status, n_kept=len(kept), n_unresolved=n_unresolved),
                status=status,
                abstain_reason=None,
            )
        )
    grounded_energy = None if grounded is None else grounded.get("energy_kcal")
    return steps, _audit(
        dish,
        steps,
        grounded_energy=grounded_energy,
        n_kept=len(kept),
        n_unresolved=n_unresolved,
    )


def _audit(
    dish: DishMeta,
    steps: list[dict[str, Any]],
    *,
    grounded_energy: float | None,
    n_kept: int,
    n_unresolved: int,
) -> dict[str, Any]:
    last = steps[-1]
    emitted = None
    if last.get("s") != "abstain":
        dish_nutri = last.get("nutri_dish") or {}
        emitted = dish_nutri.get("energy_kcal")
    abs_err = None
    if emitted is not None:
        abs_err = round(abs(float(emitted) - float(dish.energy_kcal)), 4)
    return {
        "dish_id": dish.dish_id,
        "pool": pool_of_dish(dish.dish_id),
        "s": last.get("s"),
        "n_steps": len(steps),
        "actions": [step["action_t"] for step in steps],
        "n_kept": n_kept,
        "n_unresolved": n_unresolved,
        "emitted_energy_kcal": emitted,
        "grounded_energy_kcal": grounded_energy,
        "train_gt_energy_kcal": dish.energy_kcal,
        "abs_err_kcal": abs_err,
    }


def select_train_dishes(
    *,
    split: str = TRAIN_SPLIT,
    pool: str = "fit",
    n: int | None = None,
    seed: int | None = None,
    dish_ids: Sequence[str] | None = None,
    metadata: dict[str, DishMeta] | None = None,
) -> list[DishMeta]:
    """rgb_train rows for one pool. Raises if a test id is requested."""
    refuse_test_split(split)
    if pool not in {"fit", "holdout", "all"}:
        raise ValueError(f"pool must be fit, holdout, or all; got {pool!r}")
    if n is not None and n < 1:
        raise ValueError("--n must be >= 1")
    forbidden = test_split_ids()
    train_ids = read_ids(TRAIN_SPLIT)
    leaked = [dish_id for dish_id in train_ids if dish_id in forbidden]
    if leaked:
        raise TestSplitRefused(
            f"rgb_train file contains {len(leaked)} test ids; refusing to continue"
        )
    table = metadata if metadata is not None else load_dish_metadata()
    if dish_ids is None:
        chosen = list(train_ids)
    else:
        unknown = [dish_id for dish_id in dish_ids if dish_id not in set(train_ids)]
        if unknown:
            raise TestSplitRefused(
                "weak-step teacher refused dish ids that are not in rgb_train: "
                + ", ".join(unknown[:5])
            )
        hit = [dish_id for dish_id in dish_ids if dish_id in forbidden]
        if hit:
            raise TestSplitRefused(
                "weak-step teacher refused test dish ids: " + ", ".join(hit[:5])
            )
        chosen = list(dish_ids)

    rows: list[DishMeta] = []
    for dish_id in chosen:
        slot = pool_of_dish(dish_id)
        if pool != "all" and slot != pool:
            if dish_ids is not None:
                raise ValueError(
                    f"{dish_id} is in the {slot} pool, not {pool}. "
                    "Hold-out rows are not for fitting thresholds."
                )
            continue
        dish = table.get(dish_id)
        if dish is None:
            continue
        rows.append(dish)

    if seed is not None and n is not None:
        rng = random.Random(seed)
        take = min(n, len(rows))
        rows = rng.sample(rows, take)
        rows.sort(key=lambda dish: dish.dish_id)
    elif n is not None:
        rows = rows[:n]
    return rows


def generate_split(
    *,
    split: str = TRAIN_SPLIT,
    pool: str = "fit",
    n: int | None = None,
    seed: int | None = None,
    dish_ids: Sequence[str] | None = None,
    catalog_path: str | Path | None = None,
    retrieve_fn: RetrieveFn = retrieve,
    lookup_fn: LookupFn = lookup_per_100g,
    portion_fn: PortionFn = search_portion,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate weak steps for a train pool. Test splits raise."""
    dishes = select_train_dishes(
        split=split,
        pool=pool,
        n=n,
        seed=seed,
        dish_ids=dish_ids,
    )
    steps: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    forbidden = test_split_ids()
    for dish in dishes:
        if dish.dish_id in forbidden:
            raise TestSplitRefused(f"refusing test dish {dish.dish_id}")
        dish_steps, audit = generate_dish_steps(
            dish,
            catalog_path=catalog_path,
            retrieve_fn=retrieve_fn,
            lookup_fn=lookup_fn,
            portion_fn=portion_fn,
        )
        steps.extend(dish_steps)
        audits.append(audit)
    summary = build_summary(
        audits,
        split=split,
        pool=pool,
        n_requested=n,
        seed=seed,
    )
    return steps, summary


def build_summary(
    audits: Sequence[dict[str, Any]],
    *,
    split: str,
    pool: str,
    n_requested: int | None,
    seed: int | None,
) -> dict[str, Any]:
    """Train audit. ``train_gt_energy_kcal`` here is not a controller feature."""
    actions: dict[str, int] = {}
    terminal: dict[str, int] = {}
    abs_errs: list[float] = []
    for audit in audits:
        terminal[str(audit.get("s"))] = terminal.get(str(audit.get("s")), 0) + 1
        for action in audit.get("actions") or []:
            actions[action] = actions.get(action, 0) + 1
        if audit.get("abs_err_kcal") is not None:
            abs_errs.append(float(audit["abs_err_kcal"]))
    mae = None
    if abs_errs:
        mae = round(sum(abs_errs) / len(abs_errs), 4)
    return {
        "run_id": f"weak_steps_{split}_{pool}",
        "teacher_version": TEACHER_VERSION,
        "split": split,
        "pool": pool,
        "n_requested": n_requested,
        "seed": seed,
        "n_samples": len(audits),
        "label_quality": "weak",
        "hypothesis_source": "n5k_ingredient_metadata",
        "uses_train_gt": True,
        "uses_test_gt": False,
        "vlm": False,
        "volume_tool": False,
        "legacy_regression_weights": False,
        "actions": actions,
        "terminal_s": terminal,
        "thresholds": dict(THRESHOLDS),
        "holdout_rule": "sha256(dish_id) hex[:8] % 10 == 0",
        "supervision": "action_t on pre-action IR fields; obs_t is privileged",
        "metrics": {
            "note": (
                "Train-split audit of emitted energy versus train dish GT "
                "on pass/corrected rows. Not a test score. Do not tune on rgb_test."
            ),
            "n_scored": len(abs_errs),
            "train_energy_mae": mae,
        },
        "dishes": list(audits),
    }


def default_output_path(*, pool: str, n: int | None, seed: int | None) -> Path:
    """Path under results/weak_steps. Never the frozen 507 filenames."""
    tag = f"rgb_train_{pool}"
    tag += f"_n{n}" if n is not None else "_all"
    if seed is not None:
        tag += f"_seed{seed}"
    return Path("results") / "weak_steps" / f"{tag}.jsonl"


def summary_path_for(jsonl_path: Path) -> Path:
    name = jsonl_path.name
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")] + ".summary.json"
    else:
        name = name + ".summary.json"
    return jsonl_path.with_name(name)
