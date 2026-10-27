"""B1 RAG-once baseline — DietAI24-style retrieve + aggregate.

No VLM required when hypotheses are provided via `hypotheses`.
The image path (``predict``) calls Qwen2.5-VL when hypotheses are omitted.

Before lookup, duplicate hypotheses are merged (median portion, not a sum),
then a cheese+crust style fragment pair is collapsed into one query.
Each matched code's grams are chosen on ``portion_grid`` instead of trusting
the raw VLM number. The grid adds a density penalty above 100 g so a large
hint on nuts or hard cheese cannot sit on the hint; a 100 g hint is kept.

If ``CODE_CONF_GAP`` fires, one follow-up runs before STOP:

- a second RETRIEVE when the selected name is still a prepared form
  (dip/soup/sauce/...) for a plain produce query
- a second RETRIEVE when the query named a dish form, the selected code is
  energy-dense, and a candidate that keeps the dish form is sitting behind it
- otherwise one RE_PORTION that re-scores the same grid

The follow-up does not shrink a portion that the grid already accepted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from vdr.perception.hypotheses import dedupe_hypotheses, merge_complementary_fragments
from vdr.tools.fndds_retrieve import (
    name_has_lemmas,
    name_is_prepared,
    query_asks_prepared,
    query_dish_forms,
    retrieve,
    rewrite_dish_form_query,
    rewrite_plain_query,
)
from vdr.tools.portion_grid import DENSE_KCAL_PER_100G
from vdr.tools.nutrition_lookup import (
    aggregate_dish,
    freeze_nutrition_vector,
    lookup_and_scale,
)
from vdr.tools.portion_grid import DEFAULT_PORTION_GRID_G, search_portion
from vdr.tools.rule_verify import verify


def _as_hyp_list(
    hypotheses: Sequence[dict[str, Any]] | Sequence[str] | None,
) -> list[dict[str, Any]]:
    if not hypotheses:
        return []
    out: list[dict[str, Any]] = []
    for h in hypotheses:
        if isinstance(h, str):
            out.append({"name": h, "portion_g": None})
        else:
            out.append(
                {
                    "name": h.get("name") or h.get("hyp") or h.get("query"),
                    "portion_g": h.get("portion_g"),
                }
            )
    return out


def choose_followup(verify_codes: Sequence[str], components: Sequence[dict[str, Any]]) -> str | None:
    """One extra action after the first retrieve, or None.

    CODE_CONF_GAP on a prepared-form hit for a plain produce query, or on a
    high-density ingredient that beat a dish-form candidate, gets a second
    RETRIEVE. Any other CODE_CONF_GAP gets one confirmatory RE_PORTION. Tied
    raw variants are the common case: another retrieve would not change the
    code, and shrinking those portions would bias mass.
    """
    if "CODE_CONF_GAP" not in verify_codes:
        return None
    for comp in components:
        if _needs_second_retrieve(comp) or _needs_dish_form_retrieve(comp):
            return "RETRIEVE"
    return "RE_PORTION"


def _needs_second_retrieve(comp: dict[str, Any]) -> bool:
    if not comp.get("code"):
        return False
    query = str(comp.get("query") or "")
    name = str(comp.get("selected_name") or "")
    if query_asks_prepared(query) or not name_is_prepared(name):
        return False
    return rewrite_plain_query(query) is not None


def _kcal_per_100g(comp: dict[str, Any]) -> float | None:
    full = comp.get("nutri_full") or {}
    mass = full.get("mass_g")
    if mass is None:
        mass = comp.get("portion_g")
    energy = full.get("energy_kcal")
    try:
        mass_f = float(mass)
        energy_f = float(energy)
    except (TypeError, ValueError):
        return None
    if mass_f <= 0:
        return None
    return energy_f * 100.0 / mass_f


def _dish_form_alternatives(comp: dict[str, Any]) -> list[str]:
    forms = query_dish_forms(str(comp.get("query") or ""))
    if not forms:
        return []
    selected = str(comp.get("selected_name") or "")
    if name_has_lemmas(selected, set(forms)):
        return []
    names: list[str] = []
    for cand in comp.get("cands") or []:
        if cand.get("code") == comp.get("code"):
            continue
        name = str(cand.get("name") or "")
        if name_has_lemmas(name, set(forms)):
            names.append(name)
    return names


def _needs_dish_form_retrieve(comp: dict[str, Any]) -> bool:
    """High-density top1 lost a dish-form candidate that the query asked for."""
    if not comp.get("code"):
        return False
    if not _dish_form_alternatives(comp):
        return False
    density = _kcal_per_100g(comp)
    if density is None or density < DENSE_KCAL_PER_100G:
        return False
    return rewrite_dish_form_query(
        str(comp.get("query") or ""),
        _dish_form_alternatives(comp),
    ) is not None


def _tightest_cands(components: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top-2 of the component whose score gap is smallest."""
    best: list[dict[str, Any]] | None = None
    best_gap: float | None = None
    for comp in components:
        cands = comp.get("cands") or []
        if len(cands) < 2:
            continue
        try:
            gap = float(cands[0].get("score", 0)) - float(cands[1].get("score", 0))
        except (TypeError, ValueError):
            continue
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = list(cands[:2])
    return best or []


def _portion_for(code: str, hint: float, catalog_path: str | Path | None) -> Any:
    return search_portion(code, hint_g=hint, catalog_path=catalog_path)


def _bind_hit(
    comp: dict[str, Any],
    rr: Any,
    hint: float,
    catalog_path: str | Path | None,
) -> None:
    """Fill code, grid portion, and scaled nutrition from a retrieve result."""
    comp["cands"] = list(rr.cands)
    comp["code"] = rr.selected_code
    comp["selected_name"] = rr.cands[0].get("name") if rr.cands else None
    if not rr.selected_code:
        comp["portion_g"] = hint
        comp["nutri_full"] = None
        comp["nutri"] = None
        return
    portion = _portion_for(rr.selected_code, hint, catalog_path)
    comp["portion_g"] = float(portion.portion_g)
    comp["portion_scores"] = portion.scores
    scaled = lookup_and_scale(rr.selected_code, comp["portion_g"], catalog_path=catalog_path)
    full = dict(scaled)
    full["mass_g"] = comp["portion_g"]
    comp["nutri_full"] = full
    comp["nutri"] = freeze_nutrition_vector(full)


def _resolve_component(
    hyp: dict[str, Any],
    *,
    catalog_path: str | Path | None,
    top_k: int,
    default_portion_g: float,
) -> dict[str, Any]:
    name = hyp.get("name") or ""
    hint = hyp.get("portion_g")
    if hint is None:
        hint = default_portion_g
    hint_f = float(hint)
    comp: dict[str, Any] = {
        "name": name,
        "query": name,
        "hint_g": hint_f,
        "code": None,
        "selected_name": None,
        "cands": [],
        "portion_g": hint_f,
        "nutri": None,
        "nutri_full": None,
        "previous_code": None,
        "previous_name": None,
        "requery": None,
    }
    rr = retrieve(name, top_k=top_k, catalog_path=catalog_path)
    _bind_hit(comp, rr, hint_f, catalog_path)
    return comp


def _accept_requery(
    comp: dict[str, Any],
    rr: Any,
    *,
    catalog_path: str | Path | None,
    accept,
) -> bool:
    if not rr.selected_code or not rr.cands:
        return False
    new_name = str(rr.cands[0].get("name") or "")
    old_name = str(comp.get("selected_name") or "")
    if rr.selected_code == comp.get("code") or not accept(new_name, old_name):
        return False
    comp["previous_code"] = comp.get("code")
    comp["previous_name"] = old_name
    _bind_hit(comp, rr, float(comp["hint_g"]), catalog_path)
    return True


def _second_retrieve_summary(components: Sequence[dict[str, Any]], changed: bool) -> str:
    kinds = {comp.get("requery_kind") for comp in components if comp.get("requery")}
    if "dish_form" in kinds and "prepared" in kinds:
        target = "a prepared-form code and a dense dish-form miss"
    elif "dish_form" in kinds:
        target = "a dense code that dropped the queried dish form"
    else:
        target = "a prepared-form code"
    if changed:
        return f"second retrieve after CODE_CONF_GAP on {target}"
    return f"second retrieve after CODE_CONF_GAP; {target} unchanged"


def _prepared_rewrite_ok(new_name: str, old_name: str) -> bool:
    """Leave a dip/soup/sauce hit. Switching dip → soup is not a correction."""
    return name_is_prepared(old_name) and not name_is_prepared(new_name)


def _second_retrieve(
    components: list[dict[str, Any]],
    *,
    catalog_path: str | Path | None,
    top_k: int,
) -> bool:
    """Re-query prepared-form or dish-form misses. Return True if any code changes."""
    changed = False
    for comp in components:
        if _needs_second_retrieve(comp):
            rewritten = rewrite_plain_query(str(comp.get("query") or ""))
            kind = "prepared"
            accept = _prepared_rewrite_ok
        elif _needs_dish_form_retrieve(comp):
            forms = set(query_dish_forms(str(comp.get("query") or "")))
            rewritten = rewrite_dish_form_query(
                str(comp.get("query") or ""),
                _dish_form_alternatives(comp),
            )
            kind = "dish_form"

            def accept(new_name: str, _old_name: str, forms: set[str] = forms) -> bool:
                return name_has_lemmas(new_name, forms)

        else:
            continue
        comp["requery"] = rewritten
        comp["requery_kind"] = kind
        if not rewritten:
            continue
        rr = retrieve(rewritten, top_k=top_k, catalog_path=catalog_path)
        if _accept_requery(comp, rr, catalog_path=catalog_path, accept=accept):
            changed = True
    return changed


def _confirm_portions(
    components: list[dict[str, Any]],
    *,
    catalog_path: str | Path | None,
) -> None:
    """Re-run the grid. Deterministic, so an accepted point stays put."""
    for comp in components:
        if not comp.get("code"):
            continue
        portion = _portion_for(comp["code"], float(comp["hint_g"]), catalog_path)
        comp["portion_g"] = float(portion.portion_g)
        comp["portion_scores"] = portion.scores
        scaled = lookup_and_scale(comp["code"], comp["portion_g"], catalog_path=catalog_path)
        full = dict(scaled)
        full["mass_g"] = comp["portion_g"]
        comp["nutri_full"] = full
        comp["nutri"] = freeze_nutrition_vector(full)


def _verify_state(n_agg: dict[str, Any], edges: list[dict[str, Any]], components: list[dict[str, Any]]) -> dict[str, Any]:
    portions = [comp["portion_g"] for comp in components if comp.get("code")]
    return {
        "n": n_agg,
        "evidence_edges": edges,
        "portions": portions,
        "cands": _tightest_cands(components),
    }


def _aggregate(components: list[dict[str, Any]]) -> dict[str, float | None]:
    rows = [comp["nutri"] for comp in components if comp.get("nutri")]
    if not rows:
        return freeze_nutrition_vector({})
    return freeze_nutrition_vector(aggregate_dish(rows))


def _build_graph(
    dish_id: str,
    components: list[dict[str, Any]],
    vcodes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = [{"id": "dish", "type": "dish", "label": dish_id}]
    edges: list[dict[str, Any]] = []
    for i, comp in enumerate(components):
        name = comp.get("name") or f"comp_{i}"
        hid = f"h{i}"
        cid = f"c{i}"
        nodes.append({"id": hid, "type": "hyp", "label": name})
        nodes.append({"id": f"q{i}", "type": "query", "label": comp.get("query") or name})
        if not comp.get("code"):
            nodes.append({"id": cid, "type": "food_code", "label": None, "attrs": {"code": None}})
            continue
        if comp.get("previous_code") and comp["previous_code"] != comp["code"]:
            prev_id = f"c{i}_prev"
            nodes.append(
                {
                    "id": prev_id,
                    "type": "food_code",
                    "label": comp.get("previous_name"),
                    "attrs": {"code": comp["previous_code"]},
                }
            )
            edges.append({"src": cid, "dst": prev_id, "type": "corrects"})
        nodes.append(
            {
                "id": cid,
                "type": "food_code",
                "label": comp.get("selected_name"),
                "attrs": {"code": comp["code"], "cands": comp.get("cands") or []},
            }
        )
        pid = f"p{i}"
        nid = f"n{i}"
        portion = comp["portion_g"]
        nodes.append(
            {
                "id": pid,
                "type": "portion_g",
                "label": f"{portion}g",
                "attrs": {
                    "portion_g": portion,
                    "hint_g": comp.get("hint_g"),
                    "src": "portion_grid",
                },
            }
        )
        nodes.append(
            {
                "id": nid,
                "type": "nutrition",
                "label": f"{name} macros",
                "attrs": comp.get("nutri_full") or comp.get("nutri") or {},
            }
        )
        edges.append({"src": cid, "dst": f"q{i}", "type": "code<-query"})
        edges.append({"src": cid, "dst": hid, "type": "code<-hyp"})
        edges.append({"src": pid, "dst": cid, "type": "portion<-code"})
        edges.append(
            {
                "src": nid,
                "dst": cid,
                "type": "n<-code×portion",
                "attrs": {"via_portion": pid, "portion_g": portion},
            }
        )
        edges.append({"src": "dish", "dst": nid, "type": "n<-aggregate"})
    for vc in vcodes:
        rid = f"r_{vc}"
        nodes.append({"id": rid, "type": "rule", "label": vc})
        edges.append({"src": rid, "dst": "dish", "type": "rule_trigger", "attrs": {"code": vc}})
    return nodes, edges


def predict_from_hypotheses(
    dish_id: str,
    hypotheses: Sequence[dict[str, Any]] | Sequence[str],
    catalog_path: str | Path | None = None,
    top_k: int = 5,
    default_portion_g: float = 100.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Retrieve+aggregate from component hypotheses (no VLM)."""
    hyps_in = _as_hyp_list(hypotheses)
    hyps_deduped = dedupe_hypotheses(hyps_in)
    hyps = merge_complementary_fragments(hyps_deduped)
    components = [
        _resolve_component(
            hyp,
            catalog_path=catalog_path,
            top_k=top_k,
            default_portion_g=default_portion_g,
        )
        for hyp in hyps
    ]

    # Provisional edges so the first verify sees the same binding types as the final graph.
    provisional_edges = [
        {"src": f"n{i}", "dst": f"c{i}", "type": "n<-code×portion"}
        for i, comp in enumerate(components)
        if comp.get("code")
    ]
    n_first = _aggregate(components)
    first_vcodes = verify(_verify_state(n_first, provisional_edges, components))
    followup = choose_followup(first_vcodes, components)
    changed_code = False
    if followup == "RETRIEVE":
        changed_code = _second_retrieve(components, catalog_path=catalog_path, top_k=top_k)
    elif followup == "RE_PORTION":
        _confirm_portions(components, catalog_path=catalog_path)

    n_agg = _aggregate(components)
    # Graph edges are built with the final codes; verify uses the same binding type.
    nodes, edges = _build_graph(dish_id, components, vcodes=[])
    vcodes = verify(_verify_state(n_agg, edges, components))
    # Rule nodes are appended after verify so they do not affect the evidence check.
    for vc in vcodes:
        rid = f"r_{vc}"
        nodes.append({"id": rid, "type": "rule", "label": vc})
        edges.append({"src": rid, "dst": "dish", "type": "rule_trigger", "attrs": {"code": vc}})

    matched = [comp for comp in components if comp.get("nutri")]
    changed_portion = any(
        abs(float(comp["portion_g"]) - float(comp["hint_g"])) > 1e-6 for comp in matched
    )
    s = "pass"
    abstain_reason = None
    if "UNGROUNDED" in vcodes and not matched:
        s = "abstain"
        abstain_reason = "No FNDDS match / ungrounded components"
    elif changed_code or changed_portion:
        s = "corrected"

    tau: list[dict[str, Any]] = [
        {
            "t": 0,
            "action": "RETRIEVE",
            "obs": {
                "n_components": len(hyps),
                "n_hypotheses_in": len(hyps_in),
                "n_matched": len(matched),
                "dropped_duplicates": len(hyps_in) - len(hyps_deduped),
                "merged_fragments": len(hyps_deduped) - len(hyps),
                "verify_codes": first_vcodes,
            },
            "state_summary": (
                f"rag-once matched {len(matched)}/{len(hyps)} components"
                f" after dedupe {len(hyps_in)}->{len(hyps_deduped)}"
                f" merge {len(hyps_deduped)}->{len(hyps)}"
            ),
        }
    ]
    if followup == "RETRIEVE":
        tau.append(
            {
                "t": len(tau),
                "action": "RETRIEVE",
                "obs": {
                    "verify_codes": first_vcodes,
                    "rewritten": [
                        {
                            "query": comp.get("query"),
                            "rewritten": comp.get("requery"),
                            "code_before": comp.get("previous_code") or comp.get("code"),
                            "code_after": comp.get("code"),
                        }
                        for comp in components
                        if comp.get("requery")
                    ],
                    "changed": changed_code,
                },
                "state_summary": _second_retrieve_summary(components, changed_code),
            }
        )
    elif followup == "RE_PORTION":
        tau.append(
            {
                "t": len(tau),
                "action": "RE_PORTION",
                "obs": {
                    "verify_codes": first_vcodes,
                    "portions": [
                        {
                            "code": comp.get("code"),
                            "hint_g": comp.get("hint_g"),
                            "portion_g": comp.get("portion_g"),
                        }
                        for comp in components
                        if comp.get("code")
                    ],
                },
                "state_summary": (
                    "CODE_CONF_GAP on a non-prepared code; portion grid re-scored and kept"
                ),
            }
        )
    tau.append(
        {
            "t": len(tau),
            "action": "ABSTAIN" if s == "abstain" else "STOP",
            "obs": {"s": s, "verify_codes": vcodes},
            "state_summary": f"emit {s}",
        }
    )

    return {
        "dish_id": dish_id,
        "n": n_agg,
        "G": {"nodes": nodes, "edges": edges},
        "s": s,
        "tau": tau,
        "abstain_reason": abstain_reason,
        "meta": {
            "baseline": "b1_rag_once",
            "verify_codes": vcodes,
            "catalog": "toy_or_env",
            "n_hypotheses_in": len(hyps_in),
            "n_hypotheses": len(hyps),
            "merged_fragments": len(hyps_deduped) - len(hyps),
            "hypotheses": hyps,
            "followup": followup,
            "portion_grid_g": list(DEFAULT_PORTION_GRID_G),
        },
    }


def _dish_id_from_image(image_path: str) -> str:
    parent = Path(image_path).parent.name
    if parent.startswith("dish_"):
        return parent
    return Path(image_path).stem


def predict(image_path: str, **kwargs: Any) -> dict[str, Any]:
    """Image entrypoint. Uses given hypotheses, otherwise Qwen2.5-VL."""
    hypotheses = kwargs.get("hypotheses")
    vlm_meta = None
    if not hypotheses:
        from vdr.perception.qwen25 import ingredient_hypotheses

        hypotheses, vlm_meta = ingredient_hypotheses(image_path)
    dish_id = kwargs.get("dish_id") or _dish_id_from_image(image_path)
    pred = predict_from_hypotheses(
        dish_id=dish_id,
        hypotheses=hypotheses,
        catalog_path=kwargs.get("catalog_path"),
        top_k=int(kwargs.get("top_k", 5)),
        default_portion_g=float(kwargs.get("default_portion_g", 100.0)),
    )
    if vlm_meta is not None:
        pred.setdefault("meta", {})["vlm"] = vlm_meta
    return pred
