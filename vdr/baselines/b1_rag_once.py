"""B1 RAG-once baseline — DietAI24-style single retrieve + aggregate.

No VLM required when hypotheses are provided via `hypotheses`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from vdr.tools.fndds_retrieve import retrieve
from vdr.tools.nutrition_lookup import aggregate_dish, lookup_and_scale
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


def predict_from_hypotheses(
    dish_id: str,
    hypotheses: Sequence[dict[str, Any]] | Sequence[str],
    catalog_path: str | Path | None = None,
    top_k: int = 5,
    default_portion_g: float = 100.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """One-shot retrieve+aggregate from given component hypotheses (no VLM)."""
    hyps = _as_hyp_list(hypotheses)
    nodes: list[dict[str, Any]] = [{"id": "dish", "type": "dish", "label": dish_id}]
    edges: list[dict[str, Any]] = []
    comp_nutris: list[dict[str, float]] = []
    portions_used: list[float] = []
    all_cands: list[dict[str, Any]] = []

    for i, hyp in enumerate(hyps):
        name = hyp.get("name") or f"comp_{i}"
        portion = hyp.get("portion_g")
        if portion is None:
            portion = default_portion_g
        portion = float(portion)
        portions_used.append(portion)

        hid = f"h{i}"
        cid = f"c{i}"
        pid = f"p{i}"
        nid = f"n{i}"
        nodes.append({"id": hid, "type": "hyp", "label": name})
        nodes.append({"id": f"q{i}", "type": "query", "label": name})

        rr = retrieve(name, top_k=top_k, catalog_path=catalog_path)
        all_cands.extend(rr.cands)
        if not rr.selected_code:
            # no match — leave ungrounded for this component
            nodes.append({"id": cid, "type": "food_code", "label": None, "attrs": {"code": None}})
            continue

        code = rr.selected_code
        nutri = lookup_and_scale(code, portion, catalog_path=catalog_path)
        nutri_with_mass = dict(nutri)
        nutri_with_mass["mass_g"] = portion
        comp_nutris.append(nutri_with_mass)

        nodes.append(
            {
                "id": cid,
                "type": "food_code",
                "label": rr.cands[0].get("name"),
                "attrs": {"code": code, "cands": rr.cands},
            }
        )
        nodes.append(
            {"id": pid, "type": "portion_g", "label": f"{portion}g", "attrs": {"portion_g": portion}}
        )
        nodes.append(
            {"id": nid, "type": "nutrition", "label": f"{name} macros", "attrs": nutri_with_mass}
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

    if comp_nutris:
        n_agg = aggregate_dish(comp_nutris)
    else:
        n_agg = {
            "energy_kcal": None,
            "protein_g": None,
            "carb_g": None,
            "fat_g": None,
            "mass_g": None,
        }

    # Ensure required keys exist
    for k in ("energy_kcal", "protein_g", "carb_g", "fat_g", "mass_g"):
        n_agg.setdefault(k, None)

    G = {"nodes": nodes, "edges": edges}
    state = {
        "n": n_agg,
        "G": G,
        "evidence_edges": edges,
        "portions": portions_used,
        "cands": all_cands[:2] if all_cands else [],
    }
    vcodes = verify(state)
    for vc in vcodes:
        rid = f"r_{vc}"
        nodes.append({"id": rid, "type": "rule", "label": vc})
        edges.append({"src": rid, "dst": "dish", "type": "rule_trigger", "attrs": {"code": vc}})

    s = "pass"
    abstain_reason = None
    if "UNGROUNDED" in vcodes and not comp_nutris:
        s = "abstain"
        abstain_reason = "No FNDDS match / ungrounded components"
    elif vcodes:
        # still emit numbers but mark corrected only if we had matches; else pass with flags
        s = "pass"

    tau = [
        {
            "t": 0,
            "action": "RETRIEVE",
            "obs": {
                "n_components": len(hyps),
                "n_matched": len(comp_nutris),
                "verify_codes": vcodes,
            },
            "state_summary": f"rag-once matched {len(comp_nutris)}/{len(hyps)} components",
        },
        {
            "t": 1,
            "action": "ABSTAIN" if s == "abstain" else "STOP",
            "obs": {"s": s, "verify_codes": vcodes},
            "state_summary": f"emit {s}",
        },
    ]

    return {
        "dish_id": dish_id,
        "n": n_agg,
        "G": G,
        "s": s,
        "tau": tau,
        "abstain_reason": abstain_reason,
        "meta": {
            "baseline": "b1_rag_once",
            "verify_codes": vcodes,
            "catalog": "toy_or_env",
        },
    }


def predict(image_path: str, **kwargs: Any) -> dict[str, Any]:
    """Image entrypoint: requires hypotheses= until VLM is wired."""
    hypotheses = kwargs.get("hypotheses")
    if not hypotheses:
        raise NotImplementedError(
            "b1_rag_once.predict: VLM perception not wired. "
            "Pass hypotheses=[...] for no-VLM smoke, or wait for 4090D VLM setup."
        )
    dish_id = kwargs.get("dish_id") or Path(image_path).stem
    return predict_from_hypotheses(
        dish_id=dish_id,
        hypotheses=hypotheses,
        catalog_path=kwargs.get("catalog_path"),
        top_k=int(kwargs.get("top_k", 5)),
        default_portion_g=float(kwargs.get("default_portion_g", 100.0)),
    )
