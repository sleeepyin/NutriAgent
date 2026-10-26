"""Rule verification prior.

Produces verify_codes for the controller; hard safety intercepts only.
Thresholds here are toy defaults — retune on train hold-out only (W3).
"""

from __future__ import annotations

from typing import Any, Mapping

# Default toy thresholds (NOT tuned on real N5K)
ENERGY_MIN = 0.0
ENERGY_MAX = 2500.0  # per dish rough upper bound for smoke
MACRO_MIN = 0.0
MACRO_MAX = 500.0  # g per dish
ATWATER_REL_TOL = 0.25  # |E - (4P+4C+9F)| / max(E,1) > tol -> ENERGY_ATWATER_MISMATCH
PORTION_MIN_G = 1.0
PORTION_MAX_G = 1500.0
CODE_CONF_GAP = 0.08  # top1 - top2 below this -> CODE_CONF_GAP


def _get_n(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    n = state.get("n") or state.get("nutri_dish")
    if isinstance(n, Mapping):
        return n
    return None


def check_macro_bounds(state: Mapping[str, Any]) -> list[str]:
    codes: list[str] = []
    n = _get_n(state)
    if n is None:
        return codes
    energy = n.get("energy_kcal")
    if energy is not None:
        e = float(energy)
        if e < ENERGY_MIN or e > ENERGY_MAX:
            codes.append("ENERGY_OUTLIER")
    for key, flag in (
        ("protein_g", "PROTEIN_OUTLIER"),
        ("carb_g", "CARB_OUTLIER"),
        ("fat_g", "FAT_OUTLIER"),
    ):
        v = n.get(key)
        if v is None:
            continue
        fv = float(v)
        if fv < MACRO_MIN or fv > MACRO_MAX:
            codes.append(flag)
    return codes


def check_atwater(state: Mapping[str, Any]) -> list[str]:
    """Rough energy ≈ 4P + 4C + 9F consistency check."""
    n = _get_n(state)
    if n is None:
        return []
    try:
        e = n.get("energy_kcal")
        p = n.get("protein_g")
        c = n.get("carb_g")
        f = n.get("fat_g")
        if None in (e, p, c, f):
            return []
        expected = 4.0 * float(p) + 4.0 * float(c) + 9.0 * float(f)
        denom = max(abs(float(e)), 1.0)
        if abs(float(e) - expected) / denom > ATWATER_REL_TOL:
            return ["ENERGY_ATWATER_MISMATCH"]
    except (TypeError, ValueError):
        return ["ENERGY_ATWATER_MISMATCH"]
    return []


def check_evidence(state: Mapping[str, Any]) -> list[str]:
    """Missing evidence path -> UNGROUNDED."""
    # Explicit flag
    if state.get("ungrounded") is True:
        return ["UNGROUNDED"]
    edges = state.get("evidence_edges")
    if edges is None and isinstance(state.get("G"), Mapping):
        edges = state["G"].get("edges")
    if not edges:
        # If we claim a numeric n but have no edges, ungrounded
        n = _get_n(state)
        if n and any(n.get(k) is not None for k in ("energy_kcal", "protein_g", "carb_g", "fat_g")):
            return ["UNGROUNDED"]
        return []
    has_bind = any(
        isinstance(e, Mapping) and e.get("type") == "n<-code×portion" for e in edges
    )
    if not has_bind:
        n = _get_n(state)
        if n and any(n.get(k) is not None for k in ("energy_kcal", "protein_g", "carb_g", "fat_g")):
            return ["UNGROUNDED"]
    return []


def check_portion(state: Mapping[str, Any]) -> list[str]:
    codes: list[str] = []
    portions = state.get("portions") or state.get("portion_g_list")
    if portions is None and state.get("portion_g_i") is not None:
        portions = [state.get("portion_g_i")]
    if not portions:
        return codes
    for p in portions:
        if p is None:
            codes.append("PORTION_MISSING")
            continue
        try:
            g = float(p)
        except (TypeError, ValueError):
            codes.append("PORTION_EXTREME")
            continue
        if g < PORTION_MIN_G or g > PORTION_MAX_G:
            codes.append("PORTION_EXTREME")
    return list(dict.fromkeys(codes))


def check_code_conf_gap(state: Mapping[str, Any]) -> list[str]:
    cands = state.get("cands_i") or state.get("cands")
    if not cands or len(cands) < 2:
        return []
    try:
        s0 = float(cands[0].get("score", 0))
        s1 = float(cands[1].get("score", 0))
    except (TypeError, AttributeError, ValueError):
        return []
    if (s0 - s1) < CODE_CONF_GAP:
        return ["CODE_CONF_GAP"]
    return []


def verify(state: dict[str, Any], **kwargs: Any) -> list[str]:
    """Return triggered rule codes given current IR/state."""
    triggered: list[str] = []
    for fn in (
        check_macro_bounds,
        check_atwater,
        check_evidence,
        check_portion,
        check_code_conf_gap,
    ):
        triggered.extend(fn(state))
    # stable unique order
    return list(dict.fromkeys(triggered))
