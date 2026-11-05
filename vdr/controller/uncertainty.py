"""Uncertainty features that replace the weak committed-kcal proxy.

``U_id``, ``U_mass``, ``U_nutrient``, and ``U_evidence`` are the error-source
coordinates in PLAN / Canvas. They are computed from the public IR row, the
previous tool observation, and, when it exists, the RGB Mass Estimator's
Laplace σ (``estimate_mass``). No coefficient is fit on hold-out or on
rgb_test. Train labels and rgb_test labels are not read.

The old ``abstain_score = clip(committed_kcal / 2500)`` stays in
``vdr.eval.abstain_calibration`` as the Ours-old negative control. It is not
an input here.

Timing. σ is a tool output. The step that calls the specialist does not see
that call's σ. The next step does, via ``obs_t`` (live) or via a dish-level
RGB forward supplied by the caller when the historical log has no σ (the
teacher portion tool predates ``estimate_mass``). An open component with no
reading is unmeasured: ``U_mass = 1``. A closed component with no reading is
not assigned a fake scale.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from vdr.controller.features import CAND_SLOTS, PORTION_SCALE_G, _candidates, _finite, _text
from vdr.controller.state_features import ENERGY_CAP_KCAL
from vdr.eval.fixrate import FIXRATE_EPSILON_KCAL
from vdr.tools.mass_estimator import CONF_SCALE_G
from vdr.tools.rule_verify import CODE_CONF_GAP, MACRO_MAX

# Same role as CONF_SCALE_G (100 g): a fixed squash for energy σ, not a search.
ENERGY_SIGMA_SCALE_KCAL = 100.0
# Macro σ uses the same 100-unit squash as mass grams. A few grams of σ then
# stays a small feature instead of saturating. Not fit on hold-out or rgb_test.
MACRO_SIGMA_SCALE_G = 100.0

BINDING_EDGE = "n<-code×portion"
AGGREGATE_EDGE = "n<-aggregate"
NUTRIENT_KEYS = ("energy_kcal", "protein_g", "carb_g", "fat_g")
SIGMA_KEYS = ("mass_g", "energy_kcal", "protein_g", "carb_g", "fat_g")

# Headline U_*, then disagreement / σ, then boolean gates the heuristic reads.
# Gates are state, not uncertainty: masking U_* leaves them in place.
UNCERTAINTY_NAMES: tuple[str, ...] = (
    "U_id",
    "U_mass",
    "U_nutrient",
    "U_evidence",
    "D_energy",
    "D_mass",
    "D_macro",
    "D_models",
    "sigma_mass_norm",
    "sigma_energy_norm",
    "sigma_macro_norm",
    "sigma_present",
    "gate_component_open",
    "gate_code_present",
    "gate_code_top1",
    "gate_code_in_cands",
    "gate_specialist_applied",
    "gate_dish_energy",
    "gate_portion_present",
)
N_HEADLINE = 4
N_DISAGREEMENT = 8  # D_* , sigma_* 
# U and D and σ. Gates start after this prefix.
MASKED_UNCERTAINTY = N_HEADLINE + N_DISAGREEMENT

_OUTLIER_FLAGS = (
    "ENERGY_OUTLIER",
    "PROTEIN_OUTLIER",
    "CARB_OUTLIER",
    "FAT_OUTLIER",
    "ENERGY_ATWATER_MISMATCH",
)


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"uncertainty component is not finite: {value}")
    return float(min(1.0, max(0.0, value)))


def _squash(sigma: float, scale: float) -> float:
    """σ / (σ + scale), in (0, 1). ``scale`` is a fixed positive constant."""
    if scale <= 0.0:
        raise ValueError(f"squash scale must be positive, got {scale}")
    width = max(0.0, float(sigma))
    return _clip01(width / (width + scale))


def _split_obs(
    o_prev: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if o_prev is None:
        return None
    nested = o_prev.get("obs_t")
    if isinstance(nested, Mapping):
        return nested
    return o_prev


def _verify_set(h_t: Mapping[str, Any]) -> set[str]:
    raw = h_t.get("verify_codes")
    if not isinstance(raw, (list, tuple)):
        return set()
    return {item for item in raw if isinstance(item, str) and item}


def _edge_types(h_t: Mapping[str, Any]) -> set[str]:
    raw = h_t.get("evidence_edges")
    if not isinstance(raw, (list, tuple)):
        return set()
    types: set[str] = set()
    for edge in raw:
        if isinstance(edge, Mapping) and isinstance(edge.get("type"), str):
            types.add(edge["type"])
    return types


def _nutri_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for key in ("energy_kcal", "protein_g", "carb_g", "fat_g", "mass_g"):
        number = _finite(value.get(key))
        if number is not None:
            out[key] = number
    return out


def _committed_nutri(h_t: Mapping[str, Any]) -> dict[str, float]:
    """Component nutrition if it carries energy, otherwise the dish total."""
    component = _nutri_map(h_t.get("nutri_i"))
    if "energy_kcal" in component:
        return component
    return _nutri_map(h_t.get("nutri_dish"))


def _tool_failed(o_prev: Mapping[str, Any] | None) -> bool:
    obs = _split_obs(o_prev)
    if obs is None:
        return False
    if obs.get("tool_failed") is True or obs.get("failed") is True:
        return True
    return False


def _ignore_non_rgb(estimate: Mapping[str, Any] | None) -> bool:
    if estimate is None:
        return True
    if estimate.get("depth_used") is True or estimate.get("rgb_only") is False:
        return True
    return False


def _sigma_map(estimate: Mapping[str, Any] | None) -> dict[str, float]:
    if _ignore_non_rgb(estimate):
        return {}
    assert estimate is not None
    raw = estimate.get("sigma")
    out: dict[str, float] = {}
    if isinstance(raw, Mapping):
        for key in SIGMA_KEYS:
            number = _finite(raw.get(key))
            if number is not None and number >= 0.0:
                out[key] = number
    if "mass_g" not in out:
        number = _finite(estimate.get("mass_sigma"))
        if number is not None and number >= 0.0:
            out["mass_g"] = number
    return out


def _model_nutri(estimate: Mapping[str, Any] | None) -> dict[str, float]:
    if _ignore_non_rgb(estimate):
        return {}
    assert estimate is not None
    raw = estimate.get("nutrients_model")
    if not isinstance(raw, Mapping):
        raw = estimate.get("nutrients")
    nutri = _nutri_map(raw)
    mass = _finite(estimate.get("mass_g"))
    if mass is not None and mass >= 0.0:
        nutri["mass_g"] = mass
    return nutri


def mass_estimate_from_obs(o_prev: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Pull an RGB ``estimate_mass`` payload off the previous observation.

    Returns ``None`` when the previous tool did not report a specialist
    reading. A payload that says it used depth is dropped.
    """
    obs = _split_obs(o_prev)
    if obs is None:
        return None
    if obs.get("depth_used") is True or obs.get("rgb_only") is False:
        return None
    sigma = obs.get("sigma") if isinstance(obs.get("sigma"), Mapping) else {}
    nutrients = obs.get("nutrients_model") if isinstance(obs.get("nutrients_model"), Mapping) else None
    mass_sigma = _finite(obs.get("mass_sigma"))
    mass_g = _finite(obs.get("mass_g"))
    has_sigma = any(_finite(sigma.get(key)) is not None for key in SIGMA_KEYS) or mass_sigma is not None
    if not has_sigma and nutrients is None and mass_g is None:
        return None
    payload: dict[str, Any] = {
        "rgb_only": True,
        "depth_used": False,
        "sigma": {str(key): sigma[key] for key in sigma},
    }
    if mass_sigma is not None:
        payload["mass_sigma"] = mass_sigma
        payload["sigma"].setdefault("mass_g", mass_sigma)
    if nutrients is not None:
        payload["nutrients_model"] = dict(nutrients)
    if mass_g is not None:
        payload["mass_g"] = mass_g
    return payload


def _atwater_rel(nutri: Mapping[str, float]) -> float:
    if any(key not in nutri for key in NUTRIENT_KEYS):
        return 0.0
    expected = 4.0 * nutri["protein_g"] + 4.0 * nutri["carb_g"] + 9.0 * nutri["fat_g"]
    energy = nutri["energy_kcal"]
    return _clip01(abs(energy - expected) / max(abs(energy), 1.0))


def _conflict(committed: float | None, model: float | None, sigma: float | None) -> float:
    """How far the specialist and the commitment disagree, in [0, 1].

    Gaps at or below the pre-registered 1 kcal FixRate ε are 0. A reported
    energy σ widens the denominator, so the same gap is a weaker conflict
    when the specialist is already unsure. The σ itself still raises
    ``U_nutrient`` through ``sigma_energy_norm``.
    """
    if committed is None or model is None:
        return 0.0
    gap = abs(float(committed) - float(model))
    if gap <= FIXRATE_EPSILON_KCAL:
        return 0.0
    excess = gap - FIXRATE_EPSILON_KCAL
    scale = ENERGY_CAP_KCAL if sigma is None else max(0.0, float(sigma))
    return _clip01(excess / (excess + scale))


def uncertainty_parts(
    h_t: Mapping[str, Any],
    o_prev: Mapping[str, Any] | None = None,
    mass_estimate: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Named ``U_*``, ``D_*``, σ squashes, and gates. All finite, mostly in [0, 1]."""
    if not isinstance(h_t, Mapping):
        raise TypeError(f"uncertainty_parts expects a mapping, got {type(h_t).__name__}")

    obs_estimate = mass_estimate_from_obs(o_prev)
    # A σ already written by the tool wins over a caller-injected dish forward.
    resolved = obs_estimate if obs_estimate is not None else mass_estimate

    hyp = _text(h_t.get("hyp_i"))
    code = _text(h_t.get("code_i"))
    codes, scores = _candidates(h_t.get("cands_i"))
    portion = _finite(h_t.get("portion_g_i"))
    portion_src = _text(h_t.get("portion_src_i"))
    verify = _verify_set(h_t)
    edges = _edge_types(h_t)
    committed = _committed_nutri(h_t)
    dish = _nutri_map(h_t.get("nutri_dish"))
    component_nutri = _nutri_map(h_t.get("nutri_i"))
    sigma = _sigma_map(resolved)
    model = _model_nutri(resolved)

    component_open = hyp is not None
    code_present = code is not None
    in_cands = code is not None and code in codes
    code_top1 = in_cands and code == codes[0]
    specialist_applied = portion_src == "mass_estimator"
    dish_energy = "energy_kcal" in dish
    n_scores = len(scores)
    top1 = scores[0] if n_scores else 0.0
    gap = (scores[0] - scores[1]) if n_scores >= 2 else 0.0

    if not component_open:
        u_id = 0.0
    else:
        if n_scores == 0:
            top1_u = 1.0
            gap_u = 1.0
        else:
            top1_u = max(0.0, 1.0 - top1)
            # One candidate has no runner-up. rule_verify also stays quiet then.
            if n_scores < 2 or CODE_CONF_GAP <= 0.0:
                gap_u = 0.0
            else:
                gap_u = _clip01((CODE_CONF_GAP - gap) / CODE_CONF_GAP)
        if code is None or not in_cands:
            rank_u = 1.0
        elif code_top1:
            rank_u = 0.0
        else:
            rank_u = codes.index(code) / float(max(CAND_SLOTS - 1, 1))
        conf_u = 1.0 if "CODE_CONF_GAP" in verify else 0.0
        u_id = _clip01(max(top1_u, gap_u, rank_u, conf_u))

    sigma_mass = sigma.get("mass_g")
    sigma_energy = sigma.get("energy_kcal")
    if sigma_mass is None:
        sigma_mass_norm = 0.0
        if component_open:
            u_mass = 1.0
        else:
            u_mass = 0.0
    else:
        sigma_mass_norm = _squash(sigma_mass, CONF_SCALE_G)
        u_mass = sigma_mass_norm
    if "PORTION_EXTREME" in verify:
        u_mass = 1.0
    if component_open and code_present and portion is None:
        u_mass = 1.0

    macro_sigmas = [sigma[key] for key in ("protein_g", "carb_g", "fat_g") if key in sigma]
    if macro_sigmas:
        sigma_macro_norm = sum(_squash(value, MACRO_SIGMA_SCALE_G) for value in macro_sigmas) / len(macro_sigmas)
    else:
        sigma_macro_norm = 0.0
    sigma_energy_norm = 0.0 if sigma_energy is None else _squash(sigma_energy, ENERGY_SIGMA_SCALE_KCAL)
    sigma_present = 1.0 if sigma_mass is not None or sigma_energy is not None or macro_sigmas else 0.0

    atwater = _atwater_rel(committed)
    outlier = 1.0 if any(flag in verify for flag in _OUTLIER_FLAGS) else 0.0
    conflict = _conflict(
        committed.get("energy_kcal"),
        model.get("energy_kcal"),
        sigma_energy,
    )
    u_nutrient = _clip01(max(atwater, outlier, sigma_energy_norm, conflict))

    failed = 1.0 if _tool_failed(o_prev) else 0.0
    ungrounded = 1.0 if "UNGROUNDED" in verify else 0.0
    if component_open:
        broken = (
            0.0 if code_present else 1.0,
            0.0 if portion is not None else 1.0,
            0.0 if "energy_kcal" in component_nutri else 1.0,
            0.0 if BINDING_EDGE in edges else 1.0,
            ungrounded,
            failed,
        )
    else:
        grounded = BINDING_EDGE in edges or AGGREGATE_EDGE in edges
        broken = (
            0.0 if dish_energy else 1.0,
            0.0 if grounded else 1.0,
            ungrounded,
            failed,
        )
    u_evidence = float(sum(broken) / len(broken))

    def _gap(key: str, scale: float) -> float | None:
        if key not in committed or key not in model or scale <= 0.0:
            return None
        return _clip01(abs(committed[key] - model[key]) / scale)

    d_energy = _gap("energy_kcal", ENERGY_CAP_KCAL)
    d_mass = _gap("mass_g", PORTION_SCALE_G)
    macro_gaps = [value for key in ("protein_g", "carb_g", "fat_g") if (value := _gap(key, MACRO_MAX)) is not None]
    d_energy_v = 0.0 if d_energy is None else d_energy
    d_mass_v = 0.0 if d_mass is None else d_mass
    d_macro = 0.0 if not macro_gaps else float(sum(macro_gaps) / len(macro_gaps))
    d_models = max(d_energy_v, d_mass_v, d_macro)

    parts = {
        "U_id": u_id,
        "U_mass": _clip01(u_mass),
        "U_nutrient": u_nutrient,
        "U_evidence": _clip01(u_evidence),
        "D_energy": d_energy_v,
        "D_mass": d_mass_v,
        "D_macro": _clip01(d_macro),
        "D_models": _clip01(d_models),
        "sigma_mass_norm": sigma_mass_norm,
        "sigma_energy_norm": sigma_energy_norm,
        "sigma_macro_norm": _clip01(sigma_macro_norm),
        "sigma_present": sigma_present,
        "gate_component_open": 1.0 if component_open else 0.0,
        "gate_code_present": 1.0 if code_present else 0.0,
        "gate_code_top1": 1.0 if code_top1 else 0.0,
        "gate_code_in_cands": 1.0 if in_cands else 0.0,
        "gate_specialist_applied": 1.0 if specialist_applied else 0.0,
        "gate_dish_energy": 1.0 if dish_energy else 0.0,
        "gate_portion_present": 1.0 if portion is not None else 0.0,
    }
    missing = [name for name in UNCERTAINTY_NAMES if name not in parts]
    if missing:
        raise RuntimeError(f"uncertainty parts missing {missing}")
    if any(not math.isfinite(parts[name]) for name in UNCERTAINTY_NAMES):
        raise RuntimeError("uncertainty parts produced a non-finite value")
    return parts


def run_self_check() -> None:
    """CPU checks. No checkpoint, no cafe labels, no rgb_test."""
    base = {
        "hyp_i": "apple",
        "code_i": None,
        "cands_i": [],
        "portion_g_i": None,
        "portion_src_i": None,
        "verify_codes": [],
        "evidence_edges": [],
        "nutri_i": None,
        "nutri_dish": None,
        "step_i": 0,
    }
    open_unknown = uncertainty_parts(base)
    if open_unknown["U_id"] != 1.0 or open_unknown["U_mass"] != 1.0:
        raise AssertionError((open_unknown["U_id"], open_unknown["U_mass"]))
    if open_unknown["sigma_present"] != 0.0:
        raise AssertionError("sigma appeared without a specialist reading")
    if open_unknown["gate_component_open"] != 1.0 or open_unknown["gate_code_present"] != 0.0:
        raise AssertionError("gates")

    confident = dict(base)
    confident["code_i"] = "111"
    confident["cands_i"] = [
        {"code": "111", "score": 0.95, "name": "Apple"},
        {"code": "222", "score": 0.40, "name": "Juice"},
    ]
    confident["portion_g_i"] = 100.0
    confident["portion_src_i"] = "grid_search"
    low = uncertainty_parts(confident)
    if not low["U_id"] < 0.5:
        raise AssertionError(low["U_id"])
    if low["U_mass"] != 1.0:
        raise AssertionError("unmeasured mass should stay 1 before estimate_mass")

    small = uncertainty_parts(confident, mass_estimate={"sigma": {"mass_g": 12.0}, "rgb_only": True, "depth_used": False})
    large = uncertainty_parts(confident, mass_estimate={"sigma": {"mass_g": 300.0}, "rgb_only": True, "depth_used": False})
    if not small["U_mass"] < large["U_mass"] <= 1.0:
        raise AssertionError((small["U_mass"], large["U_mass"]))
    if abs(small["U_mass"] - (12.0 / 112.0)) > 1e-9:
        raise AssertionError(small["U_mass"])
    if small["sigma_present"] != 1.0:
        raise AssertionError("sigma_present")

    depth = uncertainty_parts(
        confident,
        mass_estimate={"sigma": {"mass_g": 12.0}, "rgb_only": False, "depth_used": True},
    )
    if depth["sigma_present"] != 0.0 or depth["U_mass"] != 1.0:
        raise AssertionError("depth payload was accepted")

    obs = {
        "obs_t": {
            "mass_sigma": 12.0,
            "sigma": {"mass_g": 12.0, "energy_kcal": 40.0},
            "nutrients_model": {"energy_kcal": 180.0, "protein_g": 1.0, "carb_g": 20.0, "fat_g": 1.0},
            "mass_g": 210.0,
            "rgb_only": True,
            "depth_used": False,
            "portion_src": "mass_estimator",
        }
    }
    from_obs = uncertainty_parts(confident, obs, mass_estimate={"sigma": {"mass_g": 300.0}, "rgb_only": True})
    if abs(from_obs["U_mass"] - (12.0 / 112.0)) > 1e-9:
        raise AssertionError("obs σ should win over the injected forward")
    if from_obs["sigma_energy_norm"] <= 0.0:
        raise AssertionError("energy σ did not enter U_nutrient's scale")

    conflict_h = dict(confident)
    conflict_h["nutri_i"] = {"energy_kcal": 100.0, "protein_g": 1.0, "carb_g": 20.0, "fat_g": 1.0, "mass_g": 100.0}
    conflict_h["evidence_edges"] = [{"type": BINDING_EDGE, "src": "n", "dst": "c"}]
    near = uncertainty_parts(
        conflict_h,
        mass_estimate={
            "sigma": {"mass_g": 20.0, "energy_kcal": 30.0},
            "nutrients": {"energy_kcal": 101.0, "protein_g": 1.0, "carb_g": 20.0, "fat_g": 1.0},
            "mass_g": 100.0,
            "rgb_only": True,
            "depth_used": False,
        },
    )
    far = uncertainty_parts(
        conflict_h,
        mass_estimate={
            "sigma": {"mass_g": 20.0, "energy_kcal": 30.0},
            "nutrients": {"energy_kcal": 400.0, "protein_g": 1.0, "carb_g": 20.0, "fat_g": 1.0},
            "mass_g": 100.0,
            "rgb_only": True,
            "depth_used": False,
        },
    )
    if not far["D_energy"] > near["D_energy"]:
        raise AssertionError((near["D_energy"], far["D_energy"]))
    if not far["U_nutrient"] > near["U_nutrient"]:
        raise AssertionError((near["U_nutrient"], far["U_nutrient"]))

    labeled = dict(confident)
    labeled["train_gt_energy_kcal"] = 9999.0
    labeled["test_gt"] = 1.0
    if uncertainty_parts(labeled) != uncertainty_parts(confident):
        raise AssertionError("a label key changed U_*")

    closed = {
        "hyp_i": None,
        "code_i": None,
        "cands_i": [],
        "portion_g_i": None,
        "verify_codes": [],
        "evidence_edges": [{"type": AGGREGATE_EDGE, "src": "dish", "dst": "n"}],
        "nutri_dish": {"energy_kcal": 180.0, "protein_g": 2.0, "carb_g": 20.0, "fat_g": 5.0, "mass_g": 200.0},
        "nutri_i": None,
        "step_i": 3,
    }
    done = uncertainty_parts(closed)
    if done["U_id"] != 0.0 or done["gate_component_open"] != 0.0:
        raise AssertionError("closed component still has identity uncertainty")
    if done["U_evidence"] != 0.0:
        raise AssertionError(done["U_evidence"])
    if done["gate_dish_energy"] != 1.0:
        raise AssertionError("dish energy gate")
