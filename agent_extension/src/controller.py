"""Deterministic cost-aware controller. LLM never chooses q or weight."""
from __future__ import annotations

from typing import Any

import numpy as np

from .schemas import ACTIONS, PHYSICAL_SUB, PRIOR_COST, ActionDecision, UncertaintyFactors


class Controller:
    def __init__(
        self,
        gain_models: dict | None = None,
        fail_models: dict | None = None,
        lam: float = 0.15,
        eta: float = 0.25,
        z_lcb: float = 0.5,
        t_max: int = 2,
        mode: str = "myopic_voi",
        single_action: str | None = None,
    ):
        self.gain_models = gain_models or {}
        self.fail_models = fail_models or {}
        self.lam = float(lam)
        self.eta = float(eta)
        self.z_lcb = float(z_lcb)
        self.t_max = int(t_max)
        self.mode = mode
        self.single_action = single_action

    def _predict_gain(self, action: str, x: np.ndarray) -> tuple[float, float]:
        mdl = self.gain_models.get(action)
        if mdl is None:
            return 0.0, 1.0
        mu = float(np.asarray(mdl["predict"](x)).reshape(-1)[0])
        sd = float(mdl.get("sigma", 0.15))
        return mu, sd

    def _predict_fail(self, action: str, x: np.ndarray) -> float:
        mdl = self.fail_models.get(action)
        if mdl is None:
            return 0.05
        p = float(np.asarray(mdl["predict"](x)).reshape(-1)[0])
        return float(np.clip(p, 0.0, 1.0))

    def decide(
        self,
        features: np.ndarray,
        available: dict[str, bool],
        remaining_budget: float,
        step: int = 0,
        u: UncertaintyFactors | None = None,
        ood_score: float = 0.0,
    ) -> ActionDecision:
        if self.mode == "always_stop":
            return ActionDecision("STOP", "FIXED_STOP", 0.0, 0.0, remaining_budget, True)
        if self.mode == "always_abstain" or (ood_score >= 0.85 and remaining_budget <= 0):
            return ActionDecision("ABSTAIN", "OOD_OR_FIXED", 0.0, PRIOR_COST["ABSTAIN"], remaining_budget, True)
        if self.mode.startswith("always_"):
            alias = self.mode.split("_", 1)[1].upper()
            act = {
                "STOP": "STOP",
                "ABSTAIN": "ABSTAIN",
                "DEPTH": "REQUEST_DEPTH",
                "VIEW": "REQUEST_VIEW",
                "REFERENCE": "REQUEST_REFERENCE",
                "SEMANTIC": "ASK_SEMANTIC",
                "REQUEST_DEPTH": "REQUEST_DEPTH",
                "REQUEST_VIEW": "REQUEST_VIEW",
                "REQUEST_REFERENCE": "REQUEST_REFERENCE",
                "ASK_SEMANTIC": "ASK_SEMANTIC",
            }.get(alias)
            if act is None:
                return ActionDecision("STOP", "FIXED_UNKNOWN", 0.0, 0.0, remaining_budget, True)
            if act in ("STOP", "ABSTAIN"):
                return ActionDecision(act, "FIXED_ALWAYS", 0.0, PRIOR_COST.get(act, 0.0), remaining_budget, True)
            if available.get(act) and remaining_budget >= PRIOR_COST.get(act, 0) and step < self.t_max:
                return ActionDecision(act, "FIXED_ALWAYS", 0.0, PRIOR_COST[act], remaining_budget, True)
            return ActionDecision("STOP", "FIXED_UNAVAILABLE", 0.0, 0.0, remaining_budget, True)
        if self.mode == "triage":
            return self._triage(available, remaining_budget, ood_score, u)
        if self.mode == "single_trigger" and self.single_action:
            act = self.single_action
            if available.get(act) and remaining_budget >= PRIOR_COST.get(act, 0) and step < self.t_max:
                if u and self._trigger(act, u, ood_score):
                    return ActionDecision(act, "SINGLE_TRIGGER", 0.0, PRIOR_COST[act], remaining_budget, True)
            return ActionDecision("STOP", "SINGLE_NO_TRIGGER", 0.0, 0.0, remaining_budget, True)

        values: dict[str, float] = {}
        x = np.asarray(features, dtype=float).reshape(1, -1)
        best_a, best_v = "STOP", 0.0
        for a in ACTIONS:
            if a in ("STOP", "ABSTAIN"):
                continue
            if not available.get(a):
                continue
            c = PRIOR_COST[a]
            if c > remaining_budget or step >= self.t_max:
                continue
            g, sd = self._predict_gain(a, x)
            fail = self._predict_fail(a, x)
            val = (g - self.z_lcb * sd) - self.lam * c - self.eta * fail
            values[a] = val
            if val > best_v:
                best_a, best_v = a, val
        if best_v <= 0:
            if ood_score >= 0.7:
                return ActionDecision("ABSTAIN", "VOI_LEQ0_OOD", 0.0, PRIOR_COST["ABSTAIN"], remaining_budget, True, values)
            return ActionDecision("STOP", "VOI_LEQ0", 0.0, 0.0, remaining_budget, True, values)
        return ActionDecision(best_a, "LCBVOI", best_v, PRIOR_COST[best_a], remaining_budget, True, values)

    def _trigger(self, act: str, u: UncertaintyFactors, ood: float) -> bool:
        if act == "REQUEST_REFERENCE":
            return u.u_scale >= 0.45
        if act == "REQUEST_VIEW":
            return u.u_geometry >= 0.45
        if act == "REQUEST_DEPTH":
            return (u.u_scale + u.u_geometry) / 2 >= 0.4
        if act == "ASK_SEMANTIC":
            return u.u_density >= 0.45
        if act == "ABSTAIN":
            return ood >= 0.7
        return False

    def _triage(self, available, remaining_budget, ood, u) -> ActionDecision:
        if ood >= 0.75:
            return ActionDecision("ABSTAIN", "TRIAGE_OOD", 0.0, PRIOR_COST["ABSTAIN"], remaining_budget, True)
        u = u or UncertaintyFactors()
        need = max(u.u_scale, u.u_geometry, u.u_density, u.u_quality)
        if need < 0.4:
            return ActionDecision("STOP", "TRIAGE_OK", 0.0, 0.0, remaining_budget, True)
        # pick the cheapest available acquisition that matches the dominant factor
        order = []
        if u.u_density >= need - 1e-6:
            order.append("ASK_SEMANTIC")
        if u.u_scale >= need - 1e-6:
            order += ["REQUEST_REFERENCE", "REQUEST_DEPTH"]
        if u.u_geometry >= need - 1e-6:
            order += ["REQUEST_VIEW", "REQUEST_DEPTH"]
        order += ["ASK_SEMANTIC", "REQUEST_VIEW", "REQUEST_REFERENCE", "REQUEST_DEPTH"]
        for a in order:
            if available.get(a) and PRIOR_COST[a] <= remaining_budget:
                return ActionDecision(a, "TRIAGE_ACQUIRE", 0.0, PRIOR_COST[a], remaining_budget, True)
        return ActionDecision("ABSTAIN", "TRIAGE_NO_TOOL", 0.0, PRIOR_COST["ABSTAIN"], remaining_budget, True)


def physical_subselect(u: UncertaintyFactors, available: dict[str, bool], device_depth: bool = True) -> str:
    cands = [a for a in PHYSICAL_SUB if available.get(a) and (a != "REQUEST_DEPTH" or device_depth)]
    if not cands:
        return "STOP"
    scores = {
        "REQUEST_REFERENCE": u.u_scale,
        "REQUEST_VIEW": u.u_geometry,
        "REQUEST_DEPTH": 0.6 * u.u_geometry + 0.4 * u.u_scale,
    }
    return max(cands, key=lambda a: scores.get(a, 0.0))
