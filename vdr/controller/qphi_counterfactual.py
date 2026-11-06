"""One-step counterfactual value of the frozen coverage-constrained controller.

The fit / hold-out weak-step files are audited by ``dish_id``. A random split
of decision steps would put two steps of the same dish on opposite sides of
the cut. This module only reads those files and the frozen ``Q`` checkpoint.
It does not call ``QPhi.fit``, does not edit ``τ``, and does not write the
weak-step jsonl or the #46 / #48 / #51 / #53 result directories.

Branch execution uses the live tool semantics, not the weak-step teacher's
privileged ingredient grams or kcal/g:

- ``RETRIEVE`` is the lexical FNDDS query plus the blind 100 g portion grid
- ``CORRECT_CODE`` swaps to another code already in the retrieve window
- ``RE_PORTION`` allocates the frozen RGB mass mean. Missing component hints
  use the tool's weight-1 equal split. The gram grid is not a substitute.
- ``STOP`` and ``ABSTAIN`` do not edit ``n``
- ``RE_DECOMPOSE`` is the live no-op (no second perception call)

``STOP`` / ``ABSTAIN`` are removed from ``A_legal`` while the open component
has no ``portion_src=mass_estimator`` write. That is the coverage constraint.
The cafe label is used only after the branch, to score energy and mass error.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vdr.controller.coverage import apply_coverage_guard, component_needs_mass_write
from vdr.controller.nutrition_loss import energy_abs_error, nutrition_loss
from vdr.controller.policy_bc import BC_ACTIONS
from vdr.controller.q_calibration import calibrated_decision
from vdr.controller.q_phi import QPhi
from vdr.controller.q_state import Z_FEATURE_NAMES, build_z
from vdr.controller.utility import ACTION_COST, TIE_ORDER, argmax_value_over_cost, heuristic_utility, legal_actions
from vdr.eval.fixrate import FIXRATE_EPSILON_KCAL
from vdr.ir.io import read_ir_jsonl
from vdr.tools.mass_estimator import assemble_estimate
from vdr.weak.teacher import (
    NUTRI_KEYS,
    TOP_K,
    _public_cands,
    _scaled,
    _sum_nutri,
    candidate_rewrite,
    pool_of_dish,
)
from vdr.controller.on_policy import _alternate_code

# Numerical tie for "same absolute error". Not a threshold searched on data.
ENERGY_TIE_KCAL = 1e-4
MASS_TIE_G = 1e-4
# Same 1 kcal already used by FixRate. Not refit here. Marks states where the
# legal actions actually disagree, as a pre-specified slice beside the full set.
DECISIVE_ENERGY_KCAL = float(FIXRATE_EPSILON_KCAL)
Q_WEIGHTS = Path("results/qphi/q_phi.npz")
TAU_PATH = Path("results/qphi/calibration.json")
FIT_JSONL = Path("results/weak_steps/rgb_train_fit_all.jsonl")
HOLDOUT_JSONL = Path("results/weak_steps/rgb_train_holdout_all.jsonl")
DEVELOPMENT_JSONL = Path("results/coverage_early_stop/paired_scores.jsonl")
SIGMA_CACHE = Path("results/qphi/mass_sigma_rgb_train.json")

_GATE_OPEN = Z_FEATURE_NAMES.index("gate_component_open")
_GATE_CODE = Z_FEATURE_NAMES.index("gate_code_present")
_GATE_APPLIED = Z_FEATURE_NAMES.index("gate_specialist_applied")

PROTOCOL_ID = "qphi_counterfactual_onestep_holdout_v1"
PROTOCOL = (
    "One-step branches from frozen rgb_train hold-out weak-step states. "
    "Those dishes did not drive the #51 food-code swap or the #53 coverage "
    "guard (both were read off the rgb_test development 505). They were also "
    "not the fit rows used to train Q or to set τ. "
    "A_legal drops STOP and ABSTAIN while a component is open and has no "
    "mass-estimator write. Each legal action is executed once from that same "
    "state. The oracle minimizes post-action absolute energy error. Mass error "
    "is logged and is not the selection objective. Q is the frozen checkpoint "
    "plus the existing terminal-margin fence and coverage guard. τ is not refit. "
    "No reinforcement learning. rgb_test labels are not read."
)


class CounterfactualError(ValueError):
    """The counterfactual sample or a branch is not legal."""


def coverage_legal(row_z: np.ndarray) -> dict[str, bool]:
    """Tool preconditions, then the coverage constraint.

    ``legal_actions`` still marks ``STOP`` and ``ABSTAIN`` as available.
    While the open component has no mass write they are not in ``A_legal``.
    """
    vector = np.asarray(row_z, dtype=np.float64)
    legal = dict(legal_actions(vector))
    if component_needs_mass_write(vector):
        legal["STOP"] = False
        legal["ABSTAIN"] = False
    if not any(legal.values()):
        raise CounterfactualError("A_legal is empty")
    return legal


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_pool(path: Path) -> tuple[dict[str, list[int]], int]:
    dishes: dict[str, list[int]] = {}
    n_rows = 0
    for record in read_ir_jsonl(path):
        n_rows += 1
        dish_id = record.get("dish_id")
        step = record.get("step_i")
        if not isinstance(dish_id, str) or not dish_id:
            raise CounterfactualError(f"{path.name}: row without dish_id")
        if isinstance(step, bool) or not isinstance(step, int):
            raise CounterfactualError(f"{path.name}: {dish_id} has step_i {step!r}")
        dishes.setdefault(dish_id, []).append(step)
    return dishes, n_rows


def _step_integrity(steps: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(steps)
    unique = sorted(set(ordered))
    contiguous = unique == list(range(unique[0], unique[-1] + 1)) if unique else False
    return {
        "n_steps": len(ordered),
        "n_unique_steps": len(unique),
        "duplicate_steps": len(ordered) - len(unique),
        "starts_at_zero": bool(unique) and unique[0] == 0,
        "contiguous": contiguous and bool(unique) and unique[0] == 0,
    }


def audit_weak_step_split(fit_path: Path | str = FIT_JSONL, holdout_path: Path | str = HOLDOUT_JSONL) -> dict[str, Any]:
    """Whether fit vs hold-out is a dish cut or a decision-step cut.

    Leakage, for this gate, means one ``dish_id`` has decision steps in both
    files. The hash rule is ``sha256(dish_id)[:8] % 10 == 0`` → hold-out.
    A disagreement between that hash and the file is reported separately and
    is not repaired by retraining.
    """
    fit_file = Path(fit_path)
    hold_file = Path(holdout_path)
    fit, n_fit = _scan_pool(fit_file)
    hold, n_hold = _scan_pool(hold_file)
    both = sorted(set(fit) & set(hold))
    misfiled: list[dict[str, str]] = []
    fit_multi = 0
    hold_multi = 0
    fit_bad_steps = 0
    hold_bad_steps = 0
    for dish_id, steps in fit.items():
        slot = pool_of_dish(dish_id)
        if slot != "fit":
            misfiled.append({"dish_id": dish_id, "file": "fit", "hash_pool": slot})
        if len(steps) > 1:
            fit_multi += 1
        if not _step_integrity(steps)["contiguous"]:
            fit_bad_steps += 1
    for dish_id, steps in hold.items():
        slot = pool_of_dish(dish_id)
        if slot != "holdout":
            misfiled.append({"dish_id": dish_id, "file": "holdout", "hash_pool": slot})
        if len(steps) > 1:
            hold_multi += 1
        if not _step_integrity(steps)["contiguous"]:
            hold_bad_steps += 1
    leaked_steps = 0
    for dish_id in both:
        leaked_steps += len(fit[dish_id]) + len(hold[dish_id])
    dish_leakage = bool(both)
    conclusion = "no_dish_leakage" if not dish_leakage and not misfiled else (
        "dish_leakage" if dish_leakage else "pool_hash_disagreement"
    )
    return {
        "question": (
            "Are weak-step fit and hold-out grouped by dish_id, or do decision "
            "steps of the same dish fall on both sides?"
        ),
        "split_rule": "int(sha256(dish_id).hexdigest()[:8], 16) % 10 == 0 → holdout, else fit",
        "split_function": "vdr.weak.teacher.pool_of_dish",
        "unit": "dish_id",
        "fit_path": str(fit_file),
        "holdout_path": str(hold_file),
        "fit_rows": n_fit,
        "holdout_rows": n_hold,
        "fit_dishes": len(fit),
        "holdout_dishes": len(hold),
        "fit_dishes_with_multiple_steps": fit_multi,
        "holdout_dishes_with_multiple_steps": hold_multi,
        "fit_dishes_with_noncontiguous_steps": fit_bad_steps,
        "holdout_dishes_with_noncontiguous_steps": hold_bad_steps,
        "dishes_in_both_splits": len(both),
        "rows_belonging_to_dishes_in_both_splits": leaked_steps,
        "example_leaked_dish_ids": both[:8],
        "misfiled_dishes": len(misfiled),
        "misfiled_examples": misfiled[:8],
        "leakage_exists": dish_leakage,
        "conclusion": conclusion,
        "retrained": False,
        "note": (
            "Q training fits every supervised row of the fit file and scores "
            "the hold-out file. There is no random decision-step split in "
            "vdr.controller.q_data or scripts/qphi_train.py. "
            "The mass-estimator probe-val residue (sha256 % 10 == 1) is a "
            "second dish-grouped slice inside fit. It is not this cut. "
            "This audit does not refit Q, Mass, or τ."
        ),
    }


def load_frozen_tau(path: Path | str = TAU_PATH) -> float:
    """Read the fit-set fence. Do not recompute it."""
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tau = float(payload["tau"])
    if not math.isfinite(tau) or tau < 0.0:
        raise CounterfactualError(f"frozen τ is not usable: {tau}")
    if payload.get("holdout_used_for_selection") is not False:
        raise CounterfactualError("calibration file says hold-out was used to choose τ")
    return tau


def load_frozen_q(path: Path | str = Q_WEIGHTS) -> QPhi:
    """Load checkpoint weights. This does not call ``fit``."""
    model = QPhi.from_npz(path)
    if model.train_mse_ is not None:
        raise CounterfactualError("loading a checkpoint should not look like a new fit")
    return model


def choose_q(model: QPhi, row_z: np.ndarray, tau: float) -> tuple[str, str, dict[str, float]]:
    """Frozen ``Q/Cost``, then the existing fence, then the coverage guard."""
    vector = np.asarray(row_z, dtype=np.float64)
    predicted = model.predict_q(vector.reshape(1, -1))[0]
    values = {action: float(predicted[index]) for index, action in enumerate(BC_ACTIONS)}
    action, source = calibrated_decision(values, vector, tau)
    action, source = apply_coverage_guard(action, source, vector)
    return action, source, values


def choose_rule(row_z: np.ndarray, legal: Mapping[str, bool]) -> str:
    """Rule utility restricted to ``A_legal``. ``STOP`` cannot win on a tie while it is illegal."""
    return argmax_value_over_cost(heuristic_utility(np.asarray(row_z, dtype=np.float64)), legal)


def choose_fixed(row_z: np.ndarray, legal: Mapping[str, bool]) -> str:
    """Retrieve, then portion, then stop. Not the uncertainty rule and not ``Q``.

    The order is the stub pipeline. An action that is illegal or over a
    caller's cost mask is skipped. ``CORRECT_CODE`` is not in the pipeline
    until every earlier preference is unavailable.
    """
    vector = np.asarray(row_z, dtype=np.float64)
    open_ = float(vector[_GATE_OPEN]) >= 0.5
    code = float(vector[_GATE_CODE]) >= 0.5
    applied = float(vector[_GATE_APPLIED]) >= 0.5
    preferences: list[str] = []
    if open_ and not code:
        preferences.append("RETRIEVE")
    if open_ and code and not applied:
        preferences.append("RE_PORTION")
    if (open_ and applied) or not open_:
        preferences.append("STOP")
    if not open_:
        preferences.append("ABSTAIN")
    preferences.extend(("RETRIEVE", "RE_PORTION", "CORRECT_CODE", "STOP", "ABSTAIN", "RE_DECOMPOSE"))
    seen: set[str] = set()
    for action in preferences:
        if action in seen:
            continue
        seen.add(action)
        if legal.get(action):
            return action
    raise CounterfactualError("fixed pipeline found no legal action")


def _abs_field(nutri: Mapping[str, Any] | None, label: Mapping[str, float], key: str) -> float | None:
    if key == "energy_kcal":
        return energy_abs_error(nutri, label)
    truth = label.get(key)
    if truth is None:
        raise CounterfactualError(f"label is missing {key}")
    if nutri is None or nutri.get(key) is None:
        return None
    try:
        hat = float(nutri[key])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(hat):
        return None
    return abs(hat - float(truth))


def _residual(dish: Mapping[str, Any] | None, component: Mapping[str, Any] | None) -> dict[str, float | None] | None:
    if dish is None:
        return None
    acc: dict[str, float | None] = {}
    for key in NUTRI_KEYS:
        total = dish.get(key)
        part = None if component is None else component.get(key)
        if total is None and part is None:
            acc[key] = None
        else:
            acc[key] = float(total or 0.0) - float(part or 0.0)
    return acc


def _replace_component(
    dish: Mapping[str, Any] | None,
    component: Mapping[str, Any] | None,
    new_component: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    base = _residual(dish, component)
    parts: list[Mapping[str, Any] | None] = []
    if base is not None:
        parts.append(base)
    if new_component is not None:
        parts.append(new_component)
    if not parts:
        return None
    return _sum_nutri(parts)


@dataclass
class BranchOutcome:
    """Nutrition after one action, scored only by the caller."""

    action: str
    nutri_dish: dict[str, Any] | None
    tool_failed: bool
    changed: bool
    detail: str


@dataclass
class BranchExecutor:
    """Live tools, with caches. Labels are not arguments."""

    retrieve_fn: Callable[..., Any]
    lookup_fn: Callable[..., Any]
    portion_fn: Callable[..., Any]
    mass_by_dish: Mapping[str, Mapping[str, Any]]
    catalog_path: str | None = None
    retrieve_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    portion_cache: dict[str, float | None] = field(default_factory=dict)
    mass_hits: int = 0
    mass_misses: int = 0

    def _retrieve_cands(self, query: str) -> list[dict[str, Any]]:
        cached = self.retrieve_cache.get(query)
        if cached is not None:
            return cached
        found = self.retrieve_fn(query, top_k=TOP_K, catalog_path=self.catalog_path)
        cands = _public_cands(getattr(found, "cands", []) or [])[:TOP_K]
        self.retrieve_cache[query] = cands
        return cands

    def _blind_portion(self, code: str) -> float | None:
        if code in self.portion_cache:
            return self.portion_cache[code]
        try:
            result = self.portion_fn(code, hint_g=100.0, catalog_path=self.catalog_path)
        except KeyError:
            self.portion_cache[code] = None
            return None
        portion = getattr(result, "portion_g", None)
        if portion is None and isinstance(result, Mapping):
            portion = result.get("portion_g")
        value = None if portion is None else float(portion)
        self.portion_cache[code] = value
        return value

    def _scaled(self, code: str, portion: float) -> dict[str, Any] | None:
        return _scaled(code, portion, self.lookup_fn, self.catalog_path)

    def run(
        self,
        state: Mapping[str, Any],
        action: str,
        *,
        n_components: int,
    ) -> BranchOutcome:
        dish = state.get("nutri_dish")
        if isinstance(dish, Mapping):
            dish = dict(dish)
        else:
            dish = None
        unchanged = BranchOutcome(action, dish, tool_failed=False, changed=False, detail="no_edit")
        if action in {"STOP", "ABSTAIN"}:
            return unchanged
        if action == "RE_DECOMPOSE":
            return BranchOutcome(action, dish, tool_failed=True, changed=False, detail="re_decompose_not_a_second_vlm_call")
        if action == "RETRIEVE":
            return self._retrieve(state, dish)
        if action == "CORRECT_CODE":
            return self._correct(state, dish)
        if action == "RE_PORTION":
            return self._reportion(state, dish, n_components)
        raise CounterfactualError(f"unhandled action {action}")

    def _retrieve(self, state: Mapping[str, Any], dish: dict[str, Any] | None) -> BranchOutcome:
        hyp = state.get("hyp_i")
        if not isinstance(hyp, str) or not hyp.strip():
            return BranchOutcome("RETRIEVE", dish, True, False, "no_hypothesis")
        query = hyp.strip()
        if state.get("code_i"):
            rewrite = candidate_rewrite(query)
            if rewrite:
                query = rewrite
        cands = self._retrieve_cands(query)
        if not cands:
            return BranchOutcome("RETRIEVE", dish, True, False, "no_candidate")
        code = str(cands[0]["code"])
        portion = self._blind_portion(code)
        if portion is None:
            return BranchOutcome("RETRIEVE", dish, True, False, "portion_grid_miss")
        nutri = self._scaled(code, portion)
        if nutri is None:
            return BranchOutcome("RETRIEVE", dish, True, False, "lookup_miss")
        new_dish = _replace_component(dish, state.get("nutri_i"), nutri)
        return BranchOutcome("RETRIEVE", new_dish, False, True, "top_code_grid_portion")

    def _correct(self, state: Mapping[str, Any], dish: dict[str, Any] | None) -> BranchOutcome:
        current = state.get("code_i")
        cands = state.get("cands_i") or []
        if not isinstance(cands, Sequence):
            cands = []
        target = _alternate_code(cands, None if current is None else str(current))
        if not target or not state.get("hyp_i"):
            return BranchOutcome("CORRECT_CODE", dish, True, False, "no_alternate")
        portion = state.get("portion_g_i")
        try:
            portion_f = None if portion is None else float(portion)
        except (TypeError, ValueError):
            portion_f = None
        if portion_f is None:
            portion_f = self._blind_portion(target)
        if portion_f is None:
            return BranchOutcome("CORRECT_CODE", dish, True, False, "no_portion")
        nutri = self._scaled(target, portion_f)
        if nutri is None:
            return BranchOutcome("CORRECT_CODE", dish, True, False, "lookup_miss")
        new_dish = _replace_component(dish, state.get("nutri_i"), nutri)
        return BranchOutcome("CORRECT_CODE", new_dish, False, True, "alternate_code_same_grams")

    def _reportion(
        self,
        state: Mapping[str, Any],
        dish: dict[str, Any] | None,
        n_components: int,
    ) -> BranchOutcome:
        code = state.get("code_i")
        component_i = state.get("component_i")
        if not code or isinstance(component_i, bool) or not isinstance(component_i, int):
            return BranchOutcome("RE_PORTION", dish, True, False, "no_code")
        record = self.mass_by_dish.get(str(state.get("dish_id")))
        if not isinstance(record, Mapping) or "mean" not in record or "sigma" not in record:
            self.mass_misses += 1
            return BranchOutcome("RE_PORTION", dish, True, False, "no_frozen_mass_forward")
        self.mass_hits += 1
        if n_components < 1 or component_i < 0 or component_i >= n_components:
            return BranchOutcome("RE_PORTION", dish, True, False, "component_index")
        # Hints stay missing. Weight 1 is the frozen splitter. Ingredient grams
        # would leak the cafe label into the portion.
        estimate = assemble_estimate(
            record["mean"],
            record["sigma"],
            backbone="vit_b_16",
            candidates=state.get("cands_i"),
            component_index=component_i,
            component_hints_g=[None] * n_components,
            catalog_path=self.catalog_path,
            lookup_fn=self.lookup_fn,
        )
        if estimate.get("portion_src") != "mass_estimator" or estimate.get("rgb_only") is not True:
            return BranchOutcome("RE_PORTION", dish, True, False, "mass_contract")
        portion = estimate.get("portion_g")
        if portion is None:
            return BranchOutcome("RE_PORTION", dish, True, False, "mass_estimator_no_portion")
        nutri = self._scaled(str(code), float(portion))
        if nutri is None:
            return BranchOutcome("RE_PORTION", dish, True, False, "lookup_miss")
        new_dish = _replace_component(dish, state.get("nutri_i"), nutri)
        return BranchOutcome("RE_PORTION", new_dish, False, True, "estimate_mass_equal_split")


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _best_action(
    outcomes: Mapping[str, Mapping[str, Any]],
    *,
    field_name: str,
) -> tuple[str, list[str]]:
    """Minimize post absolute error. Missing error is worse than any finite error.

    Ties keep the cheaper action, then ``TIE_ORDER``.
    """
    finite: list[tuple[float, float, str]] = []
    missing: list[str] = []
    for action in BC_ACTIONS:
        if action not in outcomes:
            continue
        error = _finite(outcomes[action].get(field_name))
        if error is None:
            missing.append(action)
        else:
            finite.append((error, ACTION_COST[action], action))
    if finite:
        best_error = min(item[0] for item in finite)
        tied = [item for item in finite if abs(item[0] - best_error) <= (ENERGY_TIE_KCAL if field_name.startswith("post_energy") else MASS_TIE_G)]
        tied.sort(key=lambda item: (item[1], TIE_ORDER.index(item[2])))
        optimal = [item[2] for item in tied]
        return optimal[0], optimal
    if not missing:
        raise CounterfactualError("oracle had no actions")
    missing.sort(key=lambda action: (ACTION_COST[action], TIE_ORDER.index(action)))
    return missing[0], list(missing)


def score_branch(
    outcome: BranchOutcome,
    *,
    label: Mapping[str, float],
    pre_dish: Mapping[str, Any] | None,
    q_value: float,
) -> dict[str, Any]:
    post = outcome.nutri_dish
    pre_energy = _abs_field(pre_dish, label, "energy_kcal")
    post_energy = _abs_field(post, label, "energy_kcal")
    pre_mass = _abs_field(pre_dish, label, "mass_g")
    post_mass = _abs_field(post, label, "mass_g")
    pre_loss = nutrition_loss(pre_dish, label)
    post_loss = nutrition_loss(post, label)
    energy_delta = None if pre_energy is None or post_energy is None else pre_energy - post_energy
    mass_delta = None if pre_mass is None or post_mass is None else pre_mass - post_mass
    delta_l = None if pre_loss is None or post_loss is None else pre_loss - post_loss
    q_finite = float(q_value) if math.isfinite(float(q_value)) else None
    return {
        "post_energy_ae": post_energy,
        "post_mass_ae": post_mass,
        "energy_delta": energy_delta,
        "mass_delta": mass_delta,
        "delta_l": delta_l,
        "q": q_finite,
        "cost": ACTION_COST[outcome.action],
        "tool_failed": outcome.tool_failed,
        "changed": outcome.changed,
        "detail": outcome.detail,
    }


def build_decision_record(
    *,
    dish_id: str,
    step_i: int,
    component_i: int | None,
    n_components: int,
    legal: Mapping[str, bool],
    pre_energy: float | None,
    pre_mass: float | None,
    pre_loss: float | None,
    branches: Mapping[str, Mapping[str, Any]],
    q_action: str,
    q_source: str,
    rule_action: str,
    fixed_action: str,
) -> dict[str, Any]:
    oracle, optimal = _best_action(branches, field_name="post_energy_ae")
    mass_oracle, _mass_optimal = _best_action(branches, field_name="post_mass_ae")
    finite_energy = [
        float(row["post_energy_ae"])
        for row in branches.values()
        if _finite(row.get("post_energy_ae")) is not None
    ]
    if len(finite_energy) >= 2:
        energy_span = max(finite_energy) - min(finite_energy)
    elif finite_energy and any(_finite(row.get("post_energy_ae")) is None for row in branches.values()):
        energy_span = float("inf")
    else:
        energy_span = 0.0
    return {
        "dish_id": dish_id,
        "step_i": step_i,
        "component_i": component_i,
        "n_components": n_components,
        "n_legal": int(sum(1 for flag in legal.values() if flag)),
        "legal": [action for action in BC_ACTIONS if legal[action]],
        "pre_energy_ae": pre_energy,
        "pre_mass_ae": pre_mass,
        "pre_loss": pre_loss,
        "energy_span": None if math.isinf(energy_span) else energy_span,
        "decisive_energy": bool(energy_span > DECISIVE_ENERGY_KCAL),
        "oracle_action": oracle,
        "oracle_actions": optimal,
        "mass_oracle_action": mass_oracle,
        "q_action": q_action,
        "q_source": q_source,
        "q_legal": bool(legal.get(q_action, False)),
        "rule_action": rule_action,
        "fixed_action": fixed_action,
        "branches": {action: dict(row) for action, row in branches.items()},
    }


def _allowed(record: Mapping[str, Any], budget: float | None) -> list[str]:
    actions = []
    for action in record["legal"]:
        if budget is None or ACTION_COST[action] <= budget + 1e-12:
            actions.append(action)
    return actions


def _pick(record: Mapping[str, Any], policy: str, allowed: Sequence[str]) -> str | None:
    allowed_set = set(allowed)
    if policy == "oracle":
        if not allowed_set:
            return None
        subset = {action: record["branches"][action] for action in allowed}
        action, _optimal = _best_action(subset, field_name="post_energy_ae")
        return action
    if policy == "q_phi":
        action = str(record["q_action"])
        if action not in allowed_set:
            return None
        return action
    if policy == "rule_utility":
        # Re-select inside the affordable legal set. The utility values do not change.
        if not allowed_set:
            return None
        # The stored rule action was computed on the full A_legal. Recompute is
        # the caller's job when the mask changes; here we only accept it when
        # the stored action is still allowed, otherwise the caller passes a
        # recomputed action via record['rule_action_budget'].
        action = str(record.get("rule_action_override") or record["rule_action"])
        if action not in allowed_set:
            return None
        return action
    if policy == "fixed_pipeline":
        action = str(record.get("fixed_action_override") or record["fixed_action"])
        if action not in allowed_set:
            return None
        return action
    raise CounterfactualError(policy)


def _outcome_of(record: Mapping[str, Any], action: str | None) -> dict[str, Any]:
    """Idle keeps the pre-action error and costs nothing."""
    if action is None:
        pre_e = _finite(record.get("pre_energy_ae"))
        pre_m = _finite(record.get("pre_mass_ae"))
        return {
            "action": None,
            "post_energy_ae": pre_e,
            "post_mass_ae": pre_m,
            "energy_delta": 0.0 if pre_e is not None else None,
            "mass_delta": 0.0 if pre_m is not None else None,
            "delta_l": 0.0 if record.get("pre_loss") is not None else None,
            "cost": 0.0,
            "idle": True,
        }
    branch = record["branches"][action]
    return {
        "action": action,
        "post_energy_ae": _finite(branch.get("post_energy_ae")),
        "post_mass_ae": _finite(branch.get("post_mass_ae")),
        "energy_delta": _finite(branch.get("energy_delta")),
        "mass_delta": _finite(branch.get("mass_delta")),
        "delta_l": _finite(branch.get("delta_l")),
        "cost": float(branch["cost"]),
        "idle": False,
    }


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def summarize_policy(records: Sequence[Mapping[str, Any]], policy: str) -> dict[str, Any]:
    """Identification, realized energy improvement, and regret for one policy.

    Random is the uniform expectation over the allowed actions, not one draw.
    A budget mask is already written onto ``rule_action_override`` /
    ``fixed_action_override`` when the caller recomputed those selectors.
    ``Q`` is never re-argmaxed: a choice that the budget cannot pay becomes idle.
    """
    hits = 0
    exact = 0
    n = 0
    n_q_illegal = 0
    improvements: list[float] = []
    regrets: list[float] = []
    mass_improvements: list[float] = []
    mass_regrets: list[float] = []
    costs: list[float] = []
    chosen_counts: Counter[str] = Counter()
    n_idle = 0
    for record in records:
        n += 1
        allowed = list(record["allowed"])
        optimal = set(record["oracle_actions_allowed"])
        if policy == "random":
            if not allowed:
                n_idle += 1
                chosen_counts["IDLE"] += 1
                idle = _outcome_of(record, None)
                oracle = _outcome_of(record, record["oracle_action_allowed"])
                if idle["energy_delta"] is not None:
                    improvements.append(float(idle["energy_delta"]))
                if idle["post_energy_ae"] is not None and oracle["post_energy_ae"] is not None:
                    regrets.append(float(idle["post_energy_ae"]) - float(oracle["post_energy_ae"]))
                if idle["mass_delta"] is not None:
                    mass_improvements.append(float(idle["mass_delta"]))
                mass_oracle = _outcome_of(record, record["mass_oracle_action_allowed"])
                if idle["post_mass_ae"] is not None and mass_oracle["post_mass_ae"] is not None:
                    mass_regrets.append(float(idle["post_mass_ae"]) - float(mass_oracle["post_mass_ae"]))
                costs.append(0.0)
                hits += 1
                exact += 1
                continue
            id_prob = len(optimal) / len(allowed)
            hits += id_prob
            exact += (1.0 / len(allowed)) if record["oracle_action_allowed"] in allowed else 0.0
            energy_deltas = []
            posts = []
            mass_deltas = []
            mass_posts = []
            action_costs = []
            for action in allowed:
                outcome = _outcome_of(record, action)
                action_costs.append(outcome["cost"])
                if outcome["energy_delta"] is not None:
                    energy_deltas.append(outcome["energy_delta"])
                if outcome["post_energy_ae"] is not None:
                    posts.append(outcome["post_energy_ae"])
                if outcome["mass_delta"] is not None:
                    mass_deltas.append(outcome["mass_delta"])
                if outcome["post_mass_ae"] is not None:
                    mass_posts.append(outcome["post_mass_ae"])
            if len(energy_deltas) == len(allowed):
                improvements.append(float(sum(energy_deltas) / len(allowed)))
            oracle = _outcome_of(record, record["oracle_action_allowed"])
            if posts and oracle["post_energy_ae"] is not None and len(posts) == len(allowed):
                regrets.append(float(sum(posts) / len(posts)) - float(oracle["post_energy_ae"]))
            if len(mass_deltas) == len(allowed):
                mass_improvements.append(float(sum(mass_deltas) / len(allowed)))
            mass_oracle = _outcome_of(record, record["mass_oracle_action_allowed"])
            if mass_posts and mass_oracle["post_mass_ae"] is not None and len(mass_posts) == len(allowed):
                mass_regrets.append(float(sum(mass_posts) / len(mass_posts)) - float(mass_oracle["post_mass_ae"]))
            costs.append(float(sum(action_costs) / len(action_costs)))
            chosen_counts["EXPECTED"] += 1
            continue
        if policy == "q_phi" and not record.get("q_legal", True) and record.get("budget") is None:
            n_q_illegal += 1
        action = _pick(record, policy, allowed)
        if action is None:
            n_idle += 1
            chosen_counts["IDLE"] += 1
        else:
            chosen_counts[action] += 1
        outcome = _outcome_of(record, action)
        oracle = _outcome_of(record, record["oracle_action_allowed"])
        if action is not None and action in optimal:
            hits += 1
        if action is not None and action == record["oracle_action_allowed"]:
            exact += 1
        if action is None and not allowed:
            hits += 1
            exact += 1
        if outcome["energy_delta"] is not None:
            improvements.append(float(outcome["energy_delta"]))
        if outcome["post_energy_ae"] is not None and oracle["post_energy_ae"] is not None:
            regrets.append(float(outcome["post_energy_ae"]) - float(oracle["post_energy_ae"]))
        if outcome["mass_delta"] is not None:
            mass_improvements.append(float(outcome["mass_delta"]))
        mass_oracle = _outcome_of(record, record["mass_oracle_action_allowed"])
        if outcome["post_mass_ae"] is not None and mass_oracle["post_mass_ae"] is not None:
            mass_regrets.append(float(outcome["post_mass_ae"]) - float(mass_oracle["post_mass_ae"]))
        costs.append(float(outcome["cost"]))
    return {
        "n": n,
        "n_improvement": len(improvements),
        "n_regret": len(regrets),
        "n_idle": n_idle,
        "n_q_illegal_on_full_set": n_q_illegal,
        "best_action_identification_rate": _round(hits / n if n else None, 4),
        "exact_tiebreak_rate": _round(exact / n if n else None, 4),
        "mean_energy_improvement_kcal": _round(_mean(improvements), 4),
        "mean_energy_regret_kcal": _round(_mean(regrets), 4),
        "mean_mass_improvement_g": _round(_mean(mass_improvements), 4),
        "mean_mass_regret_g": _round(_mean(mass_regrets), 4),
        "mean_cost": _round(_mean(costs), 4),
        "actions": {name: int(chosen_counts[name]) for name in ("IDLE", "EXPECTED", *BC_ACTIONS) if chosen_counts[name]},
    }


def prepare_budget(record: Mapping[str, Any], budget: float | None, row_z: np.ndarray | None) -> dict[str, Any]:
    """Copy a decision and restrict the legal set to ``cost <= budget``.

    ``row_z`` is required when the budget actually drops an action, so the
    rule and the fixed pipeline can be recomputed. ``Q`` is left frozen.
    """
    allowed = _allowed(record, budget)
    allowed_map = {action: action in allowed for action in BC_ACTIONS}
    view = dict(record)
    view["allowed"] = allowed
    view["budget"] = budget
    if allowed:
        subset = {action: record["branches"][action] for action in allowed}
        oracle, optimal = _best_action(subset, field_name="post_energy_ae")
        mass_oracle, _mass_optimal = _best_action(subset, field_name="post_mass_ae")
    else:
        oracle, optimal = None, []
        mass_oracle = None
    view["oracle_action_allowed"] = oracle
    view["oracle_actions_allowed"] = optimal
    view["mass_oracle_action_allowed"] = mass_oracle
    if budget is None or row_z is None:
        view["rule_action_override"] = record["rule_action"] if record["rule_action"] in allowed else None
        view["fixed_action_override"] = record["fixed_action"] if record["fixed_action"] in allowed else None
    else:
        if allowed:
            view["rule_action_override"] = choose_rule(row_z, allowed_map)
            view["fixed_action_override"] = choose_fixed(row_z, allowed_map)
        else:
            view["rule_action_override"] = None
            view["fixed_action_override"] = None
    return view


def value_calibration(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Predicted ``Q(z, a)`` against realized ``ΔL`` on every executed legal action.

    This is not the logged-action regression. Actions the selector did not
    take are included. Non-finite ``Q`` (the masked ``RE_DECOMPOSE`` head)
    is omitted. No new map from ``Q`` to ``ΔL`` is fit for later use; the
    slope below is a description of this sample.
    """
    predicted: list[float] = []
    realized: list[float] = []
    energy: list[float] = []
    by_action: dict[str, list[tuple[float, float]]] = {name: [] for name in BC_ACTIONS}
    for record in records:
        for action, branch in record["branches"].items():
            q_value = branch.get("q")
            delta = branch.get("delta_l")
            if q_value is None or delta is None:
                continue
            predicted.append(float(q_value))
            realized.append(float(delta))
            energy_delta = branch.get("energy_delta")
            energy.append(float("nan") if energy_delta is None else float(energy_delta))
            by_action[action].append((float(q_value), float(delta)))
    pred = np.asarray(predicted, dtype=np.float64)
    real = np.asarray(realized, dtype=np.float64)
    kcal = np.asarray(energy, dtype=np.float64)
    pearson = None
    if pred.size >= 3 and float(np.std(pred)) > 1e-12 and float(np.std(real)) > 1e-12:
        pearson = float(np.corrcoef(pred, real)[0, 1])
    slope = None
    intercept = None
    if pred.size >= 3 and float(np.std(pred)) > 1e-12:
        slope_f, intercept_f = np.polyfit(pred, real, 1)
        slope = float(slope_f)
        intercept = float(intercept_f)
    finite_kcal = np.isfinite(kcal)
    energy_pearson = None
    if int(finite_kcal.sum()) >= 3 and float(np.std(pred[finite_kcal])) > 1e-12 and float(np.std(kcal[finite_kcal])) > 1e-12:
        energy_pearson = float(np.corrcoef(pred[finite_kcal], kcal[finite_kcal])[0, 1])
    bins: list[dict[str, Any]] = []
    if pred.size:
        quantiles = np.quantile(pred, np.linspace(0.0, 1.0, 11))
        edges = [float(quantiles[0])]
        for edge in quantiles[1:]:
            if edge > edges[-1] + 1e-12:
                edges.append(float(edge))
        for left, right in zip(edges[:-1], edges[1:]):
            if right == edges[-1]:
                mask = (pred >= left) & (pred <= right)
            else:
                mask = (pred >= left) & (pred < right)
            if not bool(mask.any()):
                continue
            bins.append(
                {
                    "q_lo": _round(left, 6),
                    "q_hi": _round(right, 6),
                    "n": int(mask.sum()),
                    "mean_predicted_q": _round(float(pred[mask].mean()), 6),
                    "mean_realized_delta_l": _round(float(real[mask].mean()), 6),
                    "mean_realized_energy_delta_kcal": _round(
                        float(kcal[mask][np.isfinite(kcal[mask])].mean()) if np.isfinite(kcal[mask]).any() else None,
                        4,
                    ),
                }
            )
    per_action = {}
    for action, pairs in by_action.items():
        if not pairs:
            per_action[action] = {"n": 0}
            continue
        left = np.asarray([item[0] for item in pairs], dtype=np.float64)
        right = np.asarray([item[1] for item in pairs], dtype=np.float64)
        action_pearson = None
        if left.size >= 3 and float(np.std(left)) > 1e-12 and float(np.std(right)) > 1e-12:
            action_pearson = float(np.corrcoef(left, right)[0, 1])
        per_action[action] = {
            "n": int(left.size),
            "mae": _round(float(np.mean(np.abs(left - right)))),
            "mean_predicted_q": _round(float(left.mean())),
            "mean_realized_delta_l": _round(float(right.mean())),
            "pearson": _round(action_pearson),
        }
    return {
        "target": "ΔL = L(pre) - L(post), L = min(|kcal error|, 2500) / 2500",
        "population": "every executed legal action on the reported states whose pre and post energy are both finite and whose Q is finite",
        "n": int(pred.size),
        "mae": _round(float(np.mean(np.abs(pred - real))) if pred.size else None),
        "pearson_delta_l": _round(pearson),
        "pearson_energy_kcal": _round(energy_pearson),
        "mean_predicted_q": _round(float(pred.mean()) if pred.size else None),
        "mean_realized_delta_l": _round(float(real.mean()) if pred.size else None),
        "slope_realized_on_predicted": _round(slope, 4),
        "intercept_realized_on_predicted": _round(intercept, 4),
        "bins": bins,
        "per_action": per_action,
        "used_to_retune_tau": False,
        "used_to_edit_weights": False,
    }


def belief_at(step_i: int) -> dict[str, Any]:
    return {
        "tools_used": max(int(step_i), 0),
        "step_i": int(step_i),
        "safety_force_abstain": False,
        "safety_force_stop": False,
    }


def component_count(steps: Sequence[Mapping[str, Any]]) -> int:
    indexes = [int(step["component_i"]) for step in steps if isinstance(step.get("component_i"), int)]
    if not indexes:
        return 0
    return max(indexes) + 1


def development_dish_ids(path: Path | str = DEVELOPMENT_JSONL) -> set[str]:
    """Dish ids in the frozen #53 paired file. Read-only."""
    ids: set[str] = set()
    source = Path(path)
    if not source.is_file():
        return ids
    for record in read_ir_jsonl(source):
        dish_id = record.get("dish_id")
        if isinstance(dish_id, str):
            ids.add(dish_id)
    return ids


def json_ready(value: Any) -> Any:
    """Drop non-finite floats so the result file is strict JSON."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run_self_check() -> None:
    """Synthetic split, coverage, and one-step oracle. No checkpoint and no cafe file."""
    import json
    import tempfile

    from vdr.controller.q_state import build_z

    open_row = build_z(
        {
            "hyp_i": "apple",
            "code_i": None,
            "cands_i": [],
            "portion_g_i": None,
            "verify_codes": [],
            "evidence_edges": [],
            "nutri_i": None,
            "nutri_dish": None,
            "step_i": 0,
        },
        None,
        belief_at(0),
    )
    legal = coverage_legal(open_row)
    if legal["STOP"] or legal["ABSTAIN"] or not legal["RETRIEVE"]:
        raise AssertionError(legal)
    if choose_fixed(open_row, legal) != "RETRIEVE":
        raise AssertionError("fixed pipeline should retrieve before a code exists")
    if choose_rule(open_row, legal) in {"STOP", "ABSTAIN"}:
        raise AssertionError("rule utility selected a coverage-illegal terminal")

    closed = build_z(
        {
            "hyp_i": None,
            "code_i": None,
            "cands_i": [],
            "portion_g_i": None,
            "verify_codes": [],
            "evidence_edges": [{"type": "n<-aggregate", "src": "dish", "dst": "n"}],
            "nutri_i": None,
            "nutri_dish": {"energy_kcal": 80.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0, "mass_g": 90.0},
            "step_i": 2,
        },
        {"action_t": "RE_PORTION", "obs_t": {"portion_src": "mass_estimator"}, "step_i": 1},
        belief_at(2),
    )
    closed_legal = coverage_legal(closed)
    if not closed_legal["STOP"] or closed_legal["RETRIEVE"]:
        raise AssertionError(closed_legal)

    label = {"energy_kcal": 100.0, "mass_g": 100.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0}
    pre = {"energy_kcal": 180.0, "mass_g": 150.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0}
    good = {"energy_kcal": 110.0, "mass_g": 105.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0}
    bad = {"energy_kcal": 250.0, "mass_g": 40.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0}
    branches = {
        "RETRIEVE": score_branch(
            BranchOutcome("RETRIEVE", good, False, True, "stub"),
            label=label,
            pre_dish=pre,
            q_value=0.2,
        ),
        "RE_PORTION": score_branch(
            BranchOutcome("RE_PORTION", bad, False, True, "stub"),
            label=label,
            pre_dish=pre,
            q_value=-0.1,
        ),
    }
    legal_map = {action: action in branches for action in BC_ACTIONS}
    record = build_decision_record(
        dish_id="dish_synthetic",
        step_i=0,
        component_i=0,
        n_components=1,
        legal=legal_map,
        pre_energy=_abs_field(pre, label, "energy_kcal"),
        pre_mass=_abs_field(pre, label, "mass_g"),
        pre_loss=nutrition_loss(pre, label),
        branches=branches,
        q_action="RETRIEVE",
        q_source="q_over_cost",
        rule_action="RETRIEVE",
        fixed_action="RE_PORTION",
    )
    if record["oracle_action"] != "RETRIEVE":
        raise AssertionError(record["oracle_action"])
    view = prepare_budget(record, None, open_row)
    table = {
        name: summarize_policy([view], name)
        for name in ("oracle", "q_phi", "rule_utility", "random", "fixed_pipeline")
    }
    if table["oracle"]["mean_energy_regret_kcal"] != 0.0:
        raise AssertionError(table["oracle"])
    if table["q_phi"]["best_action_identification_rate"] != 1.0:
        raise AssertionError(table["q_phi"])
    if table["fixed_pipeline"]["mean_energy_regret_kcal"] <= 0.0:
        raise AssertionError(table["fixed_pipeline"])
    if table["random"]["best_action_identification_rate"] != 0.5:
        raise AssertionError(table["random"])
    calibrated = value_calibration([record])
    if calibrated["n"] != 2 or calibrated["used_to_retune_tau"]:
        raise AssertionError(calibrated)

    fit_dish = "dish_fit_selfcheck"
    hold_dish = "dish_hold_selfcheck"
    # The hash, not the name, assigns the pool. Put each dish in its real file.
    rows = {
        "fit": fit_dish if pool_of_dish(fit_dish) == "fit" else hold_dish,
        "holdout": hold_dish if pool_of_dish(hold_dish) == "holdout" else fit_dish,
    }
    if pool_of_dish(rows["fit"]) != "fit" or pool_of_dish(rows["holdout"]) != "holdout":
        # Both synthetic ids can land in one pool. Nudge with a suffix until they split.
        for salt in range(100):
            left = f"dish_selfcheck_{salt}_a"
            right = f"dish_selfcheck_{salt}_b"
            if pool_of_dish(left) != pool_of_dish(right):
                rows = {
                    "fit": left if pool_of_dish(left) == "fit" else right,
                    "holdout": right if pool_of_dish(right) == "holdout" else left,
                }
                break
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fit_path = root / "fit.jsonl"
        hold_path = root / "hold.jsonl"
        fit_path.write_text(
            "\n".join(
                json.dumps({"dish_id": rows["fit"], "step_i": step, "action_t": "RETRIEVE"})
                for step in (0, 1)
            )
            + "\n",
            encoding="utf-8",
        )
        hold_path.write_text(
            json.dumps({"dish_id": rows["holdout"], "step_i": 0, "action_t": "STOP"}) + "\n",
            encoding="utf-8",
        )
        clean = audit_weak_step_split(fit_path, hold_path)
        if clean["leakage_exists"] or clean["conclusion"] != "no_dish_leakage":
            raise AssertionError(clean)
        leaked = hold_path.read_text(encoding="utf-8") + json.dumps(
            {"dish_id": rows["fit"], "step_i": 2, "action_t": "STOP"}
        ) + "\n"
        hold_path.write_text(leaked, encoding="utf-8")
        dirty = audit_weak_step_split(fit_path, hold_path)
        if not dirty["leakage_exists"] or dirty["dishes_in_both_splits"] != 1:
            raise AssertionError(dirty)

    # A retrieve that writes a known vector, with stub tools and no catalog.
    class _Found:
        def __init__(self) -> None:
            self.cands = [{"code": "111", "score": 0.9, "name": "Apple raw"}]

    class _Portion:
        portion_g = 100.0

    def lookup_fn(code: str, catalog_path: str | None = None) -> dict[str, float]:
        del catalog_path
        if code != "111":
            raise KeyError(code)
        return {"energy_kcal": 52.0, "protein_g": 0.3, "carb_g": 14.0, "fat_g": 0.2}

    executor = BranchExecutor(
        retrieve_fn=lambda query, top_k=5, catalog_path=None: _Found(),
        lookup_fn=lookup_fn,
        portion_fn=lambda code, hint_g=100.0, catalog_path=None: _Portion(),
        mass_by_dish={
            "dish_mass": {
                "mean": {"mass_g": 200.0, "energy_kcal": 180.0, "protein_g": 6.0, "carb_g": 20.0, "fat_g": 7.0},
                "sigma": {"mass_g": 10.0, "energy_kcal": 12.0, "protein_g": 1.0, "carb_g": 2.0, "fat_g": 1.0},
            }
        },
    )
    state = {
        "dish_id": "dish_mass",
        "hyp_i": "apple",
        "code_i": "111",
        "cands_i": [
            {"code": "111", "score": 0.91, "name": "Apple raw"},
            {"code": "222", "score": 0.2, "name": "Apple juice"},
        ],
        "portion_g_i": 100.0,
        "component_i": 0,
        "nutri_i": {"energy_kcal": 40.0, "protein_g": 0.2, "carb_g": 10.0, "fat_g": 0.1, "mass_g": 80.0},
        "nutri_dish": {"energy_kcal": 40.0, "protein_g": 0.2, "carb_g": 10.0, "fat_g": 0.1, "mass_g": 80.0},
    }
    portion = executor.run(state, "RE_PORTION", n_components=1)
    if portion.tool_failed or portion.nutri_dish is None:
        raise AssertionError(portion)
    if abs(float(portion.nutri_dish["mass_g"]) - 200.0) > 1e-6:
        raise AssertionError(portion.nutri_dish)
    if portion.detail != "estimate_mass_equal_split":
        raise AssertionError(portion.detail)
    failed = executor.run({**state, "dish_id": "missing"}, "RE_PORTION", n_components=1)
    if not failed.tool_failed or failed.changed:
        raise AssertionError("a missing mass forward must not invent grams")
    decompose = executor.run(state, "RE_DECOMPOSE", n_components=1)
    if decompose.changed or not decompose.tool_failed:
        raise AssertionError(decompose)
    stop = executor.run(state, "STOP", n_components=1)
    if stop.changed or stop.nutri_dish != state["nutri_dish"]:
        raise AssertionError(stop)


__all__ = [
    "PROTOCOL",
    "PROTOCOL_ID",
    "BranchExecutor",
    "audit_weak_step_split",
    "build_decision_record",
    "choose_fixed",
    "choose_q",
    "choose_rule",
    "coverage_legal",
    "development_dish_ids",
    "json_ready",
    "load_frozen_q",
    "load_frozen_tau",
    "prepare_budget",
    "run_self_check",
    "summarize_policy",
    "value_calibration",
]
