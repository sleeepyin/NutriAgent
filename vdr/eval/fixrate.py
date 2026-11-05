"""FixRate, WorsenRate, and tool cost for the closed-loop hold-out.

Primary contrast
---------------
``offline_policy_matched_terminal_counterfactual``

On a hold-out weak-step trajectory whose last step is ``STOP`` with a finite
``nutri_dish.energy_kcal``, a scored pair is a teacher step in
``{CORRECT_CODE, RE_PORTION, RE_DECOMPOSE}`` whose saved closed-loop policy
predicts that same action. Dish energy before and after the correction is the
STOP total with that one component swapped back to its pre-correction energy.
The reference is the train-split cafe dish label (kcal). rgb_test labels are
not read.

``ε = 1`` kcal is pre-registered. It is one dietary calorie, above the
teacher's 1e-4 kcal nutrient rounding, and it was not searched on hold-out
or on rgb_test. Fix means the absolute error fell by more than ε. Worsen
means it rose by more than ε. Every other scored pair is neutral. FixRate and
WorsenRate share that denominator, so

    FixRate − WorsenRate = (n_fix − n_worsen) / n_pairs.

The 95% interval is a percentile cluster bootstrap over dishes (pairs that
share a dish stay together), B=10000, seed=0.

Budget ``K = 32`` is ``STEP_BUDGET`` and ``TOOL_CAP``. Tool cost counts
``RETRIEVE``, ``CORRECT_CODE``, ``RE_PORTION``, and ``RE_DECOMPOSE``. ``STOP``
and ``ABSTAIN`` are not tools. ``VOLUME_ESTIMATE`` is refused.

Two labeled diagnostics use the same ε and the same bootstrap. They are not
the primary. ``diagnostic_consecutive_running_total`` pairs the corrective
step's ``nutri_dish`` with the next IR row (partial totals omit later
components). ``diagnostic_teacher_executed`` is the terminal contrast gated
on the teacher action alone.

Scope: offline logged tool results and the saved policy checkpoint.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from vdr.controller.features import STEP_BUDGET
from vdr.controller.state_features import TOOL_CAP

# Pre-registered. Not fit on hold-out or rgb_test.
FIXRATE_EPSILON_KCAL = 1.0
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
CI_ALPHA = 0.05
# Schema check only: portion rescale versus the next logged dish energy.
PAIRING_GAP_MAX_KCAL = 1e-3

if int(STEP_BUDGET) != int(TOOL_CAP):
    raise RuntimeError(f"STEP_BUDGET {STEP_BUDGET} != TOOL_CAP {TOOL_CAP}")
BUDGET_K = int(STEP_BUDGET)

CORRECTIVE_ACTIONS = frozenset({"CORRECT_CODE", "RE_PORTION", "RE_DECOMPOSE"})
TOOL_ACTIONS = frozenset({"RETRIEVE", "CORRECT_CODE", "RE_PORTION", "RE_DECOMPOSE"})
TERMINAL_ACTIONS = frozenset({"STOP", "ABSTAIN"})
KNOWN_ACTIONS = TOOL_ACTIONS | TERMINAL_ACTIONS

DEFINITION_ID = "offline_policy_matched_terminal_counterfactual"
DEFINITION = (
    "Hold-out weak steps plus the saved closed-loop policy. A scored pair is a "
    "teacher step in {CORRECT_CODE, RE_PORTION, RE_DECOMPOSE} whose policy "
    "prediction equals that action, on a dish that ends at STOP with a finite "
    "nutri_dish energy (pass or corrected). Before/after dish energy is that "
    "STOP total with the corrected component swapped to its pre-correction "
    "energy. RE_PORTION uses the obs portion ratio. CORRECT_CODE swaps the old "
    "code's component energy for the new code at the same pre-portion grams, "
    "so a later RE_PORTION on that component is not credited to the code change. "
    "Error is absolute kcal versus the train-split cafe dish label. "
    "Fix: error falls by more than epsilon. Worsen: error rises by more than "
    "epsilon. Otherwise neutral. Rates share the scored-pair denominator. "
    "Scope: offline logged tool results and the saved policy checkpoint."
)
CONSECUTIVE_NOTE = (
    "Diagnostic only. Energy is nutri_dish.energy_kcal on the corrective step "
    "and on the next IR step. A partial total omits later components, so a "
    "calorie increase can shrink |E − dish label| without a better final estimate."
)
TEACHER_NOTE = (
    "Diagnostic only. Same terminal component swap as the primary, gated on "
    "the teacher action rather than a matching policy prediction."
)
CI_METHOD = (
    "Percentile cluster bootstrap over dishes. Each draw resamples dishes "
    "with replacement and keeps every scored pair from those dishes. "
    f"B={BOOTSTRAP_DRAWS}, seed={BOOTSTRAP_SEED}, quantiles "
    f"{100 * CI_ALPHA / 2:g} and {100 * (1 - CI_ALPHA / 2):g}. "
    "Each rate block restarts the generator at the same seed. "
    "Wilson intervals are not used: fix and worsen are mutually exclusive "
    "outcomes of the same pairs, so the contrast is a paired trinomial mean."
)


class FixRateRefused(ValueError):
    """The trajectories are not a closed-loop hold-out FixRate input."""


@dataclass(frozen=True)
class ScoredPair:
    dish_id: str
    step_i: int
    action: str
    e_before: float
    e_after: float
    err_before: float
    err_after: float
    outcome: str


def classify_error_change(
    err_before: float,
    err_after: float,
    epsilon: float = FIXRATE_EPSILON_KCAL,
) -> str:
    """Return ``fix``, ``worsen``, or ``neutral`` under a strict ε deadband."""
    if epsilon < 0.0 or not math.isfinite(epsilon):
        raise ValueError(f"epsilon must be a finite value >= 0, got {epsilon}")
    delta = float(err_before) - float(err_after)
    if not math.isfinite(delta):
        raise ValueError("error change is not finite")
    if delta > epsilon:
        return "fix"
    if delta < -epsilon:
        return "worsen"
    return "neutral"


def _energy(nutri: Any) -> float | None:
    if not isinstance(nutri, Mapping):
        return None
    value = nutri.get("energy_kcal")
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _portion_ratio(obs: Any) -> float | None:
    if not isinstance(obs, Mapping):
        return None
    new_g = obs.get("portion_g")
    old_g = obs.get("from_portion_g")
    if new_g is None or old_g is None:
        return None
    try:
        new_f = float(new_g)
        old_f = float(old_g)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(new_f) or not math.isfinite(old_f) or old_f <= 0.0:
        return None
    return new_f / old_f


def _ordered_steps(steps: Sequence[Mapping[str, Any]], dish_id: str) -> list[Mapping[str, Any]]:
    ordered = sorted(steps, key=lambda row: int(row["step_i"]))
    seen: list[int] = []
    for row in ordered:
        action = row.get("action_t")
        if action == "VOLUME_ESTIMATE":
            raise FixRateRefused(f"{dish_id}: VOLUME_ESTIMATE is not a FixRate action")
        if action not in KNOWN_ACTIONS:
            raise FixRateRefused(f"{dish_id}: unknown action {action!r}")
        step_i = row.get("step_i")
        if not isinstance(step_i, int) or isinstance(step_i, bool) or step_i < 0:
            raise FixRateRefused(f"{dish_id}: invalid step_i {step_i!r}")
        seen.append(step_i)
    if len(seen) != len(set(seen)):
        raise FixRateRefused(f"{dish_id}: duplicate step_i")
    return ordered


def _require_prediction(pred: Any, dish_id: str, step_i: int) -> str:
    if not isinstance(pred, str) or pred not in KNOWN_ACTIONS:
        raise FixRateRefused(f"{dish_id} step {step_i}: prediction {pred!r} is not a known action")
    if pred == "VOLUME_ESTIMATE":
        raise FixRateRefused(f"{dish_id} step {step_i}: VOLUME_ESTIMATE prediction")
    return pred


def terminal_stop_energy(steps: Sequence[Mapping[str, Any]]) -> float | None:
    """Grounded STOP energy, or None when the dish does not emit one."""
    if not steps:
        return None
    last = steps[-1]
    if last.get("action_t") != "STOP" or last.get("s") not in ("pass", "corrected"):
        return None
    return _energy(last.get("nutri_dish"))


def _later_same_component_correction(steps: Sequence[Mapping[str, Any]], index: int) -> bool:
    component = steps[index].get("component_i")
    for later in steps[index + 1 :]:
        if later.get("component_i") != component:
            continue
        if later.get("action_t") in CORRECTIVE_ACTIONS:
            return True
    return False


def _finite_pair(e_before: float, e_after: float) -> tuple[float, float] | None:
    if not math.isfinite(e_before) or not math.isfinite(e_after):
        return None
    if e_before < -1e-3 or e_after < -1e-3:
        return None
    return e_before, e_after


def terminal_correction_energies(
    steps: Sequence[Mapping[str, Any]],
    index: int,
    e_terminal: float,
) -> tuple[float, float] | None:
    """STOP-total energy with this correction undone, then applied.

    Returns None when the logged IR does not identify that component swap.
    """
    step = steps[index]
    action = step.get("action_t")
    if action == "RE_PORTION":
        if _later_same_component_correction(steps, index):
            return None
        e_pre = _energy(step.get("nutri_i"))
        ratio = _portion_ratio(step.get("obs_t"))
        if e_pre is None or ratio is None:
            return None
        e_post = e_pre * ratio
        return _finite_pair(e_terminal - e_post + e_pre, e_terminal)

    if action != "CORRECT_CODE" or index + 1 >= len(steps):
        return None
    e_old = _energy(step.get("nutri_i"))
    if e_old is None:
        return None
    nxt = steps[index + 1]
    if nxt.get("action_t") == "RE_PORTION" and nxt.get("component_i") == step.get("component_i"):
        if _later_same_component_correction(steps, index + 1):
            return None
        e_new = _energy(nxt.get("nutri_i"))
        ratio = _portion_ratio(nxt.get("obs_t"))
        if e_new is None or ratio is None:
            return None
        e_final = e_new * ratio
    elif nxt.get("action_t") in ("RETRIEVE", "STOP") and nxt.get("component_i") != step.get("component_i"):
        if _energy(nxt.get("nutri_i")) is not None:
            return None
        if _later_same_component_correction(steps, index):
            return None
        e_dish = _energy(step.get("nutri_dish"))
        e_next = _energy(nxt.get("nutri_dish"))
        if e_dish is None or e_next is None:
            return None
        e_new = e_next - e_dish + e_old
        e_final = e_new
    else:
        return None
    return _finite_pair(e_terminal - e_final + e_old, e_terminal - e_final + e_new)


def consecutive_correction_energies(
    steps: Sequence[Mapping[str, Any]],
    index: int,
) -> tuple[float, float] | None:
    """Running ``nutri_dish`` on this step and the next IR step."""
    if steps[index].get("action_t") not in CORRECTIVE_ACTIONS or index + 1 >= len(steps):
        return None
    e_before = _energy(steps[index].get("nutri_dish"))
    e_after = _energy(steps[index + 1].get("nutri_dish"))
    if e_before is None or e_after is None:
        return None
    return _finite_pair(e_before, e_after)


def portion_rescale_gap(steps: Sequence[Mapping[str, Any]], index: int) -> float | None:
    """Absolute gap between the portion rescale and the next logged dish energy."""
    step = steps[index]
    if step.get("action_t") != "RE_PORTION" or index + 1 >= len(steps):
        return None
    nxt = steps[index + 1]
    if nxt.get("action_t") not in ("RETRIEVE", "STOP"):
        return None
    e_comp = _energy(step.get("nutri_i"))
    e_dish = _energy(step.get("nutri_dish"))
    e_next = _energy(nxt.get("nutri_dish"))
    ratio = _portion_ratio(step.get("obs_t"))
    if e_comp is None or e_dish is None or e_next is None or ratio is None:
        return None
    expected = e_dish - e_comp + e_comp * ratio
    return abs(e_next - expected)


def _score_pair(
    dish_id: str,
    step_i: int,
    action: str,
    energies: tuple[float, float],
    train_energy: float,
    epsilon: float,
) -> ScoredPair:
    e_before, e_after = energies
    err_before = abs(e_before - train_energy)
    err_after = abs(e_after - train_energy)
    return ScoredPair(
        dish_id=dish_id,
        step_i=step_i,
        action=action,
        e_before=e_before,
        e_after=e_after,
        err_before=err_before,
        err_after=err_after,
        outcome=classify_error_change(err_before, err_after, epsilon),
    )


def _policy_horizon_tools(steps: Sequence[Mapping[str, Any]], preds: Mapping[int, str]) -> int:
    n_tools = 0
    for step in steps:
        pred = preds[int(step["step_i"])]
        if pred in TERMINAL_ACTIONS:
            break
        if pred in TOOL_ACTIONS:
            n_tools += 1
    return n_tools


def _logged_tools(steps: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for step in steps if step.get("action_t") in TOOL_ACTIONS)


def bootstrap_rate_ci(
    dish_outcomes: Sequence[Sequence[str]],
    *,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = CI_ALPHA,
) -> dict[str, Any]:
    """Percentile CI for FixRate, WorsenRate, and their difference."""
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    groups = [list(group) for group in dish_outcomes if group]
    if not groups:
        return {
            "method": "cluster_bootstrap_dishes",
            "n_draws": n_draws,
            "seed": seed,
            "alpha": alpha,
            "fix_rate": None,
            "worsen_rate": None,
            "delta": None,
        }
    encoded = [
        np.array(
            [1 if item == "fix" else -1 if item == "worsen" else 0 for item in group],
            dtype=np.int8,
        )
        for group in groups
    ]
    rng = np.random.default_rng(seed)
    n_dish = len(encoded)
    stats = np.empty((n_draws, 3), dtype=np.float64)
    for draw in range(n_draws):
        pick = rng.integers(0, n_dish, size=n_dish)
        cat = np.concatenate([encoded[int(index)] for index in pick])
        n_fix = int(np.sum(cat == 1))
        n_worsen = int(np.sum(cat == -1))
        n_pairs = int(cat.size)
        stats[draw, 0] = n_fix / n_pairs
        stats[draw, 1] = n_worsen / n_pairs
        stats[draw, 2] = (n_fix - n_worsen) / n_pairs
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    lo = np.percentile(stats, lo_q, axis=0)
    hi = np.percentile(stats, hi_q, axis=0)
    return {
        "method": "cluster_bootstrap_dishes",
        "n_draws": n_draws,
        "seed": seed,
        "alpha": alpha,
        "interval": "percentile",
        "fix_rate": [float(lo[0]), float(hi[0])],
        "worsen_rate": [float(lo[1]), float(hi[1])],
        "delta": [float(lo[2]), float(hi[2])],
    }


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def rate_block(
    pairs: Sequence[ScoredPair],
    *,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    note: str | None = None,
) -> dict[str, Any]:
    """Point rates plus a dish-cluster CI. Dishes are sorted for the bootstrap."""
    by_dish: dict[str, list[ScoredPair]] = {}
    for pair in pairs:
        by_dish.setdefault(pair.dish_id, []).append(pair)
    dish_ids = sorted(by_dish)
    groups = [[pair.outcome for pair in by_dish[dish_id]] for dish_id in dish_ids]
    n_pairs = len(pairs)
    n_fix = sum(1 for pair in pairs if pair.outcome == "fix")
    n_worsen = sum(1 for pair in pairs if pair.outcome == "worsen")
    n_neutral = sum(1 for pair in pairs if pair.outcome == "neutral")
    block: dict[str, Any] = {
        "n_pairs": n_pairs,
        "n_dishes": len(dish_ids),
        "n_fix": n_fix,
        "n_worsen": n_worsen,
        "n_neutral": n_neutral,
        "fix_rate": None if n_pairs == 0 else n_fix / n_pairs,
        "worsen_rate": None if n_pairs == 0 else n_worsen / n_pairs,
        "delta": None if n_pairs == 0 else (n_fix - n_worsen) / n_pairs,
        "mean_err_before_kcal": _mean([pair.err_before for pair in pairs]),
        "mean_err_after_kcal": _mean([pair.err_after for pair in pairs]),
        "mean_err_decrease_kcal": _mean([pair.err_before - pair.err_after for pair in pairs]),
        "ci95": None if n_pairs == 0 else bootstrap_rate_ci(groups, n_draws=n_draws, seed=seed),
    }
    if note is not None:
        block["note"] = note
    return block


def summarize_tool_counts(counts: Sequence[int], *, budget_k: int = BUDGET_K) -> dict[str, Any]:
    """Mean and median tool counts, plus the same figures capped at K."""
    if budget_k < 1:
        raise ValueError("budget_k must be >= 1")
    values = [int(count) for count in counts]
    if any(count < 0 for count in values):
        raise ValueError("tool counts must be >= 0")
    if not values:
        return {
            "budget_K": budget_k,
            "n_dishes": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "mean_capped_at_K": None,
            "median_capped_at_K": None,
            "n_within_budget": 0,
            "n_over_budget": 0,
            "fraction_within_budget": None,
        }
    capped = [min(count, budget_k) for count in values]
    n_over = sum(1 for count in values if count > budget_k)
    n_within = len(values) - n_over
    return {
        "budget_K": budget_k,
        "n_dishes": len(values),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "min": min(values),
        "max": max(values),
        "mean_capped_at_K": float(sum(capped) / len(capped)),
        "median_capped_at_K": float(statistics.median(capped)),
        "n_within_budget": n_within,
        "n_over_budget": n_over,
        "fraction_within_budget": n_within / len(values),
    }


def _by_action(pairs: Sequence[ScoredPair], *, n_draws: int, seed: int) -> dict[str, Any]:
    blocks: dict[str, Any] = {}
    for action in ("CORRECT_CODE", "RE_PORTION", "RE_DECOMPOSE"):
        chosen = [pair for pair in pairs if pair.action == action]
        blocks[action] = rate_block(chosen, n_draws=n_draws, seed=seed)
    return blocks


def evaluate_fixrate(
    steps_by_dish: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[tuple[str, int], str],
    train_energy_kcal: Mapping[str, float],
    *,
    epsilon: float = FIXRATE_EPSILON_KCAL,
    budget_k: int = BUDGET_K,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Score policy-matched corrections and tool cost. Does not read a file."""
    if epsilon < 0.0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite and >= 0")
    primary: list[ScoredPair] = []
    consecutive: list[ScoredPair] = []
    teacher: list[ScoredPair] = []
    logged_counts: list[int] = []
    horizon_counts: list[int] = []
    gaps: list[float] = []
    funnel = {
        "n_dishes": 0,
        "n_steps": 0,
        "n_grounded_dishes": 0,
        "n_ungrounded_dishes": 0,
        "n_teacher_corrective_steps": 0,
        "n_teacher_corrective_on_ungrounded": 0,
        "n_policy_matched_on_grounded": 0,
        "n_policy_matched_skipped_undefined": 0,
        "n_policy_corrective_unmatched": 0,
    }

    for dish_id in sorted(steps_by_dish):
        ordered = _ordered_steps(steps_by_dish[dish_id], dish_id)
        if not ordered:
            raise FixRateRefused(f"{dish_id}: empty trajectory")
        preds: dict[int, str] = {}
        for step in ordered:
            step_i = int(step["step_i"])
            key = (dish_id, step_i)
            if key not in predictions:
                raise FixRateRefused(f"{dish_id} step {step_i}: missing policy prediction")
            preds[step_i] = _require_prediction(predictions[key], dish_id, step_i)
        funnel["n_dishes"] += 1
        funnel["n_steps"] += len(ordered)
        logged_counts.append(_logged_tools(ordered))
        horizon_counts.append(_policy_horizon_tools(ordered, preds))

        label = train_energy_kcal.get(dish_id)
        if label is None or not math.isfinite(float(label)):
            raise FixRateRefused(f"{dish_id}: missing finite train dish energy")
        train_energy = float(label)
        e_terminal = terminal_stop_energy(ordered)
        grounded = e_terminal is not None
        if grounded:
            funnel["n_grounded_dishes"] += 1
        else:
            funnel["n_ungrounded_dishes"] += 1

        for index, step in enumerate(ordered):
            action = str(step["action_t"])
            pred = preds[int(step["step_i"])]
            if pred in CORRECTIVE_ACTIONS and pred != action:
                funnel["n_policy_corrective_unmatched"] += 1
            if action not in CORRECTIVE_ACTIONS:
                continue
            funnel["n_teacher_corrective_steps"] += 1
            if not grounded:
                funnel["n_teacher_corrective_on_ungrounded"] += 1
                continue
            assert e_terminal is not None
            matched = pred == action
            if matched:
                funnel["n_policy_matched_on_grounded"] += 1
            gap = portion_rescale_gap(ordered, index)
            if gap is not None:
                gaps.append(gap)
            terminal = terminal_correction_energies(ordered, index, e_terminal)
            running = consecutive_correction_energies(ordered, index)
            if matched and terminal is None:
                funnel["n_policy_matched_skipped_undefined"] += 1
            if terminal is not None and matched:
                primary.append(
                    _score_pair(
                        dish_id,
                        int(step["step_i"]),
                        action,
                        terminal,
                        train_energy,
                        epsilon,
                    )
                )
            if terminal is not None:
                teacher.append(
                    _score_pair(
                        dish_id,
                        int(step["step_i"]),
                        action,
                        terminal,
                        train_energy,
                        epsilon,
                    )
                )
            if running is not None and matched:
                consecutive.append(
                    _score_pair(
                        dish_id,
                        int(step["step_i"]),
                        action,
                        running,
                        train_energy,
                        epsilon,
                    )
                )

    max_gap = max(gaps) if gaps else None
    if max_gap is not None and max_gap > PAIRING_GAP_MAX_KCAL:
        raise FixRateRefused(
            "RE_PORTION rescale disagrees with the next IR dish energy "
            f"by {max_gap} kcal (limit {PAIRING_GAP_MAX_KCAL})"
        )

    primary_block = rate_block(primary, n_draws=n_draws, seed=seed)
    primary_block["by_action"] = _by_action(primary, n_draws=n_draws, seed=seed)
    n_horizon_eq = sum(left == right for left, right in zip(logged_counts, horizon_counts))
    return {
        "metric": "FixRate-WorsenRate",
        "definition_id": DEFINITION_ID,
        "definition": DEFINITION,
        "epsilon_kcal": float(epsilon),
        "epsilon_rule": (
            "fix if (err_before - err_after) > epsilon; "
            "worsen if (err_after - err_before) > epsilon; otherwise neutral"
        ),
        "error": "absolute dish energy kcal versus the train-split cafe label",
        "budget_K": int(budget_k),
        "budget_K_source": (
            "vdr.controller.features.STEP_BUDGET and vdr.controller.state_features.TOOL_CAP"
        ),
        "ci_method": CI_METHOD,
        "audit": {
            "uses_test_gt": False,
            "uses_train_dish_labels": True,
            "vlm": False,
            "volume_tool": False,
            "legacy_regression_weights": False,
            "online_rollout": False,
            "policy_retrained": False,
            "rl": False,
        },
        "funnel": funnel,
        "primary": primary_block,
        "diagnostic_consecutive_running_total": rate_block(
            consecutive,
            n_draws=n_draws,
            seed=seed,
            note=CONSECUTIVE_NOTE,
        ),
        "diagnostic_teacher_executed": rate_block(
            teacher,
            n_draws=n_draws,
            seed=seed,
            note=TEACHER_NOTE,
        ),
        "tool_cost": {
            "unit": "tool actions per dish",
            "tool_actions": sorted(TOOL_ACTIONS),
            "not_tools": sorted(TERMINAL_ACTIONS),
            "logged_trajectory": summarize_tool_counts(logged_counts, budget_k=budget_k),
            "logged_trajectory_note": (
                "Tools the teacher executed on the evaluated hold-out trajectories."
            ),
            "policy_horizon_off_policy": summarize_tool_counts(horizon_counts, budget_k=budget_k),
            "n_dishes_horizon_equals_logged": n_horizon_eq,
            "policy_horizon_note": (
                "Predicted tool actions along the logged states until the policy's "
                "first STOP or ABSTAIN. States stay the teacher states. "
                "Teacher trajectories put STOP or ABSTAIN only on the last step, "
                "so this count equals the logged tool count when that first "
                "predicted terminal action is the last step."
            ),
        },
        "pairing_check": {
            "n_re_portion_compared": len(gaps),
            "max_abs_gap_kcal": max_gap,
            "limit_kcal": PAIRING_GAP_MAX_KCAL,
            "note": (
                "RE_PORTION nutri_i rescaled by obs portion_g/from_portion_g, "
                "compared with the next IR nutri_dish. Schema check, not a threshold search."
            ),
        },
    }


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _fmt_ci(bounds: Any) -> str:
    if not isinstance(bounds, list) or len(bounds) != 2:
        return "n/a"
    return f"[{bounds[0]:.4f}, {bounds[1]:.4f}]"


def _fmt_kcal(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def format_fixrate_report(payload: Mapping[str, Any]) -> str:
    """Plain-text report. Numbers come from ``evaluate_fixrate``."""
    primary = payload["primary"]
    ci = primary.get("ci95") or {}
    funnel = payload["funnel"]
    logged = payload["tool_cost"]["logged_trajectory"]
    horizon = payload["tool_cost"]["policy_horizon_off_policy"]
    consecutive = payload["diagnostic_consecutive_running_total"]
    teacher = payload["diagnostic_teacher_executed"]
    gap = payload["pairing_check"]
    lines = [
        "FixRate − WorsenRate (hold-out weak steps, closed-loop policy)",
        f"definition: {payload['definition_id']}",
        f"epsilon_kcal: {payload['epsilon_kcal']} (pre-registered; |Δerr| > epsilon)",
        f"budget_K: {payload['budget_K']} ({payload['budget_K_source']})",
        f"error: {payload['error']}",
        f"ci: {payload['ci_method']}",
        "",
        (
            f"primary n_pairs={primary['n_pairs']} n_dishes={primary['n_dishes']} "
            f"fix={primary['n_fix']} worsen={primary['n_worsen']} neutral={primary['n_neutral']}"
        ),
        f"FixRate={_fmt_rate(primary['fix_rate'])}",
        f"WorsenRate={_fmt_rate(primary['worsen_rate'])}",
        f"FixRate-WorsenRate={_fmt_rate(primary['delta'])}",
        f"95% CI delta={_fmt_ci(ci.get('delta'))}",
        f"95% CI FixRate={_fmt_ci(ci.get('fix_rate'))}",
        f"95% CI WorsenRate={_fmt_ci(ci.get('worsen_rate'))}",
        (
            "mean |err| kcal before="
            f"{_fmt_kcal(primary['mean_err_before_kcal'])} "
            f"after={_fmt_kcal(primary['mean_err_after_kcal'])} "
            f"decrease={_fmt_kcal(primary['mean_err_decrease_kcal'])}"
        ),
        "",
        "by action (primary):",
    ]
    for action, block in primary.get("by_action", {}).items():
        action_ci = (block.get("ci95") or {}).get("delta")
        lines.append(
            f"  {action}: n={block['n_pairs']} FixRate={_fmt_rate(block['fix_rate'])} "
            f"WorsenRate={_fmt_rate(block['worsen_rate'])} "
            f"delta={_fmt_rate(block['delta'])} 95% CI delta={_fmt_ci(action_ci)}"
        )
    lines.extend(
        [
            "",
            (
                "tool cost logged trajectories: "
                f"mean={_fmt_kcal(logged['mean'])} median={_fmt_kcal(logged['median'])} "
                f"min={logged['min']} max={logged['max']} "
                f"within_K={logged['n_within_budget']}/{logged['n_dishes']} "
                f"over_K={logged['n_over_budget']}"
            ),
            (
                "tool cost policy horizon (off-policy states): "
                f"mean={_fmt_kcal(horizon['mean'])} median={_fmt_kcal(horizon['median'])} "
                f"min={horizon['min']} max={horizon['max']} "
                f"within_K={horizon['n_within_budget']}/{horizon['n_dishes']} "
                f"equals_logged={payload['tool_cost']['n_dishes_horizon_equals_logged']}/"
                f"{logged['n_dishes']}"
            ),
            "",
            (
                "diagnostic consecutive running total: "
                f"n={consecutive['n_pairs']} FixRate={_fmt_rate(consecutive['fix_rate'])} "
                f"WorsenRate={_fmt_rate(consecutive['worsen_rate'])} "
                f"delta={_fmt_rate(consecutive['delta'])} "
                f"95% CI delta={_fmt_ci((consecutive.get('ci95') or {}).get('delta'))}"
            ),
            (
                "diagnostic teacher-executed terminal: "
                f"n={teacher['n_pairs']} FixRate={_fmt_rate(teacher['fix_rate'])} "
                f"WorsenRate={_fmt_rate(teacher['worsen_rate'])} "
                f"delta={_fmt_rate(teacher['delta'])} "
                f"95% CI delta={_fmt_ci((teacher.get('ci95') or {}).get('delta'))}"
            ),
            "",
            (
                "funnel: "
                f"dishes={funnel['n_dishes']} steps={funnel['n_steps']} "
                f"grounded={funnel['n_grounded_dishes']} "
                f"ungrounded={funnel['n_ungrounded_dishes']} "
                f"teacher_corrective={funnel['n_teacher_corrective_steps']} "
                f"matched_grounded={funnel['n_policy_matched_on_grounded']} "
                f"matched_undefined={funnel['n_policy_matched_skipped_undefined']} "
                f"policy_corrective_unmatched={funnel['n_policy_corrective_unmatched']}"
            ),
            (
                "pairing check: n="
                f"{gap['n_re_portion_compared']} max_abs_gap_kcal="
                f"{_fmt_kcal(gap['max_abs_gap_kcal'])}"
            ),
            "",
            DEFINITION,
            "",
            "Overall action accuracy is not this report. It is a weight-loader check when present.",
        ]
    )
    sanity = payload.get("loader_sanity")
    if isinstance(sanity, Mapping):
        lines.append(
            "loader sanity: logged-action agreement="
            f"{_fmt_rate(sanity.get('logged_action_agreement'))} "
            f"saved_holdout_accuracy={_fmt_rate(sanity.get('saved_holdout_accuracy'))} "
            f"match={sanity.get('agreement_matches_saved_metrics')}"
        )
    return "\n".join(lines) + "\n"


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_self_check() -> None:
    """Synthetic FixRate contract. Does not read weak steps or weights."""
    _check(FIXRATE_EPSILON_KCAL == 1.0, "epsilon was pre-registered at 1 kcal")
    _check(BUDGET_K == 32, "budget K was pre-registered at 32")
    _check(classify_error_change(10.0, 10.0) == "neutral", "equal error is neutral")
    _check(classify_error_change(11.0, 10.0) == "neutral", "delta == epsilon is neutral")
    _check(classify_error_change(11.0 + 1e-9, 10.0) == "fix", "delta above epsilon is fix")
    _check(classify_error_change(10.0, 11.0) == "neutral", "worsen boundary is neutral")
    _check(classify_error_change(10.0, 11.0 + 1e-9) == "worsen", "delta below -epsilon is worsen")

    y_boundary = 15.5
    e_before, e_after = 10.0, 20.0
    _check(
        classify_error_change(abs(e_before - y_boundary), abs(e_after - y_boundary)) == "neutral",
        "constructed |Δerr| == 1 kcal stays neutral",
    )

    hand = rate_block(
        [
            ScoredPair("a", 0, "RE_PORTION", 1, 0, 3, 1, "fix"),
            ScoredPair("a", 1, "RE_PORTION", 1, 0, 3, 1, "fix"),
            ScoredPair("b", 0, "CORRECT_CODE", 1, 0, 3, 1, "fix"),
            ScoredPair("c", 0, "RE_PORTION", 0, 1, 1, 3, "worsen"),
            ScoredPair("d", 0, "RE_PORTION", 1, 1, 2, 2, "neutral"),
        ],
        n_draws=200,
        seed=0,
    )
    _check(hand["n_pairs"] == 5 and hand["n_dishes"] == 4, "hand pair counts")
    _check(abs(hand["fix_rate"] - 0.6) < 1e-12, "hand FixRate")
    _check(abs(hand["worsen_rate"] - 0.2) < 1e-12, "hand WorsenRate")
    _check(abs(hand["delta"] - 0.4) < 1e-12, "hand delta")
    _check(hand["ci95"]["delta"][0] <= hand["delta"] <= hand["ci95"]["delta"][1], "CI covers delta")
    pure = rate_block(
        [ScoredPair("a", 0, "RE_PORTION", 1, 0, 2, 1, "fix")],
        n_draws=50,
        seed=0,
    )
    _check(pure["ci95"]["delta"] == [1.0, 1.0], "single-dish fix CI is the point")
    again = rate_block(
        [
            ScoredPair("a", 0, "RE_PORTION", 1, 0, 3, 1, "fix"),
            ScoredPair("b", 0, "RE_PORTION", 0, 1, 1, 3, "worsen"),
        ],
        n_draws=300,
        seed=0,
    )
    repeat = rate_block(
        [
            ScoredPair("a", 0, "RE_PORTION", 1, 0, 3, 1, "fix"),
            ScoredPair("b", 0, "RE_PORTION", 0, 1, 1, 3, "worsen"),
        ],
        n_draws=300,
        seed=0,
    )
    _check(again["ci95"] == repeat["ci95"], "bootstrap seed is stable")

    counts = summarize_tool_counts([1, 2, 10, 40], budget_k=32)
    _check(counts["n_over_budget"] == 1 and counts["max"] == 40, "over-budget count")
    _check(abs(counts["mean"] - 13.25) < 1e-12, "raw mean keeps the over-budget dish")
    _check(abs(counts["mean_capped_at_K"] - (1 + 2 + 10 + 32) / 4) < 1e-12, "capped mean")
    _check(counts["median"] == 6.0, "median of 1,2,10,40")

    def portion_step(step_i: int, component: int, pre: float, dish: float) -> dict[str, Any]:
        return {
            "step_i": step_i,
            "action_t": "RE_PORTION",
            "component_i": component,
            "nutri_i": {"energy_kcal": pre},
            "nutri_dish": {"energy_kcal": dish},
            "obs_t": {"portion_g": 200.0, "from_portion_g": 100.0},
            "s": None,
        }

    keep = [
        {
            "step_i": 0,
            "action_t": "RETRIEVE",
            "component_i": 0,
            "nutri_i": None,
            "nutri_dish": None,
            "obs_t": {},
            "s": None,
        },
        portion_step(1, 0, 50.0, 50.0),
        {
            "step_i": 2,
            "action_t": "STOP",
            "component_i": None,
            "nutri_i": None,
            "nutri_dish": {"energy_kcal": 100.0},
            "obs_t": {"s": "corrected"},
            "s": "corrected",
        },
    ]
    miss = [
        {
            "step_i": 0,
            "action_t": "RETRIEVE",
            "component_i": 0,
            "nutri_i": None,
            "nutri_dish": None,
            "obs_t": {},
            "s": None,
        },
        portion_step(1, 0, 50.0, 50.0),
        {
            "step_i": 2,
            "action_t": "STOP",
            "component_i": None,
            "nutri_i": None,
            "nutri_dish": {"energy_kcal": 100.0},
            "obs_t": {},
            "s": "corrected",
        },
    ]
    early = [
        {"step_i": 0, "action_t": "RETRIEVE", "component_i": 0, "nutri_dish": None, "obs_t": {}, "s": None},
        {"step_i": 1, "action_t": "RE_PORTION", "component_i": 0, "nutri_dish": None, "obs_t": {}, "s": None},
        {
            "step_i": 2,
            "action_t": "ABSTAIN",
            "component_i": None,
            "nutri_dish": {"energy_kcal": None},
            "obs_t": {},
            "s": "abstain",
        },
    ]
    drop = [
        {"step_i": 0, "action_t": "RETRIEVE", "component_i": 0, "nutri_dish": None, "obs_t": {}, "s": None},
        portion_step(1, 0, 50.0, 50.0),
        {
            "step_i": 2,
            "action_t": "ABSTAIN",
            "component_i": None,
            "nutri_dish": {"energy_kcal": None},
            "obs_t": {},
            "s": "abstain",
        },
    ]
    preds = {
        ("keep", 0): "CORRECT_CODE",
        ("keep", 1): "RE_PORTION",
        ("keep", 2): "STOP",
        ("miss", 0): "RETRIEVE",
        ("miss", 1): "RETRIEVE",
        ("miss", 2): "STOP",
        ("early", 0): "STOP",
        ("early", 1): "RE_PORTION",
        ("early", 2): "ABSTAIN",
        ("drop", 0): "RETRIEVE",
        ("drop", 1): "RE_PORTION",
        ("drop", 2): "ABSTAIN",
    }
    payload = evaluate_fixrate(
        {"keep": keep, "miss": miss, "early": early, "drop": drop},
        preds,
        {name: 100.0 for name in ("keep", "miss", "early", "drop")},
        n_draws=20,
        seed=0,
    )
    _check(payload["primary"]["n_pairs"] == 1, "only the matched grounded correction is primary")
    _check(payload["primary"]["n_fix"] == 1 and payload["primary"]["delta"] == 1.0, "primary fix")
    _check(payload["primary"]["mean_err_before_kcal"] == 50.0, "terminal before-error")
    _check(payload["primary"]["mean_err_after_kcal"] == 0.0, "terminal after-error")
    _check(payload["diagnostic_teacher_executed"]["n_pairs"] == 2, "teacher gate keeps the mismatch")
    _check(payload["diagnostic_consecutive_running_total"]["n_pairs"] == 1, "consecutive stays policy-matched")
    _check(payload["funnel"]["n_ungrounded_dishes"] == 2, "abstain dishes stay out of the rates")
    _check(payload["funnel"]["n_policy_corrective_unmatched"] == 1, "unmatched corrective prediction")
    _check(payload["tool_cost"]["logged_trajectory"]["mean"] == 2.0, "logged tool mean")
    _check(payload["tool_cost"]["policy_horizon_off_policy"]["mean"] == 1.5, "horizon tool mean")
    _check(payload["tool_cost"]["n_dishes_horizon_equals_logged"] == 3, "horizon equality count")
    _check(payload["pairing_check"]["max_abs_gap_kcal"] == 0.0, "synthetic rescale gap")
    _check("FixRate-WorsenRate=1.0000" in format_fixrate_report(payload), "report carries the delta")

    bad = [
        portion_step(0, 0, 50.0, 50.0),
        {
            "step_i": 1,
            "action_t": "STOP",
            "component_i": None,
            "nutri_dish": {"energy_kcal": 40.0},
            "obs_t": {},
            "s": "corrected",
        },
    ]
    raised = False
    try:
        evaluate_fixrate(
            {"bad": bad},
            {("bad", 0): "RE_PORTION", ("bad", 1): "STOP"},
            {"bad": 10.0},
            n_draws=5,
        )
    except FixRateRefused:
        raised = True
    _check(raised, "a broken portion rescale is refused")

    volume = [{"step_i": 0, "action_t": "VOLUME_ESTIMATE", "s": None}]
    raised = False
    try:
        evaluate_fixrate({"v": volume}, {("v", 0): "STOP"}, {"v": 1.0}, n_draws=5)
    except FixRateRefused:
        raised = True
    _check(raised, "VOLUME_ESTIMATE is refused")


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "BUDGET_K",
    "CORRECTIVE_ACTIONS",
    "DEFINITION_ID",
    "FIXRATE_EPSILON_KCAL",
    "FixRateRefused",
    "TOOL_ACTIONS",
    "classify_error_change",
    "evaluate_fixrate",
    "format_fixrate_report",
    "run_self_check",
    "summarize_tool_counts",
]
