"""On-policy closed-loop rollout. The saved policy picks every action.

Perception hypotheses come from a caller-supplied VLM (the rgb_test script
uses local Qwen3.8-27B-VL). Tool IR is FNDDS retrieve, nutrition lookup, the
blind portion grid on RETRIEVE, and ``estimate_mass`` on RE_PORTION.
Nutrition5K ingredient mass, ingredient kcal, and cafe dish labels are not inputs.

Caps are the existing ones, not a search on rgb_test:

- ``MAX_STEPS`` is ``STEP_BUDGET`` (32), same width the policy was trained with
- ``STUCK_LIMIT`` is 3, the same guard as ``DAGGER_STUCK_LIMIT``

``CORRECT_CODE`` switches to another Top-k code inside the teacher's existing
score window (``MIN_RETRIEVE_SCORE``, ``MAX_SCORE_DROP``). It does not compare
kcal/g to an ingredient label. ``RE_PORTION`` calls ``estimate_mass`` and writes
that RGB specialist's component grams. It does not re-score the portion grid.
``RETRIEVE`` still places the first grams with the blind 100 g grid; that guess
is what RE_PORTION replaces. ``VOLUME_ESTIMATE`` is refused. ``RE_DECOMPOSE``
is not a second VLM call: the #24 checkpoint masks that class (fit weight 0).

``ON_POLICY_DEFINITION`` still describes the frozen #34 rollout, which used the
portion grid. This module's RE_PORTION path is the P0b Mass wiring and is not
a rewrite of that run.

The emitted IR matches ``ir_step`` so ``scripts/closed_loop_mae_aurc.py``
can score it. Replaying mask-none on the written rows reproduces ``action_t``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vdr.controller.closed_loop_data import _belief_at_step
from vdr.controller.features import STEP_BUDGET
from vdr.controller.policy_bc import BC_ACTIONS
from vdr.controller.policy_state_mlp import PolicyStateMLP
from vdr.controller.selector import decision_features
from vdr.controller.state_features import TOOL_CAP
from vdr.perception.hypotheses import dedupe_hypotheses, merge_complementary_fragments
from vdr.schemas.loader import IR_STEP_SCHEMA_PATH, load_schema, validate_instance
from vdr.tools.rule_verify import verify
from vdr.weak.teacher import (
    BLIND_PORTION_HINT_G,
    MAX_SCORE_DROP,
    MIN_RETRIEVE_SCORE,
    TOP_K,
    _binding_edges,
    _null_nutri,
    _public_cands,
    _scaled,
    _sum_nutri,
    candidate_rewrite,
)

# Same integer as vdr.controller.dagger.DAGGER_STUCK_LIMIT. Not fit on rgb_test.
STUCK_LIMIT = 3
PERCEPTION_CAP = 8  # same cap as vdr.perception.qwen25.ingredient_hypotheses

if int(STEP_BUDGET) != int(TOOL_CAP):
    raise RuntimeError(f"STEP_BUDGET {STEP_BUDGET} != TOOL_CAP {TOOL_CAP}")
MAX_STEPS = int(STEP_BUDGET)

ROLLOUT_ID = "on_policy_vlm"
PROTOCOL_ID = "on_policy_vlm_closed_loop_v1"
HYPOTHESIS_SOURCE = "qwen3.8-27b-vl"
ON_POLICY_DEFINITION_ID = "on_policy_vlm_horizon_force_answer"
ON_POLICY_DEFINITION = (
    "On-policy VLM horizon scored with the same force-answer reader as the offline "
    "script. States were produced by rolling the saved closed-loop policy on rgb_test "
    "overhead images: the policy chose each action, local Qwen3.8-27B-VL wrote the "
    "component hypotheses, and FNDDS retrieve, the existing portion grid, and nutrition "
    "lookup wrote the tool IR. Nutrition5K ingredient rows and cafe labels were not used "
    "to choose actions, codes, or grams. For ours_mask_none, re-predicted actions match "
    "the executed actions, so the decision step is that rollout's STOP or ABSTAIN. "
    "B5 and the observation masks are counterfactual predictions on this trace, not "
    "separate rollouts. This is not the #28/#31 offline logged hold-out table "
    "(offline_logged_horizon_force_answer, rgb_train hold-out, n=356). No threshold "
    "was fit on rgb_test. This VLM is the closed-loop backbone, not faithful DietAI24 "
    "or FoodAgent."
)
_FORBIDDEN_KEYS = frozenset(
    {
        "train_gt_energy_kcal",
        "train_gt",
        "y_true",
        "label_kcal",
        "energy_label",
        "cafe_label",
    }
)
_TERMINAL_ACTIONS = frozenset({"STOP", "ABSTAIN"})
_SCHEMA: dict[str, Any] | None = None
_RGB_TEST_IDS: set[str] | None = None

PerceiveFn = Callable[[str | Path | None], tuple[Sequence[Mapping[str, Any]] | Sequence[str], Mapping[str, Any]]]
RetrieveFn = Callable[..., Any]
LookupFn = Callable[..., Any]
PortionFn = Callable[..., Any]


class OnPolicyRefused(ValueError):
    """The rollout is not a valid on-policy IR trajectory."""


@dataclass(frozen=True)
class Hyp:
    name: str
    portion_g: float | None


@dataclass
class RolloutResult:
    dish_id: str
    steps: list[dict[str, Any]]
    ended_reason: str
    n_hypotheses: int


def _schema() -> dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = load_schema(IR_STEP_SCHEMA_PATH)
    return _SCHEMA


def rgb_test_ids() -> set[str]:
    global _RGB_TEST_IDS
    if _RGB_TEST_IDS is None:
        from vdr.data.n5k import read_ids

        _RGB_TEST_IDS = set(read_ids("rgb_test"))
    return _RGB_TEST_IDS


def prepare_hypotheses(raw: Sequence[Mapping[str, Any]] | Sequence[str] | None) -> list[Hyp]:
    """Dedupe, merge a crust/filling pair, and cap at the existing perception width."""
    items: list[dict[str, Any]] = []
    for hyp in raw or []:
        if isinstance(hyp, str):
            items.append({"name": hyp, "portion_g": None})
        elif isinstance(hyp, Mapping):
            items.append(
                {
                    "name": hyp.get("name") or hyp.get("food") or hyp.get("hyp") or "",
                    "portion_g": hyp.get("portion_g"),
                }
            )
    merged = merge_complementary_fragments(dedupe_hypotheses(items))
    prepared: list[Hyp] = []
    for hyp in merged[:PERCEPTION_CAP]:
        name = str(hyp.get("name") or "").strip()
        if not name:
            continue
        portion = hyp.get("portion_g")
        portion_f: float | None
        try:
            portion_f = None if portion is None or portion == "" else float(portion)
        except (TypeError, ValueError):
            portion_f = None
        if portion_f is not None and portion_f <= 0:
            portion_f = None
        prepared.append(Hyp(name, portion_f))
    return prepared


def _alternate_code(cands: Sequence[Mapping[str, Any]], current: str | None) -> str | None:
    """Highest-scoring other code inside the existing retrieve window. No density target."""
    if not cands:
        return None
    try:
        top_score = float(cands[0].get("score") or 0.0)
    except (TypeError, ValueError):
        return None
    best: tuple[float, str] | None = None
    for cand in cands:
        code = str(cand.get("code") or "")
        if not code or (current and code == current):
            continue
        try:
            score = float(cand.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
        if score < MIN_RETRIEVE_SCORE or top_score - score > MAX_SCORE_DROP:
            continue
        if best is None or score > best[0]:
            best = (score, code)
    return None if best is None else best[1]


def _walk_keys(value: Any, found: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _FORBIDDEN_KEYS:
                found.add(str(key))
            _walk_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk_keys(item, found)


def assert_no_label_keys(steps: Sequence[Mapping[str, Any]]) -> None:
    found: set[str] = set()
    for row in steps:
        _walk_keys(row, found)
    if found:
        raise OnPolicyRefused("trajectory contains label keys: " + ", ".join(sorted(found)))


def assert_on_policy_marker(steps_by_dish: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Refuse teacher logs and anything that is not this rollout's IR."""
    if not steps_by_dish:
        raise OnPolicyRefused("no dishes")
    for dish_id, steps in steps_by_dish.items():
        if not steps:
            raise OnPolicyRefused(f"{dish_id}: empty trajectory")
        for row in steps:
            if row.get("action_t") == "VOLUME_ESTIMATE":
                raise OnPolicyRefused(f"{dish_id}: VOLUME_ESTIMATE")
            obs = row.get("obs_t")
            if not isinstance(obs, Mapping):
                raise OnPolicyRefused(f"{dish_id}: obs_t is not an object")
            if obs.get("rollout") != ROLLOUT_ID or obs.get("protocol") != PROTOCOL_ID:
                raise OnPolicyRefused(f"{dish_id}: trajectory is not {PROTOCOL_ID}")
            if obs.get("hypothesis_source") != HYPOTHESIS_SOURCE:
                raise OnPolicyRefused(f"{dish_id}: hypothesis_source {obs.get('hypothesis_source')!r}")
            if obs.get("privileged") is not False or obs.get("uses_test_gt") is not False:
                raise OnPolicyRefused(f"{dish_id}: privileged or test-GT observation")
            if obs.get("volume_tool") is not False:
                raise OnPolicyRefused(f"{dish_id}: volume tool flag is set")
        assert_no_label_keys(steps)


def assert_replay_matches(steps: Sequence[Mapping[str, Any]], policy: Any) -> None:
    """Features of the written IR must reproduce the executed action.

    State-aware BC replays the 79-d vector. A z-policy replays ``build_z``,
    including σ only when the previous observation stored an ``estimate_mass``
    reading. Both paths use ``decision_features``.
    """
    prev: Mapping[str, Any] | None = None
    for row in steps:
        step_i = int(row["step_i"])
        vec = decision_features(policy, row, prev, _belief_at_step(step_i))
        predicted = policy.predict_actions(np.asarray(vec, dtype=np.float64).reshape(1, -1))
        if len(predicted) != 1 or predicted[0] != row.get("action_t"):
            raise OnPolicyRefused(
                f"{row.get('dish_id')} step {step_i}: replay {predicted} != executed {row.get('action_t')}"
            )
        prev = row


def load_closed_loop_policy(weights: Path, meta_path: Path) -> PolicyStateMLP:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actions = tuple(meta.get("actions") or ())
    if actions != tuple(BC_ACTIONS):
        raise OnPolicyRefused("train_meta action order does not match the BC head")
    class_weight = meta.get("class_weight")
    if not isinstance(class_weight, dict):
        raise OnPolicyRefused("train_meta is missing class_weight")
    return PolicyStateMLP.from_npz(weights, actions=actions, class_weight=class_weight)


def annotate_on_policy_table(payload: dict[str, Any], steps_by_dish: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Relabel the force-answer table as on-policy. Does not change numeric blocks."""
    assert_on_policy_marker(steps_by_dish)
    payload["definition_id"] = ON_POLICY_DEFINITION_ID
    payload["definition"] = ON_POLICY_DEFINITION
    payload["primary_title"] = (
        "Primary — on-policy terminal STOP with s in {pass, corrected} "
        "(ABSTAIN terminals are outside this population):"
    )
    payload["terminal_audit_title"] = "on-policy rollout terminal audit (not the weak-step teacher)"
    payload["populations"] = {
        "logged_grounded_stop": (
            "On-policy trajectory ends in STOP with s in {pass, corrected} and a finite "
            "terminal nutri_dish.energy_kcal. This is not the weak-step teacher audit set "
            "and not the #28/#31 hold-out. Policy ABSTAIN terminals are outside this population."
        ),
        "any_finite_horizon_energy": (
            "Every on-policy dish with a finite force-answer energy, including an ABSTAIN "
            "terminal whose earlier steps wrote a partial total. Dishes with no finite total "
            "are unscored and are not imputed as zero."
        ),
    }
    audit = payload.get("teacher_terminal_audit")
    if isinstance(audit, dict):
        audit["note"] = (
            "Terminal nutri_dish of on-policy STOP pass/corrected trajectories versus the "
            "cafe label. Not the weak-step teacher audit and not a second threshold search."
        )
    flags = payload.get("audit")
    if isinstance(flags, dict):
        flags["online_rollout"] = True
        flags["vlm"] = True
        flags["uses_test_gt_for_thresholds"] = False
        flags["rl"] = False
        flags["volume_tool"] = False
        flags["legacy_regression_weights"] = False
    scope = payload.setdefault("scope", {})
    if isinstance(scope, dict):
        scope["on_policy"] = (
            "Trajectories are an on-policy rollout of results/closed_loop/policy_state_mlp.npz. "
            "ours_mask_none re-prediction matches the executed action. B5 and obs masks are "
            "counterfactual reads of this trace, not separate VLM rollouts."
        )
        scope["offline_holdout"] = (
            "results/closed_loop/mae_aurc_holdout.json is the #28/#31 offline logged hold-out "
            "(definition offline_logged_horizon_force_answer, n=356, train labels). "
            "Those numbers are not rows of this on-policy rgb_test table."
        )
        scope["rgb_test"] = (
            "rgb_test dishes that have an overhead rgb.png. Official rgb_test ids without an "
            "image were not imputed. Cafe labels are used only to score. No threshold was fit."
        )
        scope["vlm"] = (
            "Local Qwen3.8-27B-VL via llama.cpp on 127.0.0.1:18080 (GPU 0): "
            "Qwen3.8-27B-UD-Q5_K_M.gguf + mmproj-BF16.gguf, alias qwen3.8-27b. "
            "Closed-loop perception only. Not faithful DietAI24 (B1) or FoodAgent (B2). "
            "Historical B0 97.03 is an anchor, not a row of this table. B1 110.72 is not faithful."
        )
    return payload


class ClosedLoopRollout:
    """One dish. ``policy.predict_actions`` is the only action source."""

    def __init__(
        self,
        dish_id: str,
        hypotheses: Sequence[Hyp],
        policy: Any,
        *,
        retrieve_fn: RetrieveFn,
        lookup_fn: LookupFn,
        portion_fn: PortionFn,
        catalog_path: str | None = None,
        image_path: str | Path | None = None,
        mass_fn: Callable[..., Any] | None = None,
        vlm_alias: str = "qwen3.8-27b",
        vlm_serving: str = "llama.cpp llama-server 127.0.0.1:18080",
    ) -> None:
        if dish_id not in rgb_test_ids():
            raise OnPolicyRefused(f"{dish_id} is not an official rgb_test id")
        self.dish_id = dish_id
        self.hyps = list(hypotheses)
        self.policy = policy
        self.retrieve_fn = retrieve_fn
        self.lookup_fn = lookup_fn
        self.portion_fn = portion_fn
        self.catalog_path = catalog_path
        self.image_path = None if image_path is None else Path(image_path)
        self.mass_fn = mass_fn
        self.vlm_alias = vlm_alias
        self.vlm_serving = vlm_serving
        self.finalized: list[dict[str, Any]] = []
        self.n_unresolved = 0
        self.corrected = False
        self.step_i = 0
        self.comp_i = 0
        self.query_i = 0
        self.stuck = 0
        self.phase = "open" if self.hyps else "terminal"
        self.live_code: str | None = None
        self.live_cands: list[dict[str, Any]] = []
        self.live_portion: float | None = None
        self.live_portion_src: str | None = None
        self.live_nutri: dict[str, Any] | None = None
        self.live_edges: list[dict[str, Any]] = []
        self.accepted_query = ""
        self.prev: dict[str, Any] | None = None
        self.steps: list[dict[str, Any]] = []
        self.ended_reason = "step_cap"

    def run(self) -> RolloutResult:
        while self.step_i < MAX_STEPS:
            before = self._signature()
            view = self._view()
            action = self._predict(view)
            obs, status, reason, dish_override = self._apply(action)
            record = self._record(view, action, obs, status, reason, dish_override)
            self.steps.append(record)
            self.prev = record
            self.step_i += 1
            if action in _TERMINAL_ACTIONS:
                self.ended_reason = "policy_stop" if action == "STOP" else "policy_abstain"
                break
            if self._signature() == before:
                self.stuck += 1
                if self.stuck >= STUCK_LIMIT:
                    self.ended_reason = "stuck"
                    break
            else:
                self.stuck = 0
        return RolloutResult(self.dish_id, self.steps, self.ended_reason, len(self.hyps))

    def _predict(self, view: Mapping[str, Any]) -> str:
        vec = decision_features(self.policy, view, self.prev, _belief_at_step(self.step_i))
        predicted = self.policy.predict_actions(np.asarray(vec, dtype=np.float64).reshape(1, -1))
        if len(predicted) != 1:
            raise OnPolicyRefused(f"{self.dish_id}: policy returned {predicted!r}")
        action = str(predicted[0])
        if action == "VOLUME_ESTIMATE" or action not in BC_ACTIONS:
            raise OnPolicyRefused(f"{self.dish_id}: action {action!r} is outside the BC head")
        return action

    def _base_obs(self, **extra: Any) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "label_quality": "vlm",
            "hypothesis_source": HYPOTHESIS_SOURCE,
            "rollout": ROLLOUT_ID,
            "protocol": PROTOCOL_ID,
            "privileged": False,
            "uses_test_gt": False,
            "volume_tool": False,
            "vlm_alias": self.vlm_alias,
            "vlm_weights": "Qwen3.8-27B-UD-Q5_K_M.gguf",
            "vlm_mmproj": "mmproj-BF16.gguf",
            "vlm_serving": self.vlm_serving,
            "vlm_n_hypotheses": len(self.hyps),
        }
        obs.update(extra)
        return obs

    def _on_component(self) -> bool:
        return self.phase == "open" and 0 <= self.comp_i < len(self.hyps)

    def _view(self) -> dict[str, Any]:
        if self._on_component():
            hyp: str | None = self.hyps[self.comp_i].name
            component_i: int | None = self.comp_i
            code = self.live_code
            cands = [dict(cand) for cand in self.live_cands]
            portion = None if self.live_portion is None else round(float(self.live_portion), 4)
            portion_src = self.live_portion_src if portion is not None else None
            nutri = None if self.live_nutri is None else dict(self.live_nutri)
            edges = copy.deepcopy(self.live_edges)
        else:
            hyp = None
            component_i = None
            code = None
            cands = []
            portion = None
            portion_src = None
            nutri = None
            edges = (
                [{"src": "dish", "dst": "n_dish", "type": "n<-aggregate"}]
                if self.finalized
                else []
            )
        parts = [dict(item) for item in self.finalized]
        if self._on_component() and self.live_nutri is not None:
            parts.append(dict(self.live_nutri))
        dish = _sum_nutri(parts)
        codes = verify(
            {
                "n": nutri,
                "cands_i": cands,
                "portion_g_i": portion,
                "evidence_edges": edges,
            }
        )
        return {
            "dish_id": self.dish_id,
            "step_i": self.step_i,
            "component_i": component_i,
            "hyp_i": hyp,
            "code_i": code,
            "cands_i": cands,
            "portion_g_i": portion,
            "portion_src_i": portion_src,
            "nutri_i": nutri,
            "nutri_dish": dish,
            "verify_codes": list(codes),
            "evidence_edges": edges,
        }

    def _signature(self) -> tuple[Any, ...]:
        portion = None if self.live_portion is None else round(float(self.live_portion), 4)
        cands = tuple((cand.get("code"), cand.get("score")) for cand in self.live_cands)
        hyp = self.hyps[self.comp_i].name if self._on_component() else None
        return (self.phase, self.comp_i, hyp, self.live_code, portion, self.query_i, cands, len(self.finalized))

    def _consume_query(self) -> str | None:
        if not self._on_component():
            return None
        name = self.hyps[self.comp_i].name.strip()
        queries = [name]
        rewrite = candidate_rewrite(name)
        if rewrite:
            queries.append(rewrite)
        index = min(self.query_i, len(queries) - 1)
        if self.query_i < len(queries) - 1:
            self.query_i += 1
        return queries[index]

    def _call_portion(self, code: str, hint_g: float) -> float | None:
        try:
            result = self.portion_fn(code, hint_g=hint_g, catalog_path=self.catalog_path)
        except KeyError:
            return None
        portion = getattr(result, "portion_g", None)
        if portion is None and isinstance(result, Mapping):
            portion = result.get("portion_g")
        if portion is None:
            return None
        return float(portion)

    def _apply(self, action: str) -> tuple[dict[str, Any], str | None, str | None, dict[str, Any] | None]:
        if action == "STOP":
            status = "corrected" if self.corrected else "pass"
            n_kept = len(self.finalized) + (1 if self._on_component() and self.live_nutri is not None else 0)
            return (
                self._base_obs(s=status, n_kept=n_kept, n_unresolved=self.n_unresolved),
                status,
                None,
                None,
            )
        if action == "ABSTAIN":
            reason = "no_vlm_hypothesis" if not self.hyps else "policy_abstain"
            return (
                self._base_obs(
                    s="abstain",
                    n_kept=len(self.finalized),
                    n_unresolved=self.n_unresolved,
                    abstain_reason=reason,
                ),
                "abstain",
                reason,
                _null_nutri(),
            )
        if action == "RETRIEVE":
            return self._retrieve(), None, None, None
        if action == "CORRECT_CODE":
            return self._correct(), None, None, None
        if action == "RE_PORTION":
            return self._reportion(), None, None, None
        if action == "RE_DECOMPOSE":
            return (
                self._base_obs(tool_failed=True, note="re_decompose_not_a_second_vlm_call"),
                None,
                None,
                None,
            )
        raise OnPolicyRefused(f"unhandled action {action}")

    def _retrieve(self) -> dict[str, Any]:
        query = self._consume_query()
        if query is None:
            return self._base_obs(tool_failed=True, error="no_hypothesis")
        found = self.retrieve_fn(query, top_k=TOP_K, catalog_path=self.catalog_path)
        cands = _public_cands(getattr(found, "cands", []) or [])[:TOP_K]
        top = str(cands[0]["code"]) if cands else None
        obs = self._base_obs(query=query, cands=cands, selected_code=top)
        if not top:
            obs["tool_failed"] = True
            return obs
        portion = self._call_portion(top, BLIND_PORTION_HINT_G)
        nutri = None if portion is None else _scaled(top, portion, self.lookup_fn, self.catalog_path)
        if portion is None or nutri is None:
            obs["tool_failed"] = True
            return obs
        self.live_code = top
        self.live_cands = cands
        self.live_portion = portion
        self.live_portion_src = "grid_search"
        self.live_nutri = nutri
        self.live_edges = _binding_edges(self.comp_i, top, query, portion)
        self.accepted_query = query
        return obs

    def _correct(self) -> dict[str, Any]:
        current = self.live_code or ""
        target = _alternate_code(self.live_cands, current or None)
        if not target or not self._on_component():
            return self._base_obs(tool_failed=True, **{"from": current, "to": current})
        portion = self.live_portion
        if portion is None:
            portion = self._call_portion(target, BLIND_PORTION_HINT_G)
        nutri = None if portion is None else _scaled(target, portion, self.lookup_fn, self.catalog_path)
        if portion is None or nutri is None:
            return self._base_obs(tool_failed=True, **{"from": current, "to": target})
        query = self.accepted_query or self.hyps[self.comp_i].name
        self.live_code = target
        self.live_portion = portion
        self.live_nutri = nutri
        self.live_portion_src = self.live_portion_src or "grid_search"
        self.live_edges = _binding_edges(self.comp_i, target, query, portion)
        self.corrected = True
        return self._base_obs(**{"from": current, "to": target})

    def _call_mass(self, candidates: Sequence[Mapping[str, Any]], hints: Sequence[float | None]) -> Any:
        """RE_PORTION's only portion source. The gram grid is not consulted."""
        fn = self.mass_fn
        if fn is None:
            from vdr.tools.mass_estimator import estimate_mass

            fn = estimate_mass
        return fn(
            self.image_path,
            candidates,
            component_index=self.comp_i,
            component_hints_g=hints,
            catalog_path=self.catalog_path,
        )

    def _reportion(self) -> dict[str, Any]:
        if not self.live_code or not self._on_component():
            return self._base_obs(tool_failed=True, error="no_code")
        from vdr.tools.mass_estimator import MassEstimatorError

        candidates = [dict(cand) for cand in self.live_cands]
        if not any(str(cand.get("code")) == str(self.live_code) for cand in candidates):
            candidates.insert(
                0,
                {"code": self.live_code, "score": 1.0, "name": self.hyps[self.comp_i].name},
            )
        hints = [hyp.portion_g for hyp in self.hyps]
        try:
            estimate = self._call_mass(candidates, hints)
        except MassEstimatorError as exc:
            return self._base_obs(
                tool_failed=True,
                error="mass_estimator_failed",
                mass_error=str(exc),
                mass_tool="estimate_mass",
                rgb_only=True,
                depth_used=False,
            )
        if isinstance(estimate, Mapping) and (
            estimate.get("rgb_only") is False or estimate.get("depth_used") is True
        ):
            return self._base_obs(
                tool_failed=True,
                error="mass_estimator_not_rgb_only",
                mass_tool="estimate_mass",
            )
        new_portion = _portion_grams(estimate)
        if new_portion is None:
            return self._base_obs(tool_failed=True, error="mass_estimator_no_portion", mass_tool="estimate_mass")
        new_portion = round(float(new_portion), 4)
        nutri = _scaled(self.live_code, new_portion, self.lookup_fn, self.catalog_path)
        if nutri is None:
            return self._base_obs(tool_failed=True, error="lookup_miss", mass_tool="estimate_mass")
        old_portion = 0.0 if self.live_portion is None else float(self.live_portion)
        query = self.accepted_query or self.hyps[self.comp_i].name
        self.live_portion = new_portion
        self.live_portion_src = "mass_estimator"
        self.live_nutri = nutri
        self.live_edges = _binding_edges(self.comp_i, self.live_code, query, new_portion)
        self.corrected = True
        self.finalized.append(dict(nutri))
        self._clear_live()
        self.comp_i += 1
        self.query_i = 0
        if self.comp_i >= len(self.hyps):
            self.phase = "terminal"
        obs = self._base_obs(
            portion_g=new_portion,
            from_portion_g=round(old_portion, 4),
            **_mass_obs(estimate),
        )
        return obs

    def _clear_live(self) -> None:
        self.live_code = None
        self.live_cands = []
        self.live_portion = None
        self.live_portion_src = None
        self.live_nutri = None
        self.live_edges = []
        self.accepted_query = ""

    def _record(
        self,
        view: Mapping[str, Any],
        action: str,
        obs: Mapping[str, Any],
        status: str | None,
        reason: str | None,
        dish_override: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        dish = view.get("nutri_dish") if dish_override is None else dict(dish_override)
        record = {
            "dish_id": self.dish_id,
            "step_i": int(view["step_i"]),
            "component_i": view["component_i"],
            "hyp_i": view["hyp_i"],
            "code_i": view["code_i"],
            "cands_i": copy.deepcopy(view["cands_i"]),
            "portion_g_i": view["portion_g_i"],
            "portion_src_i": view["portion_src_i"],
            "nutri_i": copy.deepcopy(view["nutri_i"]),
            "nutri_dish": copy.deepcopy(dish) if isinstance(dish, Mapping) else None,
            "verify_codes": list(view["verify_codes"]),
            "evidence_edges": copy.deepcopy(view["evidence_edges"]),
            "action_t": action,
            "obs_t": dict(obs),
            "s": status,
            "abstain_reason": reason,
        }
        validate_instance(record, _schema())
        assert_no_label_keys([record])
        return record


def _portion_grams(estimate: Any) -> float | None:
    if isinstance(estimate, Mapping):
        value = estimate.get("portion_g", estimate.get("mass_g"))
    else:
        value = getattr(estimate, "portion_g", None)
        if value is None:
            value = getattr(estimate, "mass_g", None)
    if value is None:
        return None
    return float(value)


def _mass_obs(estimate: Any) -> dict[str, Any]:
    """Fields copied onto the RE_PORTION observation. The grid is not a field."""
    obs: dict[str, Any] = {
        "portion_src": "mass_estimator",
        "portion_note": "estimate_mass_rgb_only",
        "mass_tool": "estimate_mass",
        "rgb_only": True,
        "depth_used": False,
    }
    if not isinstance(estimate, Mapping):
        return obs
    sigma = estimate.get("sigma") if isinstance(estimate.get("sigma"), Mapping) else {}
    intervals = estimate.get("intervals") if isinstance(estimate.get("intervals"), Mapping) else {}
    obs.update(
        {
            "mass_g": estimate.get("mass_g"),
            "mass_sigma": sigma.get("mass_g"),
            "sigma": {str(key): sigma[key] for key in sigma},
            "mass_interval": intervals.get("mass_g"),
            "mass_confidence": estimate.get("confidence"),
            "nutrient_refs": estimate.get("nutrient_refs"),
            "mass_backbone": estimate.get("backbone"),
            "nutrients_model": estimate.get("nutrients"),
        }
    )
    return obs


def rollout_dish(
    dish_id: str,
    policy: Any,
    *,
    image_path: str | Path | None = None,
    perceive_fn: PerceiveFn | None = None,
    retrieve_fn: RetrieveFn | None = None,
    lookup_fn: LookupFn | None = None,
    portion_fn: PortionFn | None = None,
    mass_fn: Callable[..., Any] | None = None,
    catalog_path: str | None = None,
    check_replay: bool = True,
) -> RolloutResult:
    """Roll one rgb_test dish. The default perception function calls local Qwen3.8."""
    if dish_id not in rgb_test_ids():
        raise OnPolicyRefused(f"{dish_id} is not an official rgb_test id")
    if perceive_fn is None:
        from vdr.perception.qwen38 import ingredient_hypotheses

        perceive_fn = ingredient_hypotheses
        if image_path is None or not Path(image_path).is_file():
            raise OnPolicyRefused(f"{dish_id}: missing overhead image")
    if retrieve_fn is None or lookup_fn is None or portion_fn is None:
        from vdr.tools.fndds_retrieve import retrieve
        from vdr.tools.nutrition_lookup import lookup_per_100g
        from vdr.tools.portion_grid import search_portion

        retrieve_fn = retrieve_fn or retrieve
        lookup_fn = lookup_fn or lookup_per_100g
        portion_fn = portion_fn or search_portion
    raw, meta = perceive_fn(image_path)
    if not isinstance(meta, Mapping):
        raise OnPolicyRefused(f"{dish_id}: perception meta is not a mapping")
    alias = str(meta.get("model_alias") or "qwen3.8-27b")
    serving = str(meta.get("serving") or "llama.cpp llama-server 127.0.0.1:18080")
    env = ClosedLoopRollout(
        dish_id,
        prepare_hypotheses(raw),
        policy,
        retrieve_fn=retrieve_fn,
        lookup_fn=lookup_fn,
        portion_fn=portion_fn,
        catalog_path=catalog_path,
        image_path=image_path,
        mass_fn=mass_fn,
        vlm_alias=alias,
        vlm_serving=serving,
    )
    result = env.run()
    assert_on_policy_marker({dish_id: result.steps})
    if check_replay:
        assert_replay_matches(result.steps, policy)
    return result


def _finite_dish_kcal(steps: Sequence[Mapping[str, Any]]) -> float | None:
    last = steps[-1] if steps else None
    if last is None:
        return None
    nutri = last.get("nutri_dish")
    if not isinstance(nutri, Mapping):
        return None
    value = nutri.get("energy_kcal")
    if value is None:
        return None
    return float(value)


def run_self_check() -> None:
    """CPU contract. No HTTP, no cafe labels, no rgb_test nutrition."""
    import vdr.data.n5k as n5k

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("on-policy self-check tried to read dish labels or ingredient metadata")

    n5k.load_labels = _boom  # type: ignore[assignment]
    n5k.load_dish_metadata = _boom  # type: ignore[assignment]

    dish_id = sorted(rgb_test_ids())[0]
    hints: list[float] = []

    def retrieve_fn(query: str, top_k: int = 5, catalog_path: str | None = None) -> Any:
        del top_k, catalog_path

        class Result:
            cands = [
                {"code": "111", "score": 0.91, "name": "Apple raw"},
                {"code": "222", "score": 0.80, "name": "Apple juice"},
            ]
            selected_code = "111"

        result = Result()
        result.query = query  # type: ignore[attr-defined]
        return result

    def lookup_fn(code: str, catalog_path: str | None = None) -> dict[str, float]:
        del code, catalog_path
        return {"energy_kcal": 100.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0}

    def portion_fn(code: str, hint_g: float = 100.0, catalog_path: str | None = None) -> Any:
        del code, catalog_path
        hints.append(float(hint_g))

        class Result:
            pass

        result = Result()
        result.portion_g = 100.0 if abs(float(hint_g) - 100.0) < 1e-6 else 175.0
        return result

    class ListPolicy:
        def __init__(self, actions: list[str]) -> None:
            self.actions = actions
            self.i = 0

        def predict_actions(self, _x: Any) -> list[str]:
            action = self.actions[self.i]
            self.i += 1
            return [action]

    mass_calls: list[dict[str, Any]] = []

    def mass_fn(
        image: Any,
        candidates: Sequence[Mapping[str, Any]],
        component_index: int | None = None,
        component_hints_g: Sequence[float | None] | None = None,
        catalog_path: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del catalog_path
        mass_calls.append(
            {
                "image": None if image is None else str(image),
                "codes": [str(cand.get("code")) for cand in candidates],
                "component_index": component_index,
                "hints": [None if hint is None else float(hint) for hint in (component_hints_g or [])],
            }
        )
        return {
            "portion_g": 210.0,
            "mass_g": 210.0,
            "sigma": {"mass_g": 12.0},
            "intervals": {"mass_g": {"low": 170.0, "high": 250.0, "coverage": 0.95}},
            "confidence": 0.89,
            "nutrient_refs": {"model_rgb": {"energy_kcal": 180.0}, "fndds": []},
            "nutrients": {"energy_kcal": 180.0, "protein_g": 2.0, "carb_g": 20.0, "fat_g": 1.0},
            "rgb_only": True,
            "depth_used": False,
            "backbone": "stub",
            "portion_src": "mass_estimator",
        }

    def _run(
        actions: list[str],
        hyps: list[dict[str, Any]],
        *,
        image_path: str | None = None,
        mass: Any = mass_fn,
    ) -> RolloutResult:
        hints.clear()
        mass_calls.clear()
        return rollout_dish(
            dish_id,
            ListPolicy(actions),
            image_path=image_path,
            perceive_fn=lambda _path: (hyps, {"model_alias": "qwen3.8-27b", "serving": "self-check"}),
            retrieve_fn=retrieve_fn,
            lookup_fn=lookup_fn,
            portion_fn=portion_fn,
            mass_fn=mass,
            check_replay=False,
        )

    apple = [{"name": "apple", "portion_g": 180}]
    rolled = _run(["RETRIEVE", "RE_PORTION", "STOP"], apple, image_path="overhead/rgb.png")
    actions = [step["action_t"] for step in rolled.steps]
    if actions != ["RETRIEVE", "RE_PORTION", "STOP"]:
        raise AssertionError(actions)
    if rolled.ended_reason != "policy_stop" or rolled.steps[-1]["s"] != "corrected":
        raise AssertionError(rolled.ended_reason)
    terminal = rolled.steps[-1]["nutri_dish"]["energy_kcal"]
    # 210 g from estimate_mass, not the grid's 175 g for a 180 g hint.
    if abs(float(terminal) - 210.0) > 1e-6:
        raise AssertionError(terminal)
    if hints != [100.0]:
        raise AssertionError(hints)
    if len(mass_calls) != 1 or "111" not in mass_calls[0]["codes"]:
        raise AssertionError(mass_calls)
    if mass_calls[0]["hints"] != [180.0] or not str(mass_calls[0]["image"]).endswith("rgb.png"):
        raise AssertionError(mass_calls)
    portion_obs = rolled.steps[1]["obs_t"]
    if portion_obs.get("portion_src") != "mass_estimator" or portion_obs.get("mass_tool") != "estimate_mass":
        raise AssertionError(portion_obs)
    if portion_obs.get("rgb_only") is not True or portion_obs.get("depth_used") is not False:
        raise AssertionError(portion_obs)
    if abs(float(portion_obs["portion_g"]) - 210.0) > 1e-6:
        raise AssertionError(portion_obs["portion_g"])
    if rolled.steps[-1]["obs_t"]["privileged"] is not False:
        raise AssertionError("privileged observation")

    def boom(*_args: Any, **_kwargs: Any) -> None:
        from vdr.tools.mass_estimator import MassEstimatorError

        raise MassEstimatorError("checkpoint missing")

    failed = _run(["RETRIEVE", "RE_PORTION", "STOP"], apple, mass=boom)
    if failed.steps[1]["obs_t"].get("tool_failed") is not True:
        raise AssertionError(failed.steps[1]["obs_t"])
    if hints != [100.0]:
        raise AssertionError(hints)

    stuck = _run(["RETRIEVE", "RE_DECOMPOSE", "RE_DECOMPOSE", "RE_DECOMPOSE", "STOP"], apple)
    if "STOP" in [step["action_t"] for step in stuck.steps] or stuck.ended_reason != "stuck":
        raise AssertionError((stuck.ended_reason, [step["action_t"] for step in stuck.steps]))

    abstain = _run(["ABSTAIN"], apple)
    if abstain.steps[0]["s"] != "abstain" or abstain.steps[0]["nutri_dish"]["energy_kcal"] is not None:
        raise AssertionError("abstain should null the terminal total")
    if abstain.steps[0]["abstain_reason"] != "policy_abstain":
        raise AssertionError(abstain.steps[0]["abstain_reason"])

    empty = _run(["STOP"], [])
    if empty.steps[0]["s"] != "pass" or empty.n_hypotheses != 0:
        raise AssertionError("empty perception should still let the policy stop")

    merged = prepare_hypotheses(
        [{"name": "Cheese", "portion_g": 40}, {"name": "Crust", "portion_g": 80}]
    )
    if len(merged) != 1 or "crust" not in merged[0].name.casefold():
        raise AssertionError(merged)

    root = Path(__file__).resolve().parents[2]
    weights = root / "results" / "closed_loop" / "policy_state_mlp.npz"
    meta = root / "results" / "closed_loop" / "train_meta.json"
    policy = load_closed_loop_policy(weights, meta)
    live = rollout_dish(
        dish_id,
        policy,
        image_path=None,
        perceive_fn=lambda _path: (apple, {"model_alias": "qwen3.8-27b", "serving": "self-check"}),
        retrieve_fn=retrieve_fn,
        lookup_fn=lookup_fn,
        portion_fn=portion_fn,
        mass_fn=mass_fn,
        check_replay=True,
    )
    if not live.steps or live.steps[0]["obs_t"]["hypothesis_source"] != HYPOTHESIS_SOURCE:
        raise AssertionError("real policy rollout missed the on-policy marker")

    payload = {
        "definition_id": "offline_logged_horizon_force_answer",
        "definition": "old",
        "populations": {},
        "teacher_terminal_audit": {"n": 1, "note": "old", "mae_kcal": 3.0},
        "audit": {
            "online_rollout": False,
            "vlm": False,
            "uses_test_gt_for_thresholds": False,
            "rl": False,
            "volume_tool": False,
        },
        "scope": {},
        "methods": {"ours_mask_none": {"primary_logged_grounded_stop": {"full_set_mae_kcal": 12.5}}},
    }
    annotate_on_policy_table(payload, {dish_id: live.steps})
    if payload["definition_id"] != ON_POLICY_DEFINITION_ID:
        raise AssertionError(payload["definition_id"])
    if payload["methods"]["ours_mask_none"]["primary_logged_grounded_stop"]["full_set_mae_kcal"] != 12.5:
        raise AssertionError("annotate changed a numeric block")
    if payload["audit"]["online_rollout"] is not True or payload["audit"]["vlm"] is not True:
        raise AssertionError(payload["audit"])
    try:
        assert_on_policy_marker(
            {
                dish_id: [
                    {
                        "action_t": "STOP",
                        "obs_t": {
                            "rollout": "teacher",
                            "protocol": "weak",
                            "hypothesis_source": "n5k_ingredient_metadata",
                            "privileged": True,
                            "uses_test_gt": False,
                            "volume_tool": False,
                        },
                    }
                ]
            }
        )
    except OnPolicyRefused:
        pass
    else:
        raise AssertionError("teacher logs were accepted as on-policy")
