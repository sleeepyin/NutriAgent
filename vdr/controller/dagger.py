"""One DAgger round on the train-only weak teacher.

The learner is ``PolicyMLP``. The expert is the rule teacher in
``vdr.weak.teacher``: the same retrieve, blind portion grid, ``choose_code``,
and terminal rule. Train ingredient mass and kcal are used only inside that
teacher query. They are not written into the feature dict.

``build_features`` still reads only the pre-action fields. ``obs_t`` is the
tool return and is not a feature. ``VOLUME_ESTIMATE`` is not an action.
Rollouts stay on the ``rgb_train`` fit pool. This module does not open the
hold-out jsonl, ``rgb_test``, or ``depth_test``.

A visited state is appended when its feature key is new, or when the learner
action differs from the teacher. Agreeing states that are already in the BC
matrix are not copied again.

``DAGGER_MAX_STEPS`` and ``DAGGER_STUCK_LIMIT`` are fixed safety caps. They
were not chosen on the hold-out or on rgb_test. A teacher trajectory is at
most 32 steps, so the expert path finishes inside the cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vdr.controller.features import FEATURE_DIM, PRE_ACTION_FIELDS, build_features
from vdr.controller.policy_bc import BC_ACTIONS
from vdr.data.n5k import DishMeta, test_split_ids
from vdr.weak.teacher import (
    PORTION_REL_OK,
    TOP_K,
    UNRESOLVED_ENERGY_SHARE,
    TestSplitRefused,
    _binding_edges,
    _blind_portion,
    _null_nutri,
    _obs,
    _public_cands,
    _scaled,
    _step,
    _sum_nutri,
    _verify,
    candidate_rewrite,
    choose_code,
    keep_ingredient,
    pool_of_dish,
    select_train_dishes,
    snap_to_grid,
)

# Safety caps. Not a hold-out search. Teacher dishes emit at most 32 steps.
DAGGER_MAX_STEPS = 48
DAGGER_STUCK_LIMIT = 3
_FEATURE_KEY_DECIMALS = 6

_LEARNER_ACTIONS = frozenset(BC_ACTIONS)


class DaggerPoolRefused(ValueError):
    """Rollout asked for a dish outside the rgb_train fit pool."""


@dataclass(frozen=True)
class DaggerRoundResult:
    """New rows from one learner rollout. Hold-out is not in here."""

    n_dishes: int
    n_dishes_rolled: int
    n_dishes_skipped_agree: int
    n_visited: int
    n_mismatch: int
    n_new_rows: int
    n_novel_rows: int
    n_mismatch_copies: int
    n_duplicate_states: int
    n_label_conflicts: int
    new_X: np.ndarray
    new_y: np.ndarray
    new_dish_ids: tuple[str, ...]
    max_steps: int
    stuck_limit: int

    @property
    def uses_test_gt(self) -> bool:
        return False

    @property
    def volume_tool(self) -> bool:
        return False


def feature_key(vector: np.ndarray) -> tuple[float, ...]:
    """Rounded feature identity. Same rounding the BC smoke uses for ABSTAIN."""
    row = np.asarray(vector, dtype=np.float64).reshape(-1)
    if row.shape != (FEATURE_DIM,):
        raise ValueError(f"feature key expects {FEATURE_DIM} dims, got {row.shape}")
    return tuple(np.round(row, _FEATURE_KEY_DECIMALS).tolist())


def expert_ir_steps(
    dish: DishMeta,
    *,
    catalog_path: str | None = None,
    retrieve_fn: Callable[..., Any] | None = None,
    lookup_fn: Callable[..., Any] | None = None,
    portion_fn: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """IR steps from always taking the teacher action.

    These match ``generate_dish_steps`` when the env is faithful. The caller
    uses that equality as the regression check. ``obs_t`` on these records is
    not fed to ``build_features``.
    """
    env = WeakTeacherEnv(
        dish,
        record_ir=True,
        catalog_path=catalog_path,
        retrieve_fn=retrieve_fn,
        lookup_fn=lookup_fn,
        portion_fn=portion_fn,
    )
    while not env.done:
        env.step(env.expert_action())
    return env.emitted


def rollout_dish(
    dish: DishMeta,
    policy: Any,
    *,
    max_steps: int = DAGGER_MAX_STEPS,
    catalog_path: str | None = None,
    retrieve_fn: Callable[..., Any] | None = None,
    lookup_fn: Callable[..., Any] | None = None,
    portion_fn: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Step one fit-pool dish under ``policy``. Each visit stores ``a*``.

    The learner action drives the transition. The label is the teacher action
    on the pre-action state. ``STOP`` and ``ABSTAIN`` end the dish.
    """
    _assert_fit_dish(dish)
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    env = WeakTeacherEnv(
        dish,
        record_ir=False,
        max_steps=max_steps,
        catalog_path=catalog_path,
        retrieve_fn=retrieve_fn,
        lookup_fn=lookup_fn,
        portion_fn=portion_fn,
    )
    visits: list[dict[str, Any]] = []
    while not env.done and len(visits) < max_steps:
        state = env.feature_state()
        if set(state) != set(PRE_ACTION_FIELDS):
            raise RuntimeError(f"feature state keys drifted: {sorted(state)}")
        expert = env.expert_action()
        learner = _learner_action(policy, state)
        visits.append(
            {
                "dish_id": dish.dish_id,
                "state": state,
                "expert_action": expert,
                "learner_action": learner,
            }
        )
        env.step(learner)
    return visits


def run_dagger_round(
    policy: Any,
    dishes: Sequence[DishMeta],
    fit_X: np.ndarray,
    fit_y: Sequence[str],
    fit_dish_ids: Sequence[str],
    *,
    max_steps: int = DAGGER_MAX_STEPS,
    catalog_path: str | None = None,
) -> DaggerRoundResult:
    """Aggregate BC with teacher labels from one learner rollout.

    A visited state is appended when its feature key is new, or when the
    learner action differs from the teacher. Agreeing states already in the
    fit matrix are left as they are. Dishes whose fit-row predictions already
    match the teacher are not re-rolled. Hold-out rows are refused.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    matrix = np.asarray(fit_X, dtype=np.float64)
    labels = [str(item) for item in list(fit_y)]
    ids = [str(item) for item in list(fit_dish_ids)]
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
        raise ValueError(f"fit_X must have shape (n, {FEATURE_DIM})")
    if not (len(labels) == len(ids) == matrix.shape[0]):
        raise ValueError("fit_X, fit_y, and fit_dish_ids must have the same length")
    if not dishes:
        raise ValueError("run_dagger_round needs at least one fit dish")

    by_dish: dict[str, list[int]] = {}
    known: dict[tuple[float, ...], set[str]] = {}
    for index, (row, label, dish_id) in enumerate(zip(matrix, labels, ids)):
        by_dish.setdefault(dish_id, []).append(index)
        known.setdefault(feature_key(row), set()).add(label)

    preds = _predict_rows(policy, matrix)
    new_vectors: list[np.ndarray] = []
    new_labels: list[str] = []
    new_ids: list[str] = []
    n_rolled = 0
    n_skipped = 0
    n_visited = 0
    n_mismatch = 0
    n_novel = 0
    n_copies = 0
    n_dup = 0
    n_conflict = 0

    for dish in dishes:
        _assert_fit_dish(dish)
        rows = by_dish.get(dish.dish_id)
        if not rows:
            raise DaggerPoolRefused(
                f"{dish.dish_id} has no fit-pool weak-step rows to roll out from"
            )
        if all(preds[index] == labels[index] for index in rows):
            n_skipped += 1
            continue
        n_rolled += 1
        if len(dishes) >= 64 and (n_rolled == 1 or n_rolled % 50 == 0):
            print(
                f"dagger rollout {n_rolled} fit dishes so far, new_rows={len(new_labels)}",
                flush=True,
            )
        visits = rollout_dish(
            dish,
            policy,
            max_steps=max_steps,
            catalog_path=catalog_path,
        )
        for visit in visits:
            n_visited += 1
            expert = visit["expert_action"]
            if visit["learner_action"] != expert:
                n_mismatch += 1
            vector = build_features(visit["state"])
            key = feature_key(vector)
            have = known.get(key)
            mismatch = visit["learner_action"] != expert
            # Novel h, or a learner mistake. Agreeing states already in the
            # fit matrix stay there once. A mistake is appended even when h
            # is already a BC row: that is the extra copy DAgger queries.
            if have is None or mismatch:
                if have is None:
                    n_novel += 1
                    known[key] = {expert}
                else:
                    n_copies += 1
                    if expert not in have:
                        n_conflict += 1
                        have.add(expert)
                new_vectors.append(vector)
                new_labels.append(expert)
                new_ids.append(dish.dish_id)
            else:
                n_dup += 1

    if new_vectors:
        new_X = np.vstack(new_vectors)
    else:
        new_X = np.zeros((0, FEATURE_DIM), dtype=np.float64)
    return DaggerRoundResult(
        n_dishes=len(dishes),
        n_dishes_rolled=n_rolled,
        n_dishes_skipped_agree=n_skipped,
        n_visited=n_visited,
        n_mismatch=n_mismatch,
        n_new_rows=int(new_X.shape[0]),
        n_novel_rows=n_novel,
        n_mismatch_copies=n_copies,
        n_duplicate_states=n_dup,
        n_label_conflicts=n_conflict,
        new_X=new_X,
        new_y=np.asarray(new_labels, dtype=object),
        new_dish_ids=tuple(new_ids),
        max_steps=max_steps,
        stuck_limit=DAGGER_STUCK_LIMIT,
    )


def stack_aggregate(
    fit_X: np.ndarray,
    fit_y: Sequence[str],
    result: DaggerRoundResult,
) -> tuple[np.ndarray, np.ndarray]:
    """BC rows plus new DAgger rows. Does not touch the hold-out."""
    labels = np.asarray([str(item) for item in list(fit_y)], dtype=object)
    if result.n_new_rows == 0:
        return np.asarray(fit_X, dtype=np.float64), labels
    return np.vstack([np.asarray(fit_X, dtype=np.float64), result.new_X]), np.concatenate(
        [labels, result.new_y]
    )


def action_hits(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Per-action ``(hits, support)`` over ``BC_ACTIONS``."""
    truth = [str(item) for item in y_true]
    pred = [str(item) for item in y_pred]
    if len(truth) != len(pred):
        raise ValueError("y_true and y_pred differ in length")
    out: dict[str, tuple[int, int]] = {}
    for action in BC_ACTIONS:
        idx = [i for i, label in enumerate(truth) if label == action]
        hit = sum(1 for i in idx if pred[i] == action)
        out[action] = (hit, len(idx))
    return out


def fit_pool_dishes(n: int | None = None) -> list[DishMeta]:
    """Official-order rgb_train fit dishes. ``n`` is a prefix, not a sample."""
    if n is not None and n < 1:
        raise ValueError("n must be >= 1")
    return select_train_dishes(split="rgb_train", pool="fit", n=n, seed=None)


class WeakTeacherEnv:
    """Interactive copy of the weak-teacher transition.

    When every action is the expert action, the emitted IR matches
    ``generate_dish_steps``. A different learner action applies the same
    tools and then asks the teacher again. Train grams used to snap a
    learner ``RE_PORTION`` stay inside the teacher bookkeeping. The next
    pre-action ``portion_g_i`` is not that snapped mass: the next public
    state is the next component, or the terminal step, both with
    ``portion_g_i`` empty. That matches the teacher trajectory, where the
    snapped grams live in ``obs_t`` only.
    """

    def __init__(
        self,
        dish: DishMeta,
        *,
        record_ir: bool = False,
        max_steps: int = DAGGER_MAX_STEPS,
        catalog_path: str | None = None,
        retrieve_fn: Callable[..., Any] | None = None,
        lookup_fn: Callable[..., Any] | None = None,
        portion_fn: Callable[..., Any] | None = None,
    ) -> None:
        _assert_fit_dish(dish)
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.dish = dish
        self.record_ir = record_ir
        self.max_steps = max_steps
        self.catalog_path = catalog_path
        self.retrieve_fn = retrieve_fn
        self.lookup_fn = lookup_fn
        self.portion_fn = portion_fn
        self.emitted: list[dict[str, Any]] = []
        self.kept = [ing for ing in dish.ingredients if keep_ingredient(ing)]
        self.finalized: list[dict[str, Any]] = []
        self.unresolved_kcal = 0.0
        self.n_unresolved = 0
        self.correction = False
        self.step_i = 0
        self.comp_i = 0
        self.stuck = 0
        self.phase = "HALT"
        self.queries: list[str] = []
        self.query_i = 0
        self.failed_top: str | None = None
        self.failed_cands: list[dict[str, Any]] = []
        self.failed_query = ""
        self.accepted_code = ""
        self.accepted_decision = "none"
        self.accepted_cands: list[dict[str, Any]] = []
        self.accepted_query = ""
        self.accepted_top: str | None = None
        self.live_code: str | None = None
        self.live_portion: float | None = None
        self.live_nutri: dict[str, Any] | None = None
        self.live_edges: list[dict[str, Any]] = []
        self.oracle: float | None = None
        self.terminal_action = "STOP"
        self.terminal_status: str | None = None
        self.terminal_reason: str | None = None
        self.hyp: str | None = None
        self.code: str | None = None
        self.cands: list[dict[str, Any]] = []
        self.portion: float | None = None
        self.portion_src: str | None = None
        self.nutri: dict[str, Any] | None = None
        self.nutri_dish: dict[str, Any] | None = None
        self.edges: list[dict[str, Any]] = []
        self.component_i: int | None = None
        self.status: str | None = None
        self.abstain_reason: str | None = None
        self.verify: list[str] = []
        if not self.kept:
            self.terminal_action = "ABSTAIN"
            self.terminal_status = "abstain"
            self.terminal_reason = "no_substantial_ingredient"
            self.phase = "TERMINAL"
            self._set_public(
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
                status="abstain",
                abstain_reason="no_substantial_ingredient",
            )
        else:
            self._prepare_fresh_retrieve()

    @property
    def done(self) -> bool:
        return self.phase == "HALT"

    def expert_action(self) -> str:
        if self.phase == "HALT":
            raise RuntimeError("expert_action on a finished dish")
        if self.phase == "RETRIEVE":
            return "RETRIEVE"
        if self.phase == "CORRECT":
            return "CORRECT_CODE"
        if self.phase == "PORTION":
            return "RE_PORTION"
        if self.phase == "TERMINAL":
            return self.terminal_action
        raise RuntimeError(f"unknown teacher phase {self.phase}")

    def feature_state(self) -> dict[str, Any]:
        """Pre-action fields only. No ``obs_t``, no train GT."""
        portion = None if self.portion is None else round(float(self.portion), 4)
        return {
            "hyp_i": self.hyp,
            "code_i": self.code,
            "cands_i": [dict(cand) for cand in self.cands],
            "portion_g_i": portion,
            "verify_codes": list(self.verify),
            "step_i": self.step_i,
        }

    def step(self, action: str) -> None:
        if self.done:
            raise RuntimeError("step on a finished dish")
        if action == "VOLUME_ESTIMATE" or action not in _LEARNER_ACTIONS:
            raise ValueError(f"DAgger action must be a BC action, got {action!r}")
        if action == self.expert_action():
            self._commit_expert()
            self.stuck = 0
        else:
            before = self._signature()
            self._commit_learner(action)
            if not self.done and self._signature() == before:
                self.stuck += 1
                if self.stuck >= DAGGER_STUCK_LIMIT:
                    self.phase = "HALT"
            elif not self.done:
                self.stuck = 0
        if not self.done and self.step_i >= self.max_steps:
            self.phase = "HALT"

    def _signature(self) -> tuple[Any, ...]:
        portion = None if self.portion is None else round(float(self.portion), 4)
        cands = tuple((cand.get("code"), cand.get("score")) for cand in self.cands)
        return (
            self.phase,
            self.comp_i,
            self.hyp,
            self.code,
            portion,
            tuple(self.verify),
            cands,
            self.query_i,
        )

    def _commit_expert(self) -> None:
        if self.phase == "RETRIEVE":
            self._commit_retrieve()
        elif self.phase == "CORRECT":
            self._commit_correct()
        elif self.phase == "PORTION":
            self._commit_portion()
        elif self.phase == "TERMINAL":
            self._commit_terminal()
        else:
            raise RuntimeError(f"cannot commit phase {self.phase}")

    def _commit_learner(self, action: str) -> None:
        if action in ("STOP", "ABSTAIN"):
            self.phase = "HALT"
            return
        if self.phase == "TERMINAL":
            self.step_i += 1
            return
        if action == "RE_DECOMPOSE":
            self.step_i += 1
            self._prepare_fresh_retrieve()
            return
        if action == "RETRIEVE":
            self._learner_retrieve()
            return
        if action == "CORRECT_CODE":
            self._learner_correct()
            return
        if action == "RE_PORTION":
            self._learner_portion()
            return
        raise RuntimeError(f"unhandled learner action {action}")

    def _learner_retrieve(self) -> None:
        """One extra retrieve. The current phase already refused RETRIEVE."""
        self.step_i += 1
        ing = self.kept[self.comp_i]
        nxt = self.query_i + 1
        if nxt < len(self.queries):
            self.query_i = nxt
            query = self.queries[nxt]
        else:
            query = self.queries[self.query_i]
        self.correction = True
        cands, top = self._retrieve(query)
        code, decision = choose_code(cands, ing, self._lookup(), self.catalog_path)
        if decision == "none":
            self.failed_top = top
            self.failed_cands = cands
            self.failed_query = query
            if self.query_i + 1 < len(self.queries):
                self.query_i += 1
                self._prepare_second_retrieve(ing)
            else:
                self._mark_unresolved(ing)
                self._advance()
            return
        self.accepted_code = code
        self.accepted_decision = decision
        self.accepted_cands = cands
        self.accepted_query = query
        self.accepted_top = top
        self._after_accept(ing)

    def _learner_correct(self) -> None:
        self.step_i += 1
        if not self.cands or self.comp_i >= len(self.kept):
            return
        ing = self.kept[self.comp_i]
        code, decision = choose_code(self.cands, ing, self._lookup(), self.catalog_path)
        public = self.code or ""
        if decision != "correct" or not code or code == public:
            return
        self.correction = True
        self.accepted_code = code
        self.accepted_decision = decision
        self.accepted_cands = [dict(cand) for cand in self.cands]
        self.accepted_top = str(self.cands[0]["code"])
        query = self.accepted_query or (self.queries[self.query_i] if self.queries else ing.name.strip())
        self.accepted_query = query
        portion, nutri, edges = self._realize(code, query)
        if portion is None or nutri is None:
            self._mark_unresolved(ing)
            self._advance()
            return
        self.live_code = code
        self.live_portion = portion
        self.live_nutri = nutri
        self.live_edges = edges
        self._after_code_fixed(ing)

    def _learner_portion(self) -> None:
        """Snap the public code with the teacher gram rule, then leave the component.

        The snapped grams are not copied into the next ``portion_g_i``.
        """
        self.step_i += 1
        if not self.code or self.comp_i >= len(self.kept):
            return
        ing = self.kept[self.comp_i]
        oracle = snap_to_grid(float(ing.mass_g))
        nutri = _scaled(self.code, oracle, self._lookup(), self.catalog_path)
        if nutri is None:
            self._mark_unresolved(ing)
            self._advance()
            return
        self.correction = True
        self.live_code = self.code
        self.finalized.append(nutri)
        self._advance()

    def _commit_retrieve(self) -> None:
        ing = self.kept[self.comp_i]
        query = self.queries[self.query_i]
        cands, top_code = self._retrieve(query)
        self._emit(
            "RETRIEVE",
            _obs(query=query, cands=cands, selected_code=top_code),
        )
        code, decision = choose_code(cands, ing, self._lookup(), self.catalog_path)
        if decision != "none":
            self.accepted_code = code
            self.accepted_decision = decision
            self.accepted_cands = cands
            self.accepted_query = query
            self.accepted_top = top_code
            self._after_accept(ing)
            return
        self.failed_top = top_code
        self.failed_cands = cands
        self.failed_query = query
        self.query_i += 1
        if self.query_i >= len(self.queries):
            self._mark_unresolved(ing)
            self._advance()
            return
        self._prepare_second_retrieve(ing)

    def _commit_correct(self) -> None:
        ing = self.kept[self.comp_i]
        self.correction = True
        self._emit(
            "CORRECT_CODE",
            _obs(**{"from": self.live_code, "to": self.accepted_code}),
        )
        code_now = self.accepted_code
        portion, nutri, edges = self._realize(code_now, self.accepted_query)
        if portion is None or nutri is None:
            self._mark_unresolved(ing)
            self._advance()
            return
        self.live_code = code_now
        self.live_portion = portion
        self.live_nutri = nutri
        self.live_edges = edges
        self._after_code_fixed(ing)

    def _commit_portion(self) -> None:
        ing = self.kept[self.comp_i]
        oracle = snap_to_grid(float(ing.mass_g))
        self.correction = True
        self._emit(
            "RE_PORTION",
            _obs(
                portion_g=round(float(oracle), 4),
                from_portion_g=round(float(self.live_portion or 0.0), 4),
                portion_src="other",
                portion_note="n5k_ingredient_mass_snapped_to_grid",
            ),
        )
        nutri = _scaled(self.live_code or "", oracle, self._lookup(), self.catalog_path)
        if nutri is None:
            self._mark_unresolved(ing)
            self._advance()
            return
        self.finalized.append(nutri)
        self._advance()

    def _commit_terminal(self) -> None:
        self._emit(
            self.terminal_action,
            _obs(
                s=self.terminal_status,
                n_kept=len(self.kept),
                n_unresolved=self.n_unresolved,
            ),
            status=self.terminal_status,
            abstain_reason=self.terminal_reason,
        )
        self.phase = "HALT"

    def _after_accept(self, ing: Any) -> None:
        code_now = self.accepted_top
        if not code_now:
            self._mark_unresolved(ing)
            self._advance()
            return
        portion, nutri, edges = self._realize(code_now, self.accepted_query)
        if portion is None or nutri is None:
            self._mark_unresolved(ing)
            self._advance()
            return
        self.live_code = code_now
        self.live_portion = portion
        self.live_nutri = nutri
        self.live_edges = edges
        if self.accepted_decision == "correct" and self.accepted_code != code_now:
            self.phase = "CORRECT"
            self._set_public(
                component_i=self.comp_i,
                hyp=ing.name,
                code=code_now,
                cands=self.accepted_cands,
                portion=portion,
                portion_src="grid_search",
                nutri=nutri,
                nutri_dish=_sum_nutri([*self.finalized, nutri]),
                edges=edges,
                status=None,
                abstain_reason=None,
            )
            return
        self._after_code_fixed(ing)

    def _after_code_fixed(self, ing: Any) -> None:
        oracle = snap_to_grid(float(ing.mass_g))
        self.oracle = oracle
        portion_now = float(self.live_portion or 0.0)
        if abs(portion_now - oracle) / max(oracle, 1.0) > PORTION_REL_OK:
            self.phase = "PORTION"
            self._set_public(
                component_i=self.comp_i,
                hyp=ing.name,
                code=self.live_code,
                cands=self.accepted_cands,
                portion=self.live_portion,
                portion_src="grid_search",
                nutri=self.live_nutri,
                nutri_dish=_sum_nutri([*self.finalized, self.live_nutri]),
                edges=self.live_edges,
                status=None,
                abstain_reason=None,
            )
            return
        if self.live_nutri is not None:
            self.finalized.append(self.live_nutri)
        self._advance()

    def _prepare_fresh_retrieve(self) -> None:
        ing = self.kept[self.comp_i]
        queries = [ing.name.strip()]
        rewrite = candidate_rewrite(queries[0])
        if rewrite is not None:
            queries.append(rewrite)
        self.queries = queries
        self.query_i = 0
        self.failed_top = None
        self.failed_cands = []
        self.failed_query = queries[0]
        self.accepted_code = ""
        self.accepted_decision = "none"
        self.accepted_cands = []
        self.accepted_query = queries[0]
        self.accepted_top = None
        self.live_code = None
        self.live_portion = None
        self.live_nutri = None
        self.live_edges = []
        self.oracle = None
        self.phase = "RETRIEVE"
        self._set_public(
            component_i=self.comp_i,
            hyp=ing.name,
            code=None,
            cands=[],
            portion=None,
            portion_src=None,
            nutri=None,
            nutri_dish=_sum_nutri(self.finalized),
            edges=[],
            status=None,
            abstain_reason=None,
        )

    def _prepare_second_retrieve(self, ing: Any) -> None:
        self.correction = True
        pre_code = self.failed_top
        pre_cands = self.failed_cands
        pre_portion = None
        pre_src = None
        pre_nutri = None
        pre_edges: list[dict[str, Any]] = []
        if pre_code:
            pre_portion, pre_nutri, pre_edges = self._realize(pre_code, self.failed_query)
            if pre_portion is not None:
                pre_src = "grid_search"
        visible = list(self.finalized)
        if pre_nutri is not None:
            visible.append(pre_nutri)
        self.phase = "RETRIEVE"
        self._set_public(
            component_i=self.comp_i,
            hyp=ing.name,
            code=pre_code,
            cands=pre_cands,
            portion=pre_portion,
            portion_src=pre_src,
            nutri=pre_nutri,
            nutri_dish=_sum_nutri(visible),
            edges=pre_edges,
            status=None,
            abstain_reason=None,
        )

    def _prepare_terminal(self) -> None:
        share = self.unresolved_kcal / max(float(self.dish.energy_kcal), 1.0)
        grounded = _sum_nutri(self.finalized)
        self.phase = "TERMINAL"
        if not self.finalized or share > UNRESOLVED_ENERGY_SHARE:
            reason = "no_grounded_component" if not self.finalized else "unresolved_ingredients"
            self.terminal_action = "ABSTAIN"
            self.terminal_status = "abstain"
            self.terminal_reason = reason
            edges = [
                {
                    "src": "dish",
                    "dst": "fail",
                    "type": "abstain_from",
                    "attrs": {"reason": reason},
                }
            ]
            dish_nutri: dict[str, Any] | None = _null_nutri()
            status = "abstain"
        else:
            reason = None
            status = "corrected" if self.correction else "pass"
            self.terminal_action = "STOP"
            self.terminal_status = status
            self.terminal_reason = None
            edges = [{"src": "dish", "dst": "n_dish", "type": "n<-aggregate"}]
            dish_nutri = grounded
        self._set_public(
            component_i=None,
            hyp=None,
            code=None,
            cands=[],
            portion=None,
            portion_src=None,
            nutri=None,
            nutri_dish=dish_nutri,
            edges=edges,
            status=status,
            abstain_reason=reason,
        )

    def _advance(self) -> None:
        self.comp_i += 1
        if self.comp_i >= len(self.kept):
            self._prepare_terminal()
        else:
            self._prepare_fresh_retrieve()

    def _mark_unresolved(self, ing: Any) -> None:
        self.n_unresolved += 1
        self.unresolved_kcal += max(float(ing.energy_kcal), 0.0)

    def _realize(
        self, code: str, query: str
    ) -> tuple[float | None, dict[str, Any] | None, list[dict[str, Any]]]:
        portion_fn = self.portion_fn or _default_portion
        lookup_fn = self._lookup()
        portion = _blind_portion(code, portion_fn, self.catalog_path)
        if portion is None:
            return None, None, _binding_edges(self.comp_i, code, query, None)
        nutri = _scaled(code, portion, lookup_fn, self.catalog_path)
        if nutri is None:
            return None, None, _binding_edges(self.comp_i, code, query, None)
        return portion, nutri, _binding_edges(self.comp_i, code, query, portion)

    def _lookup(self) -> Callable[..., Any]:
        return self.lookup_fn or _default_lookup

    def _retrieve(self, query: str) -> tuple[list[dict[str, Any]], str | None]:
        fn = self.retrieve_fn or _default_retrieve
        found = fn(query, top_k=TOP_K, catalog_path=self.catalog_path)
        cands = _public_cands(found.cands)
        top = str(cands[0]["code"]) if cands else None
        return cands, top

    def _set_public(
        self,
        *,
        component_i: int | None,
        hyp: str | None,
        code: str | None,
        cands: list[dict[str, Any]],
        portion: float | None,
        portion_src: str | None,
        nutri: dict[str, Any] | None,
        nutri_dish: dict[str, Any] | None,
        edges: list[dict[str, Any]],
        status: str | None,
        abstain_reason: str | None,
    ) -> None:
        self.component_i = component_i
        self.hyp = hyp
        self.code = code
        self.cands = cands
        self.portion = portion
        self.portion_src = portion_src
        self.nutri = nutri
        self.nutri_dish = nutri_dish
        self.edges = edges
        self.status = status
        self.abstain_reason = abstain_reason
        self.verify = _verify(nutri, cands, portion, edges)

    def _emit(
        self,
        action: str,
        obs: dict[str, Any],
        *,
        status: str | None = None,
        abstain_reason: str | None = None,
    ) -> None:
        if self.record_ir:
            self.emitted.append(
                _step(
                    dish_id=self.dish.dish_id,
                    step_i=self.step_i,
                    component_i=self.component_i,
                    hyp=self.hyp,
                    code=self.code,
                    cands=self.cands,
                    portion=self.portion,
                    portion_src=self.portion_src,
                    nutri=self.nutri,
                    nutri_dish=self.nutri_dish,
                    edges=self.edges,
                    action=action,
                    obs=obs,
                    status=status,
                    abstain_reason=abstain_reason,
                )
            )
        self.step_i += 1


def _default_retrieve(query: str, top_k: int = TOP_K, catalog_path: str | None = None) -> Any:
    from vdr.tools.fndds_retrieve import retrieve

    return retrieve(query, top_k=top_k, catalog_path=catalog_path)


def _default_lookup(code: str, catalog_path: str | None = None) -> dict[str, float]:
    from vdr.tools.nutrition_lookup import lookup_per_100g

    return lookup_per_100g(code, catalog_path=catalog_path)


def _default_portion(code: str, hint_g: float = 100.0, catalog_path: str | None = None) -> Any:
    from vdr.tools.portion_grid import search_portion

    return search_portion(code, hint_g=hint_g, catalog_path=catalog_path)


def assert_dagger_output(path: Any) -> None:
    """Refuse writes onto frozen eval files or the weak-step jsonl names.

    DAgger artifacts, if a caller writes any, belong under ``results/dagger``.
    The smoke path does not write.
    """
    from pathlib import Path

    from vdr.weak.teacher import PROTECTED_RESULT_NAMES, UnsafeOutputPath, assert_safe_output

    dest = Path(path)
    assert_safe_output(dest)
    name = dest.name
    lowered = name.lower()
    if name in PROTECTED_RESULT_NAMES or "n5k_subset_b0_b1" in name or "n5k_rgb_test" in name:
        raise UnsafeOutputPath(f"refusing to write {name}: frozen eval artifact")
    if "rgb_test" in lowered or "depth_test" in lowered or "test_gt" in lowered:
        raise DaggerPoolRefused(f"refusing to write {name}: DAgger does not touch a test split")
    if "holdout" in lowered or lowered.startswith("rgb_train_"):
        raise DaggerPoolRefused(
            f"refusing to write {name}: that name belongs to the weak-step pools"
        )
    parts = {part.lower() for part in dest.parts}
    if "dagger" not in parts:
        raise DaggerPoolRefused(
            f"refusing to write {dest}: DAgger output must live under results/dagger"
        )


def _assert_fit_dish(dish: DishMeta) -> None:
    if dish.dish_id in test_split_ids():
        raise TestSplitRefused(f"refusing test dish {dish.dish_id}")
    slot = pool_of_dish(dish.dish_id)
    if slot != "fit":
        raise DaggerPoolRefused(
            f"{dish.dish_id} is in the {slot} pool, not fit. "
            "DAgger rollouts stay on the fit pool."
        )


def _learner_action(policy: Any, state: Mapping[str, Any]) -> str:
    action = policy.predict_action(build_features(state))
    if not isinstance(action, str) or action not in _LEARNER_ACTIONS:
        raise ValueError(f"policy returned an action outside the BC set: {action!r}")
    if action == "VOLUME_ESTIMATE":
        raise ValueError("VOLUME_ESTIMATE is not a DAgger action")
    return action


def _predict_rows(policy: Any, matrix: np.ndarray) -> list[str]:
    if not hasattr(policy, "predict_actions"):
        raise TypeError("DAgger agreement scan needs predict_actions")
    predicted = [str(item) for item in policy.predict_actions(matrix)]
    if len(predicted) != matrix.shape[0]:
        raise RuntimeError("predict_actions returned the wrong number of rows")
    illegal = sorted({name for name in predicted if name not in _LEARNER_ACTIONS})
    if illegal or "VOLUME_ESTIMATE" in predicted:
        raise ValueError(f"policy predicted an action outside the BC set: {illegal}")
    return predicted
