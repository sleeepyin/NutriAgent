"""Whitelist validator. LLM cannot bypass actions, q, or coverage claims."""
from __future__ import annotations

from ..schemas import ACTIONS, PRIOR_COST


def validate_action(action: str, available: dict[str, bool], remaining_budget: float) -> tuple[bool, str]:
    a = str(action or "").upper()
    if a not in ACTIONS:
        return False, "not_in_whitelist"
    if not available.get(a, False):
        return False, "masked"
    if PRIOR_COST.get(a, 0) > remaining_budget and a not in ("STOP", "ABSTAIN"):
        return False, "over_budget"
    return True, "ok"
