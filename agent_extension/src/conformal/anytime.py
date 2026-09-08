"""Anytime alpha-spending safety baseline."""
from __future__ import annotations

from .finite_sample import finite_sample_quantile


def alpha_spending_quantiles(scores_by_t: list, alpha: float = 0.10, t_max: int | None = None) -> dict:
    t_max = t_max if t_max is not None else max(0, len(scores_by_t) - 1)
    n_steps = t_max + 1
    alpha_t = alpha / max(n_steps, 1)
    qs = []
    for scores in scores_by_t:
        qs.append(finite_sample_quantile(scores, alpha_t))
    return {
        "mode": "anytime_alpha_spending",
        "alpha": alpha,
        "alpha_t": alpha_t,
        "t_max": t_max,
        "q_t": qs,
        "note": "union bound; conservative safety baseline, not the efficiency main result",
    }
