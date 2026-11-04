"""Coverage, Selective MAE, and AURC.

Formulae
--------
Coverage at a keep-set (answer when ``abstain_flag`` is false, or when
``abstain_score < τ``):

    C = |{i : keep_i}| / N

Selective MAE on that keep-set:

    MAE_sel = mean(error_i for i in keep)

AURC (area under the risk-coverage curve; lower is better):

    confidence_i = -abstain_score_i
    Sort by confidence descending, which is abstain_score ascending.
    For k = 1..N:
        c_k = k / N
        r_k = mean(error of the k most confident examples)
    AURC = trapezoidal integral of r versus c
         = sum_{k=1..N-1} 0.5 * (r_k + r_{k+1}) * (c_{k+1} - c_k)
           + 0.5 * r_1 * c_1
    The last term is the triangle from the origin to the first point
    (risk is undefined at coverage 0; this is the practical choice).

No model is fit here. Scores and errors are inputs.
"""

from __future__ import annotations

from typing import Sequence


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def coverage_and_selective_mae(
    errors: Sequence[float],
    abstain_scores: Sequence[float] | None = None,
    abstain_flags: Sequence[bool | int] | None = None,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    """Keep examples that are answered.

    If ``abstain_flags`` is set, keep where the flag is falsy.
    Otherwise keep where ``abstain_score < threshold`` (threshold required).
    """
    n = len(errors)
    if n == 0:
        return {"coverage": 0.0, "selective_mae": 0.0, "n": 0.0, "n_kept": 0.0}

    if abstain_flags is not None:
        if len(abstain_flags) != n:
            raise ValueError("abstain_flags length mismatch")
        keep = [not bool(flag) for flag in abstain_flags]
    else:
        if abstain_scores is None or threshold is None:
            raise ValueError("need abstain_flags OR (abstain_scores + threshold)")
        if len(abstain_scores) != n:
            raise ValueError("abstain_scores length mismatch")
        keep = [float(score) < float(threshold) for score in abstain_scores]

    kept_err = [float(err) for err, kept in zip(errors, keep) if kept]
    n_kept = len(kept_err)
    return {
        "coverage": n_kept / n,
        "selective_mae": _mean(kept_err),
        "n": float(n),
        "n_kept": float(n_kept),
    }


def aurc(
    errors: Sequence[float],
    abstain_scores: Sequence[float],
) -> dict[str, float]:
    """Trapezoidal area under risk versus coverage. Lower is better."""
    n = len(errors)
    if n == 0:
        return {"aurc": 0.0, "n": 0.0}
    if len(abstain_scores) != n:
        raise ValueError("length mismatch")

    # Higher abstain_score = more likely to abstain = lower confidence.
    order = sorted(range(n), key=lambda i: float(abstain_scores[i]))
    cumulative = 0.0
    risks: list[float] = []
    coverages: list[float] = []
    for k, idx in enumerate(order, start=1):
        cumulative += float(errors[idx])
        risks.append(cumulative / k)
        coverages.append(k / n)

    area = 0.0
    for i in range(n - 1):
        dc = coverages[i + 1] - coverages[i]
        area += 0.5 * (risks[i] + risks[i + 1]) * dc
    area += 0.5 * risks[0] * coverages[0]
    return {
        "aurc": area,
        "n": float(n),
        "risk_at_full_coverage": risks[-1],
        "risk_at_min_coverage": risks[0],
    }


def risk_coverage_curve(
    errors: Sequence[float],
    abstain_scores: Sequence[float],
) -> list[dict[str, float]]:
    n = len(errors)
    order = sorted(range(n), key=lambda i: float(abstain_scores[i]))
    cumulative = 0.0
    curve: list[dict[str, float]] = []
    for k, idx in enumerate(order, start=1):
        cumulative += float(errors[idx])
        curve.append({"coverage": k / n, "risk": cumulative / k, "k": float(k)})
    return curve
