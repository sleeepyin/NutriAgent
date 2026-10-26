"""Metric implementations for VDR primary tables (toy-ready)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def mae(
    y_true: Sequence[float | None],
    y_pred: Sequence[float | None],
) -> float:
    """Mean absolute error. Pairs with any None are skipped."""
    errs: list[float] = []
    if len(y_true) != len(y_pred):
        raise ValueError(f"mae length mismatch: {len(y_true)} vs {len(y_pred)}")
    for a, b in zip(y_true, y_pred):
        if a is None or b is None:
            continue
        errs.append(abs(float(a) - float(b)))
    if not errs:
        raise ValueError("mae: no valid pairs")
    return sum(errs) / len(errs)


def mape(
    y_true: Sequence[float | None],
    y_pred: Sequence[float | None],
    eps: float = 1e-6,
) -> float:
    """Mean absolute percentage error (percent). Skips None; uses |y_true|+eps denom."""
    pcts: list[float] = []
    if len(y_true) != len(y_pred):
        raise ValueError(f"mape length mismatch: {len(y_true)} vs {len(y_pred)}")
    for a, b in zip(y_true, y_pred):
        if a is None or b is None:
            continue
        fa, fb = float(a), float(b)
        pcts.append(abs(fa - fb) / (abs(fa) + eps) * 100.0)
    if not pcts:
        raise ValueError("mape: no valid pairs")
    return sum(pcts) / len(pcts)


def fix_rate(
    err_before: Sequence[float],
    err_after: Sequence[float],
) -> float:
    """Fraction of dishes where error decreased after correction."""
    if len(err_before) != len(err_after):
        raise ValueError("fix_rate length mismatch")
    if not err_before:
        return 0.0
    n_fix = sum(1 for b, a in zip(err_before, err_after) if float(a) < float(b))
    return n_fix / len(err_before)


def worsen_rate(
    err_before: Sequence[float],
    err_after: Sequence[float],
) -> float:
    """Fraction of dishes where error increased after correction."""
    if len(err_before) != len(err_after):
        raise ValueError("worsen_rate length mismatch")
    if not err_before:
        return 0.0
    n_w = sum(1 for b, a in zip(err_before, err_after) if float(a) > float(b))
    return n_w / len(err_before)


def has_evidence_path(record: Mapping[str, Any]) -> bool:
    """True if record has at least one n<-code×portion evidence edge (or explicit flag)."""
    if record.get("has_evidence_path") is True:
        return True
    if record.get("ungrounded") is True:
        return False
    edges = record.get("evidence_edges")
    if edges is None and isinstance(record.get("G"), Mapping):
        edges = record["G"].get("edges")
    if not edges:
        return False
    return any(
        isinstance(e, Mapping) and e.get("type") == "n<-code×portion" for e in edges
    )


def evidence_accuracy(
    records: Iterable[Mapping[str, Any]],
) -> float:
    """Evidence-Grounded Acc.

    A record counts as correct if it has a valid evidence path AND
    (if present) human evidence_ok is not False.
    If evidence_ok is missing, only the path check applies (toy mode).
    """
    recs = list(records)
    if not recs:
        raise ValueError("evidence_accuracy: empty records")
    ok = 0
    for r in recs:
        path_ok = has_evidence_path(r)
        human = r.get("evidence_ok")
        if human is False:
            grounded = False
        elif human is True:
            grounded = path_ok
        else:
            grounded = path_ok
        if grounded:
            ok += 1
    return ok / len(recs)


def nutrition_field_mae_mape(
    y_true: Mapping[str, Any],
    y_pred: Mapping[str, Any],
    fields: Sequence[str] = ("energy_kcal", "protein_g", "carb_g", "fat_g", "mass_g"),
) -> dict[str, dict[str, float]]:
    """Per-field MAE/MAPE for a single dish pair (returned as 1-element series)."""
    out: dict[str, dict[str, float]] = {}
    for f in fields:
        t, p = y_true.get(f), y_pred.get(f)
        if t is None or p is None:
            continue
        out[f] = {
            "mae": mae([t], [p]),
            "mape": mape([t], [p]),
        }
    return out
