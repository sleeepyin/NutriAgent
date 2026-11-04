"""Per-class precision / recall / F1 and an action confusion matrix.

Column order is the P0 report order (``CORRECT_CODE`` sits next to the
retrieve / portion swaps). It is not ``schemas/actions.json`` enum order.
Overall accuracy is reported and is not the headline; ABSTAIN is marked.
"""

from __future__ import annotations

from typing import Sequence

# Report order. VOLUME_ESTIMATE is not a head.
ACTIONS: tuple[str, ...] = (
    "RETRIEVE",
    "CORRECT_CODE",
    "RE_PORTION",
    "RE_DECOMPOSE",
    "ABSTAIN",
    "STOP",
)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = ACTIONS,
) -> list[list[int]]:
    index = {name: i for i, name in enumerate(labels)}
    n = len(labels)
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for true, pred in zip(y_true, y_pred):
        if true not in index or pred not in index:
            continue
        mat[index[true]][index[pred]] += 1
    return mat


def per_class_prf(
    mat: Sequence[Sequence[int]],
    labels: Sequence[str] = ACTIONS,
) -> dict[str, dict[str, float]]:
    n = len(labels)
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(labels):
        tp = mat[i][i]
        fp = sum(mat[r][i] for r in range(n) if r != i)
        fn = sum(mat[i][c] for c in range(n) if c != i)
        support = sum(mat[i])
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
        out[name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": float(support),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }
    return out


def macro_f1(prf: dict[str, dict[str, float]], labels: Sequence[str] = ACTIONS) -> float:
    """Macro-F1 over classes with support > 0 (RE_DECOMPOSE is often empty)."""
    vals = [prf[name]["f1"] for name in labels if prf[name]["support"] > 0]
    return sum(vals) / len(vals) if vals else 0.0


def balanced_accuracy(prf: dict[str, dict[str, float]], labels: Sequence[str] = ACTIONS) -> float:
    vals = [prf[name]["recall"] for name in labels if prf[name]["support"] > 0]
    return sum(vals) / len(vals) if vals else 0.0


def overall_accuracy(mat: Sequence[Sequence[int]]) -> float:
    total = sum(sum(row) for row in mat)
    correct = sum(mat[i][i] for i in range(len(mat)))
    return _safe_div(correct, total)


def format_report(
    mat: Sequence[Sequence[int]],
    labels: Sequence[str] = ACTIONS,
    *,
    title: str = "Action confusion matrix",
    highlight: str = "ABSTAIN",
) -> str:
    prf = per_class_prf(mat, labels)
    lines: list[str] = [title, "=" * len(title), ""]
    col_w = max(14, max(len(name) for name in labels) + 2)
    header = "gold \\ pred".ljust(col_w) + "".join(name[:12].center(12) for name in labels)
    lines.append(header)
    lines.append("-" * len(header))
    hi = labels.index(highlight) if highlight in labels else -1
    for i, name in enumerate(labels):
        row = name.ljust(col_w) + "".join(f"{mat[i][j]:12d}" for j in range(len(labels)))
        if i == hi:
            row += "  << ABSTAIN row"
        lines.append(row)
    if hi >= 0:
        preds = ", ".join(
            f"{labels[r]}→{mat[r][hi]}" for r in range(len(labels)) if mat[r][hi]
        )
        lines.append("")
        lines.append("ABSTAIN column (preds): " + (preds or "(none)"))
    lines.append("")
    lines.append(f"{'class':<16}{'P':>8}{'R':>8}{'F1':>8}{'support':>10}")
    lines.append("-" * 50)
    for name in labels:
        metrics = prf[name]
        mark = "  ** ABSTAIN" if name == highlight else ""
        lines.append(
            f"{name:<16}{metrics['precision']:8.4f}{metrics['recall']:8.4f}"
            f"{metrics['f1']:8.4f}{int(metrics['support']):10d}{mark}"
        )
    lines.append("-" * 50)
    lines.append(f"overall accuracy : {overall_accuracy(mat):.4f}  (not the headline)")
    lines.append(f"macro-F1         : {macro_f1(prf, labels):.4f}")
    lines.append(f"balanced accuracy: {balanced_accuracy(prf, labels):.4f}")
    highlight_metrics = prf.get(highlight, {"f1": 0.0, "tp": 0.0, "support": 0.0})
    lines.append(
        f"ABSTAIN F1       : {highlight_metrics['f1']:.4f}  "
        f"(recall {int(highlight_metrics['tp'])}/{int(highlight_metrics['support'])})"
    )
    return "\n".join(lines) + "\n"
