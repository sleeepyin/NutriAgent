#!/usr/bin/env python3
"""Paper-oriented figures from completed experiment artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import add_common_args, load_json, load_rows, out_dir, save_status
from agent_extension.src.paths import Paths
from agent_extension.src.runtime import log


def _bar(ax, labels, vals, title, ylabel):
    ax.bar(range(len(labels)), vals, color="#3c6e8f")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args(argv)
    paths = Paths.resolve()
    d = out_dir("e10_figures", args)
    rows = load_rows()
    tv = load_json(paths.results / "e03_tool_value" / "tool_value.json", {})
    bm = load_json(paths.results / "e07_main_benchmark" / "benchmark.json", {})
    xd = load_json(paths.results / "e08_cross_domain" / "cross_domain.json", {})

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.text(0.02, 0.75, "RGB  →  Domain/Morph/UQ  →  Controller  →  Tool  →  Terminal q", fontsize=11)
    ax.text(0.02, 0.45, "STOP / REF / VIEW / DEPTH / SEMANTIC / ABSTAIN", fontsize=10)
    ax.text(0.02, 0.18, "Estimate → Diagnose → Act   |   Level A public-data run", fontsize=9, color="#444")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(d / "fig1_architecture.png", dpi=160)
    plt.close()

    # action gains
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels, vals = [], []
    for act, rec in (tv.get("actions") or {}).items():
        for ds, ev in rec.items():
            labels.append(f"{act.split('_')[-1]}/{ds}")
            vals.append(100 * ev.get("d_mape", 0))
    if labels:
        _bar(ax, labels, vals, "Oracle MAPE drop (policy, native dataset)", "Δ MAPE (pp)")
        ax.axhline(5, color="crimson", ls="--", lw=1, label="5% gate")
        ax.legend(fontsize=8)
    else:
        ax.text(0.2, 0.5, "no tool-value yet")
    fig.tight_layout()
    fig.savefig(d / "fig2_action_gains.png", dpi=160)
    plt.close()

    # risk-cost
    fig, ax = plt.subplots(figsize=(6, 4))
    res = (bm.get("results") or {})
    for name, ev in res.items():
        x = ev.get("mean_cost", np.nan)
        y = ev.get("interval", {}).get("nwinkler", np.nan)
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter([x], [y], s=40)
            ax.annotate(name.replace("always_", "A:").replace("myopic_voi", "CAIA"), (x, y), fontsize=7)
    ax.set_xlabel("mean prior cost")
    ax.set_ylabel("normalized Winkler")
    ax.set_title("Risk–cost (Level A test, shared terminal q)")
    fig.tight_layout()
    fig.savefig(d / "fig3_risk_cost.png", dpi=160)
    plt.close()

    # coverage-width
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ev in res.items():
        c = ev.get("interval", {}).get("coverage", np.nan)
        w = ev.get("interval", {}).get("rel_width", np.nan)
        if np.isfinite(c) and np.isfinite(w):
            ax.scatter([w], [c], s=40)
            ax.annotate(name.replace("always_", ""), (w, c), fontsize=7)
    ax.axhline(0.90, color="crimson", ls="--", lw=1)
    ax.set_xlabel("relative width")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Coverage vs width (do not read CI lower bound as 90%)")
    fig.tight_layout()
    fig.savefig(d / "fig4_coverage_width.png", dpi=160)
    plt.close()

    # recovery
    fig, ax = plt.subplots(figsize=(6, 3.6))
    rec = xd.get("recovery") or []
    xs = [r["n"] for r in rec if "interval" in r]
    ys = [100 * r["interval"]["coverage"] for r in rec if "interval" in r]
    if xs:
        ax.plot(xs, ys, marker="o")
    ax.axhline(90, color="crimson", ls="--")
    ax.set_xlabel("target-domain cal n")
    ax.set_ylabel("NV coverage (%)")
    ax.set_title("Few-shot coverage recovery (empirical, not a transferred guarantee)")
    fig.tight_layout()
    fig.savefig(d / "fig5_recovery.png", dpi=160)
    plt.close()

    # morphology gains
    fig, ax = plt.subplots(figsize=(7, 3.6))
    morph = tv.get("by_morphology") or {}
    labs, vs = [], []
    for m, rec in morph.items():
        for a, st in rec.items():
            labs.append(f"{m}:{a.split('_')[-1]}")
            vs.append(st.get("mean_gain", 0))
    if labs:
        _bar(ax, labs, vs, "Mean interval-score gain by morphology (policy)", "gain")
    fig.tight_layout()
    fig.savefig(d / "fig6_morphology.png", dpi=160)
    plt.close()

    save_status(d, True, figures=sorted(x.name for x in d.glob("*.png")))
    log(f"E10 figures -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
