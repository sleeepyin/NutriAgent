#!/usr/bin/env python3
"""Unit tests for finite-sample q and MAPE/WAPE. Fail closed."""
from __future__ import annotations

import argparse
import math
import traceback

import numpy as np

from .common import add_common_args, out_dir, save_status
from agent_extension.src.conformal.finite_sample import finite_sample_quantile
from agent_extension.src.metrics import mape, wape


def test_q_order_stat():
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # n=5, alpha=0.2 -> k=ceil(6*0.8)=5 -> q=5
    q = finite_sample_quantile(s, 0.2)
    assert abs(q - 5.0) < 1e-12, q
    # k>n -> inf
    qinf = finite_sample_quantile(s, 0.0)
    assert math.isinf(qinf)
    # no interpolation: even if linear quantile would be between 4 and 5
    q2 = finite_sample_quantile(s, 0.10)
    # k=ceil(6*0.9)=6 >5 -> inf
    assert math.isinf(q2)
    s2 = np.arange(1, 21, dtype=float)
    q3 = finite_sample_quantile(s2, 0.10)
    k = math.ceil((20 + 1) * 0.9)
    expected = float(np.partition(s2, k - 1)[k - 1])
    assert abs(q3 - expected) < 1e-12
    # interpolation would differ for some arrays
    q_lin = float(np.quantile(s2, min(1.0, k / 20.0)))
    # we only require our q equals order statistic, not that it differs
    assert abs(q3 - expected) < 1e-12


def test_mape_wape():
    y = np.array([10.0, 20.0, 40.0])
    p = np.array([12.0, 16.0, 40.0])
    # MAPE = mean(0.2, 0.2, 0.0) = 0.4/3
    assert abs(mape(y, p) - (0.4 / 3.0)) < 1e-12
    # WAPE = (2+4+0)/(10+20+40)=6/70
    assert abs(wape(y, p) - 6.0 / 70.0) < 1e-12
    assert abs(mape(y, p) - wape(y, p)) > 1e-6


def test_empty_and_nan():
    q = finite_sample_quantile([], 0.1)
    assert math.isinf(q)
    q = finite_sample_quantile([np.nan, 1.0, 2.0], 0.5)
    assert np.isfinite(q)


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args(argv)
    d = out_dir("e00_tests", args)
    tests = [test_q_order_stat, test_mape_wape, test_empty_and_nan]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failed.append({"test": t.__name__, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()})
    ok = len(failed) == 0
    save_status(d, ok, n=len(tests), failed=failed)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
