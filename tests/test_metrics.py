import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from mmcfa.metrics import mape, wape


def test_mape_ne_wape():
    y = np.array([10.0, 20.0, 40.0])
    p = np.array([12.0, 16.0, 40.0])
    assert abs(mape(y, p) - (0.4 / 3.0)) < 1e-12
    assert abs(wape(y, p) - 6 / 70) < 1e-12
