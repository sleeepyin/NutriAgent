import math
import numpy as np
from agent_extension.src.conformal.finite_sample import finite_sample_quantile


def test_order_statistic():
    s = np.arange(1, 21, dtype=float)
    k = math.ceil((20 + 1) * 0.9)
    assert k == 19
    assert finite_sample_quantile(s, 0.10) == float(np.partition(s, 18)[18])


def test_k_gt_n():
    assert math.isinf(finite_sample_quantile([1, 2, 3], 0.0))
