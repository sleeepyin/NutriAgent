import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from experiments.action.a5_oracle_bound import _self_check


def test_a5_oracle_decision_rule():
    _self_check()
