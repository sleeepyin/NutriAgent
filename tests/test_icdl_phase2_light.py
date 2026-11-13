"""Protocol checks for the ICDL phase 2 light experiment. No training."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from icdl_phase2_light import self_check


def test_phase2_protocol_self_check():
    self_check()
