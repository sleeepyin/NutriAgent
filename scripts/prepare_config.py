"""Freeze the T_multi config before training."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from digi_nutri.prepare import main


if __name__ == "__main__":
    main()
