"""One-step myopic VOI wrapper around the deterministic controller."""
from __future__ import annotations

from ..controller import Controller


class MyopicVOI(Controller):
    def __init__(self, **kw):
        kw.setdefault("mode", "myopic_voi")
        super().__init__(**kw)
