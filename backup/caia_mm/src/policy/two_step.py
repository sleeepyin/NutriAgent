"""Two-step look-ahead. Only keep if policy-val gain vs myopic exceeds 2%."""
from __future__ import annotations

from ..controller import Controller


class TwoStepVOI(Controller):
    def __init__(self, **kw):
        kw.setdefault("mode", "myopic_voi")
        super().__init__(**kw)
        self.two_step = True
