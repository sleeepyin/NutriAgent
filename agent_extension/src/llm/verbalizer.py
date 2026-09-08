"""JSON action -> user-facing text. Never invents grams or coverage claims."""
from __future__ import annotations

TEMPLATES = {
    "STOP": "Current evidence is enough. I will output the estimate and interval now.",
    "REQUEST_REFERENCE": "Please place a standard card or coin on the same plane as the food and take one photo.",
    "REQUEST_VIEW": "Please take one more photo from a complementary angle (top vs ~45°), same plate and distance.",
    "REQUEST_DEPTH": "If your device supports depth, please capture an RGB-D frame of the same plate.",
    "ASK_SEMANTIC": "Please answer one short question about food form, hidden liquid/oil, or container type.",
    "ABSTAIN": "I cannot reliably estimate this sample. The next useful information would be a reference object or a clearer view.",
}


def verbalize(action: str, reason_code: str = "") -> str:
    base = TEMPLATES.get(action, TEMPLATES["ABSTAIN"])
    if reason_code:
        return f"{base} (reason: {reason_code})"
    return base
