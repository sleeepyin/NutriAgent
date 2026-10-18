"""Morphology: keyword rules (deployable if category is observed) + oracle mapping."""
from __future__ import annotations

RIGID_KW = (
    "apple", "banana", "orange", "pear", "carrot", "egg", "steak", "bread", "bagel",
    "croissant", "tomato", "cucumber", "chicken-breast", "chicken-leg", "chicken-wing",
    "toast", "corn", "lobster", "shrimp", "tofu", "pork", "rib", "lamb",
)
PREPARED_KW = (
    "soup", "stew", "rice", "pasta", "salad", "sandwich", "pizza", "cake", "lasagna",
    "hamburger", "sushi", "omelet", "curry", "noodle", "porridge", "casserole",
)
STACKED_KW = (
    "blueberry", "almond", "nut", "fries", "nugget", "grape", "bean", "cereal", "chip",
)
LIQUID_KW = ("soup", "juice", "milk", "yogurt", "smoothie", "broth", "tea", "coffee")

NV_RIGID = {
    "red-apple", "red-yellow-apple", "asian-pear", "carrot", "corn", "costco-egg",
    "lobster", "half-minced-shrimp", "chicken-breast", "chicken-leg", "chicken-wing",
    "near-whole-chicken", "steak", "full-bread-loaf", "half-bread-loaf", "plain-toast",
    "toast-with-strawberry-jam", "captain-crunch-granola-bar", "chocolate-granola-bar",
    "nature-valley-granola-bar", "stack-of-tofu-4pc", "pork-feet", "rib", "lamb-shank",
}


def classify_name(name: str | None) -> str:
    if not name:
        return "unknown"
    s = str(name).lower().replace("_", "-")
    if any(k in s for k in LIQUID_KW):
        return "liquid"
    if any(k in s for k in STACKED_KW):
        return "stacked"
    if any(k in s for k in PREPARED_KW):
        return "prepared"
    if any(k in s for k in RIGID_KW):
        return "rigid"
    return "unknown"


def classify_nv(cat: str | None) -> str:
    if not cat:
        return "unknown"
    return "rigid" if cat in NV_RIGID else "prepared"


def p_from_label(label: str) -> tuple[float, float]:
    if label == "rigid":
        return 0.85, 0.10
    if label == "prepared":
        return 0.10, 0.80
    if label == "stacked":
        return 0.25, 0.55
    if label == "liquid":
        return 0.05, 0.70
    return 0.25, 0.25
