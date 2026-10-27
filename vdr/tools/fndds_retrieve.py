"""FNDDS retrieval tool.

Default catalog: data/fndds/2019-2020/foods.json if present, else toy.
Override with VDR_FNDDS_PATH.

Plain produce queries ("red onion", "blueberry") used to lose to a shorter
prepared food that shared one token (Onion dip, Blueberry juice). Transparent
adjustments sit on top of the token score. None of them is a learned weight.

1. Light singularization so onion/onions and blueberry/blueberries share a lemma.
2. A lexical form prior. The token score is capped at 1 before this prior so
   the raw bonus below is the only way past 1.
   - If the query is a produce name or "rice", and it does not itself name a
     prepared form, foods whose name contains dip/soup/sauce/gravy/juice/milk/
     oil/pickled/fried/candied/creamed/flavored/rings lose ``PREP_PENALTY``
     (0.20). Other foods (tofu, chicken, beef) are not penalized: dropping
     "sauce" there lets an unrelated tempura code win.
   - If the query is a produce name and the food is raw/fresh — not merely
     "from raw" or "from fresh" — add ``RAW_BONUS`` (0.10).
   - If the query names a dish form (pudding, salad, pizza, ...), a food whose
     name drops that form loses ``DISH_FORM_PENALTY`` (0.20). Head F1 otherwise
     prefers the short ingredient ("Chia seeds" over any pudding).
   - A meat-analog qualifier (meatless, vegetarian, vegan, imitation) on a
     meat head ("Bacon strip, meatless") loses ``FAUX_MEAT_PENALTY`` (0.25)
     unless the query itself asked for that qualifier. Dishes that are merely
     meatless (cheese quiche, lasagna) are not meat heads, so they stay.

0.10 is just above ``rule_verify.CODE_CONF_GAP`` (0.08), so a raw produce code
separates from an exact-tied cooked sibling. 0.20 is large enough that a
substring win for dip/juice, or for chia seeds over pudding, does not survive.
0.25 drops an exact alias hit (1.0) under the real-bacon head match (~0.78).
Query text that already says "dip", "juice", "pudding", or "meatless" keeps
that reading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .fndds_catalog import iter_foods


@dataclass
class RetrieveResult:
    query: str
    cands: list[dict[str, Any]]
    selected_code: str | None


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Singularization exceptions: stripping a trailing s would change the word.
_KEEP_TOKEN = frozenset({"asparagus", "hummus", "couscous", "molasses", "oats"})
_IRREGULAR = {
    "leaves": "leaf",
    "potatoes": "potato",
    "tomatoes": "tomato",
}

# Prepared forms we refuse to prefer for a plain ingredient query.
PREPARED_FORM_TOKENS = frozenset(
    {
        "dip",
        "soup",
        "sauce",
        "gravy",
        "juice",
        "milk",
        "oil",
        "pickled",
        "fried",
        "candied",
        "creamed",
        "flavored",
        "rings",
    }
)
RAW_FORM_TOKENS = frozenset({"raw", "fresh"})

# Produce lemmas. The raw/fresh bonus applies only to these queries so a
# meat described as "from raw" is not promoted for the word chicken.
PRODUCE_LEMMAS = frozenset(
    {
        "onion",
        "tomato",
        "potato",
        "carrot",
        "broccoli",
        "spinach",
        "cauliflower",
        "cabbage",
        "lettuce",
        "cucumber",
        "pepper",
        "bean",
        "pea",
        "corn",
        "mushroom",
        "asparagus",
        "celery",
        "kale",
        "zucchini",
        "squash",
        "eggplant",
        "radish",
        "beet",
        "beetroot",
        "turnip",
        "leek",
        "scallion",
        "shallot",
        "garlic",
        "ginger",
        "apple",
        "banana",
        "grape",
        "orange",
        "berry",
        "blueberry",
        "strawberry",
        "raspberry",
        "blackberry",
        "melon",
        "watermelon",
        "cantaloupe",
        "honeydew",
        "pineapple",
        "mango",
        "papaya",
        "pear",
        "peach",
        "plum",
        "cherry",
        "lemon",
        "lime",
        "avocado",
        "olive",
        "vegetable",
        "almond",
        "peanut",
        "walnut",
        "cashew",
        "pecan",
    }
)

# Prep penalty also covers a few plain staples that are not eaten raw.
# Tofu/chicken/beef stay out of this set: penalizing "sauce" on those names
# lets an unrelated tempura or sandwich code win.
PENALTY_QUERY_LEMMAS = PRODUCE_LEMMAS | {"rice"}

# Dish forms. Head F1 scores a short ingredient that covers part of the query
# above the dish those words were naming ("Chia seeds" at 0.83 vs any pudding
# at 0.725). The penalty has to clear that gap.
DISH_FORM_TOKENS = frozenset(
    {
        "pudding",
        "salad",
        "pizza",
        "sandwich",
        "omelet",
        "omelette",
        "smoothie",
        "stew",
        "casserole",
    }
)

# Faux-meat qualifiers. Applied only when they mark a meat head (bacon,
# chicken, hot dog, ...). "Lasagna, meatless" and "Cheese quiche, meatless"
# use the same word for a dish that simply has no meat; those heads are not
# in FAUX_MEAT_HEADS, so a blanket penalty would bind "cheese quiche" to a
# cheese block.
MEAT_ALTERNATIVE_TOKENS = frozenset({"meatless", "vegetarian", "vegan", "imitation"})
FAUX_MEAT_HEADS = frozenset(
    {
        "bacon",
        "sausage",
        "chicken",
        "beef",
        "pork",
        "turkey",
        "ham",
        "meatball",
        "steak",
        "burger",
        "salami",
        "pepperoni",
        "crab",
        "frank",
        "wiener",
        "bratwurst",
    }
)

# Flat score shifts. See module docstring for why these magnitudes.
PREP_PENALTY = 0.20
RAW_BONUS = 0.10
DISH_FORM_PENALTY = 0.20
FAUX_MEAT_PENALTY = 0.25


def _lemma(token: str) -> str:
    t = token.lower()
    if t in _IRREGULAR:
        return _IRREGULAR[t]
    if t in _KEEP_TOKEN or len(t) <= 3:
        return t
    if t.endswith("ies"):
        return t[:-3] + "y"
    if t.endswith("oes"):
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _lemmas(text: str) -> set[str]:
    return {_lemma(t) for t in _TOKEN_RE.findall(text or "")}


def _norm_phrase(text: str) -> str:
    return " ".join(_lemma(t) for t in _TOKEN_RE.findall(text or ""))


def query_asks_prepared(query: str) -> bool:
    return bool(_lemmas(query) & PREPARED_FORM_TOKENS)


def name_is_prepared(name: str) -> bool:
    return bool(_lemmas(name) & PREPARED_FORM_TOKENS)


def query_dish_forms(query: str) -> list[str]:
    """Dish-form lemmas in query order, without duplicates."""
    forms: list[str] = []
    for token in _TOKEN_RE.findall(query or ""):
        lemma = _lemma(token)
        if lemma in DISH_FORM_TOKENS and lemma not in forms:
            forms.append(lemma)
    return forms


def name_has_lemmas(name: str, lemmas: set[str] | frozenset[str]) -> bool:
    return bool(_lemmas(name) & set(lemmas))


def rewrite_dish_form_query(
    query: str,
    alternative_names: Sequence[str] = (),
) -> str | None:
    """Corrective retrieve text when a dish-form query bound a dense ingredient.

    Reordering the original words keeps the same lemmas, so a second retrieve
    would repeat the first ranking. The rewrite is the dish-form word itself
    ("pudding"), which is the form the query already named.
    """
    forms = query_dish_forms(query)
    if not forms:
        return None
    alt: set[str] = set()
    for name in alternative_names:
        alt |= _lemmas(name)
    for form in forms:
        if form in alt:
            return form
    return forms[0]


def _head_lemmas(name: str) -> set[str]:
    return _lemmas((name or "").split(",")[0])


def _has_meat_alternative_qualifier(name: str) -> bool:
    """True when meatless/vegetarian/vegan/imitation is a label, not a side note.

    FNDDS writes it as its own comma segment ("Bacon strip, meatless") or as
    the first word ("Imitation crab meat"). "made with imitation crab" inside
    a salad name is not that label.
    """
    parts = [part.strip() for part in (name or "").split(",") if part.strip()]
    if not parts:
        return False
    allowed = MEAT_ALTERNATIVE_TOKENS | {"nfs"}
    for part in parts[1:]:
        tokens = _lemmas(part)
        if tokens and tokens <= allowed and (tokens & MEAT_ALTERNATIVE_TOKENS):
            return True
    first_tokens = _TOKEN_RE.findall(parts[0])
    if first_tokens and _lemma(first_tokens[0]) in MEAT_ALTERNATIVE_TOKENS:
        return True
    return False


def _is_faux_meat_head(name: str) -> bool:
    head = _head_lemmas(name)
    if head & FAUX_MEAT_HEADS:
        return True
    return {"hot", "dog"} <= head


def query_asks_meat_alternative(query: str) -> bool:
    return bool(_lemmas(query) & MEAT_ALTERNATIVE_TOKENS)


def name_is_unrequested_faux_meat(query: str, name: str) -> bool:
    """Meat analog selected for a query that did not ask for one."""
    if query_asks_meat_alternative(query):
        return False
    if not _has_meat_alternative_qualifier(name):
        return False
    return _is_faux_meat_head(name)


def rewrite_plain_query(query: str) -> str | None:
    """Second-retrieve text for a plain produce query.

    Returns ``"{query} raw"`` when the query is a produce name that does not
    already ask for a prepared or raw form. Returns None otherwise.
    """
    if not (query or "").strip():
        return None
    if query_asks_prepared(query):
        return None
    lemmas = _lemmas(query)
    if lemmas & RAW_FORM_TOKENS:
        return None
    if not (lemmas & PRODUCE_LEMMAS):
        return None
    cleaned = " ".join(_TOKEN_RE.findall(query))
    if not cleaned:
        return None
    return f"{cleaned} raw"


def _is_raw_or_fresh_form(name: str) -> bool:
    """True when raw/fresh is a comma-separated form, not a side remark.

    FNDDS writes the form after a comma ("Onions, raw", "Onions, green, raw").
    Phrases such as "from fresh", "from raw", and "NS as to fresh or frozen"
    are not that form. A single-segment toy name ("cucumber raw") still counts.
    """
    parts = [p.strip().lower() for p in (name or "").split(",")]
    if len(parts) == 1:
        return bool(_lemmas(parts[0]) & RAW_FORM_TOKENS)
    for part in parts[1:]:
        if part.startswith("ns as to") or part.startswith("from "):
            continue
        tokens = _lemmas(part)
        if not (tokens & RAW_FORM_TOKENS):
            continue
        if tokens & PREPARED_FORM_TOKENS:
            continue
        # "raw", "fresh", or "raw nfs" — not "fresh cooked ..." mixed into one clause.
        if tokens <= (RAW_FORM_TOKENS | {"nfs"}):
            return True
    return False


def _apply_form_prior(query: str, food: dict[str, Any], base: float) -> float:
    """Lexical raw-vs-prepared adjustment. See module docstring."""
    q_lemmas = _lemmas(query)
    name = str(food.get("name", ""))
    name_lemmas = _lemmas(name)
    score = min(float(base), 1.0)
    asks_prep = bool(q_lemmas & PREPARED_FORM_TOKENS)
    if (
        not asks_prep
        and (q_lemmas & PENALTY_QUERY_LEMMAS)
        and (name_lemmas & PREPARED_FORM_TOKENS)
    ):
        score -= PREP_PENALTY
    # Shared content lemma, ignoring the form words themselves. Otherwise a
    # rewritten query that ends in "raw" would bonus every raw food.
    content_overlap = (q_lemmas - RAW_FORM_TOKENS - PREPARED_FORM_TOKENS) & (
        name_lemmas - RAW_FORM_TOKENS
    )
    if (
        bool(q_lemmas & PRODUCE_LEMMAS)
        and not asks_prep
        and _is_raw_or_fresh_form(name)
        and content_overlap
    ):
        score += RAW_BONUS
    q_forms = q_lemmas & DISH_FORM_TOKENS
    if q_forms and not (name_lemmas & q_forms):
        score -= DISH_FORM_PENALTY
    if name_is_unrequested_faux_meat(query, name):
        score -= FAUX_MEAT_PENALTY
    return max(0.0, score)


def _score_food(query: str, food: dict[str, Any]) -> float:
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    name = str(food.get("name", "")).lower()
    aliases = [str(a).lower() for a in food.get("aliases", [])]
    head = name.split(",")[0].strip()
    qt = _lemmas(q)
    if not qt:
        return 0.0

    q_norm = _norm_phrase(q)
    name_norm = _norm_phrase(name)
    head_norm = _norm_phrase(head)
    alias_norms = [_norm_phrase(a) for a in aliases]

    scores: list[float] = []

    # Exact full-string matches, also under singularization.
    if q_norm and (q == name or q in aliases or q_norm == name_norm or q_norm in alias_norms):
        scores.append(1.0)
    elif q_norm and (q == head or q_norm == head_norm):
        scores.append(0.98)

    hay_fields = [name, head, *aliases]
    ht_name = _lemmas(name)
    ht_head = _lemmas(head)
    lemma_sets = [_lemmas(h) for h in hay_fields if h]
    ht_all = set().union(*lemma_sets) if lemma_sets else set()

    # Contiguous lemma phrase, on token boundaries. A character substring
    # would treat "herb" as inside "sherbet".
    for h in hay_fields:
        if not h:
            continue
        h_norm = _norm_phrase(h)
        if q_norm and f" {q_norm} " in f" {h_norm} ":
            density = len(q_norm) / max(len(h_norm), 1)
            scores.append(0.70 + 0.28 * density)

    # All query tokens present in the name.
    if ht_name and qt.issubset(ht_name):
        density = len(qt) / max(len(ht_name), 1)
        name_norm_start = _norm_phrase(name)
        start_boost = 0.08 if any(name.startswith(t) or name_norm_start.startswith(t) for t in qt) else 0.0
        head_overlap = len(qt & ht_head) / len(qt) if ht_head else 0.0
        scores.append(0.72 + 0.22 * density + start_boost + 0.05 * head_overlap)

    # Head-focused soft match.
    if ht_head:
        prec = len(qt & ht_head) / len(qt)
        rec = len(qt & ht_head) / len(ht_head)
        if prec + rec:
            f1 = 2 * prec * rec / (prec + rec)
            scores.append(0.55 + 0.35 * f1)

    # Weak Jaccard fallback.
    if ht_all:
        inter = len(qt & ht_all)
        if inter:
            jacc = inter / len(qt | ht_all)
            scores.append(0.40 * jacc + 0.15 * (inter / len(qt)))

    base = max(scores) if scores else 0.0
    if base <= 0:
        return 0.0
    # Cap the lexical score at 1. The raw bonus is the only term allowed to
    # exceed 1, so it can open a gap over an exact-tied cooked sibling.
    return _apply_form_prior(q, food, min(base, 1.0))


def retrieve(
    query: str,
    top_k: int = 5,
    catalog_path: str | Path | None = None,
    **kwargs: Any,
) -> RetrieveResult:
    """Retrieve Top-k FNDDS food codes for a text query."""
    scored: list[tuple[float, dict[str, Any]]] = []
    for food in iter_foods(catalog_path):
        s = _score_food(query, food)
        if s > 0:
            scored.append((s, food))
    scored.sort(key=lambda x: (-x[0], len(str(x[1].get("name", ""))), str(x[1]["code"])))
    cands = [
        {
            "code": f["code"],
            "score": round(s, 4),
            "name": f.get("name"),
        }
        for s, f in scored[: max(1, int(top_k))]
    ]
    selected = cands[0]["code"] if cands else None
    return RetrieveResult(query=query, cands=cands, selected_code=selected)
