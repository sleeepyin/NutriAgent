"""Ingredient tokens and static priors for T_ing."""

from __future__ import annotations

import numpy as np

from digi_nutri.protocol import MAX_INGREDIENTS
from digi_nutri.usda import USDADictionary


def build_vocab(usda: USDADictionary) -> dict:
    tokens: dict[str, int] = {}
    next_index = 1
    for name in sorted(usda.name_vector):
        tokens[name] = next_index
        next_index += 1
    for name in sorted(usda.ambiguous_names):
        for row in usda.ambiguous_names[name]:
            tokens[f"{name}#{row['id']}"] = next_index
            next_index += 1
    return {"pad": 0, "tokens": tokens}


def encode_ingredients(
    dish: dict,
    usda: USDADictionary,
    vocab: dict,
    max_ingredients: int = MAX_INGREDIENTS,
) -> tuple[list[int], list[list[float]], list[bool]]:
    """Encode unique ingredient ids. Plate grams are not an input."""
    name_ids: list[int] = []
    priors: list[list[float]] = []
    seen: set[str] = set()
    for ingredient in dish["ingredients"]:
        token, vector = usda.prior_for(ingredient["id"], ingredient["name"])
        if token in seen:
            continue
        seen.add(token)
        if token not in vocab["tokens"]:
            raise KeyError(f"token {token!r} is missing from the frozen vocabulary")
        name_ids.append(int(vocab["tokens"][token]))
        priors.append([float(value) for value in vector.tolist()])
    if len(name_ids) > max_ingredients:
        raise RuntimeError(
            f"{dish['dish_id']} has {len(name_ids)} ingredients, above the frozen cap {max_ingredients}"
        )
    valid = len(name_ids)
    while len(name_ids) < max_ingredients:
        name_ids.append(0)
        priors.append([0.0, 0.0, 0.0, 0.0])
    mask = [index < valid for index in range(max_ingredients)]
    return name_ids, priors, mask


def assert_ingredients_resolve(dish_ids: list[str], dishes: dict[str, dict], usda: USDADictionary) -> int:
    maximum = 0
    for dish_id in dish_ids:
        dish = dishes[dish_id]
        seen: set[str] = set()
        for ingredient in dish["ingredients"]:
            token, _vector = usda.prior_for(ingredient["id"], ingredient["name"])
            seen.add(token)
        maximum = max(maximum, len(seen))
    return maximum
