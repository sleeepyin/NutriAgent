"""Physical-plate groups for the pre-registered bootstrap."""

from __future__ import annotations

from digi_nutri.protocol import RESHOOT_MAX_GAP_SECONDS, RESHOOT_NUTRITION_ATOL


def dish_timestamp(dish_id: str) -> int:
    try:
        return int(dish_id.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"dish id {dish_id!r} has no unix timestamp") from exc


def nutrition_key(dish: dict) -> tuple[float, float, float, float, float]:
    nutrition = dish["nutrition"]
    return (
        float(nutrition["energy"]),
        float(dish["mass"]),
        float(nutrition["fat"]),
        float(nutrition["carb"]),
        float(nutrition["protein"]),
    )


def plate_groups(
    dish_ids: list[str],
    dishes: dict[str, dict],
    max_gap_seconds: int = RESHOOT_MAX_GAP_SECONDS,
    atol: float = RESHOOT_NUTRITION_ATOL,
) -> dict[str, str]:
    """Map each dish id to a group id.

    The group id is the lexicographically smallest dish id in the component.
    """
    parent = {dish_id: dish_id for dish_id in dish_ids}

    def find(dish_id: str) -> str:
        while parent[dish_id] != dish_id:
            parent[dish_id] = parent[parent[dish_id]]
            dish_id = parent[dish_id]
        return dish_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    indexed = []
    for dish_id in dish_ids:
        indexed.append((dish_timestamp(dish_id), dish_id, nutrition_key(dishes[dish_id])))
    indexed.sort()
    for index, (stamp, dish_id, key) in enumerate(indexed):
        for later, other_id, other_key in indexed[index + 1 :]:
            if later - stamp > max_gap_seconds:
                break
            if all(abs(a - b) <= atol for a, b in zip(key, other_key)):
                union(dish_id, other_id)
    return {dish_id: find(dish_id) for dish_id in dish_ids}


def cross_split_pairs(
    train_ids: list[str],
    test_ids: list[str],
    dishes: dict[str, dict],
    max_gap_seconds: int = RESHOOT_MAX_GAP_SECONDS,
    atol: float = RESHOOT_NUTRITION_ATOL,
) -> list[dict]:
    """Log same-session exact-label pairs that cross the official split.

    The split is not rewritten. These pairs are diagnostic only.
    """
    test_set = set(test_ids)
    train_set = set(train_ids)
    groups = plate_groups(list(train_ids) + list(test_ids), dishes, max_gap_seconds, atol)
    members: dict[str, list[str]] = {}
    for dish_id, group in groups.items():
        members.setdefault(group, []).append(dish_id)
    found = []
    for group, ids in members.items():
        splits = {("test" if dish_id in test_set else "train") for dish_id in ids}
        if splits == {"train", "test"}:
            found.append(
                {
                    "group": group,
                    "train": sorted(dish_id for dish_id in ids if dish_id in train_set),
                    "test": sorted(dish_id for dish_id in ids if dish_id in test_set),
                }
            )
    return found
