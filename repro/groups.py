"""Physical-plate groups. The official split is not rewritten."""

from __future__ import annotations


def dish_timestamp(dish_id: str) -> int:
    try:
        return int(dish_id.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"dish id {dish_id!r} has no unix timestamp") from exc


def nutrition_key(dish: dict) -> tuple[float, ...]:
    nutrition = dish["nutrition"]
    return (
        float(nutrition["energy"]),
        float(dish["mass"]),
        float(nutrition["fat"]),
        float(nutrition["carb"]),
        float(nutrition["protein"]),
    )


def plate_groups(dish_ids: list[str], dishes: dict[str, dict], config: dict) -> dict[str, str]:
    max_gap = int(config["reshoot"]["max_gap_seconds"])
    atol = float(config["reshoot"]["nutrition_atol"])
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
            if later - stamp > max_gap:
                break
            if all(abs(a - b) <= atol for a, b in zip(key, other_key)):
                union(dish_id, other_id)
    return {dish_id: find(dish_id) for dish_id in dish_ids}


def cross_split_pairs(train_ids: list[str], test_ids: list[str], dishes: dict, config: dict) -> list[dict]:
    groups = plate_groups(list(train_ids) + list(test_ids), dishes, config)
    test_set = set(test_ids)
    train_set = set(train_ids)
    buckets: dict[str, dict[str, list[str]]] = {}
    for dish_id, group in groups.items():
        side = "test" if dish_id in test_set else "train"
        if dish_id not in test_set and dish_id not in train_set:
            continue
        buckets.setdefault(group, {"train": [], "test": []})[side].append(dish_id)
    pairs = []
    for group, sides in sorted(buckets.items()):
        if sides["train"] and sides["test"]:
            pairs.append(
                {
                    "group": group,
                    "train": sorted(sides["train"]),
                    "test": sorted(sides["test"]),
                }
            )
    return pairs
