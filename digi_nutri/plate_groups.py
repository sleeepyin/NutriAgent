"""Physical-plate groups. Reshoots of one plate share a group id."""

from __future__ import annotations

from pathlib import Path


def assign_plate_groups(dish_ids: list[str], reshoot_tsv: Path | None = None) -> dict[str, str]:
    """Default group is the dish_id.

    A reshoot file, when present, has lines `group_id dish_id dish_id ...`.
    Dishes not listed keep their own dish_id. This run does not invent groups.
    """

    groups = {dish_id: dish_id for dish_id in dish_ids}
    if reshoot_tsv is None or not Path(reshoot_tsv).is_file():
        return groups
    for line in Path(reshoot_tsv).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) < 2:
            raise ValueError(f"reshoot line needs a group and a dish id: {text}")
        group_id, members = parts[0], parts[1:]
        for dish_id in members:
            if dish_id in groups:
                groups[dish_id] = group_id
    return groups
