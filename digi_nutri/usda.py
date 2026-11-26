"""Non-leaking static per-gram density dictionary."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REQUIRED_COLUMNS = ("ingr", "id", "cal/g", "fat(g)", "carb(g)", "protein(g)")
DENSITY_COLUMNS = ("cal/g", "fat(g)", "carb(g)", "protein(g)")


class USDAError(ValueError):
    pass


def canonical_id(raw_id: str) -> str:
    text = raw_id.strip()
    if text.startswith("ingr_"):
        text = text.rsplit("_", 1)[-1]
    if text.isdigit():
        return str(int(text))
    return text.lower()


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


@dataclass
class USDADictionary:
    by_id: dict[str, np.ndarray]
    id_name: dict[str, str]
    name_vector: dict[str, np.ndarray]
    ambiguous_names: dict[str, list[dict]]
    kept_duplicate_names: list[str] = field(default_factory=list)

    def prior_for(self, raw_id: str, name: str) -> tuple[str, np.ndarray]:
        """Return (token key, per-gram vector) or raise if the row is unsafe."""
        ident = canonical_id(raw_id)
        label = normalize_name(name)
        if ident not in self.by_id:
            raise USDAError(f"ingredient id {raw_id!r} is not in the static dictionary")
        row_name = self.id_name[ident]
        # A blank dish label keeps the unique dictionary row for that id.
        # A non-empty label that disagrees with the row is a conflict.
        if label and row_name != label:
            raise USDAError(
                f"ingredient id {ident} is named {row_name!r} in the dictionary "
                f"and {label!r} on the dish"
            )
        resolved_name = label or row_name
        if resolved_name in self.ambiguous_names:
            token = f"{resolved_name}#{ident}"
        else:
            token = resolved_name
        return token, self.by_id[ident].copy()


def _parse_table(text: str) -> list[dict]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise USDAError("USDA table is empty")
    header = tuple(part.strip() for part in lines[0].split(","))
    if header != REQUIRED_COLUMNS:
        raise USDAError(
            "USDA table must have exactly the columns "
            f"{REQUIRED_COLUMNS}, found {header}"
        )
    rows = []
    for line_number, line in enumerate(lines[1:], start=2):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(REQUIRED_COLUMNS):
            raise USDAError(f"line {line_number} does not have {len(REQUIRED_COLUMNS)} fields")
        record = dict(zip(REQUIRED_COLUMNS, parts))
        try:
            vector = np.asarray([float(record[col]) for col in DENSITY_COLUMNS], dtype=np.float64)
        except ValueError as exc:
            raise USDAError(f"line {line_number} has a non-numeric density") from exc
        rows.append(
            {
                "name": normalize_name(record["ingr"]),
                "id": canonical_id(record["id"]),
                "vector": vector,
            }
        )
    return rows


def load_usda_text(text: str) -> USDADictionary:
    """Build the static dictionary.

    Repeated names with the same per-gram vector keep the first row.
    Repeated names with different vectors are not name-addressable.
    Each ingredient id still has its own row. Callers must not substitute
    one conflicting vector for another.
    """
    parsed = _parse_table(text)
    by_id: dict[str, np.ndarray] = {}
    id_name: dict[str, str] = {}
    grouped: dict[str, list[dict]] = {}
    for row in parsed:
        grouped.setdefault(row["name"], []).append(row)
        if row["id"] in by_id:
            if not np.allclose(by_id[row["id"]], row["vector"], rtol=0.0, atol=0.0):
                raise USDAError(f"ingredient id {row['id']} has conflicting densities")
            if id_name[row["id"]] != row["name"]:
                raise USDAError(f"ingredient id {row['id']} has conflicting names")
            continue
        by_id[row["id"]] = row["vector"].astype(np.float32)
        id_name[row["id"]] = row["name"]

    name_vector: dict[str, np.ndarray] = {}
    ambiguous: dict[str, list[dict]] = {}
    kept: list[str] = []
    for name, rows in grouped.items():
        vectors = [row["vector"] for row in rows]
        same = all(np.allclose(vectors[0], vector, rtol=0.0, atol=0.0) for vector in vectors[1:])
        if same:
            name_vector[name] = by_id[rows[0]["id"]].copy()
            if len(rows) > 1:
                kept.append(name)
        else:
            ambiguous[name] = [
                {
                    "id": row["id"],
                    "cal_per_g": float(row["vector"][0]),
                    "fat_g_per_g": float(row["vector"][1]),
                    "carb_g_per_g": float(row["vector"][2]),
                    "protein_g_per_g": float(row["vector"][3]),
                }
                for row in rows
            ]
    return USDADictionary(
        by_id=by_id,
        id_name=id_name,
        name_vector=name_vector,
        ambiguous_names=ambiguous,
        kept_duplicate_names=kept,
    )


def load_usda(path: str | Path) -> USDADictionary:
    return load_usda_text(Path(path).read_text())
