"""Provenance Validity: ``n == code × portion/100 × per100g``.

The evidence edge ``n<-code×portion`` must recompute from a catalog.
A real catalog (``foods`` is a list, as in ``data/fndds/2019-2020/foods.json``)
is accepted only when the file identifies FNDDS **2019–2020**. The offline
mock (``foods`` is a dict, or the release text contains ``MOCK``) does not
claim to be that 5624-code release.

Nutrient keys checked: energy_kcal, protein_g, carb_g, fat_g.
Optional ``mass_g`` must match ``portion_g`` when present.
Tolerances default to abs 1e-2 and rel 1e-5. They are not fit on test.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

NUTRI_KEYS = ("energy_kcal", "protein_g", "carb_g", "fat_g")

# PLAN.md pins this release. 5624 is the documented full-catalog size.
FNDDS_PINNED_RELEASE = "2019-2020"
FNDDS_PINNED_N_CODES = 5624

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOCK_CATALOG = _REPO_ROOT / "tests" / "fixtures" / "fndds_mock_foods.json"


class CatalogPinError(ValueError):
    """Raised when a real catalog is not the pinned FNDDS 2019–2020 release."""


@dataclass(frozen=True)
class CatalogInfo:
    path: str
    kind: str
    pinned_release: str | None
    n_foods: int
    full_release: bool
    version: str | None
    release_field: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "pinned_release": self.pinned_release,
            "n_foods": self.n_foods,
            "full_release": self.full_release,
            "version": self.version,
            "release_field": self.release_field,
        }


def _identifies_pinned_release(raw: Mapping[str, Any], path: Path) -> bool:
    release = str(raw.get("release") or "")
    version = str(raw.get("version") or "")
    text = " ".join([release, version, str(path)]).replace("–", "-").replace("—", "-")
    return FNDDS_PINNED_RELEASE in text


def _is_mock(raw: Mapping[str, Any]) -> bool:
    foods = raw.get("foods")
    if isinstance(foods, Mapping):
        return True
    blob = " ".join(
        str(raw.get(key) or "") for key in ("release", "version", "note")
    ).upper()
    return "MOCK" in blob


def _per100g(entry: Mapping[str, Any]) -> dict[str, float]:
    if isinstance(entry.get("per100g"), Mapping):
        per = entry["per100g"]
    elif isinstance(entry.get("per_100g"), Mapping):
        per = entry["per_100g"]
    else:
        per = entry
    out: dict[str, float] = {}
    for key in NUTRI_KEYS:
        if key in per and per[key] is not None:
            out[key] = float(per[key])
    return out


def parse_foods(raw: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    foods = raw.get("foods", raw)
    out: dict[str, dict[str, float]] = {}
    if isinstance(foods, Mapping):
        for code, entry in foods.items():
            if isinstance(entry, Mapping):
                out[str(code)] = _per100g(entry)
        return out
    if isinstance(foods, list):
        for entry in foods:
            if not isinstance(entry, Mapping) or entry.get("code") is None:
                continue
            out[str(entry["code"])] = _per100g(entry)
        return out
    raise ValueError("catalog foods must be a list or a code-keyed object")


def resolve_catalog_file(path: Path) -> Path:
    if path.is_dir():
        return path / "foods.json"
    return path


def load_catalog(path: Path) -> tuple[dict[str, dict[str, float]], CatalogInfo]:
    """Load per-100g nutrients and pin a real catalog to FNDDS 2019–2020."""
    file_path = resolve_catalog_file(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"catalog not found: {file_path}")
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"catalog root must be an object: {file_path}")
    foods = parse_foods(raw)
    version = raw.get("version")
    release_field = raw.get("release")
    if _is_mock(raw):
        info = CatalogInfo(
            path=str(file_path),
            kind="mock",
            pinned_release=None,
            n_foods=len(foods),
            full_release=False,
            version=None if version is None else str(version),
            release_field=None if release_field is None else str(release_field),
        )
        return foods, info
    if not _identifies_pinned_release(raw, file_path):
        raise CatalogPinError(
            "real FNDDS catalog must be pinned to "
            f"{FNDDS_PINNED_RELEASE} (version, release, or path). "
            f"Got version={version!r} release={release_field!r} path={file_path}"
        )
    info = CatalogInfo(
        path=str(file_path),
        kind="fndds-2019-2020",
        pinned_release=FNDDS_PINNED_RELEASE,
        n_foods=len(foods),
        full_release=len(foods) == FNDDS_PINNED_N_CODES,
        version=None if version is None else str(version),
        release_field=None if release_field is None else str(release_field),
    )
    return foods, info


def scale_expected(per100g: Mapping[str, float], portion_g: float) -> dict[str, float]:
    """``per100g * portion_g / 100``."""
    factor = float(portion_g) / 100.0
    return {key: float(per100g[key]) * factor for key in NUTRI_KEYS if key in per100g}


def check_record(
    record: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, float]],
    *,
    atol: float = 1e-2,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Validate one nutrition claim against the catalog.

    Aliases: ``food_code`` / ``code`` / ``code_i``; ``portion_g`` /
    ``portion_g_i``; claimed nutrients in ``nutrients`` / ``nutri_i`` / ``n``.
    """
    code = record.get("food_code", record.get("code", record.get("code_i")))
    portion = record.get("portion_g", record.get("portion_g_i"))
    claimed = record.get("nutrients", record.get("nutri_i", record.get("n")))
    rid = record.get("id", record.get("dish_id", record.get("step_i", "?")))
    result: dict[str, Any] = {
        "id": rid,
        "food_code": code,
        "portion_g": portion,
        "ok": False,
        "broken_edges": [],
        "details": {},
    }
    if code is None:
        result["broken_edges"].append("missing_food_code")
        return result
    if portion is None:
        result["broken_edges"].append("missing_portion_g")
        return result
    if not isinstance(claimed, Mapping):
        result["broken_edges"].append("missing_claimed_nutrients")
        return result

    code_s = str(code)
    if code_s not in catalog:
        result["broken_edges"].append(f"code_not_in_catalog:{code_s}")
        return result

    expected = scale_expected(catalog[code_s], float(portion))
    all_ok = True
    for key in NUTRI_KEYS:
        if key not in expected:
            continue
        if claimed.get(key) is None:
            result["broken_edges"].append(f"missing_claimed:{key}")
            all_ok = False
            continue
        got = float(claimed[key])
        exp = float(expected[key])
        close = math.isclose(got, exp, rel_tol=rtol, abs_tol=atol)
        result["details"][key] = {"claimed": got, "expected": exp, "ok": close}
        if not close:
            result["broken_edges"].append(f"n<-code×portion:{key}")
            all_ok = False

    if claimed.get("mass_g") is not None:
        mass_ok = math.isclose(
            float(claimed["mass_g"]), float(portion), rel_tol=rtol, abs_tol=atol
        )
        result["details"]["mass_g"] = {
            "claimed": float(claimed["mass_g"]),
            "expected": float(portion),
            "ok": mass_ok,
        }
        if not mass_ok:
            result["broken_edges"].append("mass_g!=portion_g")
            all_ok = False

    result["ok"] = all_ok and not result["broken_edges"]
    return result


def evaluate(
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, float]],
    *,
    atol: float = 1e-2,
) -> dict[str, Any]:
    rows = [check_record(record, catalog, atol=atol) for record in records]
    n = len(rows)
    n_ok = sum(1 for row in rows if row["ok"])
    edge_counts: dict[str, int] = {}
    for row in rows:
        for edge in row["broken_edges"]:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return {
        "n_records": n,
        "n_pass": n_ok,
        "n_fail": n - n_ok,
        "pass_rate": (n_ok / n) if n else 0.0,
        "broken_edge_counts": edge_counts,
        "rows": rows,
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "Provenance Validity",
        "===================",
        f"catalog_kind    : {summary.get('catalog_kind', '')}",
        f"pinned_release  : {summary.get('pinned_release')}",
        f"n_catalog_foods : {summary.get('n_catalog_foods', '')}",
        f"n_records : {summary['n_records']}",
        f"n_pass    : {summary['n_pass']}",
        f"n_fail    : {summary['n_fail']}",
        f"pass_rate : {summary['pass_rate']:.4f}",
        f"broken_edge_counts: {json.dumps(summary['broken_edge_counts'], sort_keys=True)}",
        "",
        "per-record:",
    ]
    for row in summary["rows"]:
        status = "PASS" if row["ok"] else "FAIL"
        lines.append(
            f"  [{status}] id={row['id']} code={row['food_code']} edges={row['broken_edges']}"
        )
    return "\n".join(lines) + "\n"
