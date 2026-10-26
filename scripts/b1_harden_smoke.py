#!/usr/bin/env python3
"""CPU smoke for B1 hardening. No VLM, no Nutrition5K imagery.

Covers duplicate-hypothesis dedupe, portion-grid selection, produce retrieve
(raw onion / blueberry over dip / juice), and a schema-valid B1 emit whose
public ``n`` has no ``water_g``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.baselines.b1_rag_once import (  # noqa: E402
    _second_retrieve,
    choose_followup,
    predict,
    predict_from_hypotheses,
)
from vdr.perception.hypotheses import dedupe_hypotheses  # noqa: E402
from vdr.schemas import VDR_OUTPUT_SCHEMA_PATH, load_schema, validate_instance  # noqa: E402
from vdr.tools.fndds_retrieve import retrieve  # noqa: E402
from vdr.tools.nutrition_lookup import FROZEN_N_KEYS  # noqa: E402
from vdr.tools.portion_grid import (  # noqa: E402
    DEFAULT_PORTION_GRID_G,
    search_portion,
)

TOY = ROOT / "data" / "fndds" / "toy" / "foods.json"
REAL = ROOT / "data" / "fndds" / "2019-2020" / "foods.json"
FROZEN = set(FROZEN_N_KEYS)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    return ok


def _write_catalog(foods: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix="vdr-foods-",
        suffix=".json",
        mode="w",
        encoding="utf-8",
        delete=False,
    )
    json.dump({"version": "smoke", "foods": foods}, handle)
    handle.close()
    return Path(handle.name)


def _per(kcal: float, water: float = 50.0) -> dict:
    return {
        "energy_kcal": kcal,
        "protein_g": 1.0,
        "carb_g": 2.0,
        "fat_g": 0.5,
        "water_g": water,
    }


def test_dedupe() -> bool:
    ok = True
    merged = dedupe_hypotheses(
        [{"name": "Sweet Potato Cubes", "portion_g": 100} for _ in range(5)]
    )
    ok &= check(
        "dedupe five identical names",
        len(merged) == 1 and merged[0]["portion_g"] == 100,
        str(merged),
    )
    mixed = dedupe_hypotheses(
        [
            {"name": "Red Onion", "portion_g": 50},
            {"name": " red onion ", "portion_g": 150},
            {"name": "RED ONION", "portion_g": 100},
        ]
    )
    ok &= check(
        "dedupe casefold median not sum",
        len(mixed) == 1 and mixed[0]["portion_g"] == 100,
        str(mixed),
    )
    distinct = dedupe_hypotheses(
        [
            {"name": "Sweet Potato", "portion_g": 80},
            {"name": "Sweet Potato Cubes", "portion_g": 40},
        ]
    )
    ok &= check("distinct names stay split", len(distinct) == 2, str(distinct))
    return ok


def test_portion_grid() -> bool:
    ok = True
    ok &= check(
        "default grid documented ends",
        DEFAULT_PORTION_GRID_G[0] == 5 and DEFAULT_PORTION_GRID_G[-1] == 500 and 100 in DEFAULT_PORTION_GRID_G and 180 not in DEFAULT_PORTION_GRID_G,
        str(DEFAULT_PORTION_GRID_G),
    )
    catalog = _write_catalog(
        [
            {"code": "LOW", "name": "low energy food", "aliases": ["low"], "per_100g": _per(100)},
            {"code": "HIGH", "name": "high energy food", "aliases": ["high"], "per_100g": _per(900)},
        ]
    )
    on_hint = search_portion("LOW", hint_g=100, catalog_path=catalog)
    off_hint = search_portion("LOW", hint_g=180, catalog_path=catalog)
    runaway = search_portion("HIGH", hint_g=5000, catalog_path=catalog)
    ok &= check("grid keeps on-grid hint 100", on_hint.portion_g == 100, str(on_hint.portion_g))
    ok &= check("grid snaps 180 to 175", off_hint.portion_g == 175, str(off_hint.portion_g))
    ok &= check(
        "grid refuses runaway high-kcal hint",
        runaway.portion_g == 80,
        f"portion={runaway.portion_g} hint={runaway.hint_g}",
    )
    return ok


def test_retrieve() -> bool:
    ok = True
    onion = retrieve("red onion", top_k=3, catalog_path=TOY)
    ok &= check(
        "toy red onion prefers raw",
        onion.selected_code == "T01200100" and "dip" not in (onion.cands[0]["name"] or "").lower(),
        str(onion.cands),
    )
    dip = retrieve("onion dip", top_k=1, catalog_path=TOY)
    ok &= check(
        "toy query onion dip still selects dip",
        dip.selected_code == "T01200200",
        str(dip.cands),
    )
    berry = retrieve("blueberry", top_k=2, catalog_path=TOY)
    ok &= check(
        "toy blueberry prefers raw over juice",
        berry.selected_code == "T01300100",
        str(berry.cands),
    )
    juice = retrieve("blueberry juice", top_k=1, catalog_path=TOY)
    ok &= check(
        "toy query blueberry juice still selects juice",
        juice.selected_code == "T01300200",
        str(juice.cands),
    )
    if REAL.is_file():
        real_onion = retrieve("red onion", top_k=1, catalog_path=REAL)
        real_name = (real_onion.cands[0]["name"] or "") if real_onion.cands else ""
        ok &= check(
            "real catalog red onion is not a dip",
            "onion" in real_name.lower() and "dip" not in real_name.lower() and "soup" not in real_name.lower(),
            real_name,
        )
        real_berry = retrieve("blueberry", top_k=1, catalog_path=REAL)
        berry_name = (real_berry.cands[0]["name"] or "") if real_berry.cands else ""
        ok &= check(
            "real catalog blueberry is not juice",
            "blueberr" in berry_name.lower() and "juice" not in berry_name.lower(),
            berry_name,
        )
    else:
        print("  SKIP real catalog (data/fndds/2019-2020/foods.json missing)")
    return ok


def _assert_frozen(pred: dict, schema: dict) -> bool:
    validate_instance(pred, schema)
    keys = set(pred["n"])
    return keys == FROZEN and "water_g" not in pred["n"]


def test_b1_emit(schema: dict) -> bool:
    ok = True
    hyps = [{"name": "Sweet Potato Cubes", "portion_g": 100} for _ in range(5)]
    hyps.append({"name": "red onion", "portion_g": 50})
    pred = predict_from_hypotheses("smoke-dedupe", hyps, catalog_path=TOY)
    ok &= check("b1 deduped emit schema", _assert_frozen(pred, schema), str(pred["n"]))
    ok &= check(
        "b1 dropped duplicate sweet-potato copies",
        pred["meta"]["n_hypotheses_in"] == 6 and pred["meta"]["n_hypotheses"] == 2,
        str(pred["meta"]["n_hypotheses_in"]) + "->" + str(pred["meta"]["n_hypotheses"]),
    )
    # 100 g sweet-potato stand-in is not in the toy catalog; red onion is 40 kcal/100 g.
    # Mass must be one onion portion (50) plus whatever the unmatched or matched
    # sweet potato contributes — never 5 * 100.
    ok &= check(
        "b1 mass is not the summed duplicates",
        pred["n"]["mass_g"] is not None and pred["n"]["mass_g"] < 400,
        str(pred["n"]),
    )
    onion_codes = [
        node.get("attrs", {}).get("code")
        for node in pred["G"]["nodes"]
        if node.get("type") == "food_code"
    ]
    ok &= check(
        "b1 selected raw onion code",
        "T01200100" in onion_codes and "T01200200" not in onion_codes,
        str(onion_codes),
    )
    water_on_evidence = any(
        "water_g" in (node.get("attrs") or {})
        for node in pred["G"]["nodes"]
        if node.get("type") == "nutrition"
    )
    ok &= check("water_g stays on evidence attrs only", water_on_evidence)

    image_pred = predict(
        "/tmp/dish_smoke_rgb.png",
        dish_id="dish_smoke",
        hypotheses=[{"name": "red onion", "portion_g": 100}] * 5,
        catalog_path=TOY,
    )
    ok &= check(
        "image entry dedupes without a VLM",
        image_pred["dish_id"] == "dish_smoke"
        and image_pred["meta"]["n_hypotheses"] == 1
        and image_pred["n"]["mass_g"] == 100
        and _assert_frozen(image_pred, schema),
        str(image_pred["n"]),
    )
    import sys as _sys

    ok &= check(
        "volume tool stays unloaded",
        "vdr.tools.volume_estimate" not in _sys.modules,
    )
    return ok


def test_followup(schema: dict) -> bool:
    ok = True
    prepared = _write_catalog(
        [
            {
                "code": "DIP",
                "name": "Onion dip, light",
                "aliases": ["red onion", "onion dip"],
                "per_100g": _per(166, 36),
            },
            {
                "code": "SOUP",
                "name": "Onion soup, French",
                "aliases": ["red onion", "onion soup"],
                "per_100g": _per(56, 90),
            },
        ]
    )
    pred = predict_from_hypotheses(
        "smoke-dip",
        [{"name": "red onion", "portion_g": 100}] * 5,
        catalog_path=prepared,
    )
    actions = [step["action"] for step in pred["tau"]]
    ok &= check("prepared-only catalog schema", _assert_frozen(pred, schema), str(pred["n"]))
    ok &= check(
        "CODE_CONF_GAP triggers a second RETRIEVE before STOP",
        actions[0] == "RETRIEVE" and actions[-1] == "STOP" and actions.count("RETRIEVE") == 2,
        str(actions),
    )
    ok &= check(
        "duplicate dips are not summed",
        pred["n"]["mass_g"] == 100 and pred["meta"]["n_hypotheses"] == 1,
        str(pred["n"]["mass_g"]),
    )

    tied = _write_catalog(
        [
            {"code": "A", "name": "Widget alpha", "aliases": ["widget"], "per_100g": _per(100, 10)},
            {"code": "B", "name": "Widget beta", "aliases": ["widget"], "per_100g": _per(110, 11)},
        ]
    )
    tied_pred = predict_from_hypotheses(
        "smoke-tie",
        [{"name": "widget", "portion_g": 100}],
        catalog_path=tied,
    )
    tied_actions = [step["action"] for step in tied_pred["tau"]]
    ok &= check("tied plain codes schema", _assert_frozen(tied_pred, schema))
    ok &= check(
        "CODE_CONF_GAP on a plain code runs RE_PORTION before STOP",
        tied_actions == ["RETRIEVE", "RE_PORTION", "STOP"],
        str(tied_actions),
    )
    ok &= check(
        "confirmatory RE_PORTION keeps the on-grid portion",
        tied_pred["n"]["mass_g"] == 100 and tied_pred["s"] == "pass",
        f"s={tied_pred['s']} n={tied_pred['n']}",
    )

    # Policy unit: a prepared hit asks for RETRIEVE; a raw hit asks for RE_PORTION.
    swapped = {
        "code": "T01200200",
        "query": "red onion",
        "selected_name": "Onion dip, light",
        "cands": [{"code": "T01200200", "score": 0.8, "name": "Onion dip, light"}],
        "hint_g": 50.0,
        "name": "red onion",
    }
    changed = _second_retrieve([swapped], catalog_path=TOY, top_k=5)
    ok &= check(
        "second retrieve leaves a dip for raw onion",
        changed and swapped["code"] == "T01200100" and swapped["previous_code"] == "T01200200",
        f"code={swapped['code']} prev={swapped.get('previous_code')} requery={swapped.get('requery')}",
    )
    ok &= check(
        "choose_followup prepared vs raw",
        choose_followup(
            ["CODE_CONF_GAP"],
            [{"code": "D", "query": "red onion", "selected_name": "Onion dip, light"}],
        )
        == "RETRIEVE"
        and choose_followup(
            ["CODE_CONF_GAP"],
            [{"code": "R", "query": "red onion", "selected_name": "Onions, raw"}],
        )
        == "RE_PORTION"
        and choose_followup([], [{"code": "R", "query": "red onion", "selected_name": "Onions, raw"}])
        is None,
    )
    return ok


def main() -> int:
    schema = load_schema(VDR_OUTPUT_SCHEMA_PATH)
    ok = True
    print("dedupe")
    ok &= test_dedupe()
    print("portion_grid")
    ok &= test_portion_grid()
    print("retrieve")
    ok &= test_retrieve()
    print("b1 emit")
    ok &= test_b1_emit(schema)
    print("followup")
    ok &= test_followup(schema)
    print("b1_harden_smoke:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
