#!/usr/bin/env python3
"""CPU smoke for the train-only weak-step teacher. No VLM, no test GT.

Covers schema-valid trajectories, the rgb_test refusal, a toy catalog
(pass / corrected code / re-portion / abstain / rewrite retrieve), and a
two-dish prefix of official rgb_train when the cafe metadata and FNDDS
catalog are present.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.data.n5k import (  # noqa: E402
    DishMeta,
    Ingredient,
    load_dish_metadata,
    parse_dish_metadata_line,
    read_ids,
    test_split_ids,
)
from vdr.schemas.loader import IR_STEP_SCHEMA_PATH, load_schema, validate_instance  # noqa: E402
from vdr.tools.fndds_retrieve import RetrieveResult  # noqa: E402
from vdr.tools.portion_grid import PortionResult  # noqa: E402
from vdr.weak.teacher import (  # noqa: E402
    PROTECTED_RESULT_NAMES,
    TEACHER_ACTIONS,
    TestSplitRefused,
    UnsafeOutputPath,
    assert_safe_output,
    generate_dish_steps,
    pool_of_dish,
    refuse_test_split,
    select_train_dishes,
)

IR_SCHEMA = load_schema(IR_STEP_SCHEMA_PATH)
FORBIDDEN_OBS_KEYS = {
    "gt",
    "train_gt",
    "train_gt_energy_kcal",
    "label_energy_kcal",
    "dish_gt",
}


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    return ok


def _ing(
    name: str,
    mass: float,
    kcal: float,
    *,
    fat: float = 1.0,
    carb: float = 1.0,
    protein: float = 1.0,
    ingr_id: str = "ingr_toy",
) -> Ingredient:
    return Ingredient(ingr_id, name, mass, kcal, fat, carb, protein)


def _dish(dish_id: str, kcal: float, mass: float, ingredients: list[Ingredient]) -> DishMeta:
    return DishMeta(dish_id, kcal, mass, 1.0, 1.0, 1.0, tuple(ingredients))


def _food(code: str, name: str, kcal: float, aliases: list[str] | None = None) -> dict:
    return {
        "code": code,
        "name": name,
        "aliases": aliases or [],
        "per_100g": {
            "energy_kcal": kcal,
            "protein_g": 1.0,
            "carb_g": 2.0,
            "fat_g": 0.5,
            "water_g": 50.0,
        },
    }


def _write_catalog(foods: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix="vdr-weak-",
        suffix=".json",
        mode="w",
        encoding="utf-8",
        delete=False,
    )
    json.dump({"version": "weak-smoke", "foods": foods}, handle)
    handle.close()
    return Path(handle.name)


def _assert_steps(steps: list[dict]) -> str | None:
    actions = []
    for step in steps:
        try:
            validate_instance(step, IR_SCHEMA)
        except Exception as exc:  # jsonschema.ValidationError
            return f"schema: {exc}"
        action = step["action_t"]
        actions.append(action)
        if action not in TEACHER_ACTIONS or action == "VOLUME_ESTIMATE":
            return f"bad action {action}"
        if _has_forbidden_key(step):
            return "privileged GT key leaked into an IR step"
        obs = step["obs_t"]
        if obs.get("label_quality") != "weak":
            return "obs missing label_quality=weak"
        if obs.get("volume_tool") is not False:
            return "volume tool flag"
        if step["hyp_i"] and "kcal" in str(step["hyp_i"]).lower():
            return f"hypothesis looks like a GT string: {step['hyp_i']}"
    if not actions or actions[-1] not in {"STOP", "ABSTAIN"}:
        return f"trajectory does not end in STOP/ABSTAIN: {actions}"
    return None


def _has_forbidden_key(obj: object) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in FORBIDDEN_OBS_KEYS:
                return True
            if _has_forbidden_key(value):
                return True
    elif isinstance(obj, list):
        return any(_has_forbidden_key(item) for item in obj)
    return False


def test_parser() -> bool:
    ok = True
    line = (
        "dish_toy,10,20,1,2,3,"
        "ingr_1,foo,bar,5,4,0.1,0.2,0.3,"
        "ingr_2,rice,10,11,0.1,2,0.4"
    )
    dish = parse_dish_metadata_line(line)
    ok &= check(
        "comma inside ingredient name",
        dish is not None
        and len(dish.ingredients) == 2
        and dish.ingredients[0].name == "foo,bar"
        and dish.ingredients[0].mass_g == 5
        and dish.ingredients[1].name == "rice"
        and dish.ingredients[1].energy_kcal == 11,
        str(dish),
    )
    meta_path = ROOT / "data" / "nutrition5k" / "metadata" / "dish_metadata_cafe1.csv"
    first = meta_path.read_text(encoding="utf-8").splitlines()[0]
    real = parse_dish_metadata_line(first)
    if real is None or not real.ingredients:
        return check("cafe1 first line parses ingredients", False, str(real))
    mass = sum(ing.mass_g for ing in real.ingredients)
    rel = abs(mass - real.mass_g) / max(real.mass_g, 1.0)
    ok &= check(
        "cafe1 first line ingredient mass sums to the dish",
        rel < 0.02,
        f"sum={mass:.3f} dish={real.mass_g:.3f} rel={rel:.4f} n={len(real.ingredients)}",
    )
    return ok


def test_guards() -> bool:
    ok = True
    refused = False
    try:
        refuse_test_split("rgb_test")
    except TestSplitRefused:
        refused = True
    ok &= check("refuse rgb_test by name", refused)
    refused = False
    try:
        refuse_test_split("depth_test")
    except TestSplitRefused:
        refused = True
    ok &= check("refuse depth_test by name", refused)
    try:
        refuse_test_split("rgb_train")
        ok &= check("accept rgb_train", True)
    except TestSplitRefused as exc:
        ok &= check("accept rgb_train", False, str(exc))

    test_ids = test_split_ids()
    sample_test = next(iter(test_ids))
    blew = False
    try:
        select_train_dishes(split="rgb_train", pool="all", dish_ids=[sample_test])
    except TestSplitRefused:
        blew = True
    ok &= check("explicit test dish id is refused", blew, sample_test)

    for name in PROTECTED_RESULT_NAMES:
        blocked = False
        try:
            assert_safe_output(Path("results") / name)
        except UnsafeOutputPath:
            blocked = True
        ok &= check(f"protect {name}", blocked)

    stable = pool_of_dish("dish_1561662216")
    ok &= check(
        "hold-out pool is stable",
        stable == pool_of_dish("dish_1561662216") and stable in {"fit", "holdout"},
        stable,
    )
    return ok


def test_toy_catalog() -> bool:
    ok = True
    catalog = _write_catalog(
        [
            _food("RICE", "brown rice cooked", 111.0, ["brown rice"]),
            _food("BAN", "banana raw", 89.0, ["banana"]),
            _food("BAD", "apple", 400.0, ["apple"]),
            _food("GOOD", "apple baked", 52.0, ["apple"]),
        ]
    )
    cases = [
        (
            "pass",
            _dish("toy_pass", 89.0, 100.0, [_ing("banana", 100, 89)]),
            ["RETRIEVE", "STOP"],
            "pass",
        ),
        (
            "re-portion",
            _dish("toy_portion", 166.5, 150.0, [_ing("brown rice", 150, 166.5)]),
            ["RETRIEVE", "RE_PORTION", "STOP"],
            "corrected",
        ),
        (
            "correct code",
            _dish("toy_code", 52.0, 100.0, [_ing("apple", 100, 52)]),
            ["RETRIEVE", "CORRECT_CODE", "STOP"],
            "corrected",
        ),
        (
            "abstain",
            _dish("toy_abstain", 80.0, 50.0, [_ing("mysteryfoam", 50, 80)]),
            ["RETRIEVE", "ABSTAIN"],
            "abstain",
        ),
        (
            "drop salt",
            _dish(
                "toy_salt",
                89.0,
                100.2,
                [_ing("salt", 0.2, 0.0), _ing("banana", 100, 89)],
            ),
            ["RETRIEVE", "STOP"],
            "pass",
        ),
    ]
    for name, dish, actions, status in cases:
        steps, audit = generate_dish_steps(dish, catalog_path=catalog)
        err = _assert_steps(steps)
        got = [step["action_t"] for step in steps]
        ok &= check(f"toy {name} actions", err is None and got == actions and steps[-1]["s"] == status, f"{got} s={steps[-1]['s']} {err or ''}")
        if name == "pass":
            nutri = steps[-1]["nutri_dish"] or {}
            ok &= check("toy pass nutri has no water_g", "water_g" not in nutri, str(nutri))
        if name == "drop salt":
            hyps = [step["hyp_i"] for step in steps if step["hyp_i"]]
            ok &= check("salt is not a hypothesis", hyps == ["banana"], str(hyps))
        if name != "abstain":
            ok &= check(
                f"toy {name} audit stays out of the IR",
                "train_gt_energy_kcal" not in json.dumps(steps),
                "",
            )
        else:
            ok &= check("abstain emits null energy", steps[-1]["nutri_dish"]["energy_kcal"] is None)
        _ = audit
    return ok


def test_rewrite_retrieve() -> bool:
    """Second RETRIEVE uses the real produce rewrite; retrieve itself is fake."""

    def retrieve_fn(query: str, top_k: int = 5, catalog_path: str | None = None, **kwargs: object):
        if query == "kale":
            return RetrieveResult(
                query,
                [{"code": "DIP", "score": 0.90, "name": "Kale dip"}],
                "DIP",
            )
        if query == "kale raw":
            return RetrieveResult(
                query,
                [{"code": "RAW", "score": 0.92, "name": "Kale, raw"}],
                "RAW",
            )
        raise AssertionError(query)

    def lookup_fn(code: str, catalog_path: str | None = None, **kwargs: object) -> dict:
        table = {"DIP": 180.0, "RAW": 30.0}
        return {
            "energy_kcal": table[code],
            "protein_g": 2.0,
            "carb_g": 4.0,
            "fat_g": 0.4,
        }

    def portion_fn(code: str, hint_g: float = 100.0, catalog_path: str | None = None, **kwargs: object):
        return PortionResult(code=code, portion_g=float(hint_g), grid=[float(hint_g)], hint_g=float(hint_g))

    dish = _dish("toy_rewrite", 45.0, 150.0, [_ing("kale", 150, 45.0)])
    steps, _audit = generate_dish_steps(
        dish,
        retrieve_fn=retrieve_fn,
        lookup_fn=lookup_fn,
        portion_fn=portion_fn,
    )
    err = _assert_steps(steps)
    got = [step["action_t"] for step in steps]
    queries = [step["obs_t"].get("query") for step in steps if step["action_t"] == "RETRIEVE"]
    return check(
        "rewrite retrieve then re-portion",
        err is None and got == ["RETRIEVE", "RETRIEVE", "RE_PORTION", "STOP"] and queries == ["kale", "kale raw"],
        f"{got} queries={queries} {err or ''}",
    )


def test_real_prefix() -> bool:
    meta = ROOT / "data" / "nutrition5k" / "metadata" / "dish_metadata_cafe1.csv"
    catalog = ROOT / "data" / "fndds" / "2019-2020" / "foods.json"
    if not meta.is_file() or not catalog.is_file():
        return check("real rgb_train prefix", False, "metadata or FNDDS catalog missing")
    dishes = select_train_dishes(split="rgb_train", pool="fit", n=2)
    if len(dishes) != 2:
        return check("real rgb_train prefix selects 2", False, str(len(dishes)))
    forbidden = test_split_ids()
    train = set(read_ids("rgb_train"))
    ok = True
    # Confirm the loader saw the official train table, not a hand-built subset.
    table = load_dish_metadata()
    ok &= check("metadata covers the two train dishes", all(d.dish_id in table for d in dishes))
    for dish in dishes:
        steps, audit = generate_dish_steps(dish, catalog_path=catalog)
        err = _assert_steps(steps)
        leaked = dish.dish_id in forbidden or dish.dish_id not in train
        ok &= check(
            f"real {dish.dish_id}",
            err is None and not leaked and audit["pool"] == "fit" and steps[-1]["s"] in {"pass", "corrected", "abstain"},
            f"s={steps[-1]['s']} steps={len(steps)} kept={audit['n_kept']} {err or ''}",
        )
        if any(step["hyp_i"] for step in steps):
            ok &= check(
                f"real {dish.dish_id} hyp is an ingredient name",
                all(
                    step["hyp_i"] is None or step["hyp_i"] in {ing.name for ing in dish.ingredients}
                    for step in steps
                ),
            )
    return ok


def test_cli_writes_train_prefix() -> bool:
    """CLI round-trip on two fit-pool train dishes. Output stays out of the frozen names."""
    with tempfile.TemporaryDirectory(prefix="vdr-weak-out-") as tmp:
        out = Path(tmp) / "rgb_train_fit_n2.jsonl"
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "gen_weak_steps_train.py"),
                "--split",
                "rgb_train",
                "--pool",
                "fit",
                "--n",
                "2",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return check("CLI train prefix", False, proc.stderr.strip() or proc.stdout.strip())
        summary_path = out.with_name("rgb_train_fit_n2.summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        forbidden = test_split_ids()
        ids = {row["dish_id"] for row in summary["dishes"]}
        leaked = ids & forbidden
        steps_ok = True
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            step = json.loads(line)
            try:
                validate_instance(step, IR_SCHEMA)
            except Exception as exc:
                steps_ok = False
                detail = str(exc)
                break
        else:
            detail = proc.stdout.strip()
        return check(
            "CLI train prefix writes schema-valid steps",
            steps_ok
            and summary["split"] == "rgb_train"
            and summary["uses_test_gt"] is False
            and summary["vlm"] is False
            and not leaked
            and summary["n_samples"] == 2,
            detail,
        )


def test_cli_refuses_test_split() -> bool:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_weak_steps_train.py"), "--split", "rgb_test", "--n", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return check(
        "CLI --split rgb_test exits 2",
        proc.returncode == 2 and "rgb_test" in proc.stderr,
        proc.stderr.strip() or proc.stdout.strip(),
    )


def main() -> int:
    print("weak-step teacher smoke")
    ok = True
    ok &= test_parser()
    ok &= test_guards()
    ok &= test_toy_catalog()
    ok &= test_rewrite_retrieve()
    ok &= test_cli_refuses_test_split()
    ok &= test_cli_writes_train_prefix()
    ok &= test_real_prefix()
    if not ok:
        print("FAILED")
        return 1
    print("All weak-step smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
