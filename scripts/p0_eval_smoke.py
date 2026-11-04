#!/usr/bin/env python3
"""CPU smoke for VDR P0 offline scaffolding.

Runs the confusion --demo, provenance mock demo, and selective demo.
Checks the closed-loop feature sketch without training a policy.
Does not open a volume tool, load regressor weights, or write results/.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vdr.controller.features import FEATURE_DIM, build_features  # noqa: E402
from vdr.controller.state_features import (  # noqa: E402
    BUDGET_SLICE,
    CLOSED_LOOP_READY,
    ENERGY_CAP_KCAL,
    OBS_SLICE,
    PRE_SLICE,
    STATE_FEATURE_DIM,
    STATE_FEATURE_NAMES,
    build_features_for_ablation,
    build_state_features,
)
from vdr.eval.action_metrics import confusion_matrix, overall_accuracy, per_class_prf  # noqa: E402
from vdr.eval.provenance import (  # noqa: E402
    FNDDS_PINNED_N_CODES,
    FNDDS_PINNED_RELEASE,
    CatalogPinError,
    check_record,
    load_catalog,
)
from vdr.eval.selective import aurc, coverage_and_selective_mae  # noqa: E402

FROZEN = (
    "results/n5k_rgb_test_overhead_b0_b1.json",
    "results/n5k_rgb_test_overhead_b0_b1.jsonl",
    "results/n5k_subset_b0_b1.json",
    "results/n5k_subset_b0_b1.jsonl",
    "results/n5k_subset_b0_b1_n50_seed42.json",
    "results/n5k_subset_b0_b1_n50_seed42.jsonl",
    "results/weak_steps/rgb_train_fit_all.jsonl",
    "results/weak_steps/rgb_train_holdout_all.jsonl",
)
FROZEN_PREFIXES = {
    "results/n5k_rgb_test_overhead_b0_b1.jsonl": "a4dfdd82e3a595f4",
    "results/weak_steps/rgb_train_fit_all.jsonl": "ed47f9239bfecb3c",
}


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    return ok


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in FROZEN:
        path = ROOT / rel
        if path.is_file():
            out[rel] = _sha256(path)
    return out


def run_cli(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_confusion_demo() -> bool:
    ok = True
    sys.path.insert(0, str(ROOT / "scripts"))
    import action_confusion_matrix as confusion  # noqa: WPS433

    gold, pred = confusion.synthesize_bc_mlp_holdout()
    mat = confusion_matrix(gold, pred)
    prf = per_class_prf(mat)
    acc = overall_accuracy(mat)
    ok &= check("demo rows", len(gold) == 2771, str(len(gold)))
    ok &= check("demo accuracy 2688/2771", abs(acc - (2688 / 2771)) < 1e-12, f"{acc:.4f}")
    ok &= check("printed acc is 0.9700", f"{acc:.4f}" == "0.9700")
    ok &= check("ABSTAIN F1 is 0", prf["ABSTAIN"]["f1"] == 0.0)
    ok &= check("ABSTAIN recall 0/24", prf["ABSTAIN"]["tp"] == 0 and prf["ABSTAIN"]["support"] == 24)
    for name, (hits, support) in confusion.BC_MLP_HOLDOUT_RECALL.items():
        ok &= check(
            f"recall {name}",
            prf[name]["tp"] == hits and prf[name]["support"] == support,
            f"{int(prf[name]['tp'])}/{int(prf[name]['support'])}",
        )
    proc = run_cli("action_confusion_matrix.py", ["--demo", "--expected-abstain-f1", "0"])
    text = proc.stdout
    ok &= check("confusion CLI exit 0", proc.returncode == 0, proc.stderr.strip())
    ok &= check("CLI shows 0.9700", "overall accuracy : 0.9700" in text)
    ok &= check("CLI shows ABSTAIN 0/24", "ABSTAIN F1       : 0.0000" in text and "0/24" in text)
    ok &= check("CLI marks ABSTAIN row", "<< ABSTAIN row" in text)
    return ok


def test_provenance() -> bool:
    ok = True
    catalog, info = load_catalog(ROOT / "tests" / "fixtures" / "fndds_mock_foods.json")
    ok &= check("mock is not the pinned release", info.kind == "mock" and info.pinned_release is None)
    ok &= check("mock has 3 foods", info.n_foods == 3, str(info.n_foods))
    milk = check_record(
        {
            "id": "milk",
            "food_code": "11111000",
            "portion_g": 200.0,
            "nutrients": {
                "energy_kcal": 122.0,
                "protein_g": 6.3,
                "carb_g": 9.6,
                "fat_g": 6.5,
                "mass_g": 200.0,
            },
        },
        catalog,
    )
    ok &= check("milk 200g passes", milk["ok"] is True)
    ok &= check(
        "milk energy is per100g * portion/100",
        milk["details"]["energy_kcal"]["expected"] == 122.0,
        str(milk["details"]["energy_kcal"]["expected"]),
    )
    proc = run_cli("provenance_validity.py", ["--demo"])
    ok &= check("provenance CLI exit 0", proc.returncode == 0, proc.stderr.strip())
    ok &= check("demo pass count 2", "n_pass    : 2" in proc.stdout)
    ok &= check("demo catalog is mock", "catalog_kind    : mock" in proc.stdout)

    toy_raised = False
    try:
        load_catalog(ROOT / "data" / "fndds" / "toy" / "foods.json")
    except CatalogPinError:
        toy_raised = True
    ok &= check("toy catalog is not pinned 2019-2020", toy_raised)

    with tempfile.TemporaryDirectory() as tmp:
        other = Path(tmp) / "foods.json"
        other.write_text(
            '{"version":"fndds-2017-2018","foods":[{"code":"1","per_100g":{"energy_kcal":1}}]}',
            encoding="utf-8",
        )
        rejected = False
        try:
            load_catalog(other)
        except CatalogPinError:
            rejected = True
        ok &= check("other release is rejected", rejected)

    real = ROOT / "data" / "fndds" / "2019-2020" / "foods.json"
    if real.is_file():
        _foods, real_info = load_catalog(real)
        ok &= check(
            "real catalog pinned 2019-2020",
            real_info.kind == "fndds-2019-2020" and real_info.pinned_release == FNDDS_PINNED_RELEASE,
        )
        ok &= check(
            "real catalog has 5624 codes",
            real_info.full_release and real_info.n_foods == FNDDS_PINNED_N_CODES,
            str(real_info.n_foods),
        )
    else:
        ok &= check("real catalog present", False, "missing data/fndds/2019-2020/foods.json")
    return ok


def _independent_aurc(errors: list[float], scores: list[float]) -> float:
    order = sorted(range(len(errors)), key=lambda i: scores[i])
    risks: list[float] = []
    coverages: list[float] = []
    cumulative = 0.0
    n = len(errors)
    for k, idx in enumerate(order, start=1):
        cumulative += errors[idx]
        risks.append(cumulative / k)
        coverages.append(k / n)
    area = 0.5 * risks[0] * coverages[0]
    for i in range(n - 1):
        area += 0.5 * (risks[i] + risks[i + 1]) * (coverages[i + 1] - coverages[i])
    return area


def test_selective() -> bool:
    ok = True
    errors = [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 20.0, 30.0]
    scores = [0.1, 0.2, 0.25, 0.3, 0.6, 0.7, 0.85, 0.95]
    flags = [0, 0, 0, 0, 1, 1, 1, 1]
    flagged = coverage_and_selective_mae(errors, abstain_flags=flags)
    ok &= check("coverage 0.5", flagged["coverage"] == 0.5)
    ok &= check("selective MAE 2.5", flagged["selective_mae"] == 2.5)
    got = aurc(errors, scores)["aurc"]
    expect = _independent_aurc(errors, scores)
    ok &= check("AURC matches trapezoid", abs(got - expect) < 1e-12, f"{got:.6f}")
    ok &= check("AURC below full MAE", got < (sum(errors) / len(errors)))
    proc = run_cli("selective_metrics.py", ["--demo"])
    ok &= check("selective CLI exit 0", proc.returncode == 0, proc.stderr.strip())
    ok &= check("CLI coverage 0.5000", "coverage      : 0.5000" in proc.stdout)
    ok &= check("CLI selective MAE 2.5000", "selective MAE : 2.5000" in proc.stdout)
    return ok


def test_state_sketch() -> bool:
    ok = True
    ok &= check("closed loop is not ready", CLOSED_LOOP_READY is False)
    ok &= check("open-loop width stays 31", FEATURE_DIM == 31)
    ok &= check(
        "state names are unique",
        len(STATE_FEATURE_NAMES) == len(set(STATE_FEATURE_NAMES)) == STATE_FEATURE_DIM,
        str(STATE_FEATURE_DIM),
    )
    h = {
        "hyp_i": "milk",
        "code_i": "11111000",
        "cands_i": [{"code": "11111000", "score": 0.8}, {"code": "other", "score": 0.2}],
        "portion_g_i": 200.0,
        "verify_codes": ["CODE_CONF_GAP"],
        "step_i": 3,
        "obs_t": {"selected_code": "SHOULD_NOT_LEAK", "cands": []},
        "train_gt_energy_kcal": 9999.0,
        "action_t": "STOP",
    }
    bare = {key: h[key] for key in ("hyp_i", "code_i", "cands_i", "portion_g_i", "verify_codes", "step_i")}
    b5 = build_features(bare)
    leaked = build_features(h)
    ok &= check("build_features ignores obs and train GT", np.array_equal(b5, leaked))
    first = build_state_features(h, None, None)
    ok &= check("prefix matches B5", np.array_equal(first[PRE_SLICE], b5))
    obs_first = first[OBS_SLICE]
    missing_at = STATE_FEATURE_NAMES.index("obs_missing") - FEATURE_DIM
    ok &= check("first step sets obs_missing", obs_first[missing_at] == 1.0)
    ok &= check(
        "first step zeros the rest of obs",
        float(np.sum(np.delete(obs_first, missing_at))) == 0.0,
    )
    ok &= check("volume ban bit is 1", first[STATE_FEATURE_NAMES.index("safety_volume_disabled")] == 1.0)
    ok &= check("legacy-weight ban bit is 1", first[STATE_FEATURE_NAMES.index("safety_legacy_weights_off")] == 1.0)

    o_prev = {
        "action_t": "ABSTAIN",
        "abstain_reason": "no_support",
        "s": "abstain",
        "cands": [{"code": "11111000", "score": 0.4}, {"code": "x", "score": 0.39}],
        "selected_code": "11111000",
        "verify_codes": ["UNGROUNDED"],
        "nutrients": {"energy_kcal": 122.0, "mass_g": 200.0},
        "train_gt_energy_kcal": 5000.0,
        "portion_g": 200.0,
        "from_portion_g": 100.0,
        "from": "aaaa",
        "to": "bbbb",
        "n_unresolved": 2,
        "n_kept": 1,
    }
    b_t = {
        "tools_used": 4,
        "safety_force_abstain": 1,
        "safety_volume_disabled": 0,
        "safety_legacy_weights_off": 0,
        "prev_action": "VOLUME_ESTIMATE",
    }
    full = build_state_features(h, o_prev, b_t)
    without_gt = build_state_features(h, {k: v for k, v in o_prev.items() if k != "train_gt_energy_kcal"}, b_t)
    ok &= check("train_gt on obs does not change the vector", np.array_equal(full, without_gt))
    names = STATE_FEATURE_NAMES
    ok &= check("abstain signal from obs", full[names.index("obs_abstain_signal")] == 1.0)
    ok &= check("VOLUME_ESTIMATE is not a one-hot slot", "obs_prev_action_VOLUME_ESTIMATE" not in names)
    ok &= check(
        "unknown prev action leaves action one-hot at 0",
        float(sum(full[names.index(f"obs_prev_action_{a}")] for a in (
            "RETRIEVE", "RE_PORTION", "RE_DECOMPOSE", "ABSTAIN", "STOP", "CORRECT_CODE"
        ))) == 0.0,
    )
    ok &= check(
        "energy uses tool payload / 2500",
        abs(full[names.index("obs_nutri_energy_norm")] - (122.0 / ENERGY_CAP_KCAL)) < 1e-12,
    )
    ok &= check("hard volume bit ignores b_t", full[names.index("safety_volume_disabled")] == 1.0)
    ok &= check("hard legacy bit ignores b_t", full[names.index("safety_legacy_weights_off")] == 1.0)
    only_gt = build_state_features(h, {"train_gt_energy_kcal": 400.0}, None)
    ok &= check("train_gt alone is not a nutrient", only_gt[names.index("obs_nutri_present")] == 0.0)
    ir_row = {
        "action_t": "RETRIEVE",
        "verify_codes": ["CODE_CONF_GAP"],
        "obs_t": {"cands": [{"code": "11111000", "score": 0.9}], "selected_code": "11111000"},
        "nutri_dish": {"energy_kcal": 250.0, "mass_g": 100.0},
        "train_gt_energy_kcal": 9999.0,
    }
    from_row = build_state_features(h, ir_row, None)
    from_row_clean = build_state_features(
        h, {k: v for k, v in ir_row.items() if k != "train_gt_energy_kcal"}, None
    )
    ok &= check("row train_gt does not change the vector", np.array_equal(from_row, from_row_clean))
    ok &= check(
        "nutri_dish fills energy when obs has none",
        abs(from_row[names.index("obs_nutri_energy_norm")] - (250.0 / ENERGY_CAP_KCAL)) < 1e-12,
    )
    overridden = dict(ir_row)
    overridden["obs_t"] = {
        **ir_row["obs_t"],
        "nutrients": {"energy_kcal": 100.0},
    }
    from_obs = build_state_features(h, overridden, None)
    ok &= check(
        "obs nutrients override nutri_dish",
        abs(from_obs[names.index("obs_nutri_energy_norm")] - (100.0 / ENERGY_CAP_KCAL)) < 1e-12,
    )

    zeroed = build_state_features(h, o_prev, b_t, mask_obs="zero")
    ok &= check("mask zero clears obs", float(np.sum(np.abs(zeroed[OBS_SLICE]))) == 0.0)
    ok &= check("mask zero keeps prefix", np.array_equal(zeroed[PRE_SLICE], full[PRE_SLICE]))
    ok &= check("mask zero keeps budget", np.array_equal(zeroed[BUDGET_SLICE], full[BUDGET_SLICE]))
    shuffled = build_state_features(h, o_prev, b_t, mask_obs="shuffle", shuffle_seed=0)
    ok &= check("shuffle keeps prefix", np.array_equal(shuffled[PRE_SLICE], full[PRE_SLICE]))
    ok &= check("shuffle keeps budget", np.array_equal(shuffled[BUDGET_SLICE], full[BUDGET_SLICE]))
    ok &= check(
        "shuffle preserves obs multiset",
        sorted(np.round(shuffled[OBS_SLICE], 8).tolist()) == sorted(np.round(full[OBS_SLICE], 8).tolist()),
    )
    try:
        build_state_features(h, o_prev, b_t, mask_obs="shuffle")
        shuffle_guard = False
    except ValueError:
        shuffle_guard = True
    ok &= check("shuffle without seed raises", shuffle_guard)

    b5_dispatch = build_features_for_ablation(h, mode="B5_open_loop", o_prev=o_prev, b_t=b_t)
    ok &= check("B5 dispatch is 31-d build_features", np.array_equal(b5_dispatch, leaked) and b5_dispatch.shape == (31,))
    main = build_features_for_ablation(h, mode="main_closed_loop", o_prev=o_prev, b_t=b_t)
    ok &= check("main dispatch matches the sketch", np.array_equal(main, full))

    for rel in (
        "vdr/controller/policy_bc.py",
        "vdr/controller/policy_mlp.py",
        "vdr/controller/dagger.py",
        "vdr/controller/bc_data.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        ok &= check(f"{rel} does not call the sketch", "build_state_features" not in text)
    return ok


def main() -> int:
    print("p0 eval smoke (no GPU, no training)")
    before = frozen_hashes()
    for rel, prefix in FROZEN_PREFIXES.items():
        digest = before.get(rel, "")
        if not digest.startswith(prefix):
            print(f"  FAIL frozen prefix {rel}: {digest[:16]}")
            return 1
        print(f"  OK frozen prefix {rel}")
    ok = True
    ok &= test_confusion_demo()
    ok &= test_provenance()
    ok &= test_selective()
    ok &= test_state_sketch()
    after = frozen_hashes()
    ok &= check("frozen result hashes unchanged", before == after)
    if not ok:
        print("FAILED")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
