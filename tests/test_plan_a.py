"""Plan A structure, D_table rule, train gate, and holdout lock."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from digi_nutri.audit_rule import use_zero_exclusion
from digi_nutri.data import make_records, should_flip
from digi_nutri.depth_to_height import close_3x3, depth_raw_to_height01, table_depth_mm
from digi_nutri.frozen_teachers import FrozenB1, FrozenTIng
from digi_nutri.loss import TaskLoss
from digi_nutri.models import PlanATGeo, count_parameters
from digi_nutri.protocol import (
    FAIL,
    FROZEN_B1_PARAMS,
    FROZEN_B1_TRAIN_MAE,
    FROZEN_TING_PARAMS,
    PARAM_LIMIT,
    PASS,
)
from digi_nutri.splits import SplitError, audit_split
from digi_nutri.train import assert_plan_a_only
from digi_nutri.verdict import holdout_block_reason, q0_verdict, tgeo_train_gate


def _raw_from_mm(millimeters: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(millimeters) / 0.1).astype(np.uint16)


class DepthAuditTests(unittest.TestCase):
    def test_zeros_collapse_the_audited_median_and_exclusion_restores_it(self):
        millimeters = np.zeros((8, 8), dtype=np.float32)
        millimeters[0, 0] = 1000.0
        millimeters[3:5, 3:5] = 100.0
        raw = _raw_from_mm(millimeters)
        self.assertEqual(table_depth_mm(millimeters, exclude_zeros=False), 0.0)
        self.assertEqual(table_depth_mm(millimeters, exclude_zeros=True), 1000.0)
        audited = depth_raw_to_height01(raw, exclude_zeros=False)
        corrected = depth_raw_to_height01(raw, exclude_zeros=True)
        self.assertTrue(np.all(audited == 0))
        self.assertAlmostEqual(float(corrected[4, 4]), 1.0, places=5)
        self.assertTrue(np.array_equal(depth_raw_to_height01(raw), audited))

    def test_clip_and_closing_stay(self):
        millimeters = np.full((8, 8), 200.0, dtype=np.float32)
        millimeters[3:5, 3:5] = 0.0
        raw = _raw_from_mm(millimeters)
        height = depth_raw_to_height01(raw, exclude_zeros=True)
        self.assertAlmostEqual(float(height[4, 4]), 1.0, places=5)
        scaled = np.ones((5, 5), dtype=np.float32)
        scaled[2, 2] = 0.0
        self.assertEqual(float(close_3x3(scaled)[2, 2]), 1.0)

    def test_empty_positive_border_stays_zero(self):
        millimeters = np.zeros((6, 6), dtype=np.float32)
        self.assertEqual(table_depth_mm(millimeters, exclude_zeros=True), 0.0)

    def test_written_audit_is_65_of_2754_and_has_no_cutoff_in_code(self):
        audit = json.loads(Path("results/plan_a/d_table_audit.json").read_text())
        self.assertEqual(audit["n_train"], 2754)
        self.assertEqual(audit["n_test"], 507)
        self.assertEqual(audit["n_d_table_eq_0"], 65)
        self.assertEqual(audit["n_height_all_zero"], 65)
        self.assertEqual(audit["n_d_table_eq_0_and_height_all_zero"], 65)
        self.assertTrue(audit["apply_d_table_zero_exclusion"])
        self.assertTrue(use_zero_exclusion(audit))
        self.assertAlmostEqual(audit["proportion_d_table_eq_0"], 65 / 2754)
        source = Path("digi_nutri/audit_rule.py").read_text()
        self.assertNotIn("proportion >", source)
        self.assertNotIn("proportion <", source)
        broken = dict(audit)
        del broken["apply_d_table_zero_exclusion"]
        with self.assertRaises(RuntimeError):
            use_zero_exclusion(broken)


class GateTests(unittest.TestCase):
    def test_train_gate_is_strict_and_ignores_mass(self):
        tied = dict(FROZEN_B1_TRAIN_MAE)
        passed, failures = tgeo_train_gate(tied)
        self.assertFalse(passed)
        self.assertEqual(len(failures), 4)
        lower = {name: value - 1e-4 for name, value in FROZEN_B1_TRAIN_MAE.items()}
        lower["mass"] = 1e9
        self.assertTrue(tgeo_train_gate(lower)[0])
        one_tie = dict(lower)
        one_tie["energy"] = FROZEN_B1_TRAIN_MAE["energy"]
        self.assertFalse(tgeo_train_gate(one_tie)[0])

    def test_holdout_stays_locked_until_the_gate_passes(self):
        self.assertEqual(
            holdout_block_reason(False, False, True),
            "train gate did not pass",
        )
        self.assertEqual(
            holdout_block_reason(True, True, True),
            "holdout metrics already exist",
        )
        self.assertEqual(
            holdout_block_reason(True, False, False),
            "train metrics are missing",
        )
        self.assertIsNone(holdout_block_reason(True, False, True))

    def test_q0_ignores_mass_and_rejects_a_bound_of_zero(self):
        metrics = ("mae", "dataset_pmae", "plate_pmae")
        contrasts = {
            teacher: {
                nutrient: {metric: {"lo": -2.0, "hi": -0.1} for metric in metrics}
                for nutrient in ("energy", "fat", "carb", "protein")
            }
            for teacher in ("t_geo", "t_ing")
        }
        contrasts["t_geo"]["mass"] = {metric: {"lo": 1.0, "hi": 2.0} for metric in metrics}
        contrasts["t_ing"]["mass"] = {metric: {"lo": 1.0, "hi": 2.0} for metric in metrics}
        verdict, failures = q0_verdict(contrasts)
        self.assertEqual(verdict, PASS)
        self.assertEqual(failures, [])
        contrasts["t_ing"]["fat"]["plate_pmae"]["hi"] = 0.0
        verdict, failures = q0_verdict(contrasts)
        self.assertEqual(verdict, FAIL)
        self.assertEqual(len(failures), 1)


class ModelTests(unittest.TestCase):
    def test_separate_backbones_gamma_zero_and_param_cap(self):
        torch.manual_seed(0)
        model = PlanATGeo(200.0, [1.2, 0.06, 0.09, 0.08])
        count = count_parameters(model)
        self.assertLess(count, PARAM_LIMIT)
        config = json.loads(Path("configs/plan_a_frozen.json").read_text())
        self.assertEqual(count, int(config["param_counts"]["t_geo"]))
        self.assertTrue(config["apply_d_table_zero_exclusion"])
        self.assertEqual(float(model.gamma.detach().reshape(-1)[0]), 0.0)
        self.assertEqual(model.rgb_encoder.name, "resnet101")
        self.assertEqual(model.height_encoder.name, "resnet18")
        self.assertEqual(tuple(model.rgb_encoder.layer4[0].conv1.weight.shape), (512, 1024, 1, 1))
        self.assertEqual(tuple(model.height_encoder.layer4[0].conv1.weight.shape), (512, 256, 3, 3))
        rgb_ids = {id(parameter) for parameter in model.rgb_encoder.parameters()}
        height_ids = {id(parameter) for parameter in model.height_encoder.parameters()}
        self.assertTrue(rgb_ids.isdisjoint(height_ids))
        rgb_bn = [module for module in model.rgb_encoder.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        height_bn = [
            module for module in model.height_encoder.modules() if isinstance(module, torch.nn.BatchNorm2d)
        ]
        self.assertTrue(rgb_bn and height_bn)
        self.assertTrue({id(module) for module in rgb_bn}.isdisjoint({id(module) for module in height_bn}))
        self.assertFalse(any(isinstance(module, torch.nn.BatchNorm2d) for module in model.cab.modules()))
        model.train()
        before = model.rgb_encoder.stem[1].running_mean.clone()
        model.height_encoder(torch.zeros(2, 3, 64, 64))
        self.assertTrue(torch.equal(before, model.rgb_encoder.stem[1].running_mean))
        with torch.no_grad():
            torch.nn.init.normal_(model.mass_head.fc2.weight, std=0.02)
            torch.nn.init.normal_(model.density_head.fc2.weight, std=0.02)
        model.eval()
        rgb = torch.rand(2, 3, 64, 64)
        zeros = model(rgb, torch.zeros_like(rgb))
        ones = model(rgb, torch.ones_like(rgb))
        self.assertTrue(torch.allclose(zeros["yhat"], ones["yhat"]))
        with torch.no_grad():
            model.gamma.fill_(1.0)
            torch.nn.init.constant_(model.cab.align.weight, 0.01)
            torch.nn.init.constant_(model.cab.fc2.weight, 0.01)
        changed = model(rgb, torch.ones_like(rgb))
        self.assertFalse(torch.allclose(zeros["yhat"], changed["yhat"]))
        self.assertTrue(torch.allclose(changed["yhat"], changed["mass"] * changed["density"]))
        self.assertEqual(count_parameters(FrozenB1(200.0, [1.0, 0.05, 0.1, 0.08])), FROZEN_B1_PARAMS)
        self.assertEqual(
            count_parameters(FrozenTIng(200.0, [1.0, 0.05, 0.1, 0.08], 556)),
            FROZEN_TING_PARAMS,
        )
        assert_plan_a_only("t_geo")
        with self.assertRaises(RuntimeError):
            assert_plan_a_only("b1")
        with self.assertRaises(RuntimeError):
            assert_plan_a_only("t_ing")

    def test_eval_flip_is_off_and_teachers_do_not_read_usda(self):
        self.assertFalse(should_flip(False, 0.0))
        self.assertTrue(should_flip(True, 0.1))
        dishes = {
            "dish_1": {
                "dish_id": "dish_1",
                "mass": 10.0,
                "nutrition": {"energy": 1.0, "fat": 1.0, "carb": 1.0, "protein": 1.0},
                "ingredients": [],
            }
        }
        with self.assertRaises(ValueError):
            make_records("t_geo", ["dish_1"], dishes, {"dish_1": "dish_1"}, usda=object(), vocab={})


class LossAndSplitTests(unittest.TestCase):
    def test_loss_uses_predicted_product(self):
        nutrition = torch.tensor([[10.0, 2.0, 2.0, 2.0]])
        loss = TaskLoss([10.0, 10.0, 10.0, 10.0], 10.0)
        value = float(loss(torch.tensor([[4.0]]), nutrition / 5.0, torch.tensor([1.0]), nutrition))
        self.assertGreater(value, 0.0)
        self.assertNotIn("pmae", Path("digi_nutri/loss.py").read_text().lower())

    def test_bad_test_files_are_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dish_id in ("dish_a", "dish_b"):
                (root / "rgb" / dish_id).mkdir(parents=True)
                (root / "depth" / dish_id).mkdir(parents=True)
                (root / "rgb" / dish_id / "rgb.png").write_bytes(b"x")
                (root / "depth" / dish_id / "depth_raw.png").write_bytes(b"y")
            (root / "depth" / "dish_b" / "depth_raw.png").write_bytes(b"")
            with self.assertRaises(SplitError):
                audit_split(["dish_a"], ["dish_b"], root / "rgb", root / "depth", expected_test_n=1)


if __name__ == "__main__":
    unittest.main()
