"""Structure, height rule, loss, augmentation, and the 24-interval gate."""

from __future__ import annotations

import random
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from digi_nutri.data import apply_geometry, make_records, sample_geometry
from digi_nutri.depth_to_height import depth_raw_to_height01, table_depth_mm
from digi_nutri.loss import TaskLoss
from digi_nutri.models import build_model, count_parameters, parameter_groups, volume_features
from digi_nutri.protocol import (
    B1_STRUCTURE_PARAMS,
    BACKBONE_LR,
    DROPOUT_P,
    FAIL,
    HEAD_HIDDEN,
    HEAD_LR,
    PARAM_LIMIT,
    PASS,
    PLAN_A_TGEO_WITHOUT_VOLUME,
    TING_STRUCTURE_PARAMS,
    VOLUME_DIM,
)
from digi_nutri.verdict import holdout_block_reason, q0_verdict


def _raw_from_mm(millimeters: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(millimeters) / 0.1).astype(np.uint16)


class ProtocolTests(unittest.TestCase):
    def test_missing_depth_zeros_stay_out_of_the_median(self):
        millimeters = np.zeros((8, 8), dtype=np.float32)
        millimeters[0, 0] = 1000.0
        millimeters[3:5, 3:5] = 100.0
        raw = _raw_from_mm(millimeters)
        self.assertEqual(table_depth_mm(millimeters, exclude_zeros=False), 0.0)
        self.assertEqual(table_depth_mm(millimeters, exclude_zeros=True), 1000.0)
        height = depth_raw_to_height01(raw, exclude_zeros=True)
        self.assertAlmostEqual(float(height[4, 4]), 1.0, places=5)

    def test_loss_matches_the_frozen_formula(self):
        sigma_mean = [2.0, 4.0, 8.0, 16.0]
        sigma_mass = 10.0
        mass_hat = torch.tensor([8.0, 3.0])
        density = torch.tensor([[1.0, 0.5, 0.25, 0.125], [2.0, 1.0, 0.5, 0.25]])
        mass_gt = torch.tensor([10.0, 4.0])
        nutrition = torch.tensor([[9.0, 4.0, 2.0, 1.0], [6.0, 3.0, 1.0, 0.5]])
        loss = TaskLoss(sigma_mean, sigma_mass)(mass_hat, density, mass_gt, nutrition)
        yhat = mass_hat[:, None] * density
        nutrient = F.smooth_l1_loss(yhat, nutrition, beta=1.0, reduction="none")
        nutrient = (nutrient / torch.tensor(sigma_mean)).mean()
        mass_term = F.smooth_l1_loss(mass_hat[:, None], mass_gt[:, None], beta=1.0, reduction="none")
        mass_term = (mass_term / sigma_mass).mean()
        density_target = nutrition / torch.clamp(mass_gt[:, None], min=5.0)
        density_term = F.smooth_l1_loss(density, density_target, beta=1.0, reduction="none").mean()
        expected = nutrient + 0.5 * mass_term + 0.5 * density_term
        self.assertTrue(torch.allclose(loss, expected))

    def test_geometry_is_shared_and_leaves_labels_alone(self):
        self.assertEqual(sample_geometry(False), (0, False, False))
        image = np.arange(48, dtype=np.float32).reshape(4, 4, 3)
        height = image[:, :, 0].copy()
        label = np.array([1.0, 2.0, 3.0, 4.0, 80.0])
        before = label.copy()
        for rot_k in (0, 1, 2, 3):
            rgb = apply_geometry(image, rot_k, True, False)
            mapped = apply_geometry(height, rot_k, True, False)
            self.assertTrue(np.array_equal(rgb[:, :, 0], mapped))
        rng = random.Random(0)
        rotations = set()
        flips = set()
        for _ in range(80):
            rot_k, flip_h, flip_v = sample_geometry(True, rng)
            rotations.add(rot_k)
            flips.add((flip_h, flip_v))
        self.assertEqual(rotations, {0, 1, 2, 3})
        self.assertIn((True, False), flips)
        self.assertIn((False, True), flips)
        self.assertTrue(np.array_equal(label, before))

    def test_b1_refuses_the_usda_table(self):
        dishes = {
            "dish_1": {
                "dish_id": "dish_1",
                "mass": 10.0,
                "nutrition": {"energy": 1.0, "fat": 1.0, "carb": 1.0, "protein": 1.0},
                "ingredients": [],
            }
        }
        with self.assertRaises(ValueError):
            make_records("b1", ["dish_1"], dishes, {"dish_1": "dish_1"}, usda=object(), vocab={})

    def test_volume_features_use_positive_height_only(self):
        height = torch.zeros(1, 1, 4, 4)
        height[0, 0, 1, 1] = 1.0
        height[0, 0, 1, 2] = 1.0
        height[0, 0, 2, 2] = 0.5
        features = volume_features(height)
        self.assertEqual(tuple(features.shape), (1, VOLUME_DIM))
        self.assertAlmostEqual(float(features[0, 0]), 3.0 / 16.0, places=6)
        self.assertAlmostEqual(float(features[0, 1]), 2.5, places=6)
        self.assertAlmostEqual(float(features[0, 2]), 2.5 / 3.0, places=6)
        positive = torch.tensor([1.0, 1.0, 0.5])
        self.assertAlmostEqual(float(features[0, 3]), float(torch.quantile(positive, 0.9)), places=6)
        self.assertAlmostEqual(float(features[0, 4]), 2.5 / 16.0, places=6)
        tiny = torch.zeros(1, 1, 2, 2)
        tiny[0, 0, 0, 0] = 1e-4
        self.assertAlmostEqual(float(volume_features(tiny)[0, 0]), 0.25, places=6)

    def test_dropout_learning_rates_and_parameter_count(self):
        init_density = [1.0, 0.05, 0.1, 0.08]
        expected = {
            "b1": B1_STRUCTURE_PARAMS,
            "t_ing": TING_STRUCTURE_PARAMS,
            "t_geo": PLAN_A_TGEO_WITHOUT_VOLUME + VOLUME_DIM * HEAD_HIDDEN,
        }
        for kind, vocab in (("b1", 1), ("t_geo", 1), ("t_ing", 556)):
            model = build_model(kind, 200.0, init_density, vocab_size=vocab, pretrained=False)
            drops = [
                name for name, module in model.named_modules() if isinstance(module, torch.nn.Dropout)
            ]
            self.assertEqual(sorted(drops), ["density_head.dropout", "mass_head.dropout"])
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    self.assertEqual(module.p, DROPOUT_P)
            count = count_parameters(model)
            self.assertEqual(count, expected[kind])
            self.assertLess(count, PARAM_LIMIT)
            groups = parameter_groups(model, kind)
            self.assertEqual(groups[0]["lr"], BACKBONE_LR)
            self.assertEqual(groups[1]["lr"], HEAD_LR)
            if kind == "t_geo":
                self.assertEqual(model.mass_head.fc1.in_features, 2048 + VOLUME_DIM)
                self.assertEqual(model.density_head.fc1.in_features, 2048)
                self.assertEqual(float(model.gamma.detach()), 0.0)
                self.assertIsNot(model.rgb_encoder.layer1, model.height_encoder.layer1)

    def test_holdout_rule_and_second_read_block(self):
        def row(hi: float) -> dict:
            return {"lo": hi - 1.0, "hi": hi}

        contrasts = {
            teacher: {
                nutrient: {metric: row(-0.1) for metric in ("mae", "dataset_pmae", "plate_pmae")}
                for nutrient in ("energy", "fat", "carb", "protein")
            }
            for teacher in ("t_geo", "t_ing")
        }
        verdict, failures = q0_verdict(contrasts)
        self.assertEqual(verdict, PASS)
        self.assertEqual(failures, [])
        contrasts["t_ing"]["fat"]["mae"] = row(0.0)
        verdict, failures = q0_verdict(contrasts)
        self.assertEqual(verdict, FAIL)
        self.assertEqual(len(failures), 1)
        self.assertEqual(holdout_block_reason(True, True), "holdout metrics already exist")
        self.assertIsNone(holdout_block_reason(False, True))

    def test_package_does_not_name_locked_methods(self):
        text = "\n".join(path.read_text() for path in Path("digi_nutri").glob("*.py"))
        for banned in ("L_distill", "T_multi", "L_orth", "GOFF"):
            self.assertNotIn(banned, text)


if __name__ == "__main__":
    unittest.main()
