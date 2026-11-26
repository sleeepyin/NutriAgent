"""Structure, residual Atwater, frozen loss, and both Q0 gates."""

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
from digi_nutri.models import (
    B1,
    apply_trained_b1,
    atwater_density,
    build_b1,
    build_tmulti,
    count_parameters,
    parameter_groups,
    volume_features,
)
from digi_nutri.protocol import (
    BACKBONE_LR,
    B1_PARAMS,
    DROPOUT_P,
    FAIL,
    HEAD_LR,
    PARAM_LIMIT,
    PASS,
    RGB_BACKBONE,
    T_MULTI_PARAMS,
    VOLUME_DIM,
)
from digi_nutri.verdict import holdout_block_reason, q0_verdict


def _raw_from_mm(millimeters: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(millimeters) / 0.1).astype(np.uint16)


def _passing_pmae() -> dict[str, float]:
    return {"energy": 13.5, "mass": 15.0, "fat": 15.0, "carb": 15.0, "protein": 15.0}


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        density = [1.2, 0.06, 0.09, 0.08]
        cls.teacher = build_tmulti(200.0, density, 556, pretrained_height=False)
        cls.teacher.eval()
        cls.baseline = build_b1(200.0, density, pretrained=False)
        cls.baseline.eval()
        cls.teacher_state = {key: value.detach().clone() for key, value in cls.teacher.state_dict().items()}

    def tearDown(self):
        self.teacher.load_state_dict(self.teacher_state)
        self.teacher.b1_initialized = False
        self.teacher.eval()

    def test_parameter_count_is_under_the_cap(self):
        self.assertEqual(count_parameters(self.teacher), T_MULTI_PARAMS)
        self.assertEqual(count_parameters(self.baseline), B1_PARAMS)
        self.assertLess(T_MULTI_PARAMS, PARAM_LIMIT)
        self.assertLess(T_MULTI_PARAMS, 70_000_000)
        self.assertEqual(RGB_BACKBONE, "convnext_small")
        self.assertEqual(self.baseline.forward.__code__.co_varnames[1], "rgb")
        self.assertEqual(self.baseline.forward.__code__.co_argcount, 2)

    def test_dropout_is_only_on_regression_heads(self):
        positive = []
        for name, module in self.teacher.named_modules():
            if isinstance(module, torch.nn.Dropout) and module.p > 0:
                positive.append(name)
                self.assertEqual(module.p, DROPOUT_P)
        self.assertEqual(
            sorted(positive),
            ["density_head.dropout", "mass_head.dropout", "residual_head.dropout"],
        )
        groups = parameter_groups(self.teacher, "t_multi")
        self.assertEqual(groups[0]["lr"], BACKBONE_LR)
        self.assertEqual(groups[1]["lr"], HEAD_LR)
        names = {id(parameter): name for name, parameter in self.teacher.named_parameters()}
        backbone_names = [names[id(parameter)] for parameter in groups[0]["params"]]
        head_names = [names[id(parameter)] for parameter in groups[1]["params"]]
        self.assertTrue(all(name.startswith(("rgb_encoder.", "height_encoder.")) for name in backbone_names))
        self.assertNotIn("gamma", [name.split(".", 1)[0] for name in backbone_names])
        self.assertIn("gamma", head_names)
        self.assertTrue(self.teacher.rgb_encoder.parameters().__next__() is not None)
        rgb_ids = {id(parameter) for parameter in self.teacher.rgb_encoder.parameters()}
        height_ids = {id(parameter) for parameter in self.teacher.height_encoder.parameters()}
        self.assertTrue(rgb_ids.isdisjoint(height_ids))
        self.assertEqual(float(self.teacher.gamma.detach()), 0.0)

    def test_attention_and_gamma_are_identity_at_init(self):
        features = torch.randn(2, 768, 4, 4)
        tokens = torch.randn(2, 5, 256)
        mask = torch.ones(2, 5, dtype=torch.bool)
        updated = self.teacher.attention(features, tokens, mask)
        self.assertTrue(torch.allclose(updated, features, atol=1e-6))
        f_rgb = torch.randn(2, 768, 4, 4)
        f_h = torch.randn(2, 512, 4, 4)
        self.assertTrue(torch.allclose(self.teacher.fuse(f_rgb, f_h), f_rgb, atol=1e-6))

    def test_branches_affect_features_once_gates_open(self):
        generator = torch.Generator().manual_seed(0)
        rgb = torch.randn(2, 3, 64, 64, generator=generator)
        f_rgb = self.teacher.rgb_encoder(rgb)
        with torch.no_grad():
            self.teacher.attention.proj.weight.normal_(generator=generator)
            self.teacher.gamma.fill_(1.0)
        name_ids = torch.randint(1, 20, (2, 8), generator=generator)
        priors = torch.rand(2, 8, 4, generator=generator)
        mask = torch.ones(2, 8, dtype=torch.bool)
        tokens = self.teacher.encoder(name_ids, priors)
        other = self.teacher.encoder(torch.full_like(name_ids, 30), priors)
        semantic = self.teacher.attention(f_rgb, tokens, mask)
        other_semantic = self.teacher.attention(f_rgb, other, mask)
        self.assertGreater(float((semantic - other_semantic).detach().abs().max()), 0.0)
        height = torch.randn(2, 3, 64, 64, generator=generator)
        other_height = torch.randn(2, 3, 64, 64, generator=generator)
        fused = self.teacher.fuse(f_rgb, self.teacher.height_encoder(height))
        other_fused = self.teacher.fuse(f_rgb, self.teacher.height_encoder(other_height))
        self.assertGreater(float((fused - other_fused).detach().abs().max()), 0.0)

    def test_epoch0_matches_b1_on_mass_and_macros(self):
        baseline = B1(200.0, [1.2, 0.06, 0.09, 0.08], pretrained=False)
        baseline.load_state_dict(self.baseline.state_dict())
        with torch.no_grad():
            baseline.mass_head.fc1.weight.normal_()
            baseline.mass_head.fc2.weight.normal_()
            baseline.density_head.fc1.weight.normal_()
            baseline.density_head.fc2.weight.zero_()
            baseline.density_head.fc2.weight[0].fill_(3.0)
            baseline.density_head.fc2.weight[1:].normal_()
        teacher = build_tmulti(200.0, [1.2, 0.06, 0.09, 0.08], 556, False)
        apply_trained_b1(teacher, baseline.state_dict())
        teacher.eval()
        baseline.eval()
        generator = torch.Generator().manual_seed(1)
        rgb = torch.randn(2, 3, 64, 64, generator=generator)
        height = torch.randn(2, 3, 64, 64, generator=generator)
        other_height = torch.randn(2, 3, 64, 64, generator=generator)
        height01 = torch.rand(2, 1, 64, 64, generator=generator)
        other01 = torch.rand(2, 1, 64, 64, generator=generator)
        names = torch.randint(1, 30, (2, 6), generator=generator)
        priors = torch.rand(2, 6, 4, generator=generator)
        mask = torch.ones(2, 6, dtype=torch.bool)
        with torch.no_grad():
            left = baseline(rgb)
            right = teacher(rgb, height, height01, names, priors, mask)
            again = teacher(rgb, other_height, other01, names, priors, mask)
        self.assertTrue(torch.allclose(left["mass"], right["mass"], atol=1e-5))
        self.assertTrue(torch.allclose(left["density"][:, 1:], right["density"][:, 1:], atol=1e-5))
        self.assertTrue(torch.allclose(right["mass"], again["mass"], atol=1e-5))
        self.assertGreater(float((left["density"][:, 0] - right["density"][:, 0]).abs().mean()), 0.01)
        fat, carb, protein = right["density"][:, 1], right["density"][:, 2], right["density"][:, 3]
        expected = 9 * fat + 4 * carb + 4 * protein
        self.assertTrue(torch.allclose(right["density"][:, 0], expected, atol=1e-5))
        self.assertTrue(torch.allclose(right["delta_cal"], torch.zeros_like(right["delta_cal"])))
        self.assertTrue(torch.allclose(right["yhat"], right["mass"] * right["density"], atol=1e-5))
        built = atwater_density(torch.zeros(2, 3), torch.tensor([[0.5], [-0.25]]))
        scale = F.softplus(torch.zeros(()))
        expected_energy = (9.0 + 4.0 + 4.0) * scale + built["delta_cal"].view(-1)
        self.assertTrue(torch.allclose(built["density"][:, 0], expected_energy))

    def test_missing_depth_zeros_stay_out_of_the_median(self):
        millimeters = np.zeros((8, 8), dtype=np.float32)
        millimeters[0, 0] = 1000.0
        millimeters[3:5, 3:5] = 100.0
        raw = _raw_from_mm(millimeters)
        self.assertEqual(table_depth_mm(millimeters, exclude_zeros=False), 0.0)
        self.assertEqual(table_depth_mm(millimeters, exclude_zeros=True), 1000.0)
        height = depth_raw_to_height01(raw, exclude_zeros=True)
        self.assertAlmostEqual(float(height[4, 4]), 1.0, places=5)

    def test_loss_keeps_ltask_and_adds_the_residual_term(self):
        sigma_mean = [2.0, 4.0, 8.0, 16.0]
        sigma_mass = 10.0
        mass_hat = torch.tensor([8.0, 3.0])
        density = torch.tensor([[1.0, 0.5, 0.25, 0.125], [2.0, 1.0, 0.5, 0.25]])
        mass_gt = torch.tensor([10.0, 4.0])
        nutrition = torch.tensor([[9.0, 4.0, 2.0, 1.0], [6.0, 3.0, 1.0, 0.5]])
        loss_fn = TaskLoss(sigma_mean, sigma_mass)
        loss = loss_fn(mass_hat, density, mass_gt, nutrition)
        yhat = mass_hat[:, None] * density
        nutrient = F.smooth_l1_loss(yhat, nutrition, beta=1.0, reduction="none")
        nutrient = (nutrient / torch.tensor(sigma_mean)).mean()
        mass_term = F.smooth_l1_loss(mass_hat[:, None], mass_gt[:, None], beta=1.0, reduction="none")
        mass_term = (mass_term / sigma_mass).mean()
        density_target = nutrition / torch.clamp(mass_gt[:, None], min=5.0)
        density_term = F.smooth_l1_loss(density, density_target, beta=1.0, reduction="none").mean()
        expected = nutrient + 0.5 * mass_term + 0.5 * density_term
        self.assertTrue(torch.allclose(loss, expected))
        zeros = torch.zeros(2, 1)
        self.assertTrue(torch.allclose(loss_fn(mass_hat, density, mass_gt, nutrition, zeros), expected))
        delta = torch.ones(2, 1)
        residual = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=1.0, reduction="mean")
        total = loss_fn(mass_hat, density, mass_gt, nutrition, delta)
        self.assertTrue(torch.allclose(total, expected + 0.1 * residual))
        changed = nutrition.clone()
        changed[:, 0] += 3.0
        self.assertGreater(float(loss_fn(mass_hat, density, mass_gt, changed)), float(loss))

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

    def test_records_require_the_usda_dictionary(self):
        dishes = {
            "dish_1": {
                "dish_id": "dish_1",
                "mass": 10.0,
                "nutrition": {"energy": 1.0, "fat": 1.0, "carb": 1.0, "protein": 1.0},
                "ingredients": [],
            }
        }
        with self.assertRaises(ValueError):
            make_records(["dish_1"], dishes, {"dish_1": "dish_1"}, None, None)

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

    def test_both_gates(self):
        def row(hi: float) -> dict:
            return {"lo": hi - 1.0, "hi": hi}

        contrasts = {
            "t_multi": {
                nutrient: {metric: row(-0.1) for metric in ("mae", "dataset_pmae", "plate_pmae")}
                for nutrient in ("energy", "fat", "carb", "protein", "mass")
            }
        }
        contrasts["t_multi"]["mass"]["mae"] = row(1.0)
        verdict, details = q0_verdict(contrasts, _passing_pmae())
        self.assertEqual(verdict, PASS)
        self.assertEqual(details["relative_failures"], [])
        self.assertTrue(details["absolute"]["pass"])
        self.assertAlmostEqual(details["absolute"]["mean5_dataset_pmae"], 14.7)
        contrasts["t_multi"]["fat"]["mae"] = row(0.0)
        verdict, details = q0_verdict(contrasts, _passing_pmae())
        self.assertEqual(verdict, FAIL)
        self.assertEqual(len(details["relative_failures"]), 1)
        contrasts["t_multi"]["fat"]["mae"] = row(-0.1)
        high = _passing_pmae()
        high["energy"] = 13.5001
        verdict, details = q0_verdict(contrasts, high)
        self.assertEqual(verdict, FAIL)
        self.assertFalse(details["absolute"]["energy_ok"])
        wide = _passing_pmae()
        wide["protein"] = 20.0
        verdict, details = q0_verdict(contrasts, wide)
        self.assertEqual(verdict, FAIL)
        self.assertFalse(details["absolute"]["mean5_ok"])
        self.assertEqual(
            holdout_block_reason(True, True),
            "holdout metrics already exist; refusing another official evaluation",
        )
        self.assertIsNone(holdout_block_reason(False, True))

    def test_package_does_not_name_locked_methods(self):
        text = "\n".join(path.read_text() for path in Path("digi_nutri").glob("*.py"))
        text += "\n".join(path.read_text() for path in Path("scripts").glob("*.py"))
        for banned in ("L_distill", "L_orth", "GOFF", "B2a", "B2b", "t_geo.pt", "t_ing.pt"):
            self.assertNotIn(banned, text)


if __name__ == "__main__":
    unittest.main()
