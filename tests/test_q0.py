"""Pre-registered DiGI-Nutri v2 Q0 checks."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from digi_nutri.bootstrap import contrast_ci
from digi_nutri.data import make_records
from digi_nutri.depth_to_height import border_values, close_3x3, depth_raw_to_height01, raw_to_millimeters
from digi_nutri.ingredients import build_vocab, encode_ingredients
from digi_nutri.loss import TaskLoss
from digi_nutri.metadata import parse_dish_line
from digi_nutri.metrics import dataset_pmae, plate_pmae
from digi_nutri.models import B1, build_model, count_parameters, select_backbone
from digi_nutri.protocol import (
    EPOCHS,
    EXPECTED_TEST_N,
    IMAGE_SIZE,
    LEARNING_RATE,
    PARAM_LIMIT,
    RESHOOT_MAX_GAP_SECONDS,
)
from digi_nutri.reshoot import plate_groups
from digi_nutri.splits import SplitError, audit_split
from digi_nutri.usda import USDAError, load_usda_text
from digi_nutri.verdict import FAIL, PASS, q0_verdict, train_gate


USDA_TEXT = """ingr,id,cal/g,fat(g),carb(g),protein(g)
apple,1,0.520,0.002,0.140,0.003
apple,2,0.520,0.002,0.140,0.003
deprecated,3,1.000,0.100,0.200,0.300
deprecated,4,2.000,0.100,0.200,0.300
"""


def _dish(dish_id: str, energy: float, mass: float = 100.0) -> dict:
    return {
        "dish_id": dish_id,
        "mass": mass,
        "nutrition": {"energy": energy, "fat": 1.0, "carb": 2.0, "protein": 3.0},
        "ingredients": [],
    }


def _smooth(diff: float) -> float:
    absolute = abs(diff)
    if absolute < 1:
        return 0.5 * diff * diff
    return absolute - 0.5


class DepthTests(unittest.TestCase):
    def test_raw_unit_and_border_zeros(self):
        millimeters = np.array([[0, 10, 0], [10, 5, 10], [0, 10, 0]], dtype=np.float32)
        border = border_values(millimeters)
        self.assertEqual(border.tolist(), [0, 10, 0, 0, 10, 0, 10, 10])
        self.assertEqual(float(np.median(border)), 5.0)
        raw = np.full((4, 4), 1000, dtype=np.uint16)
        self.assertTrue(np.allclose(raw_to_millimeters(raw), 100.0))

    def test_shift_invariant_fixed_scale_and_clip(self):
        base = np.full((8, 8), 2000, dtype=np.uint16)
        base[1:-1, 1:-1] = 1700
        shifted = base + np.uint16(1000)
        height = depth_raw_to_height01(base)
        self.assertTrue(np.allclose(height, depth_raw_to_height01(shifted)))
        self.assertAlmostEqual(float(height[4, 4]), 30.0 / 150.0, places=5)
        self.assertLess(float(height[4, 4]), 0.5)
        tall = np.full((8, 8), 2000, dtype=np.uint16)
        tall[1:-1, 1:-1] = 0
        tall_height = depth_raw_to_height01(tall)
        self.assertAlmostEqual(float(tall_height[4, 4]), 1.0, places=5)
        behind = np.full((8, 8), 2000, dtype=np.uint16)
        behind[1:-1, 1:-1] = 2500
        self.assertAlmostEqual(float(depth_raw_to_height01(behind)[4, 4]), 0.0, places=5)

    def test_closing_fills_a_hole(self):
        scaled = np.ones((5, 5), dtype=np.float32)
        scaled[2, 2] = 0.0
        closed = close_3x3(scaled)
        self.assertEqual(float(closed[2, 2]), 1.0)


class USDATests(unittest.TestCase):
    def test_identical_names_keep_first_and_conflicts_refuse_name_lookup(self):
        table = load_usda_text(USDA_TEXT)
        self.assertEqual(table.kept_duplicate_names, ["apple"])
        self.assertIn("deprecated", table.ambiguous_names)
        self.assertEqual(len(table.ambiguous_names["deprecated"]), 2)
        token, vector = table.prior_for("ingr_0000000003", "deprecated")
        self.assertEqual(token, "deprecated#3")
        self.assertAlmostEqual(float(vector[0]), 1.0)
        other, other_vector = table.prior_for("4", "Deprecated")
        self.assertEqual(other, "deprecated#4")
        self.assertAlmostEqual(float(other_vector[0]), 2.0)
        self.assertAlmostEqual(float(table.name_vector["apple"][0]), 0.52)
        blank_token, blank_vector = table.prior_for("1", "")
        self.assertEqual(blank_token, "apple")
        self.assertAlmostEqual(float(blank_vector[0]), 0.52, places=5)
        with self.assertRaises(USDAError):
            table.prior_for("3", "apple")
        with self.assertRaises(USDAError):
            load_usda_text("ingr,id,cal/g,fat(g),carb(g),protein(g)\napple,1,1,0,0,0\napple,1,2,0,0,0\n")
        with self.assertRaises(USDAError):
            load_usda_text("ingr,id,grams,cal/g,fat(g),carb(g),protein(g)\napple,1,5,1,0,0,0\n")

    def test_parser_drops_plate_grams(self):
        dish = parse_dish_line(
            "dish_1,10,20,1,2,3,ingr_0000000001,Apple,9.5,8,1,1,1"
        )
        self.assertEqual(set(dish["ingredients"][0]), {"id", "name"})
        self.assertNotIn("9.5", str(dish["ingredients"]))
        table = load_usda_text(USDA_TEXT)
        vocab = build_vocab(table)
        dish["ingredients"] = [
            {"id": "ingr_0000000003", "name": "deprecated"},
            {"id": "ingr_0000000003", "name": "deprecated"},
            {"id": "1", "name": "apple"},
        ]
        name_ids, priors, mask = encode_ingredients(dish, table, vocab, max_ingredients=4)
        self.assertEqual(sum(mask), 2)
        self.assertEqual(priors[0][0], 1.0)
        self.assertAlmostEqual(priors[1][0], 0.52, places=5)
        self.assertEqual(name_ids[-1], 0)


class SplitAndReshootTests(unittest.TestCase):
    def test_official_test_count_constant(self):
        self.assertEqual(EXPECTED_TEST_N, 507)
        self.assertEqual(RESHOOT_MAX_GAP_SECONDS, 120)
        self.assertEqual(IMAGE_SIZE, 384)
        self.assertEqual(EPOCHS, 60)
        self.assertEqual(LEARNING_RATE, 1e-4)

    def test_train_exclusion_and_fatal_test_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = root / "rgb"
            depth = root / "depth"
            for dish_id in ("dish_a", "dish_b", "dish_c"):
                (rgb / dish_id).mkdir(parents=True)
                (depth / dish_id).mkdir(parents=True)
                (rgb / dish_id / "rgb.png").write_bytes(b"x")
                (depth / dish_id / "depth_raw.png").write_bytes(b"y")
            (rgb / "dish_c" / "rgb.png").unlink()
            (depth / "dish_c" / "depth_raw.png").write_bytes(b"")
            (rgb / "dish_a2").mkdir()
            (depth / "dish_a2").mkdir()
            (rgb / "dish_a2" / "rgb.png").write_bytes(b"x")
            (depth / "dish_a2" / "depth_raw.png").write_bytes(b"y")
            audit = audit_split(["dish_a", "dish_c"], ["dish_b", "dish_a2"], rgb, depth, expected_test_n=2)
            self.assertEqual(audit.train_ids, ["dish_a"])
            self.assertEqual(audit.test_ids, ["dish_b", "dish_a2"])
            self.assertEqual(audit.exclusions[0]["dish_id"], "dish_c")
            self.assertCountEqual(audit.exclusions[0]["reasons"], ["rgb_missing", "depth_zero"])
            with self.assertRaises(SplitError):
                audit_split(["dish_a", "dish_b"], ["dish_b", "dish_a2"], rgb, depth, expected_test_n=2)
            (depth / "dish_b" / "depth_raw.png").write_bytes(b"")
            with self.assertRaises(SplitError):
                audit_split(["dish_a"], ["dish_b", "dish_a2"], rgb, depth, expected_test_n=2)

    def test_reshoot_groups_are_conservative(self):
        dishes = {
            "dish_100": _dish("dish_100", 50),
            "dish_200": _dish("dish_200", 50),
            "dish_300": _dish("dish_300", 50),
            "dish_130": _dish("dish_130", 80),
            "dish_5000": _dish("dish_5000", 50),
        }
        groups = plate_groups(list(dishes), dishes)
        self.assertEqual(groups["dish_100"], groups["dish_200"])
        self.assertEqual(groups["dish_200"], groups["dish_300"])
        self.assertNotEqual(groups["dish_100"], groups["dish_130"])
        self.assertNotEqual(groups["dish_100"], groups["dish_5000"])
        self.assertEqual(groups["dish_100"], "dish_100")


class LossMetricVerdictTests(unittest.TestCase):
    def test_loss_uses_predicted_product_and_mass_floor(self):
        nutrition = torch.tensor([[10.0, 2.0, 2.0, 2.0]])
        mass_gt = torch.tensor([1.0])
        mass_hat = torch.tensor([[4.0]])
        density = nutrition / 5.0
        loss = TaskLoss([10.0, 10.0, 10.0, 10.0], 10.0)
        value = float(loss(mass_hat, density, mass_gt, nutrition))
        yhat = 4.0 * (nutrition.numpy() / 5.0)
        nutrient = np.mean(
            [_smooth(float(yhat[0, index] - nutrition[0, index])) / 10.0 for index in range(4)]
        )
        mass_term = 0.5 * (_smooth(4.0 - 1.0) / 10.0)
        expected = nutrient + mass_term
        self.assertAlmostEqual(value, expected, places=5)
        stitched = 1.0 * (nutrition.numpy() / 5.0)
        stitched_nutrient = np.mean(
            [_smooth(float(stitched[0, index] - nutrition[0, index])) / 10.0 for index in range(4)]
        )
        self.assertNotAlmostEqual(value, stitched_nutrient + mass_term, places=4)
        source = Path("digi_nutri/loss.py").read_text()
        self.assertNotIn("pmae", source.lower())

    def test_pmae_definitions(self):
        self.assertAlmostEqual(dataset_pmae([2, 4], [4, 4]), 25.0)
        self.assertAlmostEqual(plate_pmae([1, 5], [0, 5]), 0.0)

    def test_grouped_bootstrap_and_verdict(self):
        target = np.array([10.0, 10.0, 10.0])
        method = np.array([9.0, 9.0, 9.0])
        baseline = np.array([12.0, 12.0, 12.0])
        groups = ["plate", "plate", "other"]
        row = contrast_ci(method, baseline, target, groups, "mae", n_samples=40, seed=20261125)
        self.assertLess(row["hi"], 0.0)
        tie = contrast_ci(baseline, baseline, target, groups, "mae", n_samples=20, seed=1)
        self.assertEqual(tie["hi"], 0.0)
        metrics = ("mae", "dataset_pmae", "plate_pmae")
        contrasts = {
            teacher: {nutrient: {metric: {"lo": -2.0, "hi": -0.1} for metric in metrics} for nutrient in ("energy", "fat", "carb", "protein")}
            for teacher in ("t_geo", "t_ing")
        }
        contrasts["t_geo"]["mass"] = {metric: {"lo": 1.0, "hi": 2.0} for metric in metrics}
        verdict, failures = q0_verdict(contrasts)
        self.assertEqual(verdict, PASS)
        self.assertEqual(failures, [])
        contrasts["t_ing"]["fat"]["plate_pmae"]["hi"] = 0.0
        verdict, failures = q0_verdict(contrasts)
        self.assertEqual(verdict, FAIL)
        self.assertEqual(len(failures), 1)
        passed, gate_failures = train_gate(
            {
                "b1": {"energy": 2, "fat": 2, "carb": 2, "protein": 2},
                "t_geo": {"energy": 1, "fat": 1, "carb": 1, "protein": 1},
                "t_ing": {"energy": 1, "fat": 2, "carb": 1, "protein": 1},
            }
        )
        self.assertFalse(passed)
        self.assertEqual(gate_failures[0]["nutrient"], "fat")


class ModelTests(unittest.TestCase):
    def test_three_models_product_and_parameter_cap(self):
        torch.manual_seed(0)
        backbone, count = select_backbone()
        self.assertEqual(backbone, "resnet101")
        self.assertLess(count, PARAM_LIMIT)
        geo = build_model("t_geo", "resnet101", False, 200.0, [1.2, 0.06, 0.09, 0.08])
        self.assertLess(count_parameters(geo), PARAM_LIMIT)
        torch.nn.init.normal_(geo.mass_head.fc2.weight, std=0.02)
        geo.eval()
        rgb = torch.rand(1, 3, 64, 64)
        first = geo(rgb, torch.zeros(1, 3, 64, 64))
        second = geo(rgb, torch.ones(1, 3, 64, 64))
        self.assertTrue(torch.allclose(first["density"], second["density"]))
        self.assertFalse(torch.allclose(first["mass"], second["mass"]))
        self.assertTrue(torch.allclose(first["yhat"], first["mass"] * first["density"]))
        ingredient = build_model("t_ing", "resnet101", False, 200.0, [1.2, 0.06, 0.09, 0.08], vocab_size=8)
        torch.nn.init.normal_(ingredient.density_head.fc2.weight, std=0.02)
        ingredient.eval()
        images = torch.rand(2, 3, 64, 64)
        name_ids = torch.tensor([[1, 2, 0], [1, 0, 0]])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        prior = torch.rand(2, 3, 4)
        changed = prior.clone()
        changed[:, 0] += 1
        left = ingredient(images, name_ids, prior, mask)
        right = ingredient(images, name_ids, changed, mask)
        self.assertTrue(torch.allclose(left["mass"], right["mass"]))
        self.assertFalse(torch.allclose(left["density"], right["density"]))
        self.assertEqual(list(inspect.signature(B1.forward).parameters), ["self", "rgb"])
        with self.assertRaises(ValueError):
            build_model("b2a", "resnet101", False, 200.0, [1.0, 0.1, 0.1, 0.1])
        dishes = {"dish_1": _dish("dish_1", 10)}
        dishes["dish_1"]["ingredients"] = [{"id": "1", "name": "apple"}]
        with self.assertRaises(ValueError):
            make_records("b1", ["dish_1"], dishes, {"dish_1": "dish_1"}, usda=load_usda_text(USDA_TEXT), vocab={"tokens": {}})


if __name__ == "__main__":
    unittest.main()
