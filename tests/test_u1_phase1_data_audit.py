"""Unit tests for the phase-1 audit. They do not read the Nutrition5k images."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import u1_phase1_data_audit as audit  # noqa: E402


def _rgb(height: int, width: int, color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = color[0]
    image[..., 1] = color[1]
    image[..., 2] = color[2]
    return image


def _plate_scene(mixed: bool) -> np.ndarray:
    """Gray rig, white plate, and either two solid food halves or a coarse mix."""
    image = _rgb(120, 160, (80, 80, 80))
    cy, cx = 60, 80
    yy, xx = np.ogrid[:120, :160]
    plate = (yy - cy) ** 2 + (xx - cx) ** 2 <= 42**2
    image[plate] = (245, 245, 245)
    food = (yy - cy) ** 2 + (xx - cx) ** 2 <= 26**2
    if mixed:
        blocks = ((yy // 8 + xx // 8) % 2) == 0
        image[food & blocks] = (210, 30, 30)
        image[food & ~blocks] = (30, 30, 210)
    else:
        image[food & (xx < cx)] = (210, 30, 30)
        image[food & (xx >= cx)] = (30, 30, 210)
    return image


class SeparabilityTests(unittest.TestCase):
    def test_signature_takes_only_rgb(self) -> None:
        self.assertEqual(audit.visual_separability.__code__.co_varnames[:1], ("rgb",))
        self.assertEqual(audit.visual_separability.__code__.co_argcount, 1)

    def test_white_is_neutral_lab(self) -> None:
        white = np.array([[[255, 255, 255]]], dtype=np.uint8)
        lab = audit.srgb_to_lab(white)[0, 0]
        self.assertAlmostEqual(lab[0], 100.0, delta=0.2)
        self.assertAlmostEqual(lab[1], 0.0, delta=0.2)
        self.assertAlmostEqual(lab[2], 0.0, delta=0.2)

    def test_solid_halves_are_purer_than_a_mix(self) -> None:
        solid = audit.visual_separability(_plate_scene(mixed=False))
        mixed = audit.visual_separability(_plate_scene(mixed=True))
        self.assertTrue(solid["mask_ok"], solid)
        self.assertTrue(mixed["mask_ok"], mixed)
        self.assertGreater(solid["s_sep"], mixed["s_sep"] + 0.05)
        self.assertGreaterEqual(mixed["s_sep"], 0.0)
        self.assertLessEqual(solid["s_sep"], 1.0)

    def test_constants_stay_at_the_preregistered_values(self) -> None:
        self.assertEqual(audit.METRIC_NAME, "local_color_purity")
        self.assertEqual(audit.PRIMARY_K, 3)
        self.assertEqual(audit.PRIMARY_WINDOW, 9)
        self.assertEqual(audit.ROBUST_K, 5)
        self.assertEqual(audit.ROBUST_WINDOW, 15)
        self.assertEqual(audit.PLATE_GAP_SECONDS, 180)
        self.assertEqual(audit.MIN_CLEAN_TEST_PLATES, 80)


class PlateAndHoldoutTests(unittest.TestCase):
    def test_incremental_scan_links_and_a_reset_does_not(self) -> None:
        rows = [
            {"ts": 1000, "mass": 50.0, "ingredients": [("rice", 50.0)]},
            {"ts": 1040, "mass": 120.0, "ingredients": [("rice", 50.0), ("chicken", 70.0)]},
            {"ts": 2000, "mass": 40.0, "ingredients": [("salad", 40.0)]},
        ]
        self.assertTrue(audit.same_physical_plate(rows[0], rows[1], 180))
        self.assertFalse(audit.same_physical_plate(rows[1], rows[2], 180))

    def test_mass_drop_breaks_the_plate_even_inside_the_time_gap(self) -> None:
        previous = {"ts": 1000, "mass": 200.0, "ingredients": [("rice", 80.0)]}
        current = {"ts": 1030, "mass": 90.0, "ingredients": [("rice", 90.0)]}
        self.assertFalse(audit.same_physical_plate(previous, current, 180))

    def test_holdout_floor(self) -> None:
        small = audit.judge_holdout(64, 73)
        self.assertFalse(small["large_enough"])
        self.assertEqual(small["decision"], "stop_too_few_independent_plates")
        self.assertFalse(small["published_158_declared"])
        enough = audit.judge_holdout(120, 0)
        self.assertTrue(enough["large_enough"])
        leaked = audit.judge_holdout(120, 4)
        self.assertFalse(leaked["large_enough"])
        self.assertEqual(leaked["decision"], "clean_subset_meets_floor_but_split_leaks")


class TableTests(unittest.TestCase):
    def test_public_table_checks_and_nutrition_sum(self) -> None:
        dishes = [
            {
                "dish_id": "dish_10",
                "split": "train",
                "mass": "100",
                "calories": "200",
                "fat": "10",
                "carb": "20",
                "protein": "5",
            },
            {
                "dish_id": "dish_11",
                "split": "test",
                "mass": "80",
                "calories": "50",
                "fat": "1",
                "carb": "2",
                "protein": "3",
            },
        ]
        ingredients = [
            {
                "dish_id": "dish_10",
                "split": "train",
                "ann_id": "a",
                "annotation_name": "grain",
                "label_value": "1",
                "mask": "masks/dish_10.png",
                "rgb_image": "rgb_images/dish_10.png",
                "mask_pixels": "10",
                "has_nutrition": "1",
                "mass": "40",
                "calories": "80",
                "fat": "4",
                "carb": "8",
                "protein": "2",
                "cutout": "c.png",
            },
            {
                "dish_id": "dish_10",
                "split": "train",
                "ann_id": "b",
                "annotation_name": "meat",
                "label_value": "2",
                "mask": "masks/dish_10.png",
                "rgb_image": "rgb_images/dish_10.png",
                "mask_pixels": "12",
                "has_nutrition": "1",
                "mass": "60",
                "calories": "120",
                "fat": "6",
                "carb": "12",
                "protein": "3",
                "cutout": "d.png",
            },
            {
                "dish_id": "dish_11",
                "split": "test",
                "ann_id": "c",
                "annotation_name": "oil",
                "label_value": "1",
                "mask": "masks/dish_11.png",
                "rgb_image": "rgb_images/dish_11.png",
                "mask_pixels": "4",
                "has_nutrition": "0",
                "mass": "",
                "calories": "",
                "fat": "",
                "carb": "",
                "protein": "",
                "cutout": "e.png",
            },
        ]
        public = audit.audit_public_tables(dishes, ingredients)
        self.assertEqual(public["annotation_name_count"], 3)
        self.assertEqual(public["has_nutrition_counts"], {"1": 2, "0": 1})
        self.assertTrue(public["same_dish_id_set"])
        self.assertEqual(public["split_conflicts"], [])
        self.assertTrue(audit.valid_dish_id("dish_1556575124"))
        self.assertFalse(audit.valid_dish_id("dish_015"))
        meta = {
            "dish_10": {"mass": 100.0, "calories": 200.0, "fat": 10.0, "carb": 20.0, "protein": 5.0},
            "dish_11": {"mass": 80.0, "calories": 50.0, "fat": 1.0, "carb": 2.0, "protein": 3.0},
        }
        per_dish, summary = audit.nutrition_consistency(dishes, ingredients, meta)
        by_id = {row["dish_id"]: row for row in per_dish}
        self.assertTrue(by_id["dish_10"]["sum_matches_plate_all_nutrients"])
        self.assertFalse(by_id["dish_11"]["all_instances_have_nutrition"])
        self.assertEqual(summary["complete_and_sums_match"], 1)
        self.assertEqual(summary["partial_nutrition_dishes"], 1)

    def test_blob_sha_matches_git(self) -> None:
        payload = b"dish_id,split\n"
        self.assertEqual(
            audit.git_blob_sha1(payload),
            subprocess.check_output(["git", "hash-object", "--stdin"], input=payload).decode().strip(),
        )

    def test_real_hand_csv_unchanged_if_present(self) -> None:
        dish = ROOT / "data/hand2026/dish_nutrition.csv"
        ingredient = ROOT / "data/hand2026/ingredient_nutrition.csv"
        if not dish.exists() or not ingredient.exists():
            self.skipTest("local Hand tables are not on this machine")
        self.assertEqual(audit.git_blob_sha1(dish.read_bytes()), audit.DISH_BLOB_SHA)
        self.assertEqual(audit.git_blob_sha1(ingredient.read_bytes()), audit.INGREDIENT_BLOB_SHA)


if __name__ == "__main__":
    unittest.main()
