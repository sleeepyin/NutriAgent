"""Unit tests for the incomplete-ingredient-label audit. They do not open RGB images."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import u2_phase1_partial_label_audit as audit  # noqa: E402


def dish(dish_id: str, split: str, mass: str, calories: str, fat: str, carb: str, protein: str) -> dict:
    return {
        "dish_id": dish_id,
        "split": split,
        "mass": mass,
        "calories": calories,
        "fat": fat,
        "carb": carb,
        "protein": protein,
    }


def ingredient(
    dish_id: str,
    split: str,
    name: str,
    has_nutrition: str,
    mass: str,
    calories: str,
    fat: str,
    carb: str,
    protein: str,
    ann_id: str = "a",
) -> dict:
    return {
        "dish_id": dish_id,
        "split": split,
        "ann_id": ann_id,
        "annotation_name": name,
        "label_value": "1",
        "mask": f"masks/{dish_id}.png",
        "rgb_image": f"rgb_images/{dish_id}.png",
        "mask_pixels": "10",
        "has_nutrition": has_nutrition,
        "mass": mass,
        "calories": calories,
        "fat": fat,
        "carb": carb,
        "protein": protein,
        "cutout": "c.png",
    }


def meta(dish_id: str, mass: float, calories: float, fat: float, carb: float, protein: float, ingredients, cafe="cafe1"):
    return {
        "dish_id": dish_id,
        "ts": audit.dish_timestamp(dish_id),
        "cafe": cafe,
        "mass": mass,
        "calories": calories,
        "fat": fat,
        "carb": carb,
        "protein": protein,
        "ingredients": ingredients,
    }


class FrozenConstantTests(unittest.TestCase):
    def test_tolerance_and_floor_stay_frozen(self) -> None:
        self.assertEqual(audit.NUTR_ABS_TOL, 0.5)
        self.assertEqual(audit.NUTR_REL_TOL, 0.01)
        self.assertEqual(audit.PLATE_GAP_SECONDS, 180)
        self.assertEqual(audit.MIN_MASS_INCREASE_G, 1.0)
        self.assertEqual(audit.MASS_GRAM_TOL_G, 1.0)
        self.assertEqual(audit.MIN_HOLDOUT_PLATES, 80)
        self.assertEqual(audit.HOLDOUT_PERCENT, 15)
        self.assertEqual(audit.VALIDATION_PERCENT, 15)
        self.assertEqual(audit.MIN_MASS_SIGN_IMBALANCE, 0.5)
        self.assertFalse(audit.STERLING["public_corrected_labels"])

    def test_protocol_text_names_the_sets(self) -> None:
        text = audit.PROTOCOL_TEXT
        for phrase in (
            "可见实例",
            "有营养实例",
            "完整标签盘",
            "部分标签盘",
            "不能当成 0 克或 0 千卡",
            "不是「该成分质量为零」的负例",
            "盘级质量、能量和三种宏量以 Hand 盘级表为准",
        ):
            self.assertIn(phrase, text)
        self.assertIn("留出下限是 80", audit.SPLIT_RULE_TEXT)
        self.assertIn("不计算留出集上的营养误差", audit.SPLIT_RULE_TEXT)
        self.assertIn("符号不平衡度", audit.STRUCTURE_RULE_TEXT)
        self.assertNotIn("local_color_purity", Path(audit.__file__).read_text(encoding="utf-8"))
        self.assertNotIn("torch", Path(audit.__file__).read_text(encoding="utf-8"))

    def test_split_function_does_not_take_nutrition(self) -> None:
        self.assertEqual(audit.assign_plate_splits.__code__.co_varnames[:1], ("plate_ids",))
        self.assertEqual(audit.assign_plate_splits.__code__.co_argcount, 1)

    def test_source_has_no_training_loop(self) -> None:
        tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".")[0])
        for banned in ("torch", "tensorflow", "sklearn"):
            self.assertNotIn(banned, imported)


class ToleranceTests(unittest.TestCase):
    def test_absolute_and_relative_bands(self) -> None:
        self.assertTrue(audit.within_tolerance(10.0, 10.5))
        self.assertFalse(audit.within_tolerance(10.0, 10.6))
        self.assertTrue(audit.within_tolerance(99.0, 100.0))
        self.assertFalse(audit.within_tolerance(98.9, 100.0))
        self.assertTrue(audit.within_tolerance(0.0, 0.0))
        self.assertTrue(audit.within_tolerance(0.0, 0.5))
        self.assertFalse(audit.within_tolerance(0.0, 0.5001))

    def test_blank_is_not_zero(self) -> None:
        self.assertIsNone(audit.parse_optional_float(""))
        self.assertIsNone(audit.parse_optional_float("  "))
        self.assertEqual(audit.parse_optional_float("0"), 0.0)
        self.assertIsNone(audit.nutrition_values(ingredient("dish_10", "train", "oil", "0", "5", "9", "1", "0", "0")))
        self.assertIsNone(audit.nutrition_values(ingredient("dish_10", "train", "oil", "1", "", "9", "1", "0", "0")))


class PlateSplitTests(unittest.TestCase):
    def test_incremental_scan_links_and_a_reset_does_not(self) -> None:
        previous = {"ts": 1000, "mass": 50.0, "ingredients": [("rice", 50.0)]}
        nxt = {"ts": 1040, "mass": 120.0, "ingredients": [("rice", 50.0), ("chicken", 70.0)]}
        reset = {"ts": 2000, "mass": 40.0, "ingredients": [("salad", 40.0)]}
        self.assertTrue(audit.same_physical_plate(previous, nxt, 180))
        self.assertFalse(audit.same_physical_plate(nxt, reset, 180))

    def test_mass_drop_and_gram_drop(self) -> None:
        previous = {"ts": 1000, "mass": 200.0, "ingredients": [("rice", 80.0)]}
        dropped = {"ts": 1030, "mass": 90.0, "ingredients": [("rice", 90.0)]}
        self.assertFalse(audit.same_physical_plate(previous, dropped, 180))
        slight = {"ts": 1030, "mass": 210.0, "ingredients": [("rice", 79.0), ("oil", 131.0)]}
        self.assertTrue(audit.same_physical_plate(previous, slight, 180))
        too_much = {"ts": 1030, "mass": 210.0, "ingredients": [("rice", 78.9), ("oil", 131.1)]}
        self.assertFalse(audit.same_physical_plate(previous, too_much, 180))

    def test_cluster_uses_the_earliest_scan_as_the_plate_id(self) -> None:
        rows = [
            meta("dish_1000", 50, 10, 1, 1, 1, [("rice", 50)]),
            meta("dish_1040", 120, 20, 2, 2, 2, [("rice", 50), ("chicken", 70)]),
            meta("dish_3000", 40, 8, 1, 1, 1, [("salad", 40)]),
        ]
        links, dish_to_plate, plates = audit.cluster_plates(rows, 180)
        self.assertEqual(links, 1)
        self.assertEqual(dish_to_plate["dish_1000"], "dish_1000")
        self.assertEqual(dish_to_plate["dish_1040"], "dish_1000")
        self.assertEqual(dish_to_plate["dish_3000"], "dish_3000")
        self.assertEqual(plates["dish_1000"], ["dish_1000", "dish_1040"])

    def test_assignment_is_deterministic_and_keeps_a_plate_together(self) -> None:
        plate_ids = [f"dish_{1000 + index}" for index in range(10)]
        first = audit.assign_plate_splits(plate_ids)
        second = audit.assign_plate_splits(list(reversed(plate_ids)))
        self.assertEqual(first, second)
        self.assertEqual(audit.rounded_percent(10, 15), 2)
        self.assertEqual(sum(split == "holdout" for split in first.values()), 2)
        self.assertEqual(sum(split == "validation" for split in first.values()), 2)
        self.assertEqual(len(set(first.values())), 3)

    def test_published_plate_count_rounds_to_99(self) -> None:
        self.assertEqual(audit.rounded_percent(663, 15), 99)
        self.assertGreaterEqual(99, audit.MIN_HOLDOUT_PLATES)

    def test_decision_order(self) -> None:
        self.assertEqual(audit.decide(False, True, 99, 0), "stop_protocol_unclear")
        self.assertEqual(audit.decide(True, False, 99, 0), "stop_residual_not_structured")
        self.assertEqual(audit.decide(True, True, 99, 2), "stop_split_leaks")
        self.assertEqual(audit.decide(True, True, 64, 0), "stop_too_few_holdout_plates")
        self.assertEqual(audit.decide(True, True, 80, 0), "phase1_conditions_met_status_unchanged")


class StructureTests(unittest.TestCase):
    def _stats(self, fail: int, plate_side: int, sum_side: int, median: float) -> dict:
        return {
            "fail_count": fail,
            "plate_side": plate_side,
            "sum_side": sum_side,
            "median_signed": median,
        }

    def test_structured_pattern_passes(self) -> None:
        nutrients = {
            "calories": self._stats(422, 400, 22, 30.0),
            "fat": self._stats(383, 360, 23, 4.0),
            "mass": self._stats(361, 350, 11, 17.0),
            "carb": self._stats(284, 200, 84, 2.0),
            "protein": self._stats(174, 100, 74, 1.0),
        }
        ok, reasons = audit.structure_holds(nutrients)
        self.assertTrue(ok, reasons)

    def test_symmetric_mass_fails(self) -> None:
        nutrients = {
            "calories": self._stats(10, 9, 1, 5.0),
            "fat": self._stats(8, 7, 1, 1.0),
            "mass": self._stats(6, 3, 3, 0.1),
            "carb": self._stats(2, 1, 1, 0.1),
            "protein": self._stats(1, 1, 0, 0.1),
        }
        ok, reasons = audit.structure_holds(nutrients)
        self.assertFalse(ok)
        self.assertTrue(reasons)

    def test_missing_instance_is_not_added_as_zero(self) -> None:
        dishes = [
            dish("dish_10", "train", "100", "200", "10", "20", "5"),
            dish("dish_11", "test", "80", "50", "1", "2", "3"),
            dish("dish_12", "train", "40", "90", "3", "8", "4"),
        ]
        ingredients = [
            ingredient("dish_10", "train", "grain", "1", "40", "80", "4", "8", "2", "a"),
            ingredient("dish_10", "train", "meat", "1", "60", "120", "6", "12", "3", "b"),
            ingredient("dish_11", "test", "oil", "0", "80", "50", "1", "2", "3", "c"),
            ingredient("dish_12", "train", "rice", "1", "10", "40", "1", "4", "2", "d"),
            ingredient("dish_12", "train", "sauce", "0", "", "", "", "", "", "e"),
        ]
        records, notes = audit.classify_dishes(dishes, ingredients)
        by_id = {row["dish_id"]: row for row in records}
        self.assertEqual(by_id["dish_10"]["label"], "complete")
        self.assertTrue(by_id["dish_10"]["sums_align"])
        self.assertEqual(by_id["dish_11"]["label"], "partial")
        self.assertIsNone(by_id["dish_11"]["known_sum"])
        self.assertIsNone(by_id["dish_11"]["residual"]["mass"])
        self.assertEqual(by_id["dish_12"]["known_sum"]["mass"], 10.0)
        self.assertEqual(by_id["dish_12"]["label"], "partial")
        self.assertEqual(notes["has_nutrition_0_with_number"], 1)
        legacy = audit.legacy_u78_nutrition(dishes, ingredients, {})
        self.assertEqual(legacy["complete_and_sums_match"], 1)
        self.assertEqual(legacy["partial_nutrition_dishes"], 2)


class ReportTests(unittest.TestCase):
    def test_small_holdout_stops_without_claiming_phase_two(self) -> None:
        dishes = []
        ingredients = []
        metadata = []
        for index in range(10):
            dish_id = f"dish_{10000 + index * 1000}"
            mass = "70" if index < 6 else "100"
            fat = "5" if index < 8 else "20"
            carb = "5" if index < 3 else "10"
            protein = "4" if index < 1 else "8"
            dishes.append(dish(dish_id, "train", "100", "200", "20", "10", "8"))
            ingredients.append(
                ingredient(dish_id, "train", "rice", "1", mass, "100", fat, carb, protein, f"a{index}")
            )
            metadata.append(meta(dish_id, 100, 200, 20, 10, 8, [("rice", 100)]))
        images = {
            "checked": True,
            "expected": 10,
            "opened": 10,
            "missing": [],
            "unreadable": [],
            "size_counts": {"640x480": 10},
            "mode_counts": {"RGB": 10},
        }
        blobs = {
            "dish_start": "abc",
            "dish_end": "abc",
            "ingredient_start": "def",
            "ingredient_end": "def",
            "changed_during_audit": False,
        }
        summary = audit.run_from_loaded(
            dishes,
            ingredients,
            metadata,
            rgb_train=set(),
            rgb_test=set(),
            images=images,
            blobs=blobs,
        )
        self.assertEqual(summary["decision"], "stop_too_few_holdout_plates")
        self.assertTrue(summary["structure_ok"], summary["structure_reasons"])
        self.assertEqual(summary["split"]["cross_split_plates"], 0)
        self.assertLess(summary["split"]["holdout_plates"], 80)
        self.assertFalse(summary["holdout_nutrition_error_computed"])
        text = audit.render_report(summary)
        self.assertIn("停止", text)
        self.assertIn("stop_too_few_holdout_plates", text)
        self.assertIn("不宣布第二阶段解锁", text)
        self.assertNotIn("第二阶段已经解锁", text)
        self.assertIn(audit.SPLIT_RULE_TEXT.strip().splitlines()[0], text)


class LocalDataTests(unittest.TestCase):
    def test_merge_and_published_counts_when_tables_exist(self) -> None:
        dish_path = ROOT / "data/hand2026/dish_nutrition.csv"
        ingredient_path = ROOT / "data/hand2026/ingredient_nutrition.csv"
        metadata_dir = ROOT / "data/nutrition5k/metadata"
        if not dish_path.exists() or not ingredient_path.exists() or not metadata_dir.exists():
            self.skipTest("local Hand or Nutrition5k tables are not on this machine")
        self.assertEqual(audit.git_blob_sha1(dish_path.read_bytes()), audit.DISH_BLOB_SHA)
        self.assertEqual(audit.git_blob_sha1(ingredient_path.read_bytes()), audit.INGREDIENT_BLOB_SHA)
        _, dishes = audit.read_csv_rows(dish_path)
        _, ingredients = audit.read_csv_rows(ingredient_path)
        metadata = audit.load_metadata(metadata_dir)
        rgb_train = set(audit.load_id_list(ROOT / "data/nutrition5k/dish_ids/splits/rgb_train_ids.txt"))
        rgb_test = set(audit.load_id_list(ROOT / "data/nutrition5k/dish_ids/splits/rgb_test_ids.txt"))
        images = {
            "checked": False,
            "expected": len(dishes),
            "opened": 0,
            "missing": [],
            "unreadable": [],
            "size_counts": {},
            "mode_counts": {},
        }
        blobs = {
            "dish_start": audit.DISH_BLOB_SHA,
            "dish_end": audit.DISH_BLOB_SHA,
            "ingredient_start": audit.INGREDIENT_BLOB_SHA,
            "ingredient_end": audit.INGREDIENT_BLOB_SHA,
            "changed_during_audit": False,
        }
        summary = audit.run_from_loaded(dishes, ingredients, metadata, rgb_train, rgb_test, images, blobs)
        self.assertEqual(summary["merge"]["hand_physical_plates"], 663)
        self.assertEqual(summary["merge"]["library_physical_plates"], 2482)
        self.assertEqual(summary["merge"]["library_links"], 2524)
        self.assertEqual(summary["merge"]["official_cross_plates"], 2)
        self.assertEqual(summary["legacy_u78"]["mismatch_n"], 424)
        self.assertEqual(summary["legacy_u78"]["mass_plate_heavier_abs_0_5"], 418)
        self.assertEqual(summary["legacy_u78"]["mass_sum_heavier_abs_0_5"], 1)
        self.assertEqual(summary["legacy_u78"]["fail_counts"]["calories"], 422)
        self.assertEqual(summary["legacy_u78"]["fail_counts"]["fat"], 383)
        self.assertEqual(summary["split"]["holdout_plates"], 99)
        self.assertEqual(summary["split"]["validation_plates"], 99)
        self.assertEqual(summary["split"]["train_plates"], 465)
        self.assertEqual(summary["split"]["cross_split_plates"], 0)
        self.assertTrue(summary["structure_ok"], summary["structure_reasons"])
        self.assertTrue(all(row["match"] for row in summary["u78_comparison"]), summary["u78_comparison"])


if __name__ == "__main__":
    unittest.main()
