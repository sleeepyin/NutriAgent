"""Tests for the B1 / B2 / B3 contrast. They do not train and do not score a holdout."""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import u2_phase1_partial_label_audit as phase1  # noqa: E402
import u2_phase2_b123_contrasts as contrast  # noqa: E402


def plate_row(mass: float, calories: float, fat: float, carb: float, protein: float) -> dict[str, float]:
    return {"mass": mass, "calories": calories, "fat": fat, "carb": carb, "protein": protein}


def ingredient(name: str, has_nutrition: str, mass: str, calories: str, fat: str, carb: str, protein: str) -> dict:
    return {
        "annotation_name": name,
        "has_nutrition": has_nutrition,
        "mass": mass,
        "calories": calories,
        "fat": fat,
        "carb": carb,
        "protein": protein,
    }


def record(model: str, plate_id: str, dish_id: str, truth: dict[str, float], pred: dict[str, float]) -> dict:
    return {"model": model, "plate_id": plate_id, "dish_id": dish_id, "true": truth, "pred": pred}


def five(mass: float, calories: float, fat: float, carb: float, protein: float) -> dict[str, float]:
    return {"mass": mass, "calories": calories, "fat": fat, "carb": carb, "protein": protein}


class FrozenBudgetTests(unittest.TestCase):
    def test_budget_constants_match_the_protocol_text(self) -> None:
        budget = contrast.frozen_budget()
        self.assertEqual(budget["backbone"], "torchvision.models.resnet18")
        self.assertEqual(budget["weights"], "ResNet18_Weights.IMAGENET1K_V1")
        self.assertFalse(budget["mass_estimator_checkpoints_loaded"])
        self.assertEqual(budget["input_size"], 224)
        self.assertEqual(budget["epochs"], 20)
        self.assertEqual(budget["batch_size"], 16)
        self.assertEqual(budget["seed"], 0)
        self.assertEqual(budget["bootstrap_draws"], 10000)
        self.assertEqual(budget["class_count"], 40)
        self.assertEqual(tuple(budget["q2_nutrients"]), ("calories", "fat", "carb", "protein"))
        self.assertNotIn("mass", budget["q2_nutrients"])
        self.assertIn("残差", contrast.FROZEN_PROTOCOL_TEXT)
        self.assertIn("10000", contrast.FROZEN_PROTOCOL_TEXT)

    def test_cli_cannot_retune_the_budget(self) -> None:
        source = Path(contrast.__file__).read_text(encoding="utf-8")
        self.assertNotIn("--epochs", source)
        self.assertNotIn("--lr", source)
        self.assertNotIn("weights/nutrition5k", source)
        self.assertNotIn("weights/nvreal", source)
        self.assertNotIn("weights/metafood3d", source)
        self.assertNotIn("dish_ids_cafe2", source)

    def test_train_command_cannot_open_the_holdout(self) -> None:
        train = inspect_source("cmd_train")
        loop = inspect_source("train_one")
        evaluation = inspect_source("cmd_eval")
        self.assertNotIn("allow_holdout=True", train)
        self.assertNotIn("preload_images(holdout", train)
        self.assertIn('example["split"] == "train"', train)
        self.assertIn('example["split"] == "validation"', train)
        self.assertIn("allow_holdout=False", loop)
        self.assertNotIn("allow_holdout=True", loop)
        self.assertNotIn("score_split(", evaluation)
        self.assertIn("predict_split", evaluation)
        self.assertIn('example["split"] == "holdout"', evaluation)

    def test_image_path_is_overhead_rgb_only(self) -> None:
        path = contrast.overhead_rgb_path(Path("imagery"), "dish_1556575124")
        self.assertEqual(path, Path("imagery/dish_1556575124/rgb.png"))
        with self.assertRaises(ValueError):
            contrast.overhead_rgb_path(Path("imagery"), "masks/dish_1.png")


def inspect_source(name: str) -> str:
    import inspect

    return inspect.getsource(getattr(contrast, name))


class TargetTests(unittest.TestCase):
    def test_missing_numbers_are_not_added_and_b2_uses_zero(self) -> None:
        instances = [
            ingredient("rice", "1", "40", "80", "4", "10", "3"),
            ingredient("herb", "0", "99", "99", "99", "99", "99"),
        ]
        plate = plate_row(100, 200, 10, 20, 8)
        targets = contrast.build_targets(instances, plate)
        rice = contrast.CLASSES.index("rice")
        herb = contrast.CLASSES.index("herb")
        self.assertEqual(targets["b2"][rice], (40.0, 80.0, 4.0, 10.0, 3.0))
        self.assertEqual(targets["b2"][herb], (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(contrast.vector_sum(targets["b2"]), (40.0, 80.0, 4.0, 10.0, 3.0))
        self.assertTrue(targets["residual_defined"])
        self.assertEqual(targets["residual"], (60.0, 120.0, 6.0, 10.0, 5.0))
        self.assertEqual(contrast.vector_sum(targets["b3"]), (40.0, 80.0, 4.0, 10.0, 3.0))
        reconstructed = tuple(
            contrast.vector_sum(targets["b3"])[index] + targets["residual"][index] for index in range(5)
        )
        self.assertEqual(reconstructed, (100.0, 200.0, 10.0, 20.0, 8.0))

    def test_blank_nutrition_is_not_a_number(self) -> None:
        instances = [ingredient("rice", "1", "", "80", "4", "10", "3")]
        targets = contrast.build_targets(instances, plate_row(10, 80, 4, 10, 3))
        self.assertFalse(targets["residual_defined"])
        self.assertIsNone(targets["residual"])
        self.assertEqual(contrast.vector_sum(targets["b2"]), (0.0, 0.0, 0.0, 0.0, 0.0))

    def test_no_nutrition_instance_leaves_the_residual_undefined(self) -> None:
        instances = [ingredient("herb", "0", "", "", "", "", "")]
        targets = contrast.build_targets(instances, plate_row(10, 20, 1, 2, 3))
        self.assertFalse(targets["residual_defined"])
        self.assertIsNone(targets["residual"])
        self.assertEqual(targets["n_visible"], 1)
        self.assertEqual(targets["n_nutrition"], 0)

    def test_forced_sum_scales_by_nutrient_and_splits_ties_equally(self) -> None:
        instances = [
            ingredient("rice", "1", "20", "0", "1", "2", "3"),
            ingredient("tuber", "1", "30", "0", "1", "2", "3"),
            ingredient("herb", "0", "5", "5", "5", "5", "5"),
        ]
        targets = contrast.build_targets(instances, plate_row(100, 40, 4, 8, 12))
        rice = contrast.CLASSES.index("rice")
        tuber = contrast.CLASSES.index("tuber")
        herb = contrast.CLASSES.index("herb")
        self.assertAlmostEqual(targets["forced"][rice][0], 40.0)
        self.assertAlmostEqual(targets["forced"][tuber][0], 60.0)
        self.assertEqual(targets["forced"][herb][0], 0.0)
        self.assertAlmostEqual(targets["forced"][rice][1], 20.0)
        self.assertAlmostEqual(targets["forced"][tuber][1], 20.0)
        self.assertEqual(contrast.vector_sum(targets["forced"]), (100.0, 40.0, 4.0, 8.0, 12.0))

    def test_same_class_instances_add(self) -> None:
        instances = [
            ingredient("rice", "1", "10", "20", "1", "2", "3"),
            ingredient("rice", "1", "15", "25", "1", "2", "3"),
        ]
        targets = contrast.build_targets(instances, plate_row(25, 45, 2, 4, 6))
        rice = contrast.CLASSES.index("rice")
        self.assertEqual(targets["b3"][rice], (25.0, 45.0, 2.0, 4.0, 6.0))
        self.assertEqual(targets["residual"], (0.0, 0.0, 0.0, 0.0, 0.0))

    def test_mask_field_does_not_change_targets(self) -> None:
        left = ingredient("rice", "1", "10", "20", "1", "2", "3")
        right = dict(left)
        right["mask"] = "masks/should_not_be_read.png"
        plate = plate_row(10, 20, 1, 2, 3)
        self.assertEqual(contrast.build_targets([left], plate), contrast.build_targets([right], plate))

    def test_loss_weights_ignore_holdout_scans(self) -> None:
        def example(split: str, calories: float) -> dict:
            return {"split": split, "plate": (1.0, calories, 1.0, 1.0, 1.0)}

        train_only = [example("train", 10.0), example("train", 30.0)]
        with_holdout = train_only + [example("holdout", 1000.0)]
        self.assertEqual(
            contrast.train_loss_weights(train_only)["weights"],
            contrast.train_loss_weights(with_holdout)["weights"],
        )

    def test_constant_nutrient_uses_the_floor(self) -> None:
        examples = [{"split": "train", "plate": (5.0, 5.0, 5.0, 5.0, 5.0)} for _ in range(4)]
        weights = contrast.train_loss_weights(examples)
        self.assertEqual(weights["weights"]["mass"], 1.0 / contrast.STD_FLOOR)


class SplitTests(unittest.TestCase):
    def test_loader_rejects_a_leaking_plate_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plate_assignment.csv"
            rows = []
            for index, split in enumerate(("train", "validation", "holdout")):
                # Too few plates, and one dish repeated.
                rows.append(
                    {
                        "plate_id": f"dish_{index}",
                        "split": split,
                        "hand_dish_ids": "dish_1556575124",
                    }
                )
            # The loader checks duplicates before counts when the same dish repeats.
            with self.assertRaises(ValueError):
                write_and_load(path, rows)

    def test_holdout_guard(self) -> None:
        contrast.assert_split_allowed("validation", allow_holdout=False)
        with self.assertRaises(RuntimeError):
            contrast.assert_split_allowed("holdout", allow_holdout=False)
        contrast.assert_split_allowed("holdout", allow_holdout=True)

    def test_epoch_plan_is_deterministic_and_changes_with_epoch(self) -> None:
        first = contrast.epoch_plan(32, 0)
        self.assertEqual(first, contrast.epoch_plan(32, 0))
        self.assertNotEqual(first[0], contrast.epoch_plan(32, 1)[0])
        self.assertEqual(len(first[0]), 32)
        self.assertEqual(len(set(first[0])), 32)


def write_and_load(path: Path, rows: list[dict]) -> dict:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = __import__("csv").DictWriter(handle, fieldnames=["plate_id", "split", "hand_dish_ids"])
        writer.writeheader()
        writer.writerows(rows)
    return contrast.load_plate_assignment(path)


class MetricTests(unittest.TestCase):
    def test_percentile_interpolates_linearly(self) -> None:
        self.assertEqual(contrast.percentile_linear([0.0, 1.0, 2.0, 3.0], 0.5), 1.5)
        self.assertEqual(contrast.percentile_linear([0.0, 10.0], 0.0), 0.0)
        self.assertEqual(contrast.percentile_linear([0.0, 10.0], 1.0), 10.0)

    def test_plate_mean_does_not_weight_a_plate_by_its_scans(self) -> None:
        truth = five(0, 0, 0, 0, 0)
        records = [
            record("b3", "plate_a", "a1", truth, five(0, 0, 0, 0, 0)),
            record("b3", "plate_a", "a2", truth, five(0, 0, 0, 0, 0)),
            record("b3", "plate_a", "a3", truth, five(0, 90, 0, 0, 0)),
            record("b3", "plate_b", "b1", truth, five(0, 0, 0, 0, 0)),
            record("b1", "plate_a", "a1", truth, five(0, 0, 0, 0, 0)),
            record("b1", "plate_a", "a2", truth, five(0, 0, 0, 0, 0)),
            record("b1", "plate_a", "a3", truth, five(0, 0, 0, 0, 0)),
            record("b1", "plate_b", "b1", truth, five(0, 0, 0, 0, 0)),
        ]
        errors = contrast.plate_mean_errors(records, "b3", "calories")
        self.assertAlmostEqual(errors["plate_a"], 30.0)
        self.assertEqual(errors["plate_b"], 0.0)
        table = contrast.plate_mae_table(records)
        self.assertAlmostEqual(table["b3"]["calories"], 15.0)
        self.assertAlmostEqual(contrast.scan_mae(records, "b3", "calories"), 22.5)
        self.assertNotAlmostEqual(table["b3"]["calories"], contrast.scan_mae(records, "b3", "calories"))

    def test_bootstrap_uses_one_draw_per_plate(self) -> None:
        method = {"p0": 1.0, "p1": 3.0}
        control = {"p0": 1.0, "p1": 1.0}
        summary = contrast.paired_bootstrap(method, control, seed=0, draws=1000)
        self.assertAlmostEqual(summary["point"], 1.0)
        self.assertEqual(summary["n_plates"], 2)
        self.assertLess(summary["low"], summary["point"])
        self.assertGreater(summary["high"], summary["point"])
        again = contrast.paired_bootstrap(method, control, seed=0, draws=1000)
        self.assertEqual(summary, again)

    def test_upper_bound_of_zero_is_not_stably_better(self) -> None:
        # Both replicates of the paired difference are 0 when the draw count
        # collapses onto equal plates. Construct errors whose paired values are
        # (-1, 1) so the percentile interval reaches both signs, and a constant
        # negative pair whose high end stays negative.
        crossed = contrast.paired_bootstrap({"a": 0.0, "b": 2.0}, {"a": 1.0, "b": 1.0}, seed=0, draws=20)
        self.assertFalse(crossed["stably_better"])
        better = contrast.paired_bootstrap({"a": 0.0, "b": 0.0}, {"a": 2.0, "b": 2.0}, seed=0, draws=20)
        self.assertTrue(better["stably_better"])
        self.assertLess(better["high"], 0.0)

    def test_prediction_join_uses_the_network_vector_only(self) -> None:
        examples = [
            {
                "dish_id": "dish_1",
                "plate_id": "plate_1",
                "plate": (1.0, 2.0, 3.0, 4.0, 5.0),
                "residual": (9.0, 9.0, 9.0, 9.0, 9.0),
            }
        ]
        predictions = {"b1": {"dish_1": (1.5, 2.5, 3.5, 4.5, 5.5)}}
        rows = contrast.prediction_records(examples, predictions)
        self.assertEqual(rows[0]["pred"]["calories"], 2.5)
        self.assertEqual(rows[0]["true"]["calories"], 2.0)


class Q2Tests(unittest.TestCase):
    def test_only_the_eight_b3_intervals_enter(self) -> None:
        self.assertTrue(contrast.enters_q2("b3", "b1", "calories"))
        self.assertTrue(contrast.enters_q2("b3", "b2", "protein"))
        self.assertFalse(contrast.enters_q2("b3", "b1", "mass"))
        self.assertFalse(contrast.enters_q2("forced", "b1", "calories"))
        self.assertFalse(contrast.enters_q2("b1", "b2", "fat"))

    def test_mass_and_forced_wins_do_not_pass_q2(self) -> None:
        comparisons = []
        for method, control in (("b3", "b1"), ("b3", "b2"), ("forced", "b1"), ("forced", "b2")):
            for nutrient in contrast.REPORT_NUTRIENTS:
                enters = contrast.enters_q2(method, control, nutrient)
                high = 0.2 if enters and nutrient == "fat" and control == "b2" else -0.2
                comparisons.append(
                    {
                        "method": method,
                        "control": control,
                        "nutrient": nutrient,
                        "enters_q2": enters,
                        "point": -0.5,
                        "low": -1.0,
                        "high": high,
                        "stably_better": high < 0.0,
                        "n_plates": 99,
                    }
                )
        decision = contrast.q2_from_comparisons(comparisons)
        self.assertFalse(decision["pass"])
        self.assertEqual(decision["code"], "q2_fail_stop_wait_triste")
        for item in comparisons:
            if item["nutrient"] == "fat" and item["control"] == "b2" and item["method"] == "b3":
                item["high"] = -0.05
                item["stably_better"] = True
        passed = contrast.q2_from_comparisons(comparisons)
        self.assertTrue(passed["pass"])
        self.assertEqual(passed["code"], "q2_pass_stop_no_new_algorithm")

    def test_report_states_the_stop_and_keeps_status_untouched(self) -> None:
        failed = sample_summary(passed=False)
        text = contrast.render_report(failed)
        self.assertIn("Q2 不成立", text)
        self.assertIn("等待 Triste", text)
        self.assertIn("强迫加和", text)
        self.assertIn("不进入停止规则", text)
        self.assertIn("真值残差", text)
        self.assertIn("224", text)
        self.assertIn("465", text)
        self.assertNotIn("docs/STATUS.md` 这次改写为通过", text)
        passed = sample_summary(passed=True)
        passed_text = contrast.render_report(passed)
        self.assertIn("Q2 成立", passed_text)
        self.assertIn("不自动开始新算法", passed_text)
        self.assertIn("下一步没有解锁", passed_text)


def sample_summary(passed: bool) -> dict:
    comparisons = []
    for method, control in (("b3", "b1"), ("b3", "b2"), ("forced", "b1"), ("forced", "b2")):
        for nutrient in contrast.REPORT_NUTRIENTS:
            enters = contrast.enters_q2(method, control, nutrient)
            high = -0.1 if passed or not enters else 0.1
            comparisons.append(
                {
                    "method": method,
                    "control": control,
                    "nutrient": nutrient,
                    "enters_q2": enters,
                    "point": -0.4 if high < 0 else 0.1,
                    "low": -0.8,
                    "high": high,
                    "stably_better": high < 0.0,
                    "n_plates": 99,
                }
            )
    decision = contrast.q2_from_comparisons(comparisons)
    plate_mae = {
        model: {name: float(index + 1) for index, name in enumerate(phase1.NUTRIENTS)}
        for model in contrast.MODELS
    }
    return {
        "budget": contrast.frozen_budget(),
        "q2": decision,
        "test_time": {
            "inputs": "single overhead rgb",
            "reads_ground_truth_mask": False,
            "reads_ground_truth_residual": False,
        },
        "split": {
            "train_plates": 465,
            "validation_plates": 99,
            "holdout_plates": 99,
            "cross_split_plates": 0,
            "train_dishes": 738,
            "validation_dishes": 154,
            "holdout_dishes": 164,
            "matches_phase1_algorithm": True,
            "hand_column_counts": {"train": 898, "test": 158},
            "hand_cross_split_plates": 73,
        },
        "blobs": {"dish": phase1.DISH_BLOB_SHA, "ingredient": phase1.INGREDIENT_BLOB_SHA},
        "label_facts": {
            "undefined_residual_dishes": {"train": 1, "validation": 1, "holdout": 1},
        },
        "plate_mae": plate_mae,
        "comparisons": comparisons,
        "selection": {model: {"epoch": 3, "val_score": 1.5} for model in contrast.MODELS},
        "steps_per_model": 940,
    }


class RoundTripTests(unittest.TestCase):
    def test_prediction_csv_recomputes_the_same_decision(self) -> None:
        truth = five(10, 100, 5, 8, 6)
        pred = five(12, 90, 5, 9, 6)
        records = []
        for model in contrast.MODELS:
            records.append(record(model, "plate_a", "dish_a", truth, pred))
            records.append(record(model, "plate_b", "dish_b", truth, truth))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "heldout_predictions.csv"
            contrast.write_prediction_csv(path, records)
            loaded = contrast.read_prediction_csv(path)
            original = contrast.q2_from_comparisons(contrast.build_comparisons(records, draws=100))
            recomputed = contrast.q2_from_comparisons(contrast.build_comparisons(loaded, draws=100))
            self.assertEqual(original, recomputed)


class TorchTests(unittest.TestCase):
    def test_forward_uses_the_image_and_matches_the_plate_formula(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in this interpreter")
        torch.manual_seed(0)
        images = torch.zeros(2, 3, contrast.INPUT_SIZE, contrast.INPUT_SIZE)
        b1 = contrast.build_model("b1", pretrained=False)
        b3 = contrast.build_model("b3", pretrained=False)
        plate = b1(images)["plate"]
        self.assertEqual(tuple(plate.shape), (2, 5))
        self.assertNotIn("residual", b1(images))
        outputs = b3(images)
        summed = outputs["classes"].sum(dim=1) + outputs["residual"]
        self.assertTrue(torch.allclose(outputs["plate"], summed))
        self.assertEqual(tuple(outputs["classes"].shape), (2, 40, 5))

    def test_b1_loss_ignores_nan_class_targets(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in this interpreter")
        outputs = {"plate": torch.ones(2, 5)}
        batch = {
            "plate": torch.full((2, 5), 2.0),
            "classes": torch.full((2, 40, 5), float("nan")),
            "residual": torch.full((2, 5), float("nan")),
            "supervise": torch.tensor([True, True]),
        }
        weights = torch.ones(5)
        loss = contrast.loss_for_batch("b1", outputs, batch, weights)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss), 1.0)

    def test_b3_loss_skips_an_undefined_residual(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in this interpreter")
        outputs = {
            "classes": torch.zeros(2, 40, 5),
            "residual": torch.zeros(2, 5),
            "plate": torch.zeros(2, 5),
        }
        classes = torch.zeros(2, 40, 5)
        classes[1, 0, 0] = 5.0
        batch = {
            "plate": torch.zeros(2, 5),
            "classes": classes,
            "residual": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0, 0.0]]),
            "supervise": torch.tensor([False, True]),
        }
        loss = contrast.loss_for_batch("b3", outputs, batch, torch.ones(5))
        self.assertAlmostEqual(float(loss), 5.0 / 200.0 + 4.0 / 5.0)


class LocalDataTests(unittest.TestCase):
    def test_frozen_split_and_target_identities_when_tables_exist(self) -> None:
        dish_path = ROOT / "data/hand2026/dish_nutrition.csv"
        ingredient_path = ROOT / "data/hand2026/ingredient_nutrition.csv"
        assignment_path = ROOT / "results/u2_phase1_partial_label_audit/plate_assignment.csv"
        if not dish_path.exists() or not ingredient_path.exists() or not assignment_path.exists():
            self.skipTest("local Hand tables or the frozen assignment are absent")
        self.assertEqual(phase1.git_blob_sha1(dish_path.read_bytes()), phase1.DISH_BLOB_SHA)
        self.assertEqual(phase1.git_blob_sha1(ingredient_path.read_bytes()), phase1.INGREDIENT_BLOB_SHA)
        _, dishes = phase1.read_csv_rows(dish_path)
        _, ingredients = phase1.read_csv_rows(ingredient_path)
        plates = contrast.load_plate_assignment(assignment_path)
        dish_map = contrast.dish_split_map(plates)
        counts = {name: sum(plate["split"] == name for plate in plates.values()) for name in contrast.EXPECTED_PLATES}
        self.assertEqual(counts, {"train": 465, "validation": 99, "holdout": 99})
        self.assertEqual(contrast.cross_split_plates(plates), 0)
        self.assertEqual(len(dish_map), 1056)
        examples = contrast.iter_examples(dishes, ingredients, dish_map)
        leak = contrast.hand_image_split_leak(dishes, dish_map)
        self.assertEqual(leak["column_counts"], {"train": 898, "test": 158})
        self.assertEqual(leak["cross_split_plates"], 73)
        self.assertEqual(leak["test_only_plates"], 64)
        self.assertFalse(leak["used_as_evaluation_split"])
        dish_counts = {name: 0 for name in contrast.EXPECTED_PLATES}
        undefined = 0
        for example in examples:
            dish_counts[example["split"]] += 1
            b2_sum = contrast.vector_sum(example["b2"])
            if example["residual_defined"]:
                b3_sum = contrast.vector_sum(example["b3"])
                forced_sum = contrast.vector_sum(example["forced"])
                for index in range(5):
                    self.assertAlmostEqual(b3_sum[index] + example["residual"][index], example["plate"][index], places=4)
                    self.assertAlmostEqual(forced_sum[index], example["plate"][index], places=4)
            else:
                undefined += 1
                self.assertIsNone(example["residual"])
                self.assertEqual(b2_sum, (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(dish_counts, {"train": 738, "validation": 154, "holdout": 164})
        self.assertEqual(undefined, 3)
        self.assertEqual(contrast.negative_supervised_targets(examples), [])
        # The Hand image split is a different partition from the frozen plate split.
        self.assertNotEqual(dish_counts["train"], 898)


class BootstrapIndependenceTests(unittest.TestCase):
    def test_resampled_unit_is_the_plate_list(self) -> None:
        rng = random.Random(0)
        plates = ["a", "b"]
        draws = [plates[rng.randrange(len(plates))] for _ in range(8)]
        self.assertTrue(set(draws) <= set(plates))
        summary = contrast.paired_bootstrap({"a": 1.0, "b": 5.0}, {"a": 0.0, "b": 1.0}, seed=1, draws=4)
        self.assertEqual(summary["n_plates"], 2)


if __name__ == "__main__":
    unittest.main()
