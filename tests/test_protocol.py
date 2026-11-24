"""Loss, split, verdict, and holdout-lock tests."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from digi_nutri.bootstrap import plate_grouped_ci
from digi_nutri.depth import fill_invalid_and_normalize
from digi_nutri.fingerprint import code_fingerprint
from digi_nutri.holdout_guard import HoldoutLocked, assert_holdout_available, publish_holdout
from digi_nutri.losses import (
    LossWeights,
    aux_bce_loss,
    geo_affinity_loss,
    phase1_total_loss,
    semantic_distill_loss,
    task_loss,
)
from digi_nutri.metadata import load_metadata, parse_dish_line
from digi_nutri.models.backbone import RESNET101_CHANNELS
from digi_nutri.plate_groups import assign_plate_groups
from digi_nutri.protocol import HEADLINE
from digi_nutri.report import render_not_trained, render_report
from digi_nutri.schedule import distillation_scale
from digi_nutri.scoring import assemble_holdout_metrics
from digi_nutri.semantics import gram_weighted_dish_vector, names_only_dish_vector
from digi_nutri.splits import load_official_split


def _labels(groups):
    return [
        {"dish_id": f"dish_{index}", "plate_group": group, "y": [100, 100, 100, 100, 100]}
        for index, group in enumerate(groups)
    ]


def _const(value, mass, n):
    return [[value, mass, value, value, value] for _ in range(n)]


def _bundle(preds, groups):
    return assemble_holdout_metrics(_labels(groups), preds, n_boot=32, seed=0, eps=1.0)


class ProtocolTests(unittest.TestCase):
    def test_depth_fill_median_and_mean(self):
        depth = np.array([[0, 10], [20, 40]], dtype=np.float32)
        median = fill_invalid_and_normalize(depth, fill="median")
        mean = fill_invalid_and_normalize(depth, fill="mean")
        self.assertAlmostEqual(float(median[0, 1]), 0.0)
        self.assertAlmostEqual(float(median[1, 1]), 1.0)
        self.assertFalse(np.allclose(median, mean))
        self.assertTrue(np.all(fill_invalid_and_normalize(np.zeros((2, 2))) == 0))
        with self.assertRaises(ValueError):
            fill_invalid_and_normalize(depth, fill="mode")

    def test_affinity_is_not_pixel_l2(self):
        teacher = torch.zeros(1, 4, 2, 2)
        teacher[0, :, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        teacher[0, :, 0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        teacher[0, :, 1, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        teacher[0, :, 1, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        scaled = teacher * 4
        self.assertLess(geo_affinity_loss([scaled], [teacher]).item(), 1e-5)
        self.assertGreater(torch.nn.functional.mse_loss(scaled, teacher).item(), 0.1)
        corner = torch.zeros(1, 2, 2, 2)
        corner[..., 0, 0] = torch.tensor([1.0, 0.0])
        corner[..., 0, 1] = torch.tensor([1.0, 0.0])
        corner[..., 1, 0] = torch.tensor([1.0, 0.0])
        corner[..., 1, 1] = torch.tensor([0.0, 1.0])
        self.assertGreater(geo_affinity_loss([corner.flip(-1)], [corner]).item(), 1e-3)

    def test_task_and_phase1_objective(self):
        pred = torch.zeros(2, 5)
        target = torch.zeros(2, 5)
        loss, _ = task_loss(pred, target, torch.zeros(5), torch.ones(5))
        self.assertEqual(loss.item(), 0.0)
        for column in range(5):
            shifted = target.clone()
            shifted[:, column] = 4
            column_loss, _ = task_loss(pred, shifted, torch.zeros(5), torch.ones(5))
            self.assertGreater(column_loss.item(), 0.0)
        left = torch.tensor([[1.0, 0.0]])
        self.assertLess(semantic_distill_loss(left, left).item(), 1e-6)
        self.assertGreater(semantic_distill_loss(left, torch.tensor([[0.0, 1.0]])).item(), 0.1)
        self.assertAlmostEqual(
            aux_bce_loss(torch.zeros(2, 3), torch.zeros(2, 3)).item(), math.log(2), places=5
        )
        task = torch.tensor(1.0)
        geo = torch.tensor(2.0)
        sem = torch.tensor(3.0)
        aux = torch.tensor(4.0)
        weights = LossWeights(1.0, 1.0, 0.2)
        self.assertAlmostEqual(phase1_total_loss(task, geo, sem, aux, 1.0, "b1", weights).item(), 1.0)
        self.assertAlmostEqual(phase1_total_loss(task, geo, sem, aux, 1.0, "b2a", weights).item(), 3.0)
        self.assertAlmostEqual(phase1_total_loss(task, geo, sem, aux, 1.0, "b2b", weights).item(), 4.8, places=5)
        self.assertAlmostEqual(phase1_total_loss(task, geo, sem, aux, 1.0, "b3", weights).item(), 6.8, places=5)
        self.assertAlmostEqual(phase1_total_loss(task, None, None, None, 0.0, "b3", weights).item(), 1.0)
        with self.assertRaises(ValueError):
            phase1_total_loss(task, None, sem, aux, 1.0, "b3", weights)

    def test_schedule_and_name_pool(self):
        self.assertEqual(distillation_scale(0), 0.0)
        self.assertEqual(distillation_scale(4), 0.0)
        self.assertAlmostEqual(distillation_scale(5), 0.2)
        self.assertAlmostEqual(distillation_scale(9), 1.0)
        emb = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        mask = torch.tensor([[True, True]])
        pooled = names_only_dish_vector(emb, mask)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[math.sqrt(0.5), math.sqrt(0.5)]]), atol=1e-5))
        weighted = gram_weighted_dish_vector(emb, mask, torch.tensor([[1.0, 3.0]]))
        self.assertFalse(torch.allclose(pooled, weighted, atol=1e-5))
        with self.assertRaises(ValueError):
            names_only_dish_vector(emb, mask, grams=torch.ones(1, 2))

    def test_metadata_split_and_bootstrap(self):
        line = "dish_a,10,20,1,2,3,ingr_0000000001,soy sauce,3.5,1.2,0.1,0.2,0.3"
        record = parse_dish_line(line)
        self.assertEqual(set(record), {"dish_id", "y", "names"})
        self.assertNotIn("1.2", json.dumps(record))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cafe.csv").write_text(line + "\n" + line + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_metadata([root / "cafe.csv"])
            (root / "depth_train_ids.txt").write_text("dish_a\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_official_split(root, "train")
            with self.assertRaises(ValueError):
                load_official_split(root, "rgb_train")
            reshoot = root / "reshoot.tsv"
            reshoot.write_text("plateA d1 d2\n", encoding="utf-8")
            grouped = assign_plate_groups(["d1", "d2", "d3"], reshoot)
            self.assertEqual(grouped["d1"], grouped["d2"])
            self.assertEqual(grouped["d3"], "d3")
        grouped_ci = plate_grouped_ci(np.array([-1.0, -1.0]), np.array(["p", "p"]), n_boot=20, seed=0)
        self.assertEqual(grouped_ci["n_groups"], 1)
        self.assertTrue(grouped_ci["stable_below_zero"])
        zeros = plate_grouped_ci(np.zeros(4), np.array(list("abcd")), n_boot=20, seed=0)
        self.assertFalse(zeros["stable_below_zero"])

    def test_verdict_and_one_shot_holdout(self):
        groups = ["p", "p", "q", "q"]
        passed = _bundle(
            {
                "geometry_teacher": _const(100, 100, 4),
                "ingredient_teacher": _const(100, 100, 4),
                "b1": _const(40, 100, 4),
                "b2a": _const(70, 100, 4),
                "b2b": _const(70, 100, 4),
                "b3": _const(100, 0, 4),
            },
            groups,
        )
        self.assertEqual(passed["verdict"], "q1_q2_pass_stop_wait_triste")
        self.assertFalse(passed["comparisons"]["b3_minus_b1"]["mass"]["stable"])
        q0_fail = _bundle(
            {
                "geometry_teacher": _const(50, 100, 4),
                "ingredient_teacher": _const(100, 100, 4),
                "b1": _const(50, 100, 4),
                "b2a": _const(70, 100, 4),
                "b2b": _const(70, 100, 4),
                "b3": _const(100, 100, 4),
            },
            groups,
        )
        self.assertEqual(q0_fail["verdict"], "q0_fail_stop_wait_triste")
        self.assertIsNone(q0_fail["decision"]["q1_pass"])
        self.assertTrue(q0_fail["decision"]["descriptive_q1"])
        q1_fail = _bundle(
            {
                "geometry_teacher": _const(100, 100, 4),
                "ingredient_teacher": _const(100, 100, 4),
                "b1": _const(50, 100, 4),
                "b2a": _const(70, 100, 4),
                "b2b": _const(70, 100, 4),
                "b3": _const(50, 100, 4),
            },
            groups,
        )
        self.assertEqual(q1_fail["verdict"], "q1_fail_stop_wait_triste")
        q2_fail = _bundle(
            {
                "geometry_teacher": _const(100, 100, 4),
                "ingredient_teacher": _const(100, 100, 4),
                "b1": _const(40, 100, 4),
                "b2a": _const(70, 100, 4),
                "b2b": _const(55, 100, 4),
                "b3": _const(70, 100, 4),
            },
            groups,
        )
        self.assertEqual(q2_fail["verdict"], "q2_fail_stop_wait_triste")
        text = render_report(q1_fail)
        self.assertIn("STATUS.md was not edited", text)
        self.assertIn("Phase 2 was not started", text)
        self.assertIn("Mass is not a pass condition", text)
        self.assertIn("Training has not started", render_not_trained("busy"))
        self.assertEqual(len(HEADLINE), 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "holdout_metrics.json"
            publish_holdout(path, {"verdict": "q0_fail_stop_wait_triste"})
            with self.assertRaises(HoldoutLocked):
                assert_holdout_available(path)
            with self.assertRaises(HoldoutLocked):
                publish_holdout(path, {"again": True})
        root = Path(__file__).resolve().parents[1] / "digi_nutri"
        banned = ["L_orth", "l_orth", "GOFF", "GatedOrthogonal", "confidence_weight", "lambda_orth"]
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in banned:
                self.assertNotIn(token, source, msg=f"{path.name} {token}")
        self.assertNotIn("gram_weighted", (root / "engine.py").read_text(encoding="utf-8"))
        teacher = (root / "models" / "ingredient_teacher.py").read_text(encoding="utf-8")
        self.assertNotIn("gram_weighted", teacher)
        self.assertNotIn("calorie", teacher.lower())
        self.assertEqual(len(code_fingerprint()), 64)
        self.assertEqual(RESNET101_CHANNELS, (512, 1024, 2048))

    def test_official_files_if_present(self):
        splits = Path("/home/sdnu/nutriagent/data/nutrition5k/dish_ids/splits")
        if splits.is_dir():
            train_ids = load_official_split(splits, "train")
            test_ids = load_official_split(splits, "test")
            self.assertEqual(len(train_ids), 2758)
            self.assertEqual(len(test_ids), 507)
            self.assertFalse(set(train_ids) & set(test_ids))
        cafe = Path("/home/sdnu/nutriagent/data/nutrition5k/metadata/dish_metadata_cafe1.csv")
        if cafe.is_file():
            record = parse_dish_line(cafe.open(encoding="utf-8").readline())
            self.assertEqual(set(record), {"dish_id", "y", "names"})
            self.assertTrue(record["names"])
