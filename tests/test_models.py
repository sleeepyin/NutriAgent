"""Student, teacher, and dataset tests."""

from __future__ import annotations

import inspect
import math
import os
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from digi_nutri.datasets import (
    PrivilegedTrainDataset,
    RGBStudentTestDataset,
    collate_privileged,
    collate_student_test,
    load_depth,
    student_test_records,
)
from digi_nutri.engine import train_one_epoch_student, train_one_epoch_teacher
from digi_nutri.evaluate import predict_students
from digi_nutri.index_data import build_records_from_ids, index_depth_files, train_target_stats, train_vocab
from digi_nutri.losses import LossWeights
from digi_nutri.models.backbone import build_resnet101_features
from digi_nutri.models.fusion import CrossModalChannelSpatialFusion, SemanticCrossAttention
from digi_nutri.models.geometry_teacher import GeometryTeacher
from digi_nutri.models.heads import MultiScaleAddHead
from digi_nutri.models.ingredient_teacher import IngredientTeacher
from digi_nutri.models.student import Student


class Tiny(nn.Module):
    def __init__(self, in_channels: int, channels=(8, 12, 16)) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 8, kernel_size=1)
        self.maps = nn.ModuleList([nn.Conv2d(8, count, kernel_size=1) for count in channels])

    def forward(self, x):
        hidden = self.conv(x)
        return [proj(hidden) for proj in self.maps]


class ModelTests(unittest.TestCase):
    def test_student_api_and_step(self):
        self.assertEqual(list(inspect.signature(Student.forward_test).parameters), ["self", "rgb"])
        source = inspect.getsource(predict_students)
        self.assertNotIn("depth", source)
        self.assertNotIn("name_emb", source)
        self.assertIn("forward_test", source)
        with self.assertRaises(ValueError):
            MultiScaleAddHead((4, 4), fusion="gate")
        channels = (8, 12, 16)
        student = Student(Tiny(3), channels, hidden=8, n_labels=5)
        rgb = torch.rand(2, 3, 12, 12)
        self.assertEqual(set(student.forward_test(rgb)), {"nutrients"})
        with self.assertRaises(TypeError):
            student.forward_test(rgb, torch.rand(2))
        student.disable_aux_head()
        self.assertNotIn("aux_logits", student.forward_train(rgb))
        student = Student(Tiny(3), channels, hidden=8, n_labels=5)
        geo = GeometryTeacher(Tiny(3), Tiny(1), channels, hidden=8)
        ing = IngredientTeacher(Tiny(3), channels, hidden=8, text_dim=4, attn_dim=8, heads=2)
        batch = {
            "rgb": rgb,
            "depth": torch.rand(2, 1, 12, 12),
            "y": torch.rand(2, 5) + 1,
            "name_emb": torch.rand(2, 3, 4),
            "name_mask": torch.tensor([[True, True, False], [True, False, False]]),
            "multihot": torch.zeros(2, 5),
        }
        device = torch.device("cpu")
        mean = torch.zeros(5)
        std = torch.ones(5)
        weights = LossWeights(1, 1, 0.2)
        opt = torch.optim.AdamW(student.parameters(), lr=1e-2)
        warm = train_one_epoch_student(
            student, [batch], opt, device, "b3", 0.0, weights, mean, std, 1.0, geo, ing
        )
        full = train_one_epoch_student(
            student, [batch], opt, device, "b3", 1.0, weights, mean, std, 1.0, geo, ing
        )
        plain = train_one_epoch_student(
            student, [batch], opt, device, "b1", 1.0, weights, mean, std, 1.0, geo, ing
        )
        self.assertEqual(warm["geo_calls"], 0)
        self.assertEqual((full["geo_calls"], full["sem_calls"]), (1, 1))
        self.assertEqual(plain["geo_calls"], 0)
        self.assertTrue(math.isfinite(full["loss"]))
        poisoned = dict(batch)
        poisoned["grams"] = torch.ones(2, 3)
        with self.assertRaises(RuntimeError):
            train_one_epoch_student(
                student, [poisoned], opt, device, "b1", 0.0, weights, mean, std, 1.0
            )
        stats = train_one_epoch_teacher(
            geo, [batch], torch.optim.AdamW(geo.parameters(), lr=1e-3), device, mean, std, "geometry", 1.0
        )
        self.assertTrue(math.isfinite(stats["loss"]))

    def test_attention_mask_and_records(self):
        tokens = torch.rand(2, 3, 8)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        module = SemanticCrossAttention(16, 8, attn_dim=16, heads=4)
        vis = torch.rand(2, 16, 4, 4)
        changed = tokens.clone()
        changed[:, 2] += 3
        self.assertTrue(torch.allclose(module(vis, tokens, mask), module(vis, changed, mask), atol=1e-5))
        fused = CrossModalChannelSpatialFusion(8)(torch.rand(2, 8, 5, 5), torch.rand(2, 8, 5, 5))
        self.assertEqual(tuple(fused.shape), (2, 8, 5, 5))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_path = root / "rgb.png"
            depth_path = root / "depth_raw.png"
            Image.new("RGB", (8, 8), (20, 40, 60)).save(rgb_path)
            Image.fromarray(np.array([[0, 10], [30, 0]], dtype=np.uint16)).save(depth_path)
            record = {
                "dish_id": "dish_a",
                "rgb": str(rgb_path),
                "depth": str(depth_path),
                "y": [1, 2, 3, 4, 5],
                "names": ["apple"],
                "plate_group": "dish_a",
            }
            with self.assertRaises(ValueError):
                RGBStudentTestDataset([record], 8)
            item = RGBStudentTestDataset(student_test_records([record]), 8)[0]
            self.assertEqual(set(item), {"rgb", "y", "dish_id", "plate_group"})
            loaded = load_depth(depth_path, 8, "median")
            self.assertEqual(tuple(loaded.shape), (1, 8, 8))
            privileged = PrivilegedTrainDataset(
                [record], 8, include_depth=True, include_ingredients=True,
                text_cache={"apple": torch.ones(4)}, vocab_index={"apple": 0},
            )[0]
            self.assertNotIn("names", privileged)
            self.assertEqual(privileged["multihot"].tolist(), [1.0])
            batch = collate_privileged([privileged])
            self.assertNotIn("names", batch)
            self.assertEqual(train_vocab([{"names": ["b"]}, {"names": ["a"]}]), ["a", "b"])
            stats = train_target_stats([{"y": [0, 0, 0, 0, 0]}, {"y": [10, 10, 10, 10, 10]}])
            self.assertEqual(stats["mean"], [5, 5, 5, 5, 5])
            nested = root / "nest" / "dish_z"
            nested.mkdir(parents=True)
            Image.fromarray(np.zeros((2, 2), dtype=np.uint16)).save(nested / "depth_raw.png")
            self.assertIn("dish_z", index_depth_files([root / "nest"]))
            _rows, report = build_records_from_ids(["missing"], {}, root, {}, {"missing": "missing"})
            self.assertEqual(report["n_kept"], 0)

    def test_resnet101_feature_shapes(self):
        rgb = build_resnet101_features(3, pretrained=False)
        feats = rgb(torch.zeros(1, 3, 64, 64))
        self.assertEqual(
            [tuple(feat.shape) for feat in feats],
            [(1, 512, 8, 8), (1, 1024, 4, 4), (1, 2048, 2, 2)],
        )
        depth = build_resnet101_features(1, pretrained=False)
        self.assertEqual(depth(torch.zeros(1, 1, 64, 64))[-1].shape[1], 2048)
