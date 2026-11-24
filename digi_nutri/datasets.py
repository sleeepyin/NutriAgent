"""RGB test samples are a separate dataset from privileged training samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from digi_nutri.depth import fill_invalid_and_normalize
from digi_nutri.protocol import RGB_MEAN, RGB_STD

_BILINEAR = Image.Resampling.BILINEAR
STUDENT_TEST_KEYS = {"dish_id", "rgb", "y", "plate_group"}


def load_rgb(path: str | Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), _BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(RGB_MEAN, dtype=np.float32)
    std = np.asarray(RGB_STD, dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def load_depth(path: str | Path, size: int, fill: str) -> torch.Tensor:
    array = np.array(Image.open(path))
    if array.ndim != 2:
        raise ValueError(f"depth_raw must be single channel, got {array.shape} at {path}")
    filled = fill_invalid_and_normalize(array, fill=fill)
    tensor = torch.from_numpy(filled)[None, None]
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor[0].clamp(0, 1)


def student_test_records(records: list[dict]) -> list[dict]:
    return [
        {
            "dish_id": record["dish_id"],
            "rgb": record["rgb"],
            "y": record["y"],
            "plate_group": record["plate_group"],
        }
        for record in records
    ]


def multihot(names: list[str], vocab_index: dict[str, int]) -> torch.Tensor:
    target = torch.zeros(len(vocab_index), dtype=torch.float32)
    for name in names:
        index = vocab_index.get(name)
        if index is not None:
            target[index] = 1.0
    return target


class RGBStudentTestDataset(Dataset):
    """Test samples: overhead RGB, dish id, plate group, and supervision targets."""

    def __init__(self, records: list[dict], image_size: int) -> None:
        for record in records:
            extra = set(record) - STUDENT_TEST_KEYS
            if extra:
                raise ValueError(f"student test records must not include {sorted(extra)}")
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "rgb": load_rgb(record["rgb"], self.image_size),
            "y": torch.tensor(record["y"], dtype=torch.float32),
            "dish_id": record["dish_id"],
            "plate_group": record["plate_group"],
        }


def collate_student_test(batch: list[dict]) -> dict:
    allowed = {"rgb", "y", "dish_id", "plate_group"}
    for sample in batch:
        extra = set(sample) - allowed
        if extra:
            raise RuntimeError(f"student test batch has privileged fields {sorted(extra)}")
    return {
        "rgb": torch.stack([sample["rgb"] for sample in batch]),
        "y": torch.stack([sample["y"] for sample in batch]),
        "dish_id": [sample["dish_id"] for sample in batch],
        "plate_group": [sample["plate_group"] for sample in batch],
    }


class PrivilegedTrainDataset(Dataset):
    """Training samples. Depth and name embeddings are optional flags."""

    def __init__(
        self,
        records: list[dict],
        image_size: int,
        depth_fill: str = "median",
        include_depth: bool = False,
        include_ingredients: bool = False,
        text_cache: dict[str, torch.Tensor] | None = None,
        vocab_index: dict[str, int] | None = None,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.depth_fill = depth_fill
        self.include_depth = include_depth
        self.include_ingredients = include_ingredients
        self.text_cache = text_cache or {}
        self.vocab_index = vocab_index or {}
        if include_ingredients and not self.text_cache:
            raise ValueError("ingredient samples require a text cache")
        if self.text_cache:
            self.text_dim = int(next(iter(self.text_cache.values())).shape[-1])
        else:
            self.text_dim = 512

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        item = {
            "rgb": load_rgb(record["rgb"], self.image_size),
            "y": torch.tensor(record["y"], dtype=torch.float32),
            "dish_id": record["dish_id"],
            "plate_group": record["plate_group"],
        }
        if self.include_depth:
            item["depth"] = load_depth(record["depth"], self.image_size, self.depth_fill)
        if self.include_ingredients:
            names = list(record["names"])
            vectors = []
            for name in names:
                if name not in self.text_cache:
                    raise KeyError(f"no cached text embedding for {name!r}")
                vectors.append(self.text_cache[name].float())
            if vectors:
                item["name_emb"] = torch.stack(vectors, dim=0)
            else:
                item["name_emb"] = torch.zeros(0, self.text_dim)
            item["multihot"] = multihot(names, self.vocab_index)
        return item


def collate_privileged(batch: list[dict]) -> dict:
    out = {
        "rgb": torch.stack([sample["rgb"] for sample in batch]),
        "y": torch.stack([sample["y"] for sample in batch]),
        "dish_id": [sample["dish_id"] for sample in batch],
        "plate_group": [sample["plate_group"] for sample in batch],
    }
    if "depth" in batch[0]:
        out["depth"] = torch.stack([sample["depth"] for sample in batch])
    if "name_emb" in batch[0]:
        width = batch[0]["name_emb"].shape[-1]
        counts = [sample["name_emb"].shape[0] for sample in batch]
        length = max(max(counts), 1)
        emb = torch.zeros(len(batch), length, width)
        mask = torch.zeros(len(batch), length, dtype=torch.bool)
        for index, sample in enumerate(batch):
            count = sample["name_emb"].shape[0]
            if count:
                emb[index, :count] = sample["name_emb"]
                mask[index, :count] = True
        out["name_emb"] = emb
        out["name_mask"] = mask
        out["multihot"] = torch.stack([sample["multihot"] for sample in batch])
    return out
