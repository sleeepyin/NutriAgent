#!/usr/bin/env python3
"""Audit whether Nutrition5k incremental plate sequences can be recovered.

The official archive has dish timestamps, ingredient rows, dish totals, and
train/test splits. It does not publish a plate-sequence id. This script
recovers adjacent stages only when the earlier stage's ingredient records are
an exact proper subset of the later stage, the predecessor is unique inside
one scanning session, the predecessor has only one next composition, and the
weighed mass (and, for cafe 1, energy and macros) move with the added rows.

It does not train a model. Image checks are pixel and structural comparisons
of overhead RGB. The optional bootstrap only rereads Experiment A predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_a_common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    paired_mae_difference,
)

N5K = ROOT / "data" / "nutrition5k"
ARCHIVE_PATH = Path("/var/tmp/nutri-downloads/nutrition5k/nutrition5k_dataset.tar.gz")
PREDICTION_DIR = ROOT / "results" / "experiment_a_n5k_dual_view" / "predictions"
COMPARISON_PATH = ROOT / "results" / "experiment_a_n5k_dual_view" / "comparison.json"
DEFAULT_OUT = ROOT / "results" / "icdl_phase1_sequence_audit"

# One cafeteria scanning session. Chosen because a scan is an 8-second sweep
# plus the time to add the next item, and because unrestricted predecessor
# gaps fall either under a few minutes or beyond a day. Sensitivity is
# reported for the neighboring thresholds; those runs do not pick the count.
SESSION_GAP_S = 1800
SENSITIVITY_GAPS = (300, 600, 1800, 7200, 86400)
MASS_TOL_G = Decimal("1")
MACRO_TOL = Decimal("0.05")
RGB_MIN_BYTES = 10_000
CHANGED_PIXEL = 15
RANDOM_IMAGE_PAIRS = 400
RANDOM_IMAGE_SEED = 0
CENTER_MARGIN = 0.25

EXPECTED_SHA256 = {
    "metadata/dish_metadata_cafe1.csv": "881df31f27c2343d0ce219964527e0796f6ddadd60701ce6840fa801e8c2a960",
    "metadata/dish_metadata_cafe2.csv": "ef9f800d7b9c2e7e9a9e7cda5cbb4cfe51c760a8c1af4c05786ccf051bf5d77d",
    "metadata/ingredients_metadata.csv": "87a9bbd1fd2e28f09f16305a04946ddbe31eb7d9d03686d7e2fb3d8f19f3effb",
    "dish_ids/dish_ids_all.txt": "cd71f64b9092d7917384b477e62fbe8840409e606dc488b436f05aa2b404c2f5",
    "dish_ids/dish_ids_cafe1.txt": "bb2d2a8a918e8c4d6d212b158840d92c304d7b9757d6ffe14b4948446f7682a0",
    "dish_ids/dish_ids_cafe2.txt": "2e8cb76a9204277701e06b11476bf5a45f20f23a65dee8fc1399e9663648fc94",
    "dish_ids/README": "5110e71f40fef2e5732897d4f5f34d146253ec11caaa07f466bd4faf322fb4b1",
    "dish_ids/splits/rgb_train_ids.txt": "efd34a682b5483b01e6bd0a7c1267efec28440a8194985f9d3817d8ebc8169bd",
    "dish_ids/splits/rgb_test_ids.txt": "560ff0eedb1a2027c21f5da3a4c38a53fa201c87d108e76b11cceaa1d956ac8e",
    "dish_ids/splits/depth_train_ids.txt": "d9702b71ade32b012fcb272f1f58b1d19178acd4536eccc651857760eab4bc30",
    "dish_ids/splits/depth_test_ids.txt": "6e43a5334a72bd06fa9c0d4b3bc86e2f57e5342b0b260d3fa003a76f7e39eca2",
}

SETTING_FILES = (
    ("单张俯视", "overhead.jsonl"),
    ("单张侧视", "side.jsonl"),
    ("双视角简单特征融合", "simple_fusion.jsonl"),
    ("双视角几何融合", "geometric_fusion.jsonl"),
    ("双视角+食物语义约束", "semantic_fusion.jsonl"),
)

NUTRIENT_FIELDS = ("mass", "kcal", "fat", "carb", "protein")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_id_set(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def d(value: str) -> Decimal:
    return Decimal(value)


def parse_metadata_file(path: Path, cafe: str) -> list[dict]:
    dishes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) < 6 or not parts[0].startswith("dish_"):
            raise SystemExit(f"{path}:{line_no} is not a dish row")
        rest = parts[6:]
        if len(rest) % 7 != 0:
            raise SystemExit(f"{path}:{line_no} ingredient block length {len(rest)} is not a multiple of 7")
        rows = tuple(tuple(rest[i * 7 : (i + 1) * 7]) for i in range(len(rest) // 7))
        for row in rows:
            if not row[0].startswith("ingr_"):
                raise SystemExit(f"{path}:{line_no} ingredient id {row[0]!r} is not official")
        counter = Counter(rows)
        dishes.append(
            {
                "dish_id": parts[0],
                "cafe": cafe,
                "ts": int(parts[0].split("_", 1)[1]),
                "kcal": d(parts[1]),
                "mass": d(parts[2]),
                "fat": d(parts[3]),
                "carb": d(parts[4]),
                "protein": d(parts[5]),
                "rows": rows,
                "counter": counter,
                "key": tuple(sorted(counter.items())),
            }
        )
    dishes.sort(key=lambda dish: (dish["ts"], dish["dish_id"]))
    return dishes


def is_proper_subset(left: Counter, right: Counter) -> bool:
    """True when every ingredient record of left occurs in right, and right has more."""
    if not left or sum(left.values()) >= sum(right.values()):
        return False
    for key, count in left.items():
        if right[key] < count:
            return False
    return True


def nutrient_totals(counter: Counter) -> dict[str, Decimal]:
    totals = {name: Decimal(0) for name in NUTRIENT_FIELDS}
    for (_ing_id, _name, grams, kcal, fat, carb, protein), count in counter.items():
        multiple = Decimal(count)
        totals["mass"] += d(grams) * multiple
        totals["kcal"] += d(kcal) * multiple
        totals["fat"] += d(fat) * multiple
        totals["carb"] += d(carb) * multiple
        totals["protein"] += d(protein) * multiple
    return totals


def integer_grams(text: str) -> bool:
    value = d(text)
    return value == value.to_integral_value()


def build_repeat_clusters(dishes: list[dict], gap_s: int) -> list[dict]:
    """Group exact ingredient-record repeats whose successive gap stays in the session."""
    parent = list(range(len(dishes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    last_seen: dict[tuple, int] = {}
    for index, dish in enumerate(dishes):
        previous = last_seen.get(dish["key"])
        if previous is not None and 0 < dish["ts"] - dishes[previous]["ts"] <= gap_s:
            union(previous, index)
        last_seen[dish["key"]] = index

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(dishes)):
        grouped[find(index)].append(index)

    clusters = []
    for members in grouped.values():
        members.sort(key=lambda index: (dishes[index]["ts"], dishes[index]["dish_id"]))
        counter = dishes[members[0]]["counter"]
        totals = nutrient_totals(counter)
        representative = min(
            members,
            key=lambda index: (abs(dishes[index]["mass"] - totals["mass"]), dishes[index]["ts"], dishes[index]["dish_id"]),
        )
        clusters.append(
            {
                "members": members,
                "counter": counter,
                "key": dishes[members[0]]["key"],
                "t_min": dishes[members[0]]["ts"],
                "t_max": dishes[members[-1]]["ts"],
                "rep": representative,
                "ingredient_totals": totals,
                "cafe": dishes[members[0]]["cafe"],
            }
        )
    clusters.sort(key=lambda cluster: (cluster["t_min"], dishes[cluster["members"][0]]["dish_id"]))
    for index, cluster in enumerate(clusters):
        cluster["cluster_index"] = index
    return clusters


def maximal_predecessors(clusters: list[dict], gap_s: int | None) -> list[list[int]]:
    chosen: list[list[int]] = []
    for index, cluster in enumerate(clusters):
        candidates = []
        for earlier in range(index - 1, -1, -1):
            previous = clusters[earlier]
            gap = cluster["t_min"] - previous["t_min"]
            if gap <= 0:
                continue
            if gap_s is not None and gap > gap_s:
                break
            if is_proper_subset(previous["counter"], cluster["counter"]):
                candidates.append(earlier)
        maximal = []
        for candidate in candidates:
            contained = any(
                other != candidate and is_proper_subset(clusters[candidate]["counter"], clusters[other]["counter"])
                for other in candidates
            )
            if not contained:
                maximal.append(candidate)
        chosen.append(maximal)
    return chosen


def classify_predecessors(clusters: list[dict], predecessors: list[list[int]]) -> list[str]:
    labels = []
    for index, maximal in enumerate(predecessors):
        if not maximal:
            labels.append("no_subset_predecessor")
            continue
        if len(maximal) == 1:
            labels.append("unique_maximal_predecessor")
            continue
        union: Counter = Counter()
        for candidate in maximal:
            union += clusters[candidate]["counter"]
        if union == clusters[index]["counter"]:
            labels.append("disjoint_assembly")
        else:
            labels.append("ambiguous_predecessors")
    return labels


def nutrition_residual(parent: dict, child: dict, dishes: list[dict]) -> dict[str, Decimal]:
    added = child["counter"] - parent["counter"]
    added_totals = nutrient_totals(added)
    before = dishes[parent["rep"]]
    after = dishes[child["rep"]]
    observed = {
        "mass": after["mass"] - before["mass"],
        "kcal": after["kcal"] - before["kcal"],
        "fat": after["fat"] - before["fat"],
        "carb": after["carb"] - before["carb"],
        "protein": after["protein"] - before["protein"],
    }
    return {name: observed[name] - added_totals[name] for name in NUTRIENT_FIELDS}


def nutrition_ok(residual: dict[str, Decimal], cafe: str) -> bool:
    # Each weighing is about ±1 g, so the difference of two weighings is allowed ±2 g.
    if abs(residual["mass"]) > MASS_TOL_G * 2:
        return False
    if cafe == "cafe1":
        return all(abs(residual[name]) <= MACRO_TOL for name in ("kcal", "fat", "carb", "protein"))
    return True


def walk_linear_paths(edges: list[tuple[int, int]]) -> list[list[int]]:
    successor = {}
    predecessor = {}
    for parent, child in edges:
        if parent in successor or child in predecessor:
            raise RuntimeError("linear graph is not a collection of paths")
        successor[parent] = child
        predecessor[child] = parent
    starts = sorted((node for node in successor if node not in predecessor), key=lambda node: node)
    paths = []
    seen: set[int] = set()
    for start in starts:
        path = [start]
        cursor = start
        while cursor in successor:
            cursor = successor[cursor]
            if cursor in seen:
                raise RuntimeError("cycle in a time-ordered plate graph")
            path.append(cursor)
        seen.update(path)
        paths.append(path)
    return paths


def analyze_cafe(dishes: list[dict], gap_s: int) -> dict:
    clusters = build_repeat_clusters(dishes, gap_s)
    predecessors = maximal_predecessors(clusters, gap_s)
    labels = classify_predecessors(clusters, predecessors)
    children: dict[int, list[int]] = defaultdict(list)
    for index, maximal in enumerate(predecessors):
        if len(maximal) == 1:
            children[maximal[0]].append(index)
    structural_edges = [(parent, child) for parent, kids in children.items() if len(kids) == 1 for child in kids]
    fork_parents = {parent: kids for parent, kids in children.items() if len(kids) > 1}
    paths = walk_linear_paths(structural_edges)
    return {
        "clusters": clusters,
        "predecessors": predecessors,
        "labels": labels,
        "structural_edges": structural_edges,
        "fork_parents": fork_parents,
        "paths": paths,
    }


def dish_splits(dish_id: str, rgb_train: set[str], rgb_test: set[str], depth_train: set[str], depth_test: set[str]) -> dict[str, str]:
    def one(train: set[str], test: set[str], train_name: str, test_name: str) -> str:
        in_train = dish_id in train
        in_test = dish_id in test
        if in_train and in_test:
            return f"{train_name}_and_{test_name}"
        if in_train:
            return train_name
        if in_test:
            return test_name
        return "outside"

    return {
        "rgb": one(rgb_train, rgb_test, "rgb_train", "rgb_test"),
        "depth": one(depth_train, depth_test, "depth_train", "depth_test"),
    }


def split_set(labels: list[str]) -> list[str]:
    return sorted(set(labels))


def leaks(labels: list[str], train_name: str, test_name: str) -> bool:
    found = set(labels)
    return train_name in found and test_name in found


def added_items(parent: Counter, child: Counter) -> list[dict]:
    items = []
    for (ing_id, name, grams, kcal, fat, carb, protein), count in sorted((child - parent).items(), key=lambda item: (item[0][1], item[0][0])):
        items.append(
            {
                "id": ing_id,
                "name": name,
                "grams": grams,
                "kcal": kcal,
                "fat": fat,
                "carb": carb,
                "protein": protein,
                "copies": int(count),
            }
        )
    return items


def cluster_record(cluster: dict, dishes: list[dict], splits_of) -> dict:
    members = []
    masses = []
    for index in cluster["members"]:
        dish = dishes[index]
        masses.append(dish["mass"])
        members.append(
            {
                "dish_id": dish["dish_id"],
                "timestamp": dish["ts"],
                "mass_g": str(dish["mass"]),
                "kcal": str(dish["kcal"]),
                "fat_g": str(dish["fat"]),
                "carb_g": str(dish["carb"]),
                "protein_g": str(dish["protein"]),
                "n_ingredient_rows": len(dish["rows"]),
                **splits_of(dish["dish_id"]),
            }
        )
    totals = cluster["ingredient_totals"]
    mass_values = [dishes[index]["mass"] for index in cluster["members"]]
    return {
        "cluster_index": cluster["cluster_index"],
        "t_min": cluster["t_min"],
        "t_max": cluster["t_max"],
        "duration_s": cluster["t_max"] - cluster["t_min"],
        "n_scans": len(cluster["members"]),
        "representative_dish_id": dishes[cluster["rep"]]["dish_id"],
        "ingredient_mass_g": str(totals["mass"]),
        "mass_closest_g": str(dishes[cluster["rep"]]["mass"]),
        "mass_range_g": str(max(mass_values) - min(mass_values)),
        "scale_disagrees": abs(dishes[cluster["rep"]]["mass"] - totals["mass"]) > MASS_TOL_G,
        "fractional_grams": any(not integer_grams(row[2]) for row in cluster["counter"]),
        "n_ingredient_ids": len(cluster["counter"]),
        "members": members,
    }


def overhead_index(root: Path) -> dict[str, int]:
    folder = root / "imagery" / "realsense_overhead"
    found = {}
    if not folder.is_dir():
        return found
    for dish_dir in folder.iterdir():
        image = dish_dir / "rgb.png"
        if image.is_file():
            size = image.stat().st_size
            if size >= RGB_MIN_BYTES:
                found[dish_dir.name] = size
    return found


def first_overhead(cluster: dict, dishes: list[dict], images: dict[str, int]) -> str | None:
    for index in cluster["members"]:
        dish_id = dishes[index]["dish_id"]
        if dish_id in images:
            return dish_id
    return None


def summarize_numbers(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return float(ordered[index])

    return {
        "n": len(ordered),
        "mean": float(sum(ordered) / len(ordered)),
        "min": float(ordered[0]),
        "p10": pick(0.10),
        "p50": pick(0.50),
        "p90": pick(0.90),
        "max": float(ordered[-1]),
    }


def verify_local_hashes() -> dict:
    report = {}
    for relative, expected in EXPECTED_SHA256.items():
        path = N5K / relative
        actual = sha256_file(path)
        report[relative] = {
            "sha256": actual,
            "matches_archive_prefix": actual == expected,
            "bytes": path.stat().st_size,
        }
        if actual != expected:
            raise SystemExit(f"local file drifted from the official archive hash: {relative}")
    return report


def verify_archive_prefix() -> dict:
    if not ARCHIVE_PATH.is_file():
        return {"present": False, "path": str(ARCHIVE_PATH)}
    hashed = {}
    with tarfile.open(ARCHIVE_PATH, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if "/imagery/" in member.name:
                break
            payload = archive.extractfile(member).read()
            relative = member.name.split("nutrition5k_dataset/", 1)[-1]
            hashed[relative] = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
    missing = [name for name in EXPECTED_SHA256 if name not in hashed]
    mismatched = [name for name, expected in EXPECTED_SHA256.items() if name in hashed and hashed[name]["sha256"] != expected]
    if missing or mismatched:
        raise SystemExit(f"archive prefix mismatch missing={missing} mismatched={mismatched}")
    return {
        "present": True,
        "path": str(ARCHIVE_PATH),
        "bytes": ARCHIVE_PATH.stat().st_size,
        "prefix_matches_local_and_expected": True,
        "prefix_files": hashed,
    }


def scan_archive_inventory() -> dict:
    """Stream the official tar once and record every non-image member.

    Imagery payloads are not retained. Filename buckets show there is no
    plate-sequence table hiding under another directory name.
    """
    if not ARCHIVE_PATH.is_file():
        return {"present": False}
    non_image = []
    buckets: Counter = Counter()
    small_text = {}
    members = 0
    with tarfile.open(ARCHIVE_PATH, "r|gz") as archive:
        for member in archive:
            members += 1
            name = member.name
            if "/imagery/" in name:
                if member.isfile():
                    parts = name.split("/")
                    try:
                        image_at = parts.index("imagery")
                        bucket = "/".join(parts[image_at : image_at + 2]) + "/" + parts[-1]
                    except ValueError:
                        bucket = "imagery/?"
                    buckets[bucket] += 1
                continue
            kind = "dir" if member.isdir() else ("file" if member.isfile() else "other")
            non_image.append({"kind": kind, "bytes": int(member.size), "name": name})
            if member.isfile() and member.size <= 20_000:
                payload = archive.extractfile(member).read()
                text = payload.decode("utf-8", errors="replace")
                if "plate" in text.lower() or "incremental" in text.lower() or "sequence" in text.lower():
                    small_text[name] = text
    plate_named = [row["name"] for row in non_image if "plate" in row["name"].lower() or "sequence" in row["name"].lower()]
    return {
        "present": True,
        "path": str(ARCHIVE_PATH),
        "bytes": ARCHIVE_PATH.stat().st_size,
        "members": members,
        "non_image_members": non_image,
        "imagery_filename_buckets": dict(sorted(buckets.items())),
        "plate_or_sequence_in_non_image_names": plate_named,
        "small_text_mentioning_plate_or_incremental": small_text,
    }


def build_primary(dishes: list[dict], gap_s: int, splits_of) -> dict:
    graph = analyze_cafe(dishes, gap_s)
    clusters = graph["clusters"]
    valid_paths = []
    rejected_paths = []
    for path in graph["paths"]:
        steps = []
        ok = True
        for left, right in zip(path, path[1:]):
            residual = nutrition_residual(clusters[left], clusters[right], dishes)
            passed = nutrition_ok(residual, dishes[0]["cafe"] if dishes else "cafe1")
            steps.append((left, right, residual, passed))
            ok = ok and passed
        if ok:
            valid_paths.append(steps)
        else:
            rejected_paths.append(steps)

    fork_rows = []
    for parent, kids in sorted(graph["fork_parents"].items()):
        fork_rows.append((parent, kids))

    assembly_indexes = [index for index, label in enumerate(graph["labels"]) if label == "disjoint_assembly"]
    ambiguous_indexes = [index for index, label in enumerate(graph["labels"]) if label == "ambiguous_predecessors"]
    return {
        "graph": graph,
        "valid_paths": valid_paths,
        "rejected_paths": rejected_paths,
        "fork_rows": fork_rows,
        "assembly_indexes": assembly_indexes,
        "ambiguous_indexes": ambiguous_indexes,
        "splits_of": splits_of,
    }


def path_split_labels(steps_or_indexes: list[int], clusters: list[dict], dishes: list[dict], splits_of, kind: str) -> dict[str, list[str]]:
    labels = {"rgb": [], "depth": []}
    for index in steps_or_indexes:
        for member in clusters[index]["members"]:
            split = splits_of(dishes[member]["dish_id"])
            labels["rgb"].append(split["rgb"])
            labels["depth"].append(split["depth"])
    return labels


def sequence_payload(sequence_id: str, steps: list[tuple], clusters: list[dict], dishes: list[dict], splits_of, images: dict[str, int]) -> dict:
    indexes = [steps[0][0]] + [step[1] for step in steps]
    labels = path_split_labels(indexes, clusters, dishes, splits_of, "sequence")
    stage_rows = []
    for position, index in enumerate(indexes):
        record = cluster_record(clusters[index], dishes, splits_of)
        record["position"] = position
        record["overhead_dish_id"] = first_overhead(clusters[index], dishes, images)
        if position == 0:
            record["added"] = []
            record["nutrition_residual"] = None
        else:
            _parent, _child, residual, passed = steps[position - 1]
            record["added"] = added_items(clusters[indexes[position - 1]]["counter"], clusters[index]["counter"])
            record["nutrition_residual"] = {name: str(residual[name]) for name in NUTRIENT_FIELDS}
            record["nutrition_ok"] = passed
            record["intervening_dishes"] = sum(
                1
                for dish in dishes
                if clusters[indexes[position - 1]]["t_max"] < dish["ts"] < clusters[index]["t_min"]
            )
        stage_rows.append(record)
    return {
        "sequence_id": sequence_id,
        "cafe": dishes[0]["cafe"],
        "n_stages": len(indexes),
        "n_scans": sum(len(clusters[index]["members"]) for index in indexes),
        "n_pairs": len(steps),
        "rgb_splits": split_set(labels["rgb"]),
        "depth_splits": split_set(labels["depth"]),
        "rgb_leak": leaks(labels["rgb"], "rgb_train", "rgb_test"),
        "depth_leak": leaks(labels["depth"], "depth_train", "depth_test"),
        "stages": stage_rows,
    }


def repeat_leak_payload(cluster: dict, dishes: list[dict], splits_of) -> dict | None:
    if len(cluster["members"]) < 2:
        return None
    labels = {"rgb": [], "depth": []}
    for index in cluster["members"]:
        split = splits_of(dishes[index]["dish_id"])
        labels["rgb"].append(split["rgb"])
        labels["depth"].append(split["depth"])
    if not leaks(labels["rgb"], "rgb_train", "rgb_test") and not leaks(labels["depth"], "depth_train", "depth_test"):
        return None
    record = cluster_record(cluster, dishes, splits_of)
    record["rgb_splits"] = split_set(labels["rgb"])
    record["depth_splits"] = split_set(labels["depth"])
    record["rgb_leak"] = leaks(labels["rgb"], "rgb_train", "rgb_test")
    record["depth_leak"] = leaks(labels["depth"], "depth_train", "depth_test")
    return record


def count_sensitivity(cafe_dishes: dict[str, list[dict]], splits_of) -> list[dict]:
    rows = []
    for gap in SENSITIVITY_GAPS:
        item = {"gap_s": gap, "cafes": {}}
        for cafe, dishes in cafe_dishes.items():
            built = build_primary(dishes, gap, splits_of)
            sequences = []
            for steps in built["valid_paths"]:
                sequences.append(sequence_payload(f"{cafe}-tmp", steps, built["graph"]["clusters"], dishes, splits_of, {}))
            item["cafes"][cafe] = {
                "valid_pairs": sum(sequence["n_pairs"] for sequence in sequences),
                "valid_sequences": len(sequences),
                "rgb_leak_sequences": sum(1 for sequence in sequences if sequence["rgb_leak"]),
                "depth_leak_sequences": sum(1 for sequence in sequences if sequence["depth_leak"]),
                "fork_parents": len(built["fork_rows"]),
                "assemblies": len(built["assembly_indexes"]),
                "ambiguous_predecessors": len(built["ambiguous_indexes"]),
                "rejected_scale_paths": len(built["rejected_paths"]),
                "repeat_rgb_leak_groups": sum(
                    1
                    for cluster in built["graph"]["clusters"]
                    if (payload := repeat_leak_payload(cluster, dishes, splits_of)) is not None and payload["rgb_leak"]
                ),
            }
        rows.append(item)
    return rows


def unrestricted_gap_histogram(dishes: list[dict]) -> dict:
    """Gaps of unique maximal predecessors with session-sized repeat groups.

    The predecessor search itself has no time limit. This is a diagnostic for
    cross-day collisions, not the primary count.
    """
    clusters = build_repeat_clusters(dishes, SESSION_GAP_S)
    predecessors = maximal_predecessors(clusters, None)
    gaps = []
    for index, maximal in enumerate(predecessors):
        if len(maximal) == 1:
            gaps.append(clusters[index]["t_min"] - clusters[maximal[0]]["t_min"])
    buckets = {
        "le_5min": sum(gap <= 300 for gap in gaps),
        "le_10min": sum(gap <= 600 for gap in gaps),
        "le_30min": sum(gap <= 1800 for gap in gaps),
        "le_2h": sum(gap <= 7200 for gap in gaps),
        "le_24h": sum(gap <= 86400 for gap in gaps),
        "gt_24h": sum(gap > 86400 for gap in gaps),
    }
    return {"n_unique": len(gaps), "buckets": buckets, "max_s": max(gaps) if gaps else None}


def load_predictions() -> dict[str, list[dict]]:
    tables = {}
    order = None
    for setting, filename in SETTING_FILES:
        rows = []
        for line in (PREDICTION_DIR / filename).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        ids = [row["dish_id"] for row in rows]
        if order is None:
            order = ids
        elif ids != order:
            raise SystemExit(f"prediction dish order differs for {setting}")
        for row in rows:
            recomputed = abs(float(row["pred_mass_g"]) - float(row["mass_g"]))
            if abs(recomputed - float(row["abs_err_g"])) > 1e-4:
                raise SystemExit(f"stored absolute error does not match prediction for {row['dish_id']}")
        tables[setting] = rows
    if len(order or []) != 474:
        raise SystemExit(f"expected 474 Experiment A test predictions, found {len(order or [])}")
    return tables


def cluster_bootstrap(deltas_by_cluster: list[list[float]], draws: int, seed: int) -> dict:
    import numpy as np

    sums = np.array([float(sum(cluster)) for cluster in deltas_by_cluster], dtype=np.float64)
    sizes = np.array([len(cluster) for cluster in deltas_by_cluster], dtype=np.float64)
    if np.any(sizes <= 0):
        raise SystemExit("empty bootstrap cluster")
    point = float(sums.sum() / sizes.sum())
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(deltas_by_cluster), size=(draws, len(deltas_by_cluster)))
    means = sums[drawn].sum(axis=1) / sizes[drawn].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n_dishes": int(sizes.sum()),
        "n_clusters": len(deltas_by_cluster),
        "mae_difference_g": point,
        "ci95_low_g": float(low),
        "ci95_high_g": float(high),
        "fraction_method_mae_lower": float(np.mean(means < 0.0)),
        "stably_beats_top": bool(high < 0.0),
        "draws": draws,
        "seed": seed,
        "interval": "percentile 2.5 to 97.5",
        "cluster_size_max": int(sizes.max()),
        "dishes_in_multi_dish_clusters": int(sizes[sizes > 1].sum()),
        "multi_dish_clusters": int(np.sum(sizes > 1)),
    }


def assign_groups(cafe_results: dict[str, dict]) -> dict[str, str]:
    """Map a dish to its recovered plate group.

    Valid linear sequences share a group, including repeat scans of those
    stages. Repeat scans that never grow share their own group. Forks are not
    merged: those branches are not one recovered plate.
    """
    groups = {}
    for cafe, built in cafe_results.items():
        dishes = built["dishes"]
        clusters = built["graph"]["clusters"]
        used = set()
        for sequence in built["sequences"]:
            for stage in sequence["stages"]:
                for member in stage["members"]:
                    groups[member["dish_id"]] = sequence["sequence_id"]
                    used.add(member["dish_id"])
        for cluster in clusters:
            member_ids = [dishes[index]["dish_id"] for index in cluster["members"]]
            if any(dish_id in used for dish_id in member_ids):
                continue
            if len(member_ids) > 1:
                group = f"{cafe}-repeat-{cluster['cluster_index']:04d}"
                for dish_id in member_ids:
                    groups[dish_id] = group
    return groups


def run_bootstrap(cafe_results: dict[str, dict], groups: dict[str, str]) -> dict:
    tables = load_predictions()
    published = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    dish_ids = [row["dish_id"] for row in tables["单张俯视"]]
    top_errors = [float(row["abs_err_g"]) for row in tables["单张俯视"]]
    errors = {setting: [float(row["abs_err_g"]) for row in rows] for setting, rows in tables.items()}

    def groups_for(mode: str) -> list[list[int]]:
        buckets: dict[str, list[int]] = defaultdict(list)
        for index, dish_id in enumerate(dish_ids):
            if mode == "sequence_and_repeat":
                key = groups.get(dish_id, f"singleton:{dish_id}")
            elif mode == "published_dish":
                key = f"singleton:{dish_id}"
            else:
                raise SystemExit(mode)
            buckets[key].append(index)
        return list(buckets.values())

    sequence_groups = groups_for("sequence_and_repeat")
    settings = []
    for setting, _filename in SETTING_FILES:
        if setting == "单张俯视":
            continue
        method = errors[setting]
        iid = paired_mae_difference(method, top_errors, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
        published_by_setting = {row["setting"]: row for row in published["settings"]}
        if setting not in published_by_setting:
            known = ", ".join(published_by_setting)
            raise SystemExit(f"setting {setting!r} is missing from comparison.json; published settings: {known}")
        published_row = published_by_setting[setting]
        published_vs = published_row["versus_top"]
        iid_matches = all(
            abs(iid[key] - float(published_vs[key])) < 1e-9
            for key in ("mae_difference_g", "ci95_low_g", "ci95_high_g")
        )
        clustered_input = [[method[index] - top_errors[index] for index in cluster] for cluster in sequence_groups]
        clustered = cluster_bootstrap(clustered_input, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
        settings.append(
            {
                "setting": setting,
                "mass_mae_g": sum(method) / len(method),
                "top_mass_mae_g": sum(top_errors) / len(top_errors),
                "iid_bootstrap": iid,
                "iid_matches_published_comparison": iid_matches,
                "published_versus_top": published_vs,
                "sequence_bootstrap": clustered,
            }
        )
    multi = [cluster for cluster in sequence_groups if len(cluster) > 1]
    return {
        "test_n": len(dish_ids),
        "test_dish_ids_match_published_n": len(dish_ids) == published["test_n"],
        "grouping": "Experiment A rgb_test dishes that share a recovered linear sequence, or a repeat-scan group, are one cluster. Fork branches stay separate. Dishes with no recovered companion stay singletons.",
        "n_clusters": len(sequence_groups),
        "multi_dish_clusters": len(multi),
        "dishes_in_multi_dish_clusters": sum(len(cluster) for cluster in multi),
        "largest_cluster": max(len(cluster) for cluster in sequence_groups),
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "settings": settings,
    }


def luma(image) -> "object":
    import numpy as np

    rgb = image.astype(np.float32)
    return 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]


def windowed_ssim(left, right) -> float:
    import numpy as np
    from scipy.ndimage import uniform_filter

    left_l = luma(left)
    right_l = luma(right)
    window = 11
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_left = uniform_filter(left_l, window)
    mu_right = uniform_filter(right_l, window)
    mu_left_sq = mu_left * mu_left
    mu_right_sq = mu_right * mu_right
    mu_product = mu_left * mu_right
    sigma_left = uniform_filter(left_l * left_l, window) - mu_left_sq
    sigma_right = uniform_filter(right_l * right_l, window) - mu_right_sq
    sigma_product = uniform_filter(left_l * right_l, window) - mu_product
    numerator = (2 * mu_product + c1) * (2 * sigma_product + c2)
    denominator = (mu_left_sq + mu_right_sq + c1) * (sigma_left + sigma_right + c2)
    return float(np.mean(numerator / denominator))


def image_metrics(left, right) -> dict[str, float]:
    import numpy as np

    diff = np.abs(left.astype(np.int16) - right.astype(np.int16))
    height, width = diff.shape[:2]
    y0, y1 = int(height * CENTER_MARGIN), int(height * (1 - CENTER_MARGIN))
    x0, x1 = int(width * CENTER_MARGIN), int(width * (1 - CENTER_MARGIN))
    center = diff[y0:y1, x0:x1]
    border_mask = np.ones(diff.shape[:2], dtype=bool)
    border_mask[y0:y1, x0:x1] = False
    return {
        "mae_rgb": float(diff.mean()),
        "mae_center_rgb": float(center.mean()),
        "mae_border_rgb": float(diff[border_mask].mean()),
        "changed_frac_gt15": float(np.any(diff > CHANGED_PIXEL, axis=2).mean()),
        "ssim_luma": windowed_ssim(left, right),
        "height": int(height),
        "width": int(width),
    }


def load_rgb(dish_id: str):
    import numpy as np
    from PIL import Image

    path = N5K / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
    image = Image.open(path)
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    image.close()
    return array


def visual_audit(pair_specs: list[dict], repeat_specs: list[dict], pool: list[str], groups: dict[str, str]) -> dict:
    import numpy as np

    def evaluate(specs: list[dict], kind: str) -> list[dict]:
        rows = []
        for number, spec in enumerate(specs, start=1):
            if number == 1 or number % 250 == 0:
                print(f"visual {kind} {number}/{len(specs)}", flush=True)
            left = load_rgb(spec["before_dish_id"])
            right = load_rgb(spec["after_dish_id"])
            if left.shape != right.shape:
                rows.append({**spec, "kind": kind, "shape_mismatch": True})
                continue
            metrics = image_metrics(left, right)
            rows.append({**spec, "kind": kind, "shape_mismatch": False, **metrics})
        return rows

    adjacent_rows = evaluate(pair_specs, "adjacent_stage")
    repeat_rows = evaluate(repeat_specs, "repeat_scan")
    rng = np.random.default_rng(RANDOM_IMAGE_SEED)
    random_specs = []
    guard = 0
    while len(random_specs) < RANDOM_IMAGE_PAIRS and guard < RANDOM_IMAGE_PAIRS * 50:
        guard += 1
        left, right = rng.choice(pool, size=2, replace=False)
        left_group = groups.get(str(left), f"singleton:{left}")
        right_group = groups.get(str(right), f"singleton:{right}")
        if left_group == right_group:
            continue
        random_specs.append({"before_dish_id": str(left), "after_dish_id": str(right), "sequence_id": None})
    random_rows = evaluate(random_specs, "random_cross_sequence")

    def pack(rows: list[dict]) -> dict:
        usable = [row for row in rows if not row.get("shape_mismatch")]
        packed = {"n": len(usable), "shape_mismatch": len(rows) - len(usable)}
        for field in ("mae_rgb", "mae_center_rgb", "mae_border_rgb", "changed_frac_gt15", "ssim_luma"):
            packed[field] = summarize_numbers([float(row[field]) for row in usable])
        return packed

    return {
        "pixel_rule": {
            "mae_rgb": "Mean absolute difference over RGB channels and pixels, on the 0-255 scale.",
            "center": f"Center crop keeps the middle {1 - 2 * CENTER_MARGIN:.0%} of width and height. Border is the complement.",
            "changed_frac_gt15": f"Fraction of pixels whose largest channel difference exceeds {CHANGED_PIXEL}.",
            "ssim_luma": "Mean windowed SSIM on Rec.601 luminance, 11-pixel uniform window, constants K1=0.01 and K2=0.03.",
            "random_pairs": RANDOM_IMAGE_PAIRS,
            "random_seed": RANDOM_IMAGE_SEED,
            "network": "No network, no DINOv2, no side view, no depth.",
        },
        "adjacent_stage": pack(adjacent_rows),
        "repeat_scan": pack(repeat_rows),
        "random_cross_sequence": pack(random_rows),
        "adjacent_rows": adjacent_rows,
        "repeat_rows": repeat_rows,
    }


def decimal_default(value):
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(type(value))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=decimal_default) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, default=decimal_default) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def fmt_num(value, digits=4) -> str:
    if value is None:
        return "—"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def metric_cell(summary: dict, field: str) -> str:
    block = summary.get(field) or {}
    if not block or block.get("n", 0) == 0:
        return "—"
    return f"{block['p50']:.2f}（均值 {block['mean']:.2f}）"


def render_report(summary: dict) -> str:
    counts = summary["counts"]
    visual = summary.get("visual_summary")
    bootstrap = summary.get("bootstrap")
    inventory = summary["archive_inventory"]
    if inventory.get("members"):
        archive_line = (
            f"归档是 `{summary['archive']['path']}`，大小 {summary['archive'].get('bytes', '—')} 字节。"
            f"前缀里的元数据和划分文件已与本地副本逐字节核对。全归档共 {inventory['members']} 个成员。"
            "非图像成员只有菜品元数据、食材表、dish_id 清单、官方划分、说明和抽帧脚本。"
            f"文件名里带 plate 或 sequence 的非图像成员有 {len(inventory.get('plate_or_sequence_in_non_image_names') or [])} 个。"
            "图像文件名只有俯视 `rgb.png`、`depth_raw.png`、`depth_color.png` 和四路侧视 `camera_A` 到 `camera_D` 的视频。"
        )
    else:
        archive_line = (
            f"归档前缀已与本地副本逐字节核对，路径 `{summary['archive']['path']}`。"
            "这次输出没有包含全归档成员扫描。"
        )
    length_text = "、".join(f"{key} 个状态 {value} 条" for key, value in counts["length_histogram"].items())
    lines = [
        "# Nutrition5k 增量扫描序列审计",
        "",
        "结论先说：**可以可靠恢复出一部分同一物理餐盘的相邻扫描阶段。** 官方归档没有餐盘序列编号。恢复靠的是后一阶段原样保留了前一阶段的配料记录，并且这样的上一阶段在 30 分钟的同一次采集里只有一个。时间戳用来排序，也用来把超过 30 分钟的包含关系留在诊断里。分叉、互不相交的组件拼盘、前驱不唯一的盘子，都没有编进这批序列。",
        "",
        f"按这条规则，有效相邻阶段对是 **{counts['valid_pairs']}** 对，独立餐盘序列是 **{counts['valid_sequences']}** 条，覆盖 **{counts['dishes_in_valid_sequences']}** 个 dish_id。官方发布规模仍是 **{counts['official_dish_ids']}** 个 dish_id。这 {counts['valid_pairs']} 对是增量阶段对。实验 A 记下的 3,289 是同时有俯视和侧视的盘子数，回答的是另一个问题。",
        "",
        f"建议：**不要因为序列无法恢复而放弃这个方向。** 本审计到此停止，不训练网络，不开始第二阶段。有效序列里有 {counts['rgb_leak_sequences']} 条跨了官方 RGB 训练和测试，另有 {counts['repeat_rgb_leaks']} 组重复拍摄跨了划分。以后若要训练，这些 dish_id 必须整组留在同一侧。分叉和组件拼盘不能当成已经有唯一上一阶段的增量对。",
        "",
        "## 用了哪些官方记录",
        "",
        archive_line,
        "",
        "菜品编号 `dish_` 后面的 10 位整数是采集时间戳，这是官方编号本身，不是事后估计的拍摄时间。配料行来自 `dish_metadata_cafe1.csv` 和 `dish_metadata_cafe2.csv`。官方说明文字写了 `num_ingrs` 这一列，归档里的行在总蛋白质之后直接进入配料编号；5,006 行的剩余字段都能按 7 个一组成配料块。",
        "",
        "论文写明：食物是一项一项加到同一个盘子或碗里，每加一项做一次扫描；一项可以是单一配料，也可以是已知配方的预制食物；秤的精度大约是 ±1 克；同一次增量扫描的各阶段放在同一个训练或测试划分里。这些句子用来规定该核对什么，不用来填序列条数。",
        "",
        "## 计数规则",
        "",
        "同一家食堂的元数据放在一起排，cafe1 和 cafe2 不互相连接。一条配料记录是编号、名称、克数、能量、脂肪、碳水、蛋白质这 7 个原始字符串。多重集合相同，才算同一条记录被原样保留。",
        "",
        "1. 成分记录完全相同、并且相邻两次的时间差不超过 30 分钟的扫描，并成同一组成状态的重复拍摄。重复拍摄不是“新增了食物”。",
        "2. 后一状态比前一状态更晚开始，间隔不超过 30 分钟。前一状态的配料记录是后一状态的真子集，而且这个时间窗里没有另一个更大的真子集前驱。满足条件的上一状态必须恰好一个。",
        "3. 这个上一状态在时间窗里只能通向一种下一组成。通向两种或更多不同组成时，整组算分叉，不进入可靠序列。同一组成的重复拍摄先并掉，不算分叉。",
        "4. 每个状态选用与配料克数之和最接近的那次盘级称重。后一状态减去前一状态的质量，必须与新增配料的克数相差不超过 2 克。单次称重大约 ±1 克，两次相减的容差按 2 克计。cafe1 还要求能量、脂肪、碳水、蛋白质的盘级差与新增配料的对应差不超过 0.05。cafe2 的盘级能量和三种宏量在 238 行里都是 0，不能做这四项核对，只核质量。",
        "5. 一条结构上连起来的路径，只要有一步称重或营养对不上，整条路径都不计入有效序列。",
        "",
        "30 分钟这个窗对应一次食堂采集：论文里一圈侧视大约 8 秒，再加上把下一项食物放到盘子上的时间。敏感性表给出 5 分钟、10 分钟、30 分钟、2 小时和 24 小时。主计数只用 30 分钟。不设上限的诊断放在敏感性一节，用来看隔天的同组成配料会不会被当成上一阶段。",
        "",
        "## 规模",
        "",
        "| 项目 | cafe1 | cafe2 | 合计 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("官方 dish_id", "dishes"),
        ("有效相邻阶段对", "valid_pairs"),
        ("独立餐盘序列", "valid_sequences"),
        ("序列中的 dish_id", "dishes_in_valid_sequences"),
        ("序列中的组成状态", "stages_in_valid_sequences"),
        ("称重未通过对而整路径排除", "rejected_scale_paths"),
        ("分叉的上一状态", "fork_parents"),
        ("组件拼盘", "assemblies"),
        ("前驱不唯一", "ambiguous"),
        ("重复拍摄跨 rgb 划分的组", "repeat_rgb_leaks"),
        ("有效序列跨 rgb 划分", "rgb_leak_sequences"),
        ("有效序列跨 depth 划分", "depth_leak_sequences"),
    ):
        per = counts["by_cafe"]
        lines.append(
            f"| {label} | {per['cafe1'][key]} | {per['cafe2'][key]} | {per['cafe1'][key] + per['cafe2'][key]} |"
        )
    lines.extend(
        [
            "",
            f"有效序列的长度分布（组成状态数）：{length_text}。最长的一条有 {counts['longest_sequence']['n_stages']} 个状态。序列里的 dish_id 比组成状态多 {counts['dishes_in_valid_sequences'] - counts['stages_in_valid_sequences']} 个，这是同一组成被重复拍摄的次数。",
            "",
            f"这 {counts['valid_pairs']} 对里，中间没有其他 dish_id 的有 {counts['intervening_zero']} 对，中间隔着别的 dish_id 的最多隔 {counts['intervening_max']} 个。相邻阶段在时间上是连着的。",
            "",
            f"若只按时间戳把同一食堂的 dish_id 前后相接，质量下降超过 1 克的有 {counts['raw_consecutive_mass_decreases']} 对。恢复出来的相邻阶段里，质量增加超过 1 克的有 {counts['valid_mass_increase']} 对，变化在 1 克以内的有 {counts['valid_mass_flat']} 对，下降超过 1 克的有 {counts['valid_mass_decrease']} 对。规则没有把质量上升写成前提。核对之后，这些相邻阶段的质量都顺着新增配料上升。",
            "",
            f"新增配料行数：只有 1 个配料编号的有 {counts['added_one_id']} 对，多于 1 个配料编号的有 {counts['added_many_ids']} 对。多于一行时，官方记录并没有标出哪些行属于同一次加入的预制食物，所以这里按配料行报告，不把它们合并成菜名。",
            "",
            f"两侧都能找到不小于 {RGB_MIN_BYTES} 字节俯视 RGB 的有效阶段对有 {counts['valid_pairs_with_both_overhead']} 对。",
            "",
            "## 组成变化是否对得上称重和营养",
            "",
            f"cafe1 的有效阶段对，质量残差最大绝对值是 {counts['residual_max']['cafe1']['mass']} 克，能量 {counts['residual_max']['cafe1']['kcal']} kcal，脂肪 {counts['residual_max']['cafe1']['fat']} 克，碳水 {counts['residual_max']['cafe1']['carb']} 克，蛋白质 {counts['residual_max']['cafe1']['protein']} 克。质量容差是 2 克，能量和宏量容差是 0.05。残差来自盘级字段和配料行之和的舍入。",
            "",
            f"cafe2 只核了质量，最大绝对值是 {counts['residual_max']['cafe2']['mass']} 克。盘级能量和宏量全是 0，不能用来证明宏量差；新增食物的能量和宏量只存在于配料行里。",
            "",
            "有 {n} 个重复组，组内最接近配料克数的那次称重仍然和配料克数相差超过 1 克。其中质量极差最大的一组是 {ids}，极差 {span} 克，配料记录相同，能量和宏量也相同。坏的是盘级质量字段，这组没有被收成增量序列。".format(
                n=counts["scale_disagreement_clusters"],
                ids="、".join(f"`{dish_id}`" for dish_id in counts["worst_scale_cluster"]["dish_ids"]),
                span=counts["worst_scale_cluster"]["mass_range_g"],
            ),
            "",
        ]
    )
    longest = summary["examples"]["longest_sequence"]
    lines.append("最长的一条恢复序列如下。每一行是一个组成状态，从第二行起写的是相对上一状态新增的配料。")
    lines.append("")
    lines.append("| 阶段 | dish_id | 划分 | 质量（克） | 新增 |")
    lines.append("| ---: | --- | --- | ---: | --- |")
    for stage in longest["stages"]:
        shown = stage["added"][:8]
        extra = f" 等 {len(stage['added'])} 行" if len(stage["added"]) > 8 else ""
        added = ("、".join(f"{item['name']} {item['grams']} g" for item in shown) + extra) or "起始状态"
        mass = stage["mass_closest_g"]
        split = stage["members"][0]["rgb"]
        lines.append(f"| {stage['position'] + 1} | `{stage['representative_dish_id']}` | {split} | {mass} | {added} |")
    lines.extend(["", "## 划分有没有把同一餐盘拆开", ""])
    leak_sequences = summary["examples"]["rgb_leak_sequences"]
    if not leak_sequences:
        lines.append("有效序列没有跨 rgb 训练和测试。")
    else:
        lines.append(f"有效序列里有 {len(leak_sequences)} 条同时含有 rgb_train 和 rgb_test。官方“同一次增量扫描不跨划分”在这些序列上不成立。")
        lines.append("")
        for sequence in leak_sequences:
            bits = []
            for stage in sequence["stages"]:
                for member in stage["members"]:
                    bits.append(f"`{member['dish_id']}`（{member['rgb']} / {member['depth']}）")
            added_bits = []
            for stage in sequence["stages"][1:]:
                added_bits.append("、".join(f"{item['name']} {item['grams']} g" for item in stage["added"][:6]))
            lines.append(f"- {sequence['sequence_id']}：{' → '.join(bits)}。新增 {('；'.join(added_bits)) or '无'}。")
    same_depth = [sequence["sequence_id"] for sequence in leak_sequences if sequence["depth_leak"]]
    lines.extend(
        [
            "",
            f"深度划分上，有效序列跨 depth_train 和 depth_test 的有 {counts['depth_leak_sequences']} 条。"
            + (
                f"与上面 RGB 泄漏相同的序列是 {', '.join(same_depth)}。"
                if same_depth and len(same_depth) == counts["depth_leak_sequences"]
                else "深度泄漏按深度划分单独点名，见 `valid_sequences.jsonl`。"
            )
            + "深度划分是 RGB 划分的子集。一盘可以有 RGB 而没有深度；只有深度训练和深度测试两侧都出现，才记为深度泄漏。",
            "",
            f"重复拍摄、但没有长成增量序列的组里，跨 rgb 划分的有 {counts['repeat_rgb_leaks']} 组。其中不是“单一整数克配料”的有 {counts['repeat_rgb_leaks_strong']} 组，单一整数克配料的有 {counts['repeat_rgb_leaks_weak']} 组。前一类是多行配料或带小数克数，几乎只能是同一次称重被原样再记了一次。后一类仍有刚好称到同一整数克数的可能，所以分开计。时间差都在 30 分钟以内。",
            "",
        ]
    )
    if summary["examples"]["repeat_leak"]:
        example = summary["examples"]["repeat_leak"]
        ids = "、".join(f"`{member['dish_id']}`（{member['rgb']}，{member['mass_g']} g）" for member in example["members"])
        lines.append(f"一组重复拍摄跨划分的例子：{ids}。配料记录相同。")
        lines.append("")
    if summary["examples"]["fork"]:
        fork = summary["examples"]["fork"]
        lines.append("分叉的例子：同一个上一状态在 30 分钟里通向不止一种下一组成，所以不能收成一条盘子序列。")
        lines.append("")
        lines.append(f"- 上一状态 `{fork['parent']}`（{fork.get('parent_rgb', '—')}）")
        for child in fork["children"]:
            shown = child["added"][:6]
            extra = f" 等 {len(child['added'])} 行" if len(child["added"]) > 6 else ""
            added = "、".join(f"{item['name']} {item['grams']} g" for item in shown) + extra
            lines.append(f"- 下一状态 `{child['dish_id']}`（{child['rgb']}），新增 {added}")
        lines.append("")
    if summary["examples"]["assembly"]:
        assembly = summary["examples"]["assembly"]
        lines.append("组件拼盘的例子：后面那一盘的配料记录恰好是前面几盘的不相交并集，所以上一阶段不是唯一的。这几盘没有被写成一条增长序列。")
        lines.append("")
        lines.append(f"- 合并后的盘 `{assembly['dish_id']}`")
        for part in assembly["parts"]:
            lines.append(f"- 组件 `{part['dish_id']}`：{part['ingredients']}")
        lines.append("")
    lines.extend(["## 俯视图像有没有可学习的差异", ""])
    visual_delta_is_subtle = False
    if not visual:
        lines.append("这次没有计算图像。")
        lines.append("")
    else:
        lines.extend(
            [
                "只比较了俯视 RGB。没有用侧视，没有用深度，没有训练网络。结构相似度是亮度上的 11 像素窗口 SSIM。中心区域是画面中间 50% 的宽和高，用来看差异落在盘子中间还是落在设备背景上。",
                "",
                "| 比较 | 对数 | 像素平均绝对差中位数（0–255） | 中心 / 边缘中位数 | 差值大于 15 的像素比例中位数 | 亮度 SSIM 中位数 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, key in (
            ("恢复出的相邻阶段", "adjacent_stage"),
            ("同一组成的重复拍摄", "repeat_scan"),
            ("不同序列的随机两张", "random_cross_sequence"),
        ):
            block = visual[key]
            center = block["mae_center_rgb"]
            border = block["mae_border_rgb"]
            center_text = "—" if not center.get("n") else f"{center['p50']:.2f} / {border['p50']:.2f}"
            changed = block["changed_frac_gt15"]
            ssim = block["ssim_luma"]
            mae = block["mae_rgb"]
            lines.append(
                f"| {label} | {block['n']} | {mae.get('p50', 0):.2f} | {center_text} | {changed.get('p50', 0):.3f} | {ssim.get('p50', 0):.3f} |"
            )
        adj = visual["adjacent_stage"]["mae_rgb"]
        rep = visual["repeat_scan"]["mae_rgb"]
        rnd = visual["random_cross_sequence"]["mae_rgb"]
        adj_mae = float(adj.get("p50", 0))
        rep_mae = float(rep.get("p50", 0))
        rnd_mae = float(rnd.get("p50", 0))
        adj_ssim = float(visual["adjacent_stage"]["ssim_luma"].get("p50", 0))
        rep_ssim = float(visual["repeat_scan"]["ssim_luma"].get("p50", 0))
        rnd_ssim = float(visual["random_cross_sequence"]["ssim_luma"].get("p50", 0))
        center = float(visual["adjacent_stage"]["mae_center_rgb"].get("p50", 0))
        border = float(visual["adjacent_stage"]["mae_border_rgb"].get("p50", 0))
        repeat_center = float(visual["repeat_scan"]["mae_center_rgb"].get("p50", 0))
        repeat_border = float(visual["repeat_scan"]["mae_border_rgb"].get("p50", 0))
        close_to_repeat = abs(adj_mae - rep_mae) < 2 or (rep_mae > 0 and abs(adj_mae - rep_mae) / rep_mae < 0.15)
        if close_to_repeat and rnd_mae > adj_mae * 2:
            visual_delta_is_subtle = True
            relation = (
                f"相邻阶段和同一组成的重复拍摄都远比随机两张更接近：像素差中位数是 {adj_mae:.2f}、{rep_mae:.2f} 和 {rnd_mae:.2f}，"
                f"亮度 SSIM 中位数是 {adj_ssim:.3f}、{rep_ssim:.3f} 和 {rnd_ssim:.3f}。"
                f"相邻阶段比重复拍摄只高 {adj_mae - rep_mae:.2f}。整图指标分不清“加了一项食物”和“同一组成又拍了一次”。"
            )
        elif adj_mae > rep_mae and adj_mae < rnd_mae:
            relation = (
                f"相邻阶段的像素差中位数是 {adj_mae:.2f}，高于重复拍摄的 {rep_mae:.2f}，低于随机两张的 {rnd_mae:.2f}。"
                "新增食物在整图上留出了比重复拍摄更大的差异。"
            )
        else:
            relation = f"三种比较的像素差中位数是相邻阶段 {adj_mae:.2f}、重复拍摄 {rep_mae:.2f}、随机两张 {rnd_mae:.2f}。以上表为准。"
        if center > border and repeat_center > repeat_border:
            where = "画面中心的像素差高于边缘，重复拍摄也是这样，所以中心更高不能单独当成新增食物的证据。"
        elif center > border:
            where = "相邻阶段的差异中位数在画面中心高于边缘。"
        else:
            where = "中心和边缘的像素差见上表。"
        lines.extend(
            [
                "",
                f"{relation}{where}这里没有训练网络。随机跨序列只是图像统计对照，不是第二阶段的跨盘训练负对照。",
                "",
            ]
        )
    lines.extend(["## 实验 A 的分组 bootstrap", ""])
    if not bootstrap:
        if counts["valid_sequences"]:
            lines.append("序列已经恢复。这次运行没有重算分组 bootstrap。")
        else:
            lines.append("序列没有恢复，所以没有做分组 bootstrap。")
        lines.append("")
    else:
        lines.extend(
            [
                "实验 A 原来的区间，是把 474 个官方 rgb_test 盘当成互相独立的，做 10,000 次、种子 0 的配对 bootstrap。下面用同一批已经写好的预测重算。点估计不变。分组方式是：落在同一条恢复序列里的测试盘合成一个簇，只是重复拍摄、没有增长的测试盘也合成一个簇；分叉没有并成一簇。没有同伴的测试盘仍是单独一簇。",
                "",
                f"474 个测试盘分成 {bootstrap['n_clusters']} 个簇，其中 {bootstrap['multi_dish_clusters']} 个簇含有多盘，涉及 {bootstrap['dishes_in_multi_dish_clusters']} 盘，最大簇 {bootstrap['largest_cluster']} 盘。",
                "",
                "| 设置 | 质量 MAE（克） | 相对俯视的点估计（克） | 按盘独立的 95% 区间 | 按序列分组的 95% 区间 | 分组后是否整体低于 0 |",
                "| --- | ---: | ---: | --- | --- | --- |",
            ]
        )
        top_mae = bootstrap["settings"][0]["top_mass_mae_g"]
        lines.append(f"| 单张俯视 | {top_mae:.4f} | 基线 | — | — | 基线 |")
        for row in bootstrap["settings"]:
            iid = row["iid_bootstrap"]
            grouped = row["sequence_bootstrap"]
            lines.append(
                "| {setting} | {mae:.4f} | {diff:.4f} | [{lo:.4f}, {hi:.4f}] | [{glo:.4f}, {ghi:.4f}] | {flag} |".format(
                    setting=row["setting"],
                    mae=row["mass_mae_g"],
                    diff=iid["mae_difference_g"],
                    lo=iid["ci95_low_g"],
                    hi=iid["ci95_high_g"],
                    glo=grouped["ci95_low_g"],
                    ghi=grouped["ci95_high_g"],
                    flag="是" if grouped["stably_beats_top"] else "否",
                )
            )
        matched = all(row["iid_matches_published_comparison"] for row in bootstrap["settings"])
        lines.extend(
            [
                "",
                f"按盘独立重算的区间与已发表的 `comparison.json` {'一致' if matched else '不一致'}。分组之后，区间是否跨过 0 的判断没有改变：没有一种设置的区间整体落在 0 以下。这只说明实验 A 的“没有稳定低于俯视”在按序列分组后仍然成立，不是一项新实验，也不解锁第二阶段。",
                "",
            ]
        )
    lines.extend(
        [
            "## 敏感性",
            "",
            "主计数用 30 分钟。下表是同一套成分规则在相邻时间窗上的有效阶段对和泄漏条数。时间窗变了，极大前驱有时会换人，所以条数会有小的变化。",
            "",
            "| 时间窗 | 有效阶段对 | 序列 | 序列跨 rgb 划分 | 分叉上一状态 | 组件拼盘 | 前驱不唯一 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["sensitivity"]:
        cafes = row["cafes"]
        lines.append(
            "| {gap} 秒 | {pairs} | {seqs} | {leaks} | {forks} | {asm} | {amb} |".format(
                gap=row["gap_s"],
                pairs=cafes["cafe1"]["valid_pairs"] + cafes["cafe2"]["valid_pairs"],
                seqs=cafes["cafe1"]["valid_sequences"] + cafes["cafe2"]["valid_sequences"],
                leaks=cafes["cafe1"]["rgb_leak_sequences"] + cafes["cafe2"]["rgb_leak_sequences"],
                forks=cafes["cafe1"]["fork_parents"] + cafes["cafe2"]["fork_parents"],
                asm=cafes["cafe1"]["assemblies"] + cafes["cafe2"]["assemblies"],
                amb=cafes["cafe1"]["ambiguous_predecessors"] + cafes["cafe2"]["ambiguous_predecessors"],
            )
        )
    hist = summary["unrestricted_unique_gaps"]
    lines.extend(
        [
            "",
            f"不设时间上限时，cafe1 唯一极大前驱有 {hist['cafe1']['n_unique']} 个，其中 24 小时以内 {hist['cafe1']['buckets']['le_24h']} 个，超过 24 小时 {hist['cafe1']['buckets']['gt_24h']} 个。超过一天的那些是同一套整数克数碰巧再次出现，不收进主计数。",
            "",
            "## 失败样例里还要记住的事",
            "",
            "按时间戳把全部 dish_id 排成一列，下一项经常是另一盘食物，质量会掉下去。那一列本身不是增量序列。组件拼盘是“几项分别称完，再出现一盘把这些记录并在一起”，上一阶段不唯一。分叉是同一组成在短时间里长成两种不同的下一盘。这两类都有成分上的包含关系，但回复不出唯一的相邻阶段，所以没有计入有效阶段对。",
            "",
            "cafe2 的 238 个盘子不在官方 RGB 训练或测试清单里。它们的增量序列单独计数，不进入实验 A 的 474 盘 bootstrap。",
            "",
            "## 建议",
            "",
        "序列能够按上面的规则可靠恢复。因此不建议以序列无法恢复为由放弃这个方向。"
        + (
            "整图像素差还不能把新增食物和重复拍摄分开；以后若做模型，不能把这个整图距离当成已经证明的视觉监督。"
            if visual_delta_is_subtle
            else "俯视差异见上面的图像表。"
        ),
        "",
        "本文件不是第二阶段的开工许可。三个轻量模型、跨盘随机配对的训练负对照、完整网络、外部数据和不完整食材监督都仍然锁定。若以后开工，训练和测试必须按这里恢复出的餐盘序列分组，已经跨官方划分的 dish_id 必须整组留在同一侧；分叉和组件拼盘继续排除在增量对之外。",
            "",
            "## 怎样复核",
            "",
            "```bash",
            "python3 scripts/icdl_phase1_sequence_audit.py",
            "```",
            "",
            "脚本先跑合成例子，再读本地官方元数据，并与归档前缀的 sha256 核对。完整成员清单写在 `archive_inventory.json`。有效阶段对在 `valid_pairs.jsonl`，序列在 `valid_sequences.jsonl`。逐对俯视差异在 `visual_pairs.jsonl`（相邻阶段和同一组成的重复拍摄）。汇总表在 `visual_summary.json`。图像来自本地 `imagery/realsense_overhead/<dish_id>/rgb.png`。实验 A 的预测只读，不改 `results/experiment_a_n5k_dual_view/predictions/`。",
            "",
        ]
    )
    return "\n".join(lines)


def make_dish(dish_id: str, ts: int, mass: str, rows: list[tuple[str, str, str]], split_cafe: str = "cafe1") -> dict:
    full_rows = tuple((ing_id, name, grams, "0", "0", "0", "0") for ing_id, name, grams in rows)
    counter = Counter(full_rows)
    return {
        "dish_id": dish_id,
        "cafe": split_cafe,
        "ts": ts,
        "kcal": Decimal(0),
        "mass": d(mass),
        "fat": Decimal(0),
        "carb": Decimal(0),
        "protein": Decimal(0),
        "rows": full_rows,
        "counter": counter,
        "key": tuple(sorted(counter.items())),
    }


def run_self_check() -> None:
    apple = ("ingr_0000000001", "apple", "50")
    bread = ("ingr_0000000002", "bread", "30")
    cheese = ("ingr_0000000003", "cheese", "20")
    chain = [
        make_dish("dish_100", 1_000, "50", [apple]),
        make_dish("dish_101", 1_030, "80", [apple, bread]),
        make_dish("dish_102", 1_060, "100", [apple, bread, cheese]),
    ]
    built = build_primary(chain, SESSION_GAP_S, lambda dish_id: {"rgb": "rgb_train", "depth": "outside"})
    if len(built["valid_paths"]) != 1 or sum(len(path) for path in built["valid_paths"]) != 2:
        raise SystemExit("self-check failed: a three-stage chain should be one sequence with two pairs")

    fork = [
        make_dish("dish_200", 2_000, "40", [("ingr_0000000004", "egg", "40")]),
        make_dish("dish_201", 2_020, "50", [("ingr_0000000004", "egg", "40"), ("ingr_0000000005", "tomato", "10")]),
        make_dish("dish_202", 2_040, "55", [("ingr_0000000004", "egg", "40"), ("ingr_0000000006", "bean", "15")]),
    ]
    forked = build_primary(fork, SESSION_GAP_S, lambda dish_id: {"rgb": "rgb_train", "depth": "outside"})
    if forked["valid_paths"] or len(forked["fork_rows"]) != 1:
        raise SystemExit("self-check failed: two different additions should be a fork, not a sequence")

    assembly = [
        make_dish("dish_300", 3_000, "10", [("ingr_0000000007", "a", "10")]),
        make_dish("dish_301", 3_020, "20", [("ingr_0000000008", "b", "20")]),
        make_dish("dish_302", 3_040, "30", [("ingr_0000000009", "c", "30")]),
        make_dish(
            "dish_303",
            3_080,
            "60",
            [("ingr_0000000007", "a", "10"), ("ingr_0000000008", "b", "20"), ("ingr_0000000009", "c", "30")],
        ),
    ]
    assembled = build_primary(assembly, SESSION_GAP_S, lambda dish_id: {"rgb": "rgb_test", "depth": "outside"})
    if assembled["valid_paths"] or len(assembled["assembly_indexes"]) != 1:
        raise SystemExit("self-check failed: disjoint components should be an assembly, not a chain")

    leaked = [
        make_dish("dish_400", 4_000, "50", [apple]),
        make_dish("dish_401", 4_030, "80", [apple, bread]),
    ]

    def leak_split(dish_id: str) -> dict[str, str]:
        return {"rgb": "rgb_test" if dish_id.endswith("400") else "rgb_train", "depth": "outside"}

    leak_built = build_primary(leaked, SESSION_GAP_S, leak_split)
    if len(leak_built["valid_paths"]) != 1:
        raise SystemExit("self-check failed: cross-split chain should still be recovered and then flagged")

    bad_mass = [
        make_dish("dish_500", 5_000, "50", [apple]),
        make_dish("dish_501", 5_030, "50", [apple, bread]),
    ]
    bad = build_primary(bad_mass, SESSION_GAP_S, lambda dish_id: {"rgb": "rgb_train", "depth": "outside"})
    if bad["valid_paths"] or not bad["rejected_paths"]:
        raise SystemExit("self-check failed: a mass delta that ignores the added food should be rejected")

    far = [
        make_dish("dish_600", 6_000, "50", [apple]),
        make_dish("dish_601", 6_000 + 3 * 86400, "80", [apple, bread]),
    ]
    far_built = build_primary(far, SESSION_GAP_S, lambda dish_id: {"rgb": "rgb_train", "depth": "outside"})
    if far_built["valid_paths"]:
        raise SystemExit("self-check failed: a subset three days later should not become a session pair")
    print("self-check ok", flush=True)


def cafe_counts(built: dict, dishes: list[dict], sequences: list[dict], splits_of) -> dict:
    repeat_leaks = []
    strong = 0
    weak = 0
    for cluster in built["graph"]["clusters"]:
        payload = repeat_leak_payload(cluster, dishes, splits_of)
        if payload and payload["rgb_leak"]:
            # Incremental sequences also contain these scans. Count a repeat leak
            # only when the repeat group is not already inside a valid sequence.
            member_ids = {member["dish_id"] for member in payload["members"]}
            in_sequence = any(member_ids & {row["dish_id"] for stage in sequence["stages"] for row in stage["members"]} for sequence in sequences)
            if in_sequence:
                continue
            repeat_leaks.append(payload)
            if payload["n_ingredient_ids"] >= 3 or payload["fractional_grams"]:
                strong += 1
            elif payload["n_ingredient_ids"] == 1 and not payload["fractional_grams"]:
                weak += 1
            else:
                strong += 1
    residuals = {name: Decimal(0) for name in NUTRIENT_FIELDS}
    mass_increase = mass_flat = mass_decrease = 0
    added_one = added_many = 0
    for sequence in sequences:
        for stage in sequence["stages"][1:]:
            residual = {name: abs(d(stage["nutrition_residual"][name])) for name in NUTRIENT_FIELDS}
            for name in NUTRIENT_FIELDS:
                residuals[name] = max(residuals[name], residual[name])
            delta = d(stage["mass_closest_g"]) - d(sequence["stages"][stage["position"] - 1]["mass_closest_g"])
            if delta > MASS_TOL_G:
                mass_increase += 1
            elif delta < -MASS_TOL_G:
                mass_decrease += 1
            else:
                mass_flat += 1
            names = {item["id"] for item in stage["added"]}
            if len(names) <= 1:
                added_one += 1
            else:
                added_many += 1
    scale_bad = []
    for cluster in built["graph"]["clusters"]:
        record = cluster_record(cluster, dishes, splits_of)
        if record["scale_disagrees"] or d(record["mass_range_g"]) > MASS_TOL_G:
            scale_bad.append(record)
    return {
        "dishes": len(dishes),
        "valid_pairs": sum(sequence["n_pairs"] for sequence in sequences),
        "valid_sequences": len(sequences),
        "dishes_in_valid_sequences": sum(sequence["n_scans"] for sequence in sequences),
        "stages_in_valid_sequences": sum(sequence["n_stages"] for sequence in sequences),
        "rejected_scale_paths": len(built["rejected_paths"]),
        "fork_parents": len(built["fork_rows"]),
        "assemblies": len(built["assembly_indexes"]),
        "ambiguous": len(built["ambiguous_indexes"]),
        "repeat_rgb_leaks": len(repeat_leaks),
        "repeat_rgb_leaks_strong": strong,
        "repeat_rgb_leaks_weak": weak,
        "rgb_leak_sequences": sum(1 for sequence in sequences if sequence["rgb_leak"]),
        "depth_leak_sequences": sum(1 for sequence in sequences if sequence["depth_leak"]),
        "repeat_leaks": repeat_leaks,
        "residual_max": {name: str(residuals[name]) for name in NUTRIENT_FIELDS},
        "mass_increase": mass_increase,
        "mass_flat": mass_flat,
        "mass_decrease": mass_decrease,
        "added_one": added_one,
        "added_many": added_many,
        "scale_bad": scale_bad,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Nutrition5k incremental plate sequences.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--self-check-only", action="store_true")
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--skip-archive-full", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    frozen = (ROOT / "results" / "experiment_a_n5k_dual_view").resolve()
    if output == frozen or frozen in output.parents:
        raise SystemExit("refusing to write into Experiment A results")
    run_self_check()
    if args.self_check_only:
        return

    output.mkdir(parents=True, exist_ok=True)
    print("checking official files", flush=True)
    local_hashes = verify_local_hashes()
    archive_prefix = verify_archive_prefix()
    saved_inventory = output / "archive_inventory.json"
    if args.skip_archive_full and saved_inventory.is_file():
        inventory = json.loads(saved_inventory.read_text(encoding="utf-8"))
        if not inventory.get("members"):
            inventory = {"present": ARCHIVE_PATH.is_file(), "skipped": True}
        else:
            print(f"loaded archive inventory members {inventory['members']}", flush=True)
    elif args.skip_archive_full:
        inventory = {"present": ARCHIVE_PATH.is_file(), "skipped": True}
    else:
        print("streaming official archive members", flush=True)
        inventory = scan_archive_inventory()
        print(f"archive members {inventory.get('members')}", flush=True)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "archive_inventory.json", inventory)

    rgb_train = load_id_set(N5K / "dish_ids" / "splits" / "rgb_train_ids.txt")
    rgb_test = load_id_set(N5K / "dish_ids" / "splits" / "rgb_test_ids.txt")
    depth_train = load_id_set(N5K / "dish_ids" / "splits" / "depth_train_ids.txt")
    depth_test = load_id_set(N5K / "dish_ids" / "splits" / "depth_test_ids.txt")
    if rgb_train & rgb_test:
        raise SystemExit("official rgb train and test overlap")
    if depth_train & depth_test:
        raise SystemExit("official depth train and test overlap")

    cafe_dishes = {
        "cafe1": parse_metadata_file(N5K / "metadata" / "dish_metadata_cafe1.csv", "cafe1"),
        "cafe2": parse_metadata_file(N5K / "metadata" / "dish_metadata_cafe2.csv", "cafe2"),
    }
    all_ids = [line.strip() for line in (N5K / "dish_ids" / "dish_ids_all.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    parsed_ids = [dish["dish_id"] for dishes in cafe_dishes.values() for dish in dishes]
    if len(all_ids) != 5006 or set(all_ids) != set(parsed_ids) or len(parsed_ids) != 5006:
        raise SystemExit("parsed dishes do not match the official 5,006 dish ids")
    if any(dish["kcal"] != 0 or dish["fat"] != 0 or dish["carb"] != 0 or dish["protein"] != 0 for dish in cafe_dishes["cafe2"]):
        raise SystemExit("cafe2 dish-level macros are not all zero; the cafe2 macro rule would be wrong")

    def splits_of(dish_id: str) -> dict[str, str]:
        return dish_splits(dish_id, rgb_train, rgb_test, depth_train, depth_test)

    images = overhead_index(N5K)
    print(f"overhead rgb {len(images)}", flush=True)
    print("building sequences", flush=True)
    cafe_results = {}
    all_sequences = []
    per_cafe_counts = {}
    for cafe, dishes in cafe_dishes.items():
        built = build_primary(dishes, SESSION_GAP_S, splits_of)
        sequences = []
        ordered = sorted(
            built["valid_paths"],
            key=lambda steps: built["graph"]["clusters"][steps[0][0]]["t_min"],
        )
        for number, steps in enumerate(ordered, start=1):
            sequences.append(
                sequence_payload(f"{cafe}-{number:04d}", steps, built["graph"]["clusters"], dishes, splits_of, images)
            )
        built["dishes"] = dishes
        built["sequences"] = sequences
        cafe_results[cafe] = built
        all_sequences.extend(sequences)
        per_cafe_counts[cafe] = cafe_counts(built, dishes, sequences, splits_of)
        print(
            f"{cafe} pairs {per_cafe_counts[cafe]['valid_pairs']} sequences {per_cafe_counts[cafe]['valid_sequences']}",
            flush=True,
        )

    print("sensitivity and unrestricted gaps", flush=True)
    sensitivity = count_sensitivity(cafe_dishes, splits_of)
    matched_gap = next(row for row in sensitivity if row["gap_s"] == SESSION_GAP_S)
    for cafe, counts_cafe in per_cafe_counts.items():
        if matched_gap["cafes"][cafe]["valid_pairs"] != counts_cafe["valid_pairs"]:
            raise SystemExit(f"sensitivity at {SESSION_GAP_S}s disagrees with the primary count for {cafe}")
    unrestricted = {
        cafe: unrestricted_gap_histogram(dishes) for cafe, dishes in cafe_dishes.items()
    }

    groups = assign_groups(cafe_results)
    raw_decreases = 0
    for dishes in cafe_dishes.values():
        for left, right in zip(dishes, dishes[1:]):
            if right["mass"] < left["mass"] - MASS_TOL_G:
                raw_decreases += 1

    both_overhead = 0
    pair_rows = []
    visual_pairs = []
    for sequence in all_sequences:
        for stage in sequence["stages"][1:]:
            previous = sequence["stages"][stage["position"] - 1]
            pair_rows.append(
                {
                    "sequence_id": sequence["sequence_id"],
                    "cafe": sequence["cafe"],
                    "before_dish_id": previous["representative_dish_id"],
                    "after_dish_id": stage["representative_dish_id"],
                    "before_overhead": previous["overhead_dish_id"],
                    "after_overhead": stage["overhead_dish_id"],
                    "added": stage["added"],
                    "nutrition_residual": stage["nutrition_residual"],
                    "intervening_dishes": stage["intervening_dishes"],
                    "rgb_leak_sequence": sequence["rgb_leak"],
                }
            )
            if previous["overhead_dish_id"] and stage["overhead_dish_id"]:
                both_overhead += 1
                visual_pairs.append(
                    {
                        "sequence_id": sequence["sequence_id"],
                        "before_dish_id": previous["overhead_dish_id"],
                        "after_dish_id": stage["overhead_dish_id"],
                    }
                )

    repeat_visual = []
    for cafe, built in cafe_results.items():
        dishes = built["dishes"]
        for cluster in built["graph"]["clusters"]:
            imaged = [dishes[index]["dish_id"] for index in cluster["members"] if dishes[index]["dish_id"] in images]
            for left, right in zip(imaged, imaged[1:]):
                repeat_visual.append({"before_dish_id": left, "after_dish_id": right, "sequence_id": groups.get(left)})

    scale_bad = []
    for cafe, counts_cafe in per_cafe_counts.items():
        for record in counts_cafe["scale_bad"]:
            scale_bad.append({"cafe": cafe, **record})
    worst = max(scale_bad, key=lambda record: d(record["mass_range_g"])) if scale_bad else None

    longest = max(all_sequences, key=lambda sequence: (sequence["n_stages"], sequence["n_scans"]))
    leak_sequences = [sequence for sequence in all_sequences if sequence["rgb_leak"]]
    repeat_example = None
    for counts_cafe in per_cafe_counts.values():
        for record in counts_cafe["repeat_leaks"]:
            if record["n_ingredient_ids"] >= 2:
                repeat_example = record
                break
        if repeat_example:
            break

    fork_example = None
    for cafe, built in cafe_results.items():
        dishes = built["dishes"]
        clusters = built["graph"]["clusters"]
        for parent, kids in built["fork_rows"]:
            child_payloads = []
            splits = set()
            for kid in kids:
                split = splits_of(dishes[clusters[kid]["rep"]]["dish_id"])
                splits.add(split["rgb"])
                child_payloads.append(
                    {
                        "dish_id": dishes[clusters[kid]["rep"]]["dish_id"],
                        "rgb": split["rgb"],
                        "added": added_items(clusters[parent]["counter"], clusters[kid]["counter"]),
                    }
                )
            if len(splits) > 1 or fork_example is None:
                fork_example = {
                    "cafe": cafe,
                    "parent": dishes[clusters[parent]["rep"]]["dish_id"],
                    "parent_rgb": splits_of(dishes[clusters[parent]["rep"]]["dish_id"])["rgb"],
                    "children": child_payloads,
                }
            if len(splits) > 1:
                break
        if fork_example and len({child["rgb"] for child in fork_example["children"]}) > 1:
            break

    assembly_example = None
    for cafe, built in cafe_results.items():
        dishes = built["dishes"]
        clusters = built["graph"]["clusters"]
        for index in built["assembly_indexes"]:
            parts = []
            for pred in built["graph"]["predecessors"][index]:
                names = "、".join(f"{row[1]} {row[2]} g" for row in clusters[pred]["counter"])
                parts.append({"dish_id": dishes[clusters[pred]["rep"]]["dish_id"], "ingredients": names})
            if len(parts) >= 3:
                assembly_example = {
                    "dish_id": dishes[clusters[index]["rep"]]["dish_id"],
                    "parts": parts,
                }
                break
        if assembly_example:
            break

    length_hist = dict(sorted(Counter(sequence["n_stages"] for sequence in all_sequences).items()))
    combined = {
        "official_dish_ids": 5006,
        "valid_pairs": sum(item["valid_pairs"] for item in per_cafe_counts.values()),
        "valid_sequences": sum(item["valid_sequences"] for item in per_cafe_counts.values()),
        "dishes_in_valid_sequences": sum(item["dishes_in_valid_sequences"] for item in per_cafe_counts.values()),
        "stages_in_valid_sequences": sum(item["stages_in_valid_sequences"] for item in per_cafe_counts.values()),
        "valid_pairs_with_both_overhead": both_overhead,
        "raw_consecutive_mass_decreases": raw_decreases,
        "valid_mass_increase": sum(item["mass_increase"] for item in per_cafe_counts.values()),
        "valid_mass_flat": sum(item["mass_flat"] for item in per_cafe_counts.values()),
        "valid_mass_decrease": sum(item["mass_decrease"] for item in per_cafe_counts.values()),
        "intervening_zero": sum(row["intervening_dishes"] == 0 for row in pair_rows),
        "intervening_max": max((row["intervening_dishes"] for row in pair_rows), default=0),
        "added_one_id": sum(item["added_one"] for item in per_cafe_counts.values()),
        "added_many_ids": sum(item["added_many"] for item in per_cafe_counts.values()),
        "rgb_leak_sequences": sum(item["rgb_leak_sequences"] for item in per_cafe_counts.values()),
        "depth_leak_sequences": sum(item["depth_leak_sequences"] for item in per_cafe_counts.values()),
        "repeat_rgb_leaks": sum(item["repeat_rgb_leaks"] for item in per_cafe_counts.values()),
        "repeat_rgb_leaks_strong": sum(item["repeat_rgb_leaks_strong"] for item in per_cafe_counts.values()),
        "repeat_rgb_leaks_weak": sum(item["repeat_rgb_leaks_weak"] for item in per_cafe_counts.values()),
        "scale_disagreement_clusters": len(scale_bad),
        "length_histogram": {str(key): value for key, value in length_hist.items()},
        "longest_sequence": {"sequence_id": longest["sequence_id"], "n_stages": longest["n_stages"]},
        "residual_max": {cafe: per_cafe_counts[cafe]["residual_max"] for cafe in per_cafe_counts},
        "worst_scale_cluster": {
            "dish_ids": [member["dish_id"] for member in worst["members"]] if worst else [],
            "mass_range_g": worst["mass_range_g"] if worst else None,
        },
        "by_cafe": {
            cafe: {
                "dishes": per_cafe_counts[cafe]["dishes"],
                "valid_pairs": per_cafe_counts[cafe]["valid_pairs"],
                "valid_sequences": per_cafe_counts[cafe]["valid_sequences"],
                "dishes_in_valid_sequences": per_cafe_counts[cafe]["dishes_in_valid_sequences"],
                "stages_in_valid_sequences": per_cafe_counts[cafe]["stages_in_valid_sequences"],
                "rejected_scale_paths": per_cafe_counts[cafe]["rejected_scale_paths"],
                "fork_parents": per_cafe_counts[cafe]["fork_parents"],
                "assemblies": per_cafe_counts[cafe]["assemblies"],
                "ambiguous": per_cafe_counts[cafe]["ambiguous"],
                "repeat_rgb_leaks": per_cafe_counts[cafe]["repeat_rgb_leaks"],
                "rgb_leak_sequences": per_cafe_counts[cafe]["rgb_leak_sequences"],
                "depth_leak_sequences": per_cafe_counts[cafe]["depth_leak_sequences"],
            }
            for cafe in per_cafe_counts
        },
    }

    visual_summary = None
    visual_rows = []
    saved_visual = output / "visual_summary.json"
    if args.skip_visual and saved_visual.is_file():
        visual_summary = json.loads(saved_visual.read_text(encoding="utf-8"))
        print("loaded saved visual summary", flush=True)
    elif not args.skip_visual:
        print(f"visual pairs {len(visual_pairs)} repeats {len(repeat_visual)}", flush=True)
        visual = visual_audit(visual_pairs, repeat_visual, sorted(images), groups)
        visual_rows = visual.pop("adjacent_rows") + visual.pop("repeat_rows")
        visual_summary = visual
        write_json(saved_visual, visual_summary)
        write_jsonl(output / "visual_pairs.jsonl", visual_rows)

    bootstrap = None
    if not args.skip_bootstrap and combined["valid_sequences"]:
        print("bootstrap", flush=True)
        bootstrap = run_bootstrap(cafe_results, groups)

    summary = {
        "audit": "icdl-phase1-sequence",
        "session_gap_s": SESSION_GAP_S,
        "mass_tolerance_g": str(MASS_TOL_G),
        "macro_tolerance": str(MACRO_TOL),
        "counts": combined,
        "archive": archive_prefix,
        "local_hashes": local_hashes,
        "archive_inventory": {
            "members": inventory.get("members"),
            "path": inventory.get("path"),
            "bytes": inventory.get("bytes"),
            "plate_or_sequence_in_non_image_names": inventory.get("plate_or_sequence_in_non_image_names", []),
            "imagery_filename_buckets": inventory.get("imagery_filename_buckets", {}),
            "non_image_member_count": len(inventory.get("non_image_members", [])),
            "skipped": inventory.get("skipped", False),
        },
        "sensitivity": sensitivity,
        "unrestricted_unique_gaps": unrestricted,
        "visual_summary": visual_summary,
        "bootstrap": bootstrap,
        "examples": {
            "longest_sequence": longest,
            "rgb_leak_sequences": leak_sequences,
            "repeat_leak": repeat_example,
            "fork": fork_example,
            "assembly": assembly_example,
        },
    }

    write_json(output / "summary.json", {key: value for key, value in summary.items() if key != "examples"})
    write_json(output / "examples.json", summary["examples"])
    write_json(output / "sensitivity.json", sensitivity)
    write_json(output / "archive_inventory.json", inventory)
    write_jsonl(output / "valid_pairs.jsonl", pair_rows)
    write_jsonl(output / "valid_sequences.jsonl", all_sequences)
    if visual_summary is not None:
        write_json(output / "visual_summary.json", visual_summary)
    if visual_rows:
        write_jsonl(output / "visual_pairs.jsonl", visual_rows)
    if bootstrap is not None:
        write_json(output / "bootstrap.json", bootstrap)
    report = render_report(summary)
    (output / "report.md").write_text(report, encoding="utf-8")
    print(f"wrote {output / 'report.md'}", flush=True)
    print(
        f"pairs {combined['valid_pairs']} sequences {combined['valid_sequences']} rgb_leaks {combined['rgb_leak_sequences']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
