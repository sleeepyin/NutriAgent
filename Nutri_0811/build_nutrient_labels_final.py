#!/usr/bin/env python3
"""
Build nutrient labels for the deep crops dataset.
Maps each crop in deep_crops_cache.npz to its dish_id, then looks up
nutrient values (calories, fat, carb, protein) from dish_metadata CSVs.

Output: 
  nutrient_labels.npz — absolute nutrients (calories, fat_g, carb_g, protein_g)
  nutrient_density.npz — per-gram density (cal/g, fat/g, carb/g, protein/g)
"""
import csv, os, sys
import numpy as np

# Use sklearn from minimax env for exact split match
sys.path.insert(0, '/opt/miniconda3/envs/minimax/lib/python3.10/site-packages')
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
BASE_DIR = '/home/nutrition5k/yolo11-seg/final_729'
NUTRI_DIR = '/home/nutrition5k/nutri_dataset/nutrition5k_dataset/metadata'
OUT_DIR = '/home/Nutri_0811/models'

# ==================== Load dish_weights_full.csv ====================
print("[1/5] Loading dish weights...")
dish_ids_ordered = []
dish_weights = {}
with open(os.path.join(BASE_DIR, 'dish_weights_full.csv'), 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        did = row[0]
        w = float(row[1])
        if w > 0:
            dish_ids_ordered.append(did)
            dish_weights[did] = w

print(f"  Valid dishes: {len(dish_ids_ordered)}")

# ==================== Load dish_metadata ====================
print("[2/5] Loading nutrient metadata...")

def parse_dish_metadata(filepath):
    dishes = {}
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            dish_id = row[0]
            total_mass = float(row[1])
            if total_mass <= 0:
                continue
            dishes[dish_id] = {
                'total_mass_g': total_mass,
                'calories': float(row[2]),
                'fat_g': float(row[3]),
                'carb_g': float(row[4]),
                'protein_g': float(row[5]),
                'cal_g': float(row[2]) / total_mass,
                'fat_density': float(row[3]) / total_mass,
                'carb_density': float(row[4]) / total_mass,
                'protein_density': float(row[5]) / total_mass,
            }
    return dishes

nutri_meta = {}
nutri_meta.update(parse_dish_metadata(os.path.join(NUTRI_DIR, 'dish_metadata_cafe1.csv')))
nutri_meta.update(parse_dish_metadata(os.path.join(NUTRI_DIR, 'dish_metadata_cafe2.csv')))
print(f"  Dishes with nutrient data: {len(nutri_meta)}")

# ==================== Recreate train/test split ====================
print("[3/5] Recreating train/test split...")
unique_dishes = np.unique(dish_ids_ordered)
train_dishes, test_dishes = train_test_split(unique_dishes, test_size=0.2, random_state=SEED)
train_dishes = set(train_dishes)
test_dishes = set(test_dishes)

train_idx = [i for i, d in enumerate(dish_ids_ordered) if d in train_dishes]
test_idx = [i for i, d in enumerate(dish_ids_ordered) if d in test_dishes]
print(f"  Train dishes: {len(train_idx)}, Test dishes: {len(test_idx)}")

# ==================== Map crops to nutrients ====================
print("[4/5] Mapping crops to nutrients...")

data = np.load(os.path.join(BASE_DIR, 'deep_crops_cache.npz'))
print(f"  X_train: {data['X_train'].shape}, y_train: {data['y_train'].shape}")
print(f"  X_test: {data['X_test'].shape}, y_test: {data['y_test'].shape}")

def build_labels(indices, density=False):
    labels = []
    missing = 0
    for idx in indices:
        dish_id = dish_ids_ordered[idx]
        if dish_id in nutri_meta:
            n = nutri_meta[dish_id]
            if density:
                labels.append([n['cal_g'], n['fat_density'], n['carb_density'], n['protein_density']])
            else:
                labels.append([n['calories'], n['fat_g'], n['carb_g'], n['protein_g']])
        else:
            missing += 1
            labels.append([0.0, 0.0, 0.0, 0.0])
    if missing > 0:
        print(f"  WARNING: {missing} dishes missing nutrient data (filled with 0)")
    return np.array(labels, dtype=np.float32)

# Absolute nutrients
y_train_nutr = build_labels(train_idx)
y_test_nutr = build_labels(test_idx)

# Nutrient density
y_train_density = build_labels(train_idx, density=True)
y_test_density = build_labels(test_idx, density=True)

# ==================== Save ====================
print("[5/5] Saving...")
os.makedirs(OUT_DIR, exist_ok=True)
np.savez(os.path.join(OUT_DIR, 'nutrient_labels.npz'),
         y_train_nutr=y_train_nutr, y_test_nutr=y_test_nutr)
np.savez(os.path.join(OUT_DIR, 'nutrient_density.npz'),
         y_train_density=y_train_density, y_test_density=y_test_density)
print(f"  Saved to {OUT_DIR}/nutrient_labels.npz and nutrient_density.npz")

# ==================== Summary stats ====================
print("\n" + "=" * 60)
print("Nutrient Density Statistics")
print("=" * 60)
col_names = ['Cal/g', 'Fat/g', 'Carb/g', 'Protein/g']
for i, name in enumerate(col_names):
    tr = y_train_density[:, i]
    te = y_test_density[:, i]
    print(f"\n{name}:")
    print(f"  Train: min={tr.min():.4f}, max={tr.max():.3f}, mean={tr.mean():.3f}, std={tr.std():.3f}")
    print(f"  Test:  min={te.min():.4f}, max={te.max():.3f}, mean={te.mean():.3f}, std={te.std():.3f}")

print("\nDone!")