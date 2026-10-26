# 数据获取清单

## 状态（box）

| 资源 | 状态 | 路径 |
|------|------|------|
| N5K 元数据 + 官方 split | **已就绪** | `data/nutrition5k/{dish_ids,metadata}/` |
| N5K 全量 imagery | **未下载**（体积大；需大盘/4090D 机器） | 预期 `data/nutrition5k/imagery/` |
| FNDDS 原始表 / FDC Survey JSON | **已就绪** | `data/fndds/2019-2020/` |
| VDR catalog `foods.json` | **已构建** | `data/fndds/2019-2020/foods.json`（默认加载） |
| Toy 回退 catalog | 保留 | `data/fndds/toy/foods.json` |

重建 catalog：

```bash
python scripts/build_fndds_catalog.py
# 覆盖路径：
# export VDR_FNDDS_PATH=/path/to/foods.json
```

冒烟：

```bash
python scripts/fndds_retrieve_smoke.py
python scripts/n5k_metadata_summary.py
```

## 1. Nutrition5K

| 项 | 说明 |
|----|------|
| 官方来源 | https://github.com/google-research-datasets/Nutrition5k |
| Split（本仓库已有） | `dish_ids/splits/rgb_{train,test}_ids.txt` 等 |
| 元数据（已有） | `metadata/dish_metadata_cafe{1,2}.csv`、`ingredients_metadata.csv` |
| 全量 RGB/Depth | **请在大盘机器下载**；本 box 不拉全量图 |
| 许可 | 遵守官方 license；人工子集默认只发 dish_id+标签 |

官方 RGB split 计数（摘要脚本）：train **4059** / test **709**（见 `scripts/n5k_metadata_summary.py`）。

## 2. FNDDS / FDC Survey Foods

| 项 | 说明 |
|----|------|
| 官方入口 | USDA FNDDS Download Databases；FDC Survey Food JSON |
| 本仓库文件 | `FoodData_Central_survey_food_json_2022-10-28.json`、`FNDDS_Nutrient_Values.xlsx`、`Foods_and_Beverages.xlsx`、`Portions_and_Weights.xlsx` |
| Catalog | `scripts/build_fndds_catalog.py` → `foods.json`（与 toy schema 兼容） |
| 加载优先级 | 参数 / `VDR_FNDDS_PATH` → `2019-2020/foods.json` → `toy/foods.json` |
| 许可 | 遵守 USDA / FNDDS / FDC 条款 |

## 3. 目录约定

```
data/
  toy/                       # 虚构冒烟 dish
  nutrition5k/
    dish_ids/                # 已就绪
    metadata/                # 已就绪
    imagery/                 # 可选；全量未下
  fndds/
    toy/foods.json           # 回退
    2019-2020/foods.json     # 真实 catalog（默认）
```

## 4. 禁止项

- 不新建私有大规模采集作为主基准。
- 不把测试集 GT 用于步骤弱标签。
- 不加载旧营养回归权重。
- 不声称全量 N5K 图像已在 box 就绪。
