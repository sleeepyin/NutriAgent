# Toy FNDDS fragment

Fictional FNDDS-like foods for unit/smoke tests, including a raw-onion vs dip/soup pair and a raw-blueberry vs juice pair. **Not** USDA data.

## Real FNDDS replacement path

1. Download USDA FNDDS 2019–2020 from the official ARS page (see `docs/data_acquisition.md`).
2. Unpack tables under `data/fndds/2019-2020/`.
3. Set env `VDR_FNDDS_PATH` to that directory (or a converted `foods.json` compatible with this schema), **or** pass `catalog_path=` to `vdr.tools.fndds_catalog.load_catalog`.
4. Until then, tools default to `data/fndds/toy/foods.json`.
