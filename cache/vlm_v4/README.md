v4.0 的看图缓存。三个有条件的数据集和 NV-Real 用同一份提示词，一次调用同时要毫米/像素、参照物框、克/毫升和每 100 克含量。按「数据集 + 样本 + 原图路径 + 提示词哈希」存放。

NV-Real 的缓存可以先写，但区间打分要等 ECUSTFD、MetaFood3D、Nutrition5K 的 λ 写死。旧提示词的 JSON 不从 `cache/vlm_v3/` 或 `backup/v3_excluded/cache/vlm/` 读入。
