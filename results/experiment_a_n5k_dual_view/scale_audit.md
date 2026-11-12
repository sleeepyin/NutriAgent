# 实验 A 规模核对

这次核对只回答一个问题：官方 Nutrition5k 归档里，有多少盘同时有顶视 RGB 和侧视 RGB，并且能按 dish_id 对齐。
侧视不是度量双相机。归档没有公布侧视位姿，深度不进入可用配对规则。

## 用了哪些官方文件

- 归档：`/var/tmp/nutri-downloads/nutrition5k/nutrition5k_dataset.tar.gz`
- 归档大小：194804050002 字节
- 盘号清单：归档内 `nutrition5k_dataset/dish_ids/dish_ids_all.txt`（sha256 `cd71f64b9092d7917384b477e62fbe8840409e606dc488b436f05aa2b404c2f5`）
- 质量标签：归档内 `metadata/dish_metadata_cafe1.csv` 与 `metadata/dish_metadata_cafe2.csv`
- 顶视 RGB：归档内 `imagery/realsense_overhead/<dish_id>/rgb.png`
- 侧视 RGB：归档内 `imagery/side_angles/<dish_id>/camera_{A,B,C,D}.h264`，再解出一帧
- 官方 RGB 划分只用于后面的对照实验，不用于定义配对规模：`splits/rgb_train_ids.txt`、`splits/rgb_test_ids.txt`

本地 `data/nutrition5k/` 里的清单和元数据与归档逐字节比对。顶视 png 若与归档 sha256 一致，就沿用本地文件；不一致或不存在时，从归档抽出一份放到本实验结果目录，不覆盖本地原文件。

## 纳入规则

1. dish_id 在官方 `dish_ids_all.txt` 中。官方盘数是 5006。
2. 顶视 `rgb.png` 存在，且大小至少 10000 字节。
3. 至少一路侧视 `camera_A/B/C/D.h264` 大小至少 100000 字节。摄像头按 A、B、C、D 的顺序取第一路达标的，不按画面质量挑选。
4. 该侧视视频能解出至少一帧。保留解码序列的中间帧；超过 300 帧就在上限处停下，保留上限内的中间帧。原始码流没有可用时间戳。
5. 官方元数据里的总质量是有限数值，并且严格大于 0 克。

深度文件只在附录里计数。GitHub issue #12 的约 3490 不是输入，也不是本计数的目标。论文里约 3.5k / 5,006 盘含顶视 RGB-D，是论文约数，不是俯视 RGB 与侧视 RGB 的可用配对数。

## 计数

- 官方盘号：5006
- 元数据里有质量字段的盘：5006
- 质量大于 0 克：5006
- 达标顶视 RGB（官方盘号内）：3490
- 达标侧视视频（官方盘号内，至少一路摄像头）：4793
- 归档文件级同时具备两路、且质量大于 0（尚未要求解码成功）：3289
- 侧视解码失败：0
- **可用配对（两路 RGB 对齐、解码成功、质量大于 0）：3289**

可用配对再按官方 RGB 划分切开，供五项对照使用：

- 官方 rgb_train：2638
- 官方 rgb_test：474
- 不在官方 RGB 划分里（不进入五项对照，也不拿来调方法）：177

互斥的未纳入原因（只统计官方盘号，每盘一个主原因）：

- overhead_rgb_missing：1516
- side_video_missing：201

侧视摄像头选择（可用配对）：

- camera_C：10
- camera_A：2863
- camera_B：416

## 附录：深度（不进入主表）

可用 RGB 配对里，同时有达标深度文件的盘：3289。
其中 `depth_raw.png` ≥ 1000 字节：3288。
`depth_color.png` ≥ 1000 字节：3289。
这些深度没有读进实验，也没有参与配对定义。

## 本地顶视文件核对

- 本地 `realsense_overhead/*/rgb.png`：3490
- 与归档 sha256 一致并被沿用：3289
- 因缺失或哈希不一致而从归档另存：0

官方文本文件与本地副本是否逐字节一致：

- metadata/dish_metadata_cafe1.csv：一致
- metadata/dish_metadata_cafe2.csv：一致
- rgb_train_ids.txt：一致
- rgb_test_ids.txt：一致
- dish_ids_all.txt：一致

非官方盘号（出现在影像路径里，但不在 `dish_ids_all.txt`）不计入可用配对：
- 仅顶视：0
- 仅侧视：0
