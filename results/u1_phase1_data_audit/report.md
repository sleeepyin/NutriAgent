# 自适应营养估计粒度：第一阶段数据审计

「自适应营养估计粒度」的第一阶段审计只回答三件事：标注能否按 dish_id 对齐，有没有一个事先写死且测试时不泄漏真值的视觉可分离度，以及留出样本够不够做盘级不确定区间。对齐可靠。主指标 `local_color_purity` 已写死，测试时只读一张俯视 RGB。公开测试划分有 158 个 dish_id；收成物理餐盘并去掉跨划分的盘子之后，干净留出是 64 只，下限是 80 只。因此审计停止，等待人工判断。本报告不修改 docs/STATUS.md，不宣布第一阶段通过，也不解锁第二阶段。

## 1. 审计范围

这次只做数据审计。脚本读 Hand 等人公开的两张营养表，以及本机已经解出的 Nutrition5k 俯视 RGB。推理协议仍然是单张俯视 RGB。没有使用深度、第二视角、Agent、检索或强化学习。没有训练网络，没有写第二阶段模型，没有把 505 盘子集或 Cafe2 当作验证集或调参集，也没有把 ICDL 的 1400 对相邻阶段组成差当作监督或起点。公开 CSV 只读，脚本结束时重新计算了 git blob 摘要，确认字节没有被改写。

盘级表 blob `32ae4b8cd7c625856b5f5cf50c10392f6df0dfd3`，食材表 blob `2cd1176104461379b04436f68b6d03f75477021a`。两份文件来自标注仓库 `liamhand/ingredient-level-nutrition-regression` 默认分支，许可是 CC BY 4.0。原始 RGB 用的是本机 `data/nutrition5k/imagery/realsense_overhead/<dish_id>/rgb.png`，没有从标注仓库转存图像。标注表里的 `rgb_image` 列只是 `rgb_images/<dish_id>.png` 这个文件名，审计不打开它。

## 2. 公开表复核

STATUS 里 2026-11-13 的字段记录，在本机文件上逐项复核如下。

| 项目 | 复核结果 | 与字段记录一致 |
|---|---:|:---:|
| 盘级行数 | 1056 | 是 |
| 盘级 train / test | 898 / 158 | 是 |
| 食材实例行数 | 2922 | 是 |
| 食材实例 train / test | 2253 / 669 | 是 |
| 食材表覆盖的盘，train / test | 898 / 158 | 是 |
| `annotation_name` 类数 | 40 | 是 |
| `has_nutrition` 为 0 / 1 | 98 / 2824 | 是 |

两表的 `dish_id` 集合相同。同一 `dish_id` 的 `split` 冲突有 0 个。不符合 `dish_<数字>` 的标识有 0 个。盘级表里重复的 `dish_id` 有 0 个。`rgb_image` 与 `rgb_images/<dish_id>.png` 不一致的实例有 0 个。40 个类别的名称和频数写在 `annotation_classes.csv`。

## 3. 与原始俯视 RGB、官方划分的交集

本机俯视目录有 3490 个盘子目录，每个目录一张 `rgb.png`。Hand 的 1056 个 `dish_id` 里，图像文件存在的是 1056 个，缺失 0 个，能打开并转成 RGB 的是 1056 个。图像尺寸：640x480 有 1056 张。Cafe2 标识与 Hand 的交集是 0 个，因此这 1056 盘全部来自 Cafe1 的标识范围。

官方 RGB 划分用的是数据集发布里的 `dish_ids/splits/rgb_train_ids.txt` 与 `rgb_test_ids.txt`。Nutrition5k 的 GitHub 代码仓库只放说明和脚本，不附带这些标识表；标识表在数据发布包里，本机 `dish_ids/README` 写明论文模型只用 Cafe1。复核：官方 train 4059 条，test 709 条，两表交集 0 条，并集与 `dish_ids_cafe1.txt` 相同 （Cafe1 4768 条）。深度划分是 RGB 划分的子集：depth train 有 0 条不在 RGB 划分里，depth test 有 0 条不在 RGB 划分里。

| Hand 划分 | 官方 RGB train | 官方 RGB test | 不在官方 RGB 划分 |
|---|---:|---:|---:|
| train | 487 | 411 | 0 |
| test | 83 | 75 | 0 |

Hand 内部，同一个 `dish_id` 同时出现在 train 和 test 的有 0 个。官方 RGB 划分内部，同一个 `dish_id` 同时出现在两边的有 0 个。若把两套划分混用，Hand 测试盘里有 83 盘落在官方训练集，Hand 训练盘里有 411 盘落在官方测试集。本审计沿用 Hand 的 `split`，不另划测试集。上面的交叉只说明两套划分不是同一刀，不能拿来调阈值。

## 4. 盘级与食材级是否一致，以及缺失营养

`has_nutrition` 为 0 的实例是 98 个，占全部 2922 个实例的 3.35%。这些格子在质量、能量和三种宏量上是空字符串，不是数值 0。按盘计，至少有一个实例缺营养的盘子有 86 个，全部实例都有营养的盘子有 970 个。训练盘里缺营养的盘子 62 个，测试盘里 24 个。

一致性规则事先写死：只把 `has_nutrition=1` 的实例相加，与盘级表比较。某个营养素算对齐，当绝对差不超过 0.5，或不超过较大绝对值的 1%，二者取更宽的那个。五个营养素同时对齐的完整盘有 546 / 970。部分标注盘里五个营养素同时对齐的有 18 / 86。部分标注盘通常对不齐，因为缺营养的实例没有进入求和，盘级总量仍包含它们。

部分标注盘上，盘级质量减去已标注实例质量和，中位数是 19.72 克。盘子比已标注实例重出容差的有 68 盘，已标注实例比盘子重出容差的有 0 盘。

全部实例都有营养的 970 盘里，五个营养素同时对齐的是 546 盘，对不齐的是 424 盘。这 424 盘里，盘子比已标注实例更重的有 418 盘，已标注实例更重的有 1 盘。质量差（盘级减求和）的中位数是 17.73 克，相对盘级质量的绝对差中位数是 9.7%。对不齐的次数：能量 422，脂肪 383，质量 361，碳水 284，蛋白质 174。所以，即使每一行标注都有数字，食材求和仍然经常小于盘级总量。这个剩余对应没有画进掩膜的成分，或掩膜没有覆盖整盘食物。Hand 盘级质量相对 Nutrition5k 元数据的绝对差，中位数 0.0000 克，最大 0.0000 克，落在 0.5 克以内的有 1056 / 1056 盘。盘级表与原始元数据是同一套总量。

## 5. 重拍、物理餐盘，以及留出区间

Nutrition5k 的采集是增量扫描：同一只物理餐盘每加一次菜就再扫一次，每次扫描有自己的 `dish_id`。论文写明，同一只盘子的各次扫描必须留在同一个官方划分里，避免训练和测试看到同一只盘子。STATUS 不预设重拍是否存在。本审计用元数据把这个结构算出来，没有读取 ICDL 的成对列表。

合并规则：把 Cafe1 与 Cafe2 的 5006 条元数据按时间戳排序。相邻两条属于同一只盘子，当且仅当时间差不超过 180 秒，质量至少增加 1 克，后一条的食材名称多重集包含前一条，并且同名食材的克数下降不超过 1 克。再做传递闭包。180 秒写在「食材嵌套且质量增加」的相邻时间差之上：这样的相邻对有 2544 对，中位 33 秒，99 分位 157 秒。其余相邻对的时间差长得多。阈值在看到 Hand 留出泄漏数字之前按这个分布固定，之后没有改。

全库得到 2482 只物理餐盘，其中多次扫描的有 1142 只，覆盖 3666 条 `dish_id`。链接 2524 条。这套合并跨过官方 RGB train/test 的盘子有 2 只。官方划分本来就要把同一只盘子放在一边，这个很小的交叉用来检查合并没有把无关盘子粘在一起。交叉盘子的标识写在 `official_cross_plates.csv`。这 2 只盘子沿时间质量上升：dish_1559844458：dish_1559844458（test，81 克，1 种食材）、dish_1559844479（train，133 克，2 种食材）、dish_1559844490（test，180 克，2 种食材）；dish_1561662814：dish_1561662814（test，20 克，9 种食材）、dish_1561662842（test，172 克，17 种食材）、dish_1561662869（test，192 克，18 种食材）、dish_1561662926（test，213 克，18 种食材）、dish_1561662948（train，230 克，19 种食材）。它们更像官方划分没有完全守住的增量扫描。

Hand 的 1056 条 `dish_id` 落在 663 只物理餐盘上。其中 208 只盘子有多于一条 Hand 标注，共 601 条。测试划分的 158 条 `dish_id` 落在 137 只盘子上，训练划分落在 599 只盘子上。同时含有 Hand 训练扫描和 Hand 测试扫描的盘子有 73 只。只出现在 Hand 测试、没有 Hand 训练扫描的干净盘子有 64 只，上面共有 67 条测试 `dish_id`。字节级完全相同的俯视图像，在 Hand 子集里有 0 组。时间差不超过 180 秒且 32×24 灰度相关系数不低于 0.995 的 Hand 图像对有 1 对。增量扫描会改变盘面，近重复图像很少。

同一规则在另外两个时间差下的灵敏度：

| 时间差上限（秒） | 全库盘子数 | 跨官方划分的盘子 | Hand 测试盘子 | 跨 Hand 划分的盘子 | 干净测试盘子 |
|---:|---:|---:|---:|---:|---:|
| 90 | 2529 | 2 | 137 | 71 | 66 |
| 180 | 2482 | 2 | 137 | 73 | 64 |
| 600 | 2468 | 3 | 137 | 76 | 61 |

配对单位应是合并重拍之后的物理餐盘，而不是单独的 Hand `dish_id`。同一只盘子的多次扫描不是独立样本。Hand 的图像级划分没有遵守 Nutrition5k 自己的约束，有 73 只盘子一边在训练、一边在测试。本审计不因此重划测试集。

158 这个公开测试盘数，本报告不把它预先判成通过或不足。把它收成物理餐盘之后，干净留出是 64 只。事先写下的下限是 80 只独立餐盘。Hand 等人在 158 盘上发表的食材级减整盘平均 PMAE 差值，95% 区间是 [1.51, 5.98]，半宽 2.24 个百分点。若残差尺度不变，按干净餐盘数外推的半宽大约是 3.51 个百分点。这只说明区间会更宽，不是一次新的模型比较。干净餐盘数低于下限，留出样本不够做带不确定区间的盘级对照。

## 6. 预先登记的视觉可分离度

主指标名称是局部色纯净度，字段名 `local_color_purity`，版本 `2026-11-13-phase1`。它在任何营养误差对照之前写进脚本常量。本脚本没有营养预测器，也没有在算出 Spearman 相关之后改这些常数。

计算只使用当时可见的一张俯视 RGB。长边缩放到 160 像素，转到 CIE Lab。外圈 8% 像素的 Lab 中位数当作背景。亮度不低于 72、彩度不超过 16、与背景的 CIE76 色差不低于 12 的像素视为浅色盘面，取最大 8 连通域并填洞。食物区域是这只盘子盘面围住的孔。若孔太小，就退回填洞后的整只盘子，用来容纳米饭这类浅色食物。若找不到浅色盘子，就退回与背景色差不低于 18 的最大连通域。在食物区域上对 Lab 做半径 2 的盒式模糊，用最远点初始化、固定 3 个颜色簇、20 轮聚类。每个食物像素的局部纯净度，是它周围 9×9 窗口里与它同簇的食物像素所占比例。主指标是这些比例的平均。越高表示颜色越成块，视觉上越分得开。食材真值个数、真值掩膜、食物码和营养标签都不进入这个分数。

稳健性对照只有两个。其一把颜色簇数改成 5，窗口仍是 9。其二颜色簇数仍是 3，窗口改成 15。它们不是候选的新主指标。

高低两档的切分规则也先写死，供以后使用：以 Hand 训练集上主指标的中位数为阈值，大于等于阈值的是高可分离，低于阈值的是低可分离，掩膜失败的不进档。阈值只用训练盘的分数来定。本审计不按这个阈值训练，也不在测试集上挑选阈值。本次训练盘中位数是 0.8496。因为有 73 只物理餐盘同时出现在训练和测试，这个中位数不是干净的训练餐盘阈值。

1056 张图里，掩膜可用并得到主指标的有 1056 张，失败 0 张。找到浅色盘子的有 942 张。主指标在全部成功样本上的最小、中位、最大分别是 0.4820、0.8476、1.0000。测试盘成功 158 张，其中高可分离 71 张，低可分离 87 张。这些是按 `dish_id` 计数的描述，不是独立物理餐盘的样本数。主指标与对照 1 的 Spearman ρ 是 0.902，与对照 2 的 ρ 是 0.989。

Oracle 只用 Hand 的实例数和每盘类别数，看分数是否落在同一根轴上。Oracle 不是切换规则。在测试盘上，主指标对实例数的 Spearman ρ 是 0.030（n=158，正态近似双侧 p=0.7066），对类别数的 ρ 是 0.051。训练盘上对实例数的 ρ 是 0.205，对类别数的 ρ 是 0.215。若实例数代表「更难分开」，相关应为负。测试盘上的相关接近 0，训练盘上是弱的正相关。因此这次预先登记的分数还没有落在 Hand 实例数那根轴上。主指标保持这次写死的定义。

## 7. 与 archive/v4.0 的差别

只读打开的提交是 `ce1dc0616002a2f76fcffcc8a3ef87addd8c8358`（2026-10-26 11:35:32 +0800，Save the v4.0 plan, code, and results before starting a new scheme.）。`backup/v3_excluded/experiments/` 下的条目是：action, appendix, diagnosis, figures, run_v2_remaining.py, run_v2_remaining.sh, vlm。行动目录 `experiments/action` 里有 16 个文件。目录说明的第一句是「行动层」。逐个文件的开头和一句用途如下。

| 文件 | 用途 | 开头一句 |
|---|---|---|
| `backup/v3_excluded/experiments/action/README.md` | 行动层说明：分组路由、名称先验、逐样本 oracle 上界。 | 行动层 |
| `backup/v3_excluded/experiments/action/__init__.py` | 行动层包说明，没有实验正文。 | Chapter 6 action scripts. |
| `backup/v3_excluded/experiments/action/a1_group_routing.py` | 按数据集分组选择动作，评的是 ECUSTFD 与 MetaFood3D 的称重误差。 | A1: group-conditional action routing on public ECUSTFD + MetaFood3D. |
| `backup/v3_excluded/experiments/action/a2_name_prior.py` | 只看食物名称的质量先验，不看图像。 | A2a: Qwen3.8-27B category-mass prior, frozen on the policy split. |
| `backup/v3_excluded/experiments/action/a2b_vlm_name.py` | 用视觉语言模型给出的食物名替换标注类别名。 | A2b: replace the annotated category with the VLM food name. Tau stays at 4. |
| `backup/v3_excluded/experiments/action/a3_auto_reference/README.md` | ECUSTFD 上的硬币参照物定标，说明里写脚本尚未完成。 | A3 自动参照物 |
| `backup/v3_excluded/experiments/action/a3_auto_reference/__init__.py` | 参照物实验的空包文件。 |  |
| `backup/v3_excluded/experiments/action/a3_auto_reference/run.py` | ECUSTFD 上比较视觉语言模型硬币框、标注硬币框和霍夫圆。 | A3: VLM coin box versus the annotated coin and a Hough circle, on ECUSTFD. |
| `backup/v3_excluded/experiments/action/a4_ood_reject/README.md` | Nutrition5k 与 NV-Real 上的域外拒答，说明里写脚本尚未完成。 | A4 域外检测与选择性拒答 |
| `backup/v3_excluded/experiments/action/a4_ood_reject/__init__.py` | 域外拒答实验的空包文件。 |  |
| `backup/v3_excluded/experiments/action/a4_ood_reject/run.py` | 用共形分数和域外标记做域移检测。 | A4: domain-shift detection on the Nutrition5K eval half and NV-Real val. |
| `backup/v3_excluded/experiments/action/a5_mixed_flow/README.md` | 混合留出流上的端到端路由，说明里写端到端尚未完成。 | A5 端到端混合流 |
| `backup/v3_excluded/experiments/action/a5_mixed_flow/__init__.py` | 混合流实验的空包文件。 |  |
| `backup/v3_excluded/experiments/action/a5_mixed_flow/run.py` | 按诊断章的固定如果-那么规则在混合留出流上选动作。 | A5: fixed diagnosis-to-action routing on a mixed hold-out stream. |
| `backup/v3_excluded/experiments/action/a5_oracle_bound.py` | 分组动作冻结之后，逐样本最优动作还能变多少的上界。 | E15: within-group routing opportunity on the E13/E14 public tables. |
| `backup/v3_excluded/experiments/action/common.py` | 行动层脚本的共享参数工具。 | Shared helpers for the action-layer experiments. |

这些文件做的是分组路由、名称先验、参照物定标、域外拒答和逐样本动作上界。行动目录里，检索「segmentation granularity」「ingredient-level segmentation」「食材级」「整盘回归」「粒度实验」的命中数是 0。为避免只看目录名而漏掉实验，又在 `backup/v3_excluded` 里排除第三方 DINOv2 树之后检索了 granularity、ingredient-level、食材级、整盘、PMAE。命中并逐条读过的路径有 6 条，其中还没有解释的是 0 条。

- `backup/v3_excluded/experiments/diagnosis/d6_capacity/dinov2_ablation.py` 第 216 行摘录：return {'pmae': mean_absolute_percentage_error(trues, preds) * 100,。这是质量回归的骨干容量消融。PMAE 在这里是质量的百分比误差，输入包含深度，不是粒度实验。
- `backup/v3_excluded/logs/d6_dinov2.log` 第 28 行摘录：Ep    ValLoss    ValPMAE。这是上面那次质量回归消融的训练日志，ValPMAE 是验证集质量百分比误差。
- `backup/v3_excluded/logs/n5k_r0.log` 第 16757 行摘录：Ep    TrLoss     ValLoss    ValPMAE    ValMAE     LR。这是 Nutrition5k 质量回归的训练日志，ValPMAE 是质量百分比误差。
- `backup/v3_excluded/logs/other_datasets.log` 第 8 行摘录：[A] 全量物体 (对照: Nutrition5K 单目全 hold-out PMAE=22.0%)。这是 MetaFood3D 上的质量误差预算。文中的 PMAE 是单目质量百分比误差，不是食材级对整盘的粒度对照。
- `backup/v3_excluded/logs/remaining.log` 第 58 行摘录：Ep    TrLoss     ValLoss    ValPMAE    ValMAE     LR。这是质量回归训练日志，ValPMAE 是质量百分比误差。
- `backup/v3_excluded/logs/remaining_status.txt` 第 12 行摘录：2026-10-23 16:57:00 stopped: v2.0 drops ingredient-level NV-Real and does not retrain DINOv2。这是停止再训 DINOv2 的记录，对象是 NV-Real 的食材级路线，不是整盘对食材级的营养估计实验。

诊断目录里的 `d6_capacity` 是质量回归容量消融，PMAE 指的是克重百分比误差，而且用了深度通道。它不是整盘对食材级分割粒度。日志里「ingredient-level NV-Real」是停止再训 DINOv2 的记录，也不是这个粒度问题。没有把这些脚本拷进本分支，也没有把旧路由接回来。

U1 不是对 v4.0 路由或动作选择的重跑。

自适应营养估计粒度不是把 v4.0 的路由或动作选择再跑一遍。

## 8. 附录：部分标签的标注协议

这一节只重新定义以后若做部分标签学习时，标签该怎么读。它不训练，也不解锁该方向。依据有两处。Hand 等人写明，油和糖这类在 RGB 里分不出来的成分没有画进掩膜。Sterling 等人在 bioRxiv 2026.07.26.740845 报告，四名标注者复核了 440 张 Nutrition5k 图像，原始食材标签系统性地漏掉了看不见的成分；修正之后，某一个视觉语言模型的食材重叠从 0.62 外推到大约 0.82。

若以后使用部分标签，协议定义如下。

1. 可见实例是人能在这张俯视 RGB 上描出边界的食材。Hand 的 2922 行就是这个集合，类别是 40 个粗类。
2. 有营养的实例是 `has_nutrition=1` 且质量、能量、脂肪、碳水、蛋白质都不是空值的行。空值是缺失，不能当成 0 克或 0 千卡。
3. 一只盘子是完整标签，当且仅当每个可见实例都有营养，并且这些实例在五个营养素上的求和与盘级表都落在第 4 节的容差内。每一行都有营养数字的盘子有 970 个，其中求和也对得上的完整盘有 546 个。其余 510 个按这个定义是部分标签。
4. 部分标签盘子上，盘级总量减去已标注实例之和，是未标注剩余。剩余可以来自没有营养字段的可见实例，也可以来自油、酱料这类没有画出来的成分。剩余不是「该成分质量为零」的负例，也不能在这次审计里当作新的监督。
5. 盘级质量、能量和宏量仍以 Hand 盘级表为准，并且它们与 Nutrition5k 元数据一致。部分标签只影响食材分解，不影响盘级总量本身。

## 9. 判断

按 `dish_id` 对齐可靠：缺失图像 0，读图失败 0，标识和划分冲突 0。

主可分离度已经预先写死，测试时的输入只有俯视 RGB。字段名 `local_color_purity`。两个稳健性对照不替代它。

留出是否够做盘级不确定区间：不够。判定码 `stop_too_few_independent_plates`。干净物理餐盘 64，下限 80。

对齐可靠，主指标已经预先登记。干净的留出物理餐盘是 64 只，低于下限 80。另有 73 只物理餐盘同时出现在 Hand 的训练和测试里。留出样本不够做盘级不确定区间。审计在这里停止，等待人工判断。不改测试集，不训练，不修改 docs/STATUS.md，不解锁第二阶段。

## 10. 复现

在仓库根目录、已安装 NumPy 和 Pillow 的环境里运行：

```bash
python scripts/u1_phase1_data_audit.py --repo-root . --out results/u1_phase1_data_audit
python -m unittest tests/test_u1_phase1_data_audit.py
```

输入路径：`data/hand2026/dish_nutrition.csv`，`data/hand2026/ingredient_nutrition.csv`，`data/nutrition5k/imagery/realsense_overhead`，`data/nutrition5k/dish_ids/splits/rgb_train_ids.txt`，`data/nutrition5k/dish_ids/splits/rgb_test_ids.txt`，`data/nutrition5k/metadata/`。输出是本目录的 `report.md`、`summary.json` 和若干表。公开 CSV 与 Nutrition5k 图像都不写入版本库。
