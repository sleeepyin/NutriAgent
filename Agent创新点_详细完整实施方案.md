# MM-CFA × Agent 期刊论文新增创新点：详细完整实施方案

> 方案版本：v1.2  
> 制定日期：2026-09-04  
> 项目目录：`/home/mm-cfa-repro`  
> 推荐创新名称：**CAIA-MM（Conformal Active Information Acquisition Agent for Multimodal Food Analysis）**  
> 中文名称：**面向多模态食物分析的共形风险约束主动信息获取 Agent**

---

## 0. 一页结论

### 0.1 能否结合 Agent 增加一个创新点？

**可以，但不建议简单增加一个“LLM 对话/报告生成 Agent”。**

MM-CFA 现有实验已经反复证明：

1. 单张 RGB/RGB-D 图像中的重量信息存在尺度和域先验瓶颈；
2. 制备食物的营养误差主要来自视觉不可观测的密度、配方、含水量、孔隙率等信息；
3. 冻结的共形校准器在跨域数据上会因可交换性破坏而失效；
4. 增大视觉模型容量、增加普通区间候选或继续调校准层，无法创造输入中不存在的信息。

因此，最自然且最有论文说服力的下一步不是“再换一个大模型”，而是把当前的被动预测系统升级为：

> **一个由共形不确定性驱动、能够主动决定“还需要什么信息”的 Agent。它在单张图足够时直接停止；在信息不足时，从追加视角、物理参照物、深度信息、结构化语义问题和拒答中选择成本最低且最有价值的动作；最后对冻结策略的完整交互轨迹重新共形化，给出有限样本覆盖保证。**

若能够采集“同一 episode 的全部候选工具状态”，进一步采用**全工具状态 × 五输出联合共形包络**：在校准时同时覆盖所有预定义状态，使 Agent 改变路由或用户在允许状态中提前停止时仍保持联合边际覆盖。该版本理论主张更强，但区间更保守，必须与冻结策略终端校准并列比较。

这个创新与现有论文逻辑高度一致：

```text
现有论文：发现不确定性 → 定位信息缺失 → 证明单目天花板
新增创新：识别缺哪类信息 → 主动获取最有价值的信息 → 成本受控地降低不确定性
```

### 0.2 最终建议

推荐将新增创新定为：

**CAIA-MM：共形风险约束的主动信息获取 Agent**

但截至 2026-09 的文献边界已经很明确：**“共形触发采集/提问/停止”本身不是新颖点**。可投稿的创新必须是以下组合同时成立：

1. 连续重量与四营养素的多输出回归；
2. 物理尺度/几何动作与语义密度/配方动作的机制化分工；
3. 同一餐的完整配对动作数据；
4. 真实时间、失败、拒答和用户回答噪声；
5. 独立策略校准与联合覆盖审计；
6. 相对最佳固定策略和直接近邻方法具有显著成本收益。

如果这些条件只能完成一部分，应把工作定位为 feasibility study 或拆成后续论文，不应仅借 “Agent” 热点提高贡献等级。

核心研究问题是：

> 在食物重量与营养估计中，能否让 Agent 针对每个样本自适应选择最小的信息获取动作，在维持可靠覆盖率的同时，取得优于固定拍摄流程的“精度—区间质量—用户成本”帕累托前沿？

### 0.3 不建议作为核心创新的做法

以下方向可以作为系统功能或消融，但不应单独作为论文核心创新：

- 让 LLM 根据 MM-CFA 数值生成营养报告；
- 把重量模型、营养模型、共形模型分别命名为多个 Agent；
- 让多个 LLM 投票后平均重量；
- 让 LLM 直接看图猜克数；
- 用提示词让 LLM 自报置信度；
- 仅增加“高不确定时转人工”的阈值规则；
- 仅用 Agent 自动运行现有脚本或自动调参。

这些方案容易被审稿人评价为“工程包装”“Agent 名称替换”或“没有新的可证伪方法”。

---

## 1. 本方案所依据的项目现状

### 1.1 现有核心方法

当前项目已经包含以下主要组成：

1. 4 通道 RGB-D 重量估计；
2. ResNet18 与 EfficientNet-B0 深度集成；
3. AEC-S 分层自适应集成共形预测；
4. 分层权重的 empirical shrinkage 稳定稀疏重量区间（现稿 “Bayesian” 用词需收敛）；
5. 重量 × 密度的两阶段营养素预测；
6. AEC-S2m-N 四营养素共形区间；
7. double-dipping 泄漏分析与诚实校准协议；
8. NutritionVerse-Real 跨域评估；
9. MetaFood3D 密度瓶颈和单图 3D 失败分析；
10. ECUSTFD 参照物正向验证；
11. 多域联合训练对域差的修复实验。

### 1.2 与 Agent 方案直接相关的现有证据

#### 证据 A：深度是一种有价值但有成本的附加模态

- Nutrition5K 4 通道严格 hold-out 约为 21%–22% WAPE、31.2% MAPE；
- RGB-only 3 通道域内脚本报告约 31.4% WAPE；
- 深度信息具有可测量价值，但并非所有设备、场景和样本都能提供。

这使“是否请求深度”天然成为 Agent 的可选动作。

#### 证据 B：域偏移需要主动处理，而不是盲目输出

- NutritionVerse-Real 零样本 WAPE 从域内约 31.4% 退化到 103.7%；
- Spearman 相关降至 0.16；
- 冻结 Split 区间覆盖率从 85.9% 降至 32.8%；
- 营养素区间覆盖降至 44%–56%。

这说明 Agent 必须有 OOD/域门控和拒答能力，不能在域外继续宣称 90% 覆盖。

#### 证据 C：不同食物需要不同信息

- ECUSTFD 刚体食物使用硬币真值框定标后，annotation-assisted oracle 重量 MAPE 为 12.9%；
- MetaFood3D 制备食物即使使用真值体积，类别均值密度仍产生约 25.0% MAPE；
- 蓝莓、坚果、培根等堆叠/多孔食物同时存在体积和有效密度不确定性。

因此：

- 刚体食物更适合请求参照物、深度或额外视角；
- 制备、混合、液体、堆叠食物更需要语义澄清或直接拒答；
- “对所有样本固定要求同一种额外输入”不是最优策略。

#### 证据 D：已有不确定性信号不能直接当作精度提升

项目已发现：

- 模型分歧并不稳定等价于真实误差；
- k-NN 距离对密度误差的相关性弱；
- 加权共形在域内无可检测偏移时只会引入噪声；
- 增加候选后若重复使用同一校准集，会产生容量相关的欠覆盖。

因此 Agent 的动作策略不能只用一个未经验证的“不确定性分数”。必须：

1. 在独立策略训练集上学习动作价值；
2. 冻结策略后再使用独立数据做最终共形校准；
3. 对策略、校准器和最终测试做严格隔离。

### 1.3 执行前需要处理的工程前置问题

1. 现有脚本大量使用 `/home/Nutri_0811`、`/home/nvreal` 等绝对路径，而当前项目位于 `/home/mm-cfa-repro`。新增代码应统一使用环境变量 `MM_CFA_ROOT`，避免继续扩散绝对路径。
2. 当前可见工作树搜索未检出 README 中声明的 `.npz`、`.pkl`、`.pth` 大文件。执行前必须依据 `sha256sum.txt` 核验这些资产是否位于外部实验目录、被忽略、未解压或未拉取。
3. 现有 4 通道训练脚本记录过增强 `clamp(0,1)` 作用于 ImageNet 标准化数据的 bug。Agent 的“RGB 与 RGB-D 模态价值”主实验不能直接混用有 bug 和无 bug 的训练结果，必须使用同一划分、同一增强实现重新训练一组干净的 3ch/4ch 对照模型。
4. 所有多视角数据必须按 dish/object/instance 划分，禁止把同一物体不同帧分到训练和测试两侧。

### 1.4 必须先修复的评估协议风险

代码审计显示，重量侧现有“10 次 50/20/30 随机试验”不能直接作为新 Agent 实验的基础：

1. `train_deep_v2_final.py` 明确在 `deep_crops_cache.npz` 的 `X_train` 上训练并保存固定模型；
2. `aec_s_v2_experiment.py` 和 `conformal_deep_experiment_final.py` 随后把 `X_train` 与原始 `X_test` 合并；
3. 两个脚本从合并后的 3,260 个样本重新随机切分所谓 train/cal/test；
4. 但每次 trial 并没有在该 trial 的训练折上重训基模型，而是继续使用已经看过原固定 `X_train` 标签的权重；
5. 因此随机 test 中会包含基模型训练时见过的样本，9.44% PMAE 不能解释为对未见样本的交叉验证泛化误差，也不应称为“每折重新训练的 10 折 CV”；
6. 该随机混合协议下得到的区间效率也不能替代原始 652 个 `X_test` 上的严格部署评估。

另一个独立问题是：

- `aec_s_v2_experiment.py` 中 AEC-S2 权重在 `y_cal` 上优化后，`conformalize`/`conformalize_mondrian` 又在同一 `y_cal` 上计算 q；
- 因为区间构造本身已经依赖这些校准标签，所以不能仅凭第二次取分位数就声称标准 split-conformal 有限样本保证；
- 这正是营养素脚本已经观察到的 double-dipping 机制，重量侧也必须使用独立的 policy/weight-fit 与 final-calibration 数据。

还必须同时修复以下口径和实现问题：

1. `conformal_deep_experiment_final.py` 的早期 AEC 路径曾直接使用 `y_test` 搜索集成权重；这类结果只能保留为失效负对照，不能作为正式基线；
2. 项目脚本把 `mean(|e_i|/y_i)` 和 `MAE/mean(y)` 都称为 PMAE。前者是 MAPE，后者是 WAPE；审计复算中严格 652 hold-out 约为 31.2% MAPE、21.0% WAPE，不能与 9.44% 只按名称直接比较；
3. 有限样本共形 q 必须取 \(k=\lceil(n+1)(1-\alpha)\rceil\) 对应的顺序统计量；不能使用 `np.quantile` 默认线性插值。若 \(k>n\)，按保守约定令 q 为 \(+\infty\)；
4. 当前发布缓存只有数组而缺少 dish、plate、incremental scan 和原图 ID，因而无法仅凭缓存证明同一盘/同一扫描没有跨 split。必须从原始数据重建可审计 manifest；
5. 五个输出各自达到 90% 边际覆盖，不等于“重量与四营养素同时 90% 覆盖”；若作联合主张，必须采用联合 nonconformity score 或经过证明的多目标误差预算；
6. 按真实重量事后分箱只能作为 coverage audit，不能称为部署时可用的 Mondrian 条件保证；
7. 若实现只有 \(n/(n+\tau)\) 的经验插值而没有显式先验、似然和后验，应写作 `empirical shrinkage`，避免过度使用 “Bayesian”；
8. \(R^2\) 未解释方差、较差的单个 backbone 结果或跨数据集误差分解都不能构成严格的“信息论下界”，建议改称 empirical ceiling/bottleneck evidence。

新增 Agent 工作必须选择以下严谨协议之一：

#### 协议 A：冻结模型 + 真正未见数据

- 固定现有 base 模型；
- 只把原始 `X_test`、外部数据和新采集数据用于 policy/cal/test；
- 任何参与 base 模型训练的 `X_train` 样本都不能进入 Agent 最终测试；
- 由于原始 `X_test` 仅 652 例，建议把它用于机制验证，主 Agent 结果使用额外配对数据。

#### 协议 B：外层交叉拟合

- 按 dish/object 做外层 K 折；
- 每一外折从头训练 base 模型；
- 只在该折未见 episode 上生成动作结果；
- 再在外折未见部分中隔离 policy-fit、terminal-calibration 和 test；
- 汇总每个 episode 恰好一次的 out-of-fold 预测。

#### 协议 C：固定公开 base + 新数据验证

- 保留当前公开模型作为历史基线；
- 不再使用 N5K `X_train` 评价新贡献；
- 用 MM-CFA-Interact 或其他从未参与训练的数据完成 Agent 策略校准与最终评估。

在修复前，建议论文中：

- 将 22.0% 的原始 652 hold-out 结果作为更可信的未见数据重量结果；
- 将 9.44% 标为“包含 base-training 样本的重切分结果/开发结果”，而不是严格泛化结果；
- 不把 AEC-S2c/AEC-S2m 的“二次重标定”写成有限样本保证，除非权重拟合与 q 校准已真正分离；
- 同一批 652/3,260 样本上的 10 次重叠随机划分不是 10 个独立数据集，不能只对 10 个 seed 均值做普通 t 检验后忽略 episode 重用；新实验应使用每 episode 一次的 out-of-fold 预测和 episode-level bootstrap；
- 对正文中“10 折交叉验证”“所有结果均为严格 hold-out”等表述逐项与代码对账。

这是投稿前的强制 Gate。若不先修复，即使 Agent 部分设计正确，审稿人也可能因基础评估污染而否定整篇论文。

---

## 2. 文献定位与真正可声明的研究空白

### 2.1 已有相邻方向

截至 2026-09-04，至少存在以下相邻工作：

1. **Agent 化营养估计**：Open-KNEAD 使用 Agent 分解、FNDDS 检索和配方先验做营养估计；
2. **食物领域 Agent 基准**：OmniFood-Bench 直接讨论视觉语义与物理质量之间的缺口；
3. **营养专家路由**：FoodAgent 已探索由 LLM 路由不同营养估计专家，不能把“营养工具路由”本身写成首创；
4. **食品主动澄清与交互**：PIDA 已构建多轮食品澄清；FoodSage 以真实人机对话处理识别不确定性；CVPRW 2026 也已有交互式营养 Agent 管线；
5. **共形化 Agent/生成模型**：Conf-Gen 研究生成模型、澄清问题和 Agent 输出的共形风险控制；
6. **共形问题选择**：Conformal Information Pursuit（C-IP）已使用共形集合大小选择用户问题；
7. **共形主动多模态获取与停止**：ALMA 已结合共形集合和主动多模态获取，MiCP 已研究多轮 Agent 的自适应停止；
8. **策略后重校准与路径认证**：BCEA 已指出朴素插入证据采集会破坏保证并重校准完整策略；RouteCert 已研究策略选择终止模态组后的路径认证；
9. **共形主动感知**：Conformal Policy Learning、Co-GLANCE 已将共形量化用于机器人策略切换或主动感知；
10. **多目标联合有效性**：DCP-MT 等工作已研究多输出回归的联合覆盖与误差预算；
11. **主动特征获取**：AFA 已形成独立研究方向，包含生成式代理、非贪心策略、策略梯度、Bellman risk-to-go 和标准化基准。

所以不能声称：

- “首个营养 Agent”；
- “首个把共形预测用于 Agent”；
- “首个共形主动多模态获取或自适应停止方法”；
- “首个共形驱动澄清问题或食品主动问询方法”；
- “首个冻结策略后重新共形校准的方法”；
- “首个不确定性驱动主动感知”；
- “首个主动特征获取方法”。

### 2.2 建议声明的窄而强的空白

建议把研究空白限定为：

> 现有营养 Agent 主要进行语义分解、专家路由、检索和推理；通用工作也已覆盖共形主动多模态获取、自适应停止和多目标联合有效性。仍缺少的是一个面向食物重量与营养连续回归的完整实证框架：围绕尺度—几何—密度失效机制，在物理参照、额外视角、深度和结构化语义之间做实例级成本路由，并在真实工具失败和用户响应下，对预定义工具状态与五个输出进行联合覆盖审计。

建议主文直接使用不依赖 “first” 的表述：

> “We present a systematic benchmark of mechanism-aware, cost-sensitive physical and semantic information acquisition with joint conformal auditing for continuous food weight and nutrient estimation.”

只有在 Scopus/Web of Science/Google Scholar 等系统检索完成且保存证据后，才考虑增加 “To the best of our knowledge”；不建议把 first claim 作为贡献成立的必要条件。

本方案制定时的 arXiv 快照检索中：

- `"nutrition estimation" AND "active feature acquisition"` 返回 0 个直接结果；
- `"food portion" AND "active perception"` 返回 0 个直接结果；
- `"food" AND "conformal" AND "agent"` 未找到相关食物测量工作；
- 但分别搜索 Agent nutrition、conformal agent、conformal active perception 和 AFA 均有大量近邻工作。

此外，更宽检索已发现 FoodAgent、PIDA、C-IP、ALMA、BCEA、RouteCert、MiCP 和 DCP-MT 等强近邻。因此“精确检索式返回 0”只能说明交叉点稀缺，绝不能作为 first claim 的证据，更不能替代 Scopus、Web of Science、Google Scholar、IEEE Xplore、ACM DL 和 PubMed 的系统检索。投稿时应保存完整检索式、日期、纳入排除标准和结果清单。

### 2.3 与最接近工作的差异

#### 与 Open-KNEAD 的差异

- Open-KNEAD：单图输入下做 Agent 化食物分解、FNDDS 检索和配方先验；
- CAIA-MM：根据不确定性主动决定是否获取新物理/视觉/语义信息，并考虑交互成本和最终覆盖；
- Open-KNEAD 可作为 CAIA-MM 的“语义工具”或强基线，而不是需要回避的工作。

#### 与 Conf-Gen 的差异

- Conf-Gen：一般生成任务、澄清问题是否足够、Agent 输出正确性；
- CAIA-MM：连续数值回归、物理传感动作、多输出营养区间、食物领域成本和域偏移；
- 冻结获取策略后的终止轨迹共形化应定位为必要有效性协议，而不是单独理论首创。

#### 与 FoodAgent / ALMA / MiCP 的差异

- FoodAgent 的强项是 LLM 专家路由；CAIA-MM 必须证明真实新增观测对物理尺度和密度误差的因果式、分工具收益，而不是只换专家；
- ALMA 已覆盖“共形集合 + 主动多模态获取”的一般交叉点，因此 CAIA-MM 的新颖性必须来自食物测量的尺度—几何—密度机制、连续五输出、真实成本和配对工具 benchmark；
- MiCP 已讨论多轮 Agent 自适应停止，因此“会停止”不是贡献；CAIA-MM 应比较冻结策略终端校准与全工具状态联合包络，并明确各自的保证和保守性；
- 不把这些近邻仅列在 related work：实验中至少实现与其核心思想相匹配的公平基线，或解释任务/代码不适用的具体原因。

#### 与 C-IP / BCEA / RouteCert / PIDA 的差异

- C-IP 已覆盖共形驱动的问题选择；CAIA-MM 不能把“区间宽就提问”本身当作创新，必须展示物理与语义动作之间的实例级收益差异；
- BCEA 已覆盖完整采集策略重校准；终端校准是合规底线，CAIA-MM 的价值来自连续食物测量、五输出和真实工具 benchmark；
- RouteCert 已研究策略选择校准组后的路径保证；CAIA-MM 应将其改造成回归路径基线，并与更保守的全状态包络比较；
- PIDA 已进入食品主动澄清，因此仅做多轮食物问答不够；必须加入真实新观测、称重真值、回答噪声、负担和共形区间；
- CVPRW 2026 的交互式营养 Agent 存在模型基于同图自问自答的设置；CAIA-MM 必须清楚区分“增加推理 token”与“真正获取新物理/用户证据”。

#### 与 Conformal Policy Learning / Co-GLANCE 的差异

- 相邻工作：机器人控制、视觉遮挡、策略切换或机器人分配；
- CAIA-MM：食物质量与营养估计中的尺度—几何—密度信息分解；
- 动作包含人机问答、参照物和视觉模态，不是单一机器人再感知动作；
- 评价目标是覆盖、区间效率、点估计误差和用户负担的联合帕累托前沿。

#### 与通用 AFA 的差异

- 通用 AFA 多在完整表格上遮蔽特征并模拟逐列获取；
- CAIA-MM 获取的是异构工具输出：新图片、深度、物理尺度、语义答案和拒答；
- 每个动作具有不同失败模式、时间成本和可用性；
- CAIA-MM 引入策略级终端共形校准，避免把策略训练误差错误地解释成覆盖保证。

---

## 3. 候选 Agent 创新方向比较

| 方向 | 新颖性 | 与现有项目一致性 | 可验证性 | 风险 | 建议 |
|---|---:|---:|---:|---:|---|
| LLM 自动生成营养报告 | 低 | 中 | 低 | 高度工程化 | 不作为创新 |
| 多 Agent 分别负责识别、重量、营养、审稿 | 中低 | 中 | 中 | 容易变成模块重命名 | 仅系统消融 |
| LLM + FNDDS 检索直接修正营养 | 中 | 高 | 中 | 与 Open-KNEAD 高度相邻 | 可作为工具 |
| Agent 自动域适配/重训练 | 中高 | 高 | 中 | 在线泄漏、灾难遗忘、成本高 | 后续工作 |
| **机制感知的物理—语义主动食物测量** | **中等，依赖完整组合** | **极高** | **高** | ALMA/C-IP/BCEA/RouteCert/PIDA 已覆盖部分思想，必须有成对数据和真实成本 | **主方案** |

### 3.1 为什么主方案最适合当前论文

1. 它直接承接当前论文的“信息天花板”结论；
2. 它能利用已有 N5K、NV-Real、MetaFood3D、ECUSTFD 四类证据；
3. 它提供清楚的 Agent 定义，而不是把普通流水线改名；
4. 它有明确的动作、状态、成本、奖励和停止条件；
5. 它能提出可证伪假设；
6. 成功时是建设性突破，失败时也能量化“何时 Agent 没有必要”；
7. 只有“连续多输出 + 物理语义混合动作 + 真实交互 + 联合审计”的完整组合才足以形成期刊扩展；单个组件不再具有高新颖性。

---

## 4. 方法名称、论文主张与研究问题

### 4.1 建议方法名称

英文：

> **CAIA-MM: Conformal Active Information Acquisition for Multimodal Food Analysis**

中文：

> **CAIA-MM：面向多模态食物分析的共形主动信息获取 Agent**

如更强调用户交互，可用：

> **CAIA-Food: A Cost-Aware Conformal Agent for Interactive Food Measurement**

考虑到 “Agent” 与 “conformal acquisition” 已有大量直接先例，更稳妥的投稿标题/方法表述是：

> **Mechanism-Aware Active Food Measurement with Post-Policy Conformal Risk Control**

内部系统仍可称 CAIA-MM，但论文标题优先突出可检验的“机制感知测量”，避免让审稿人把工作理解为追逐 Agent 热点。

### 4.2 一句话方法定义

> CAIA-MM 是一个受共形风险约束的工具调用 Agent，它基于当前图像、域外程度、食物形态和预测区间状态，估计每种附加信息的期望价值，选择停止、请求参照物、请求额外视角、请求深度、提出结构化语义问题或拒答，并对冻结策略产生的最终轨迹重新校准。

### 4.3 建议研究问题

#### RQ1：成本—性能

在相同平均交互成本下，自适应 Agent 是否比固定输入协议取得更低的点误差和区间分数？

#### RQ2：动作异质性

最有价值的动作是否随刚体/制备、域内/域外、重量区间和密度不确定性而变化？

#### RQ3：自适应决策后的覆盖

如果动作和停止时间由数据自适应决定，普通冻结共形区间是否仍可靠？策略终端重新共形化能否恢复名义覆盖？

#### RQ4：域偏移

Agent 的 OOD 门控、主动获取和拒答能否减少 NutritionVerse-Real 上的灾难性错误？需要多少目标域校准样本才能恢复可声明覆盖？

#### RQ5：LLM 的真实作用

LLM 是否在结构化语义解析和问题表达上有价值？若移除 LLM，仅保留确定性控制器，核心数值性能是否保持？

### 4.4 建议预注册假设

#### H1

CAIA-MM 在经验覆盖接近名义 90% 的条件下，相对最佳成本匹配固定策略降低至少 10% 的 normalized Winkler/interval score。

#### H2

参照物/深度动作在刚体食物上的收益显著高于制备或堆叠食物；语义动作在制备食物上的收益更高。

#### H3

直接复用动作前校准器或在同一校准集上同时优化策略与分位数，会产生欠覆盖；冻结策略后的独立终端校准恢复边际覆盖。

#### H4

在目标域完全无校准时，系统只声明 OOD 检测和选择性风险结果，不声明 90% 覆盖；加入少量目标域校准后，覆盖随样本数单调恢复。

#### H5

LLM-only planner 的复现方差和无效动作率高于带动作约束的混合 Agent；混合 Agent 的主要数值收益来自工具选择，而不是 LLM 直接估数。

---

## 5. Agent 的严格形式化

### 5.1 一个样本是一个 episode

对第 \(i\) 个食物实例，定义：

\[
E_i = (X_i^{(0)}, \mathcal{O}_i, Y_i)
\]

其中：

- \(X_i^{(0)}\)：初始单张 RGB 图像；
- \(\mathcal{O}_i\)：该实例可按动作逐步揭示的额外信息；
- \(Y_i\)：真实重量及营养素向量。

目标向量可写为：

\[
Y_i = (W_i, K_i, F_i, C_i, P_i)
\]

分别代表重量、热量、脂肪、碳水和蛋白质。

### 5.2 状态

第 \(t\) 步状态：

\[
S_t =
\left[
Z_t,\ \hat{Y}_t,\ C_t,\ d_{\mathrm{OOD}},\
p_{\mathrm{rigid}},\ p_{\mathrm{prepared}},\
\Delta_{\mathrm{ens}},\ M_t,\ B_t
\right]
\]

含义：

- \(Z_t\)：当前可见模态的嵌入；
- \(\hat{Y}_t\)：重量和营养素点预测；
- \(C_t\)：当前路由用不确定性摘要；
- \(d_{\mathrm{OOD}}\)：域外分数或 conformal p-value；
- \(p_{\mathrm{rigid}}\)、\(p_{\mathrm{prepared}}\)：形态概率；
- \(\Delta_{\mathrm{ens}}\)：模型分歧，仅作为一个候选特征；
- \(M_t\)：已经获取的信息掩码；
- \(B_t\)：剩余预算。

注意：供策略使用的 \(C_t\) 是在策略训练阶段拟合并冻结的“路由不确定性”，不能与最终保证用校准器共用数据。

#### 因素化不确定性分解（方法核心）

不要只给控制器一个总区间宽度。应把项目已经识别出的失效机制编码为：

\[
\mathbf{u}_t =
\left[
u_{\mathrm{scale}},
u_{\mathrm{geometry}},
u_{\mathrm{density}},
u_{\mathrm{domain}},
u_{\mathrm{quality}}
\right]
\]

可部署代理信号示例：

- \(u_{\mathrm{scale}}\)：无参照物、相机内参未知、预测重量对裁剪尺度敏感；
- \(u_{\mathrm{geometry}}\)：单视角遮挡、food mask 不完整、视角间预测不一致；
- \(u_{\mathrm{density}}\)：形态为 mixed/stacked/liquid、food-code 熵、视觉密度与检索先验不一致；
- \(u_{\mathrm{domain}}\)：OOD 距离、域分类概率、conformal p-value；
- \(u_{\mathrm{quality}}\)：模糊、曝光、遮挡和参照物检测质量。

每个动作有物理上合理的目标因素：

| 动作 | 主要减少因素 | 通常不能解决 |
|---|---|---|
| Reference | scale，部分 geometry | prepared-food density |
| Extra view | geometry，部分 occlusion | hidden oil/water |
| Depth | geometry，部分 scale | class-conditional density |
| Semantic question | density、food identity | metric scale |
| Abstain | 下游风险 | 不减少预测误差 |

该矩阵只是结构先验，不应硬编码为结论。真实动作收益仍由 \(D_{\mathrm{policy}}\) 的配对数据学习和验证。

这一步是 CAIA-MM 区别于普通“看区间宽就再拍一张”的关键：Agent 不只判断“不确定”，还判断**不确定性的来源与哪种动作物理上匹配**。

### 5.3 动作

\[
\mathcal{A} =
\{
a_{\mathrm{stop}},
a_{\mathrm{ref}},
a_{\mathrm{view}},
a_{\mathrm{depth}},
a_{\mathrm{semantic}},
a_{\mathrm{abstain}}
\}
\]

为避免 400 例数据被过多稀疏路径切碎，主实验建议采用两层动作空间：

1. 一级只选择 `PHYSICAL / SEMANTIC / STOP / ABSTAIN`；
2. `PHYSICAL` 内再根据设备可用性和 \(u_{\mathrm{scale}},u_{\mathrm{geometry}}\) 选择 reference、view 或 depth；
3. 只有各物理子动作拥有足够配对样本时，才在主策略中把三者完全展开；
4. 一级与二级决策均计入轨迹和成本，不能把二级选择隐藏成无成本预处理。

这样既保留完整工具体系，又能避免路径条件校准和策略学习因样本过少失稳。

#### `STOP`

直接输出当前预测和最终区间。

#### `REQUEST_REFERENCE`

要求用户把已知尺寸的银行卡大小卡片、ArUco 标记或硬币放在食物同一平面，再拍一张照片。

#### `REQUEST_VIEW`

要求用户补拍标准视角，例如：

- 当前为俯视图时补拍约 45° 侧视图；
- 当前为侧视图时补拍俯视图；
- 保持同一盘食物和相近距离。

#### `REQUEST_DEPTH`

当设备支持深度或 RGB-D 传感器时获取深度；不支持时动作被 mask。

#### `ASK_SEMANTIC`

只允许从预定义问题库中选择一个低负担问题，例如：

1. 这是完整单块食物、松散堆叠、压实混合物还是汤/液体？
2. 主要食物类别是什么？
3. 是否含有较多不可见油、酱汁或水？
4. 容器是否为已知标准容量？
5. 当前份量是否来自有规格的包装或标准餐具？

禁止询问：

- “它有多少克？”；
- “它有多少卡路里？”；
- 任何直接泄露预测目标的问题。

#### `ABSTAIN`

当 OOD 严重、输入质量失败、预算用尽且区间仍不可操作时，输出“无法可靠估计”，并说明需要的下一项信息。

### 5.4 工具

Agent 至少调用以下工具：

1. `BaseMMCFATool`：现有重量和营养管线；
2. `DomainAuditTool`：域外与输入质量检测；
3. `MorphologyTool`：刚体/制备/堆叠/液体形态识别；
4. `ReferenceScaleTool`：参照物检测、尺度恢复和几何估计；
5. `MultiViewTool`：多视角特征或预测融合；
6. `SemanticDensityTool`：把结构化答案映射到密度/营养先验；
7. `ConformalSafetyTool`：最终终端轨迹校准；
8. `TraceLogger`：记录动作、成本、工具版本和输出。

### 5.5 成本

每个动作成本不应由作者随意指定为 1、2、3 后只报告单一结论。建议同时报告：

1. **真实时间成本**：从提示出现到成功获取输入的秒数；
2. **交互次数**；
3. **失败重试次数**；
4. **用户负担评分**：1–5 Likert；
5. **设备成本**：是否需要深度传感器；
6. **隐私成本**：是否需要把图片发送到外部 API。

策略训练初期可以使用归一化先验成本：

| 动作 | 试运行成本单位 |
|---|---:|
| STOP | 0 |
| ASK_SEMANTIC | 1 |
| REQUEST_VIEW | 2 |
| REQUEST_REFERENCE | 3 |
| REQUEST_DEPTH | 3 |
| ABSTAIN | 1 |

主结果必须用实际采集时间和用户负担做敏感性分析。

### 5.6 动作价值

定义动作 \(a\) 的价值：

\[
\widehat{\mathrm{VOI}}(a \mid S_t)
=
\widehat{\mathbb{E}}
\left[
\mathrm{IS}_t - \mathrm{IS}_{t+1}
\mid S_t,a
\right]
- \lambda c(a)
- \eta r_{\mathrm{fail}}(a \mid S_t)
\]

其中：

- \(\mathrm{IS}\) 为 interval score/Winkler score；
- \(c(a)\) 为交互成本；
- \(r_{\mathrm{fail}}\) 为动作获取失败概率；
- \(\lambda,\eta\) 在策略训练集上确定。

Agent 选择：

\[
a_t^* = \arg\max_{a \in \mathcal{A}_{\mathrm{valid}}(S_t)}
\widehat{\mathrm{VOI}}(a \mid S_t)
\]

停止条件：

\[
\max_a \widehat{\mathrm{VOI}}(a \mid S_t) \le 0
\]

或满足：

- 当前区间已达到可操作阈值；
- 达到最大步数 \(T_{\max}\)；
- 达到时间/动作预算；
- 所有可用动作均失败。

### 5.7 推荐的策略复杂度

第一版不要直接上大型强化学习。

推荐顺序：

1. **一阶 myopic VOI**：每次预测下一动作的期望收益；
2. **两步 look-ahead**：仅在实验表明动作存在互补性时加入；
3. **BRiG-AFA/ACO/NM-PPG**：作为高阶基线，而不是首个实现。

理由：

- 当前动作只有 4–5 个；
- 每个 episode 最多 2 次交互即可形成论文；
- 数据规模不足以稳定训练复杂 RL；
- 可解释的收益模型更符合项目一贯的诚实性风格。

### 5.8 端到端算法伪代码

```text
Algorithm CAIA-MM-TRAIN
Input:
  D_base, D_policy, D_cal
  action set A, maximum steps T_max
  miscoverage alpha, cost coefficients lambda/eta
  calibration_mode in {terminal_policy, all_tool_states}

1. Fit or freeze all base prediction tools using D_base only.
2. On D_policy:
   2.1 Generate grouped out-of-fold predictions for every legal action.
   2.2 Build route-state features without exposing unselected outcomes.
   2.3 Compute action gain labels and observed acquisition costs.
   2.4 Fit gain model G_a(s) and failure model F_a(s) for each action.
   2.5 Select lambda/eta and stopping threshold on policy-validation only.
3. Freeze controller pi, tool versions, prompts, ontology and randomization.
4. For each episode in D_cal:
   if calibration_mode == terminal_policy:
     4.1 Run the complete frozen policy pi from the initial image.
     4.2 Reveal only outcomes of actions selected by pi.
     4.3 Compute R_i = max_j |Y_ij-m_j(H_i)|/h_j(H_i).
   else:
     4.4 Evaluate every pre-collected reachable state m in M_i.
     4.5 Compute R_i = max_m max_j |Y_ij-m_mj|/h_mj.
5. Let k = ceil((n_cal+1)*(1-alpha)); take the kth order statistic
   exactly (q=+infinity if k>n_cal), without linear interpolation.
6. Freeze and hash {pi, tools, calibration_mode, q, splits, configs}.
Output: locked CAIA-MM agent.
```

```text
Algorithm CAIA-MM-INFER
Input:
  initial image x0, locked agent, available-action mask, budget B

1. Observe x0 and run base/domain/morphology tools.
2. For t = 0 ... T_max:
   2.1 Construct state s_t from currently observed information only.
   2.2 For each legal action a, compute:
       value[a] = gain_LCB[a] - lambda*cost[a] - eta*failure[a].
   2.3 If all acquisition values <= 0:
       choose STOP if deployment validity checks pass;
       otherwise choose ABSTAIN.
   2.4 Validate action against the whitelist and remaining budget.
   2.5 Execute the selected tool, log cost/failure, and update the state.
3. At termination, compute terminal center m(H) and scale h(H).
4. Return [max(0, m-q*h), m+q*h], point estimate and trace.
   Under all_tool_states, q simultaneously covers every predefined
   state/output; under terminal_policy, changing pi requires recalibration.
5. If an OOD/validity condition invalidates the calibration claim,
   label the interval as empirical only or abstain; do not display
   "90% guaranteed".
```

---

## 6. 共形保证：本方案最关键的方法学部分

### 6.1 为什么普通做法可能失效

如果使用同一批数据：

1. 学习动作策略；
2. 选择停止阈值；
3. 优化区间宽度；
4. 计算最终共形分位数；

那么会重现项目已经发现的 double-dipping：策略专门适配了校准样本，最终分位数偏小，测试时欠覆盖。

另一个风险是：Agent 根据当前区间宽度选择是否继续，最后只保留“看起来容易”的路径。动作选择改变了终端样本分布，不能直接把动作前的分位数当作动作后的保证。

### 6.2 必须采用的四段隔离

建议数据角色严格分为：

| 数据角色 | 用途 | 禁止用途 |
|---|---|---|
| \(D_{\mathrm{base}}\) | 训练视觉/重量/营养模型 | 不训练最终策略 |
| \(D_{\mathrm{policy}}\) | 训练动作收益模型、成本模型和停止规则 | 不计算最终 q |
| \(D_{\mathrm{cal}}\) | Agent 完全冻结后，跑完整轨迹并计算终端 q | 不调策略 |
| \(D_{\mathrm{test}}\) | 一次性最终评估 | 不做任何选择 |

如果需要路由区间，另从 \(D_{\mathrm{policy}}\) 内切出 `route-fit/route-val`，不要占用 \(D_{\mathrm{cal}}\)。

### 6.3 冻结策略的终端轨迹校准

设冻结 Agent 策略为 \(\pi\)，它对每个 episode 产生终止历史：

\[
H_i^{\pi}
=
(X_i^{(0)},A_{i,1},O_{i,1},\ldots,A_{i,T_i},O_{i,T_i})
\]

终端预测器给出中心 \(m(H_i^\pi)\) 和尺度 \(h(H_i^\pi)>0\)。

定义校准分数：

\[
R_i^\pi =
\frac{|Y_i-m(H_i^\pi)|}{h(H_i^\pi)}
\]

在 \(D_{\mathrm{cal}}\) 上计算：

\[
\hat q_\pi =
R^\pi_{(\lceil(n_{\mathrm{cal}}+1)(1-\alpha)\rceil)}
\]

最终区间为：

\[
C_\pi(H^\pi)
=
\left[
\max(0,m(H^\pi)-\hat q_\pi h(H^\pi)),
m(H^\pi)+\hat q_\pi h(H^\pi)
\right]
\]

若满足：

1. episode 在校准和测试间可交换；
2. 策略 \(\pi\)、预测器 \(m,h\) 在查看 \(D_{\mathrm{cal}}\) 标签前已冻结；
3. Agent 在校准和测试上使用完全相同的动作规则；
4. 随机策略使用独立同分布随机种子；
5. 用户回答、重拍行为和工具失败是 episode 数据生成机制的一部分，校准与测试阶段的响应机制没有系统变化；
6. 离线模拟器只向策略揭示其实际选择的动作结果，不能读取未选择动作的 counterfactual 结果；

则终端分数可交换，从而：

\[
\Pr\{Y_{\mathrm{test}}\in C_\pi(H_{\mathrm{test}}^\pi)\}
\ge 1-\alpha
\]

这是**策略级边际覆盖**，不是对每种任意子群都保证 90%。

若上线用户比校准参与者更常拒绝回答、使用不同设备或采取不同拍摄方式，上述可交换性仍会破坏；此时必须重新校准或只报告选择性风险，不能把旧保证机械迁移。

### 6.4 强化方案：全工具状态 × 多输出联合共形包络

若 Level B 数据为每个 episode 采集了全部候选动作结果，可以不只校准冻结策略实际走到的终点，而是同时覆盖**所有预定义可达工具状态**。

设：

- \(\mathcal M_i\)：episode \(i\) 可达的工具状态集合；
- \(j\in\{W,K,F,C,P\}\)：重量与四个营养输出；
- \(m_{m,j}(X_{i,m})\)、\(h_{m,j}(X_{i,m})>0\)：在状态 \(m\) 下已经冻结的中心与误差尺度。

定义每个 episode 的单一最大分数：

\[
R_i^{\forall}
=
\max_{m\in\mathcal M_i}
\max_j
\frac{|Y_{ij}-m_{m,j}(X_{i,m})|}
{h_{m,j}(X_{i,m})+\epsilon}
\]

令：

\[
k=\left\lceil(n_{\mathrm{cal}}+1)(1-\alpha)\right\rceil,
\qquad
\hat q_{\forall}=R^{\forall}_{(k)}
\]

实现时必须直接取顺序统计量：

```python
k = math.ceil((len(scores) + 1) * (1 - alpha))
q = math.inf if k > len(scores) else np.partition(scores, k - 1)[k - 1]
```

对 Agent 最终选择的任意允许状态 \(m_\tau\)，输出：

\[
C_{m_\tau,j}
=
\left[
\max(0,m_{m_\tau,j}-\hat q_{\forall}h_{m_\tau,j}),
\;
m_{m_\tau,j}+\hat q_{\forall}h_{m_\tau,j}
\right]
\]

在 episode 可交换、状态集合和所有状态预测器预先冻结、工具响应机制一致的条件下，可声明：

\[
\Pr\left\{
\forall m\in\mathcal M_{\mathrm{test}},
\forall j,\;
Y_j\in C_{m,j}
\right\}
\ge 1-\alpha
\]

因此 Agent 可依据已观察历史切换路由，用户也可在允许状态中提前停止，而不需要针对每个具体策略重新计算 q。这里保证的仍是**新 episode 的联合边际覆盖**，不是菜系、重量段或域外条件覆盖。

该方案的实施边界：

1. 只有实际采集到同一 episode 的完整状态结果，才能计算该最大分数；不能让模拟器凭空生成未观测 counterfactual；
2. 若动作顺序会影响用户回答或拍摄结果，应枚举“路径状态”而不是仅枚举无序模态子集；
3. 工具失败必须作为可达状态保留，不能删除失败 episode；
4. 部署时增加新工具、修改状态预测器或改变语义问题，会超出原保证；
5. 状态数和输出数增多会放大最大分数，使区间变宽。

建议把两种校准都作为正式方法：

| 校准方式 | 优点 | 代价 | 推荐角色 |
|---|---|---|---|
| 冻结策略终端校准 \(R^\pi\) | 区间通常更窄；所需数据较少 | 改策略后必须重校准 | 主效率结果 |
| 全工具状态联合包络 \(R^\forall\) | 允许预定义状态内换路由和提前停止；五输出同时覆盖 | 需要动作完备 episode；更保守 | 主安全结果/方法强化 |

若 \(R^\forall\) 宽到没有实用意义，应诚实降级到 \(R^\pi\)，不能只保留其理论表述而隐藏效率代价。

### 6.5 路径条件校准

可以按不使用真值的终止路径分组：

- `STOP@0`；
- `SEM→STOP`；
- `REF→STOP`；
- `VIEW→STOP`；
- `VIEW→SEM→STOP`；
- `ABSTAIN`。

对样本量足够的路径做 Mondrian 校准：

\[
\hat q_g = Q_{1-\alpha}\{R_i^\pi:G(H_i^\pi)=g\}
\]

要求：

- 分组规则预先固定；
- 组别只依赖观测和动作，不依赖真实标签；
- 每组至少建议 50 个校准 episode；
- 小样本组回退到全局 \(\hat q_\pi\)；
- 不能把“每路径经验覆盖”误写为任意条件覆盖。

### 6.6 任意时刻停止的保守方案

如果允许用户在任意一步主动结束，而不是严格服从冻结策略，可使用 alpha spending：

\[
\sum_{t=0}^{T_{\max}}\alpha_t \le \alpha
\]

每一步分别构造 \(1-\alpha_t\) 区间。由 union bound：

\[
\Pr\{\forall t,\ Y\in C_t\}
\ge 1-\sum_t\alpha_t
\ge 1-\alpha
\]

因此任何停止时刻选择的区间也被同时覆盖。

简单设置：

\[
\alpha_t=\frac{\alpha}{T_{\max}+1}
\]

该方法保守，应作为 safety baseline。效率主结果使用冻结策略终端校准；若具备动作完备数据，安全主结果使用全工具状态联合包络。

### 6.7 多输出联合覆盖

现有四营养素结果主要是逐输出边际覆盖。若新增创新需要声明“重量和四营养素同时可靠”，可定义：

\[
R_i^{\mathrm{joint}}
=
\max_j
\frac{|Y_{ij}-m_j(H_i^\pi)|}{h_j(H_i^\pi)}
\]

在校准集上只计算一个联合分位数。这样可保证：

\[
\Pr\{\forall j,\ Y_j\in C_j(H^\pi)\}\ge 1-\alpha
\]

代价是区间更宽。建议：

- 若论文声称“重量和四营养素同时可靠”，主文必须报告联合覆盖、联合 interval score/超矩形体积，不能只放附录；
- 同时逐项报告五个边际覆盖，定位是哪一维导致联合失效；
- 与 Bonferroni、联合 max-score 和 DCP-MT 类误差预算比较效率；
- 若新采数据只有重量真值，则重量作为该数据集主终点，五输出联合结论只能在可靠营养标签子集上报告；
- 不要混淆边际、联合、条件和域外覆盖。

### 6.8 域偏移下不能过度承诺

NutritionVerse-Real 已经实证说明 N5K 与目标域不可交换。因此：

- 零目标域校准时，不能声明 90% finite-sample coverage；
- OOD gate 只能触发获取或拒答，不能自动恢复数学保证；
- 加权共形只有在密度比和协变量偏移假设成立时才合理；
- 少量目标域校准实验应设置 \(n=25,50,100,200\) 的恢复曲线；
- 在线 ACI 如使用，只能按其实际定理声明长期平均性质，不能替换成单样本覆盖。

### 6.9 拒答后的覆盖口径

`ABSTAIN` 会引入选择性预测问题。标准全体边际覆盖不能自动解释为“非拒答样本中的 90% 覆盖”。

推荐两种合规方式：

#### 方式 A：所有 episode 都定义区间

- 自动接受样本输出正常区间；
- 拒答样本输出明确标记的保守 fallback 区间或“不提供数值”；
- 全体边际覆盖和 risk-coverage 分开报告；
- 不把全体覆盖转述成接受子集覆盖。

#### 方式 B：冻结接受规则后分组校准

- `accept/abstain` 规则必须在 \(D_{\mathrm{policy}}\) 上冻结；
- 该规则只能依赖观测、动作和路由特征，不能依赖 \(D_{\mathrm{cal}}\) 真值或最终 q；
- 在 \(D_{\mathrm{cal}}\) 中仅用被固定规则接受的 episode 计算接受组 q；
- 在可交换且接受组校准样本足够时，报告“固定选择规则下接受分布的边际覆盖”；
- 拒答组只报告拒答率、错误捕获率和后续人工流程。

主文必须同时报告：

- coverage among accepted；
- acceptance/automation rate；
- selective risk；
- catastrophic error recall；
- 平均动作和时间成本。

否则可以通过大量拒答轻易得到看似很高的准确率或覆盖率。

---

## 7. 系统架构

### 7.1 推荐架构

```text
初始 RGB
   │
   ├── Base MM-CFA ──────── 点预测/候选区间
   ├── Domain Audit ─────── OOD、质量、域置信
   ├── Morphology Tool ──── 刚体/制备/堆叠/液体
   └── Route UQ ─────────── 当前风险摘要
             │
             ▼
      Cost-aware Controller
       ├── STOP
       ├── REQUEST_REFERENCE ── ReferenceScaleTool
       ├── REQUEST_VIEW ─────── MultiViewTool
       ├── REQUEST_DEPTH ────── RGB-D Tool
       ├── ASK_SEMANTIC ─────── SemanticDensityTool
       └── ABSTAIN
             │
             ▼
       更新状态，最多两轮
             │
             ▼
   Frozen-policy Terminal Conformalizer
             │
             ▼
   点预测 + 校准区间 + 轨迹 + 适用范围声明
```

### 7.2 为什么建议“确定性控制器 + 可选 LLM”

LLM 适合：

- 把用户自然语言映射到固定语义标签；
- 将机器动作转换为易理解的问题；
- 解释为什么需要下一张图；
- 对无效输入进行自然语言反馈。

LLM 不应负责：

- 直接输出重量；
- 直接决定共形分位数；
- 修改覆盖率阈值；
- 绕过动作白名单；
- 在测试期间改写策略；
- 将自报信心当作统计保证。

核心控制器应输出严格 JSON：

```json
{
  "action": "REQUEST_REFERENCE",
  "reason_code": "SCALE_UNCERTAINTY_RIGID",
  "expected_gain": 0.18,
  "estimated_cost_seconds": 14.2,
  "remaining_budget": 1,
  "allowed": true
}
```

LLM 只把该结构转成用户可读文本。这样即使没有 API，核心实验仍可完全复现。

### 7.3 建议轨迹日志

每个 episode 保存：

```json
{
  "episode_id": "dataset|dish_or_object_id",
  "split": "policy|cal|test",
  "initial_domain": "n5k|nvreal|mf3d|ecustfd|prospective",
  "policy_version": "caia-mm-v1",
  "base_model_hash": "...",
  "calibrator_hash": "...",
  "steps": [
    {
      "t": 0,
      "available_modalities": ["rgb"],
      "point_prediction": {},
      "route_uncertainty": {},
      "ood_score": 0.0,
      "action": "REQUEST_VIEW",
      "action_cost_seconds": 0.0,
      "tool_success": true
    }
  ],
  "terminal_prediction": {},
  "terminal_interval": {},
  "abstained": false
}
```

禁止在日志中存储未脱敏的人脸、位置、EXIF 或用户健康身份信息。

---

## 8. 数据方案

### 8.1 两级实施策略

#### Level A：只用现有公开数据做机制验证

目标：先证明“不同动作对不同样本的收益确实不同”，再决定是否采新数据。

限制：各工具来自不同数据集时，数据集难度与动作效果混杂；Level A 只能支持逐工具 proof-of-concept，不能独立支撑“统一 Agent 优于固定策略”或全工具状态联合保证。

#### Level B：采集动作完备的配对交互数据

目标：让同一个食物实例同时具有所有候选动作的结果，才能严格训练和评估多工具 Agent。

期刊版本建议至少完成 Level A；若要把 Agent 作为主贡献而不是附加实验，强烈建议完成 Level B。

### 8.2 现有数据如何使用

#### Nutrition5K

用途：

- 初始 RGB vs 请求 depth 的配对动作；
- 域内重量与营养预测；
- 现有 AEC-S/AEC-S2m-N 作为基础；
- 食材和菜品元数据用于设计语义工具。

要求：

- 干净重训同协议 3ch/4ch；
- 不能用有增强 bug 的 4ch 与修复后的 3ch 做主结论；
- 从原始元数据重建带 `dish_id/plate_id/scan_id/frame_id` 的 manifest；当前仅含数组的缓存不足以证明无泄漏；
- 同一次 incremental plate scan 的全部帧、深度和视角必须作为一个 group，不能只对裁剪图随机划分。

#### NutritionVerse-Real

用途：

- 每菜约 3.5 张图，可构建“首图→请求额外视角”；
- 域外门控和拒答；
- 多域训练模型的工具输出；
- 目标域少样本校准恢复曲线。

要求：

- 以 dish 为 episode；
- 一个 dish 的所有视角只能属于同一数据角色；
- 首图选择规则固定，例如按文件名排序的第一张，不能按测试误差选择；
- 第二视角选择可用固定规则或训练于 \(D_{\mathrm{policy}}\) 的视角质量模型。

#### MetaFood3D

用途：

- 多视角 RGB-D；
- 刚体/制备/堆叠形态；
- 真值体积、重量、密度；
- 测试几何动作和语义动作的收益边界；
- 建立“哪些动作无法解决密度问题”的反例。

要求：

- 下载真实 RGB-D/多视角原始数据；
- 按 object 划分，禁止帧泄漏；
- GT volume 只能做 oracle 上界，不能作为可部署 Agent 工具；
- 类别密度必须 LOO 或仅由训练角色统计。

#### ECUSTFD

用途：

- 同一图像中“忽略参照物”与“使用硬币尺度”的配对比较；
- 刚体食物的参照物工具；
- 验证 Agent 是否在刚体样本上优先选择 reference。

要求：

- 按 instance 划分；
- 无参照物条件只能禁用 `coin_px`，不能让训练模型通过文件名或类别间接获得测试尺度；
- 当前使用 XML 真值框得到的 12.9% 只能标作 annotation-assisted oracle；可部署主结果必须加入真实硬币/卡片检测器，并将检测失败计入；
- 主结果报告 GT 体积、估计体积和重量三层误差。

### 8.3 新建 Agent 配对数据集：推荐方案

建议命名：

> **MM-CFA-Interact**

#### 推荐样本量

- Pilot：60 个独立食物实例；
- 最低可用：200 个独立实例；
- 推荐期刊规模：400 个独立实例；
- 更强覆盖审计：800 个以上。

90% 覆盖率的普通二项近似下，95% 置信区间半宽约为：

\[
1.96\sqrt{\frac{0.9(1-0.9)}{n}}
\]

因此：

- \(n=200\)：约 ±4.2 个百分点；
- \(n=400\)：约 ±2.9 个百分点；
- \(n=800\)：约 ±2.1 个百分点。

所以若只有 100 个测试实例，不能仅凭“观测到 90%”就宣称实证覆盖非常稳定。

#### 每个实例需要采集的内容

1. 初始自然 RGB 图；
2. 同场景放入标准尺寸卡片/ArUco 后的图；
3. 标准俯视图；
4. 约 45° 侧视图；
5. 若设备支持，RGB-D 图；
6. 结构化语义标签；
7. 电子秤真实重量；
8. 食物类别和制备状态；
9. 可选：真实配方或包装营养标签；
10. 每个动作所需时间、重拍次数和负担评分。

#### 标签范围与主次终点

- 400 例全部必须有电子秤重量，重量是新采数据的主终点；
- 营养素只在具有可靠配方称重、包装标签或可审计数据库映射的子集上作为次要终点；
- 不能让 LLM 生成营养值后再把该值当作 ground truth；
- 对自制混合菜，如无法称量各原料和烹饪增减水/油，应标为 `nutrient_gt_unavailable`，不能用粗略数据库值伪装精确真值；
- 新数据的核心 Agent 结论可以先聚焦重量区间和动作成本，四营养素结论由 N5K/NV-Real 的可靠标签补充。

#### 推荐语义标签

- `morphology`：whole-solid / cut-solid / loose-pile / compact-mixture / liquid / soup；
- `container`：none / plate / bowl / cup / package；
- `preparation`：raw / boiled / steamed / fried / baked / mixed；
- `hidden_liquid`：none / low / high；
- `added_fat`：none / low / high / unknown；
- `primary_food_code`：FNDDS/本地食物编码；
- `reference_available`；
- `depth_available`。

#### 数据采集原则

- 同一食物在全部动作条件下保持重量不变；
- 参照物与食物尽量处于同一平面；
- 不改变食物形态后再补拍；
- 先完整采集所有动作，后续离线模拟 Agent 只看到其选择的子集；
- 真实在线用户实验再让 Agent 决定实际采哪些信息；
- 若涉及参与者，先确认伦理审批/知情同意要求；
- 去除 EXIF、面部、地址和个人健康信息。

### 8.4 为什么必须尽量使用配对动作数据

若把 N5K 的深度收益、NV-Real 的多视角收益和 ECUSTFD 的参照物收益直接横向比较，动作与数据集完全混杂：

```text
“reference 更好”可能只是 ECUSTFD 更容易，
而不是 reference 动作本身更好。
```

所以：

- 公开数据用于逐工具机制证据；
- 同实例多动作数据用于训练统一路由策略；
- 不能把跨数据集差异当作动作因果效应。

---

## 9. 数据划分与防泄漏协议

### 9.1 推荐主协议

如有 400 个配对实例，使用 5 折外层评估：

1. 每折 20% 作为外层 test；
2. 其余 80% 再按组分为 policy 和 calibration；
3. base 模型尽量固定使用公开数据训练，不消耗新数据；
4. 每个实例只作为外层 test 一次；
5. 汇总 400 个 out-of-fold test 预测。

每折内部示例：

- 200 个 policy；
- 120 个 calibration；
- 80 个 test。

若样本量允许，calibration 应提高到 150–200。

### 9.2 分组键

必须按以下最高层级划分：

- N5K：官方 split 下的 plate/incremental-scan group；同一扫描的全部帧与裁剪不可拆分；
- NV-Real：dish id；
- MetaFood3D：object id；
- ECUSTFD：food instance id；
- 新数据：recipe/physical instance id；
- 用户实验：participant id 与 meal id 双重检查。

### 9.3 禁止事项

- 同一菜不同视角跨 split；
- 同一物体渲染帧跨 split；
- 用 test 标签选“最好视角”；
- 用 \(D_{\mathrm{cal}}\) 调 VOI 阈值；
- 看到主 test 结果后修改动作成本；
- LLM 在 test 阶段自动更新 prompt；
- 根据 test 覆盖重新选择 q；
- 用真值食物类别作为部署输入，却不把“询问类别”计入动作成本。

### 9.4 预注册

在第一次运行外层 test 前冻结：

- 主要假设；
- 主指标；
- 动作集合；
- 成本定义；
- 最大交互次数；
- 策略超参数；
- 校准协议；
- 失败判据；
- 主表结构。

建议保存为：

`agent_extension/preregistration/caia_mm_prereg_v1.md`

并计算 SHA256。

---

## 10. 各工具的具体实现

### 10.1 BaseMMCFATool

输入：

- RGB 或 RGB-D；
- 可选多视角；
- 可选结构化语义。

输出：

- 重量点预测；
- 四营养素点预测；
- 原始候选区间；
- 特征向量；
- 模型分歧；
- 工具版本。

第一阶段复用：

- `Nutri_0811/aec_s_v2_experiment.py`；
- `Nutri_0811/nutrient_conformal_final.py`；
- `Nutri_0811/test_100_samples_final.py`。

重构要求：

- 提取纯推理 API；
- 所有路径由配置传入；
- 禁止 import 时直接运行实验；
- 模型只加载一次；
- 输出统一 dataclass/schema。

### 10.2 DomainAuditTool

建议同时提供三个信号：

1. 特征空间 OOD 距离；
2. N5K/NV-Real 域分类概率；
3. 输入质量：模糊、过暗、食物截断、mask 占比。

注意：

- 历史实验表明 k-NN 距离不是可靠密度误差代理；
- 但它仍可能是域外检测信号；
- 必须单独验证 OOD AUROC/AUPRC 与灾难误差召回率；
- 不得因为 OOD score 高就声称知道误差具体有多大。

建议输出：

```python
DomainAudit(
    ood_score: float,
    ood_pvalue: float,
    quality_flags: list[str],
    recommended_action_mask: dict[str, bool],
)
```

### 10.3 MorphologyTool

类别：

- rigid/whole；
- prepared/mixed；
- loose/stacked；
- liquid/soup；
- unknown。

实现优先级：

1. 使用已有人工类别映射作为 oracle/规则基线；
2. 冻结视觉编码器 + 线性分类头；
3. 本地 VLM zero/few-shot；
4. 商业 VLM 作为外部对照。

主路由不能直接使用测试真值形态。人工映射若依赖真实类别，类别必须来自用户语义动作或部署可用识别器。

### 10.4 ReferenceScaleTool

第一版复用 ECUSTFD 的逻辑：

1. 检测硬币/标准卡片；
2. 估计 mm/pixel；
3. 得到定标面积；
4. 结合形态/类别高度先验估体积；
5. 结合训练集 LOO 密度估重量；
6. 输出工具特有的不确定性。

新数据建议使用 ArUco 或标准卡片，原因：

- 尺寸统一；
- 检测稳定；
- 不受不同国家硬币尺寸影响；
- 可估计透视姿态；
- 更容易做质量检查。

失败条件：

- 参照物与食物不共面；
- 参照物遮挡；
- 检测置信不足；
- 透视过大；
- food mask 不完整。

失败时必须返回结构化失败，而不是继续给伪精确数值。

### 10.5 MultiViewTool

第一版先实现简单、稳定、可解释的方法：

1. 每视角独立预测；
2. 对点预测取稳健加权中位数；
3. 权重由图像质量和 mask 完整度决定；
4. 视角间差异作为状态特征；
5. 用多视角专用校准器输出区间。

然后再比较：

- feature pooling；
- attention pooling；
- 多视角几何；
- TSDF/点云体积。

项目已证明几何不是所有食物的主瓶颈，因此不建议一开始投入复杂 3D 重建。

### 10.6 DepthTool

必须重训干净对照：

- `RGB-clean`；
- `RGBD-clean`；
- 相同训练样本；
- 相同随机种子；
- 相同增强；
- 相同骨干；
- 同一 hold-out。

建议报告：

- depth 的平均收益；
- 按形态收益；
- 按 OOD 程度收益；
- 按重量区间收益；
- 设备可用率；
- 单位推理成本。

### 10.7 SemanticDensityTool

该工具不直接让 LLM 猜密度，而是：

1. 将用户答案解析为固定 ontology；
2. 在训练数据/FNDDS 中检索相应密度和营养分布；
3. 与视觉密度预测做训练集上确定的融合；
4. 对融合后的终端输出重新校准。

推荐结构：

```text
自然语言答案
  → 受约束解析器
  → {food_code, morphology, preparation, hidden_liquid, added_fat}
  → 版本化营养数据库
  → 密度先验分布
  → 与视觉预测融合
```

必须记录：

- 数据库版本；
- 匹配候选；
- 最终 food code；
- 拒绝匹配原因；
- 用户是否确认。

Open-KNEAD 应作为该工具的强基线之一。

---

## 11. 策略学习方案

### 11.1 构建 counterfactual action table

对 \(D_{\mathrm{policy}}\) 中每个完整配对 episode，离线运行所有合法动作，得到：

```text
episode_id
state_before
action
prediction_after
interval_score_after
absolute_error_after
action_cost
tool_success
```

因为动作数据是成对采集的，同一 episode 可以计算真实收益：

\[
G_{i,a}
=
\mathrm{IS}_{i,\mathrm{before}}
-\mathrm{IS}_{i,\mathrm{after}(a)}
\]

构建该表时还要防止策略训练内部的乐观偏差：

- `prediction_after` 应尽量使用 grouped out-of-fold 预测，而不是工具模型对自身训练 episode 的拟合值；
- provisional route interval 必须由 \(D_{\mathrm{policy}}\) 内部交叉拟合或独立 `route-fit` 子集生成；
- 各动作的 provisional interval 必须按相同名义覆盖和相同评分定义比较，不能让欠覆盖动作凭借虚假窄区间获得高 gain；
- 计算 \(G_{i,a}\) 可以使用 policy 标签，因为它就是策略监督目标，但这些 episode 之后不得进入 terminal calibration 或 test；
- 动作失败也必须保留为结果，不能只在成功样本上计算平均收益；
- 对未选择动作的完整结果只用于离线训练和 oracle 分析，部署轨迹绝不能访问。

### 11.2 动作收益模型

为每个动作拟合：

\[
\hat G_a(S_t)=f_a(S_t)
\]

第一版建议：

- LightGBM/HistGradientBoosting；
- 限制深度；
- grouped cross-validation；
- 预测均值收益和失败概率；
- 使用校准后的分位数回归作为保守收益下界是可选增强。

选择动作时可使用保守价值：

\[
\mathrm{LCBVOI}_a
=
\hat G_a-z\hat\sigma_a-\lambda c(a)
\]

这样 Agent 在收益不确定时更倾向 STOP/ABSTAIN。

### 11.3 两步策略

最大交互步数建议先设为 2。

若动作存在互补性，例如：

- 先询问“刚体还是混合物”；
- 再决定参照物还是视角；

可在 \(D_{\mathrm{policy}}\) 上学习：

\[
Q_2(S,a)
=
\mathbb{E}
\left[
G(S,a)
+\max_{a'}Q_1(S',a')
\right]
\]

若两步模型对独立 policy validation 的收益小于 2%，保留一阶策略，避免复杂化。

### 11.4 LLM planner 基线

给 LLM 相同状态摘要和动作白名单，让它选择动作。要求：

- temperature 固定；
- prompt 固定；
- 模型版本固定；
- 每 episode 重复 3–5 次测方差；
- 不提供真值；
- 不提供其他方法测试结果；
- 无效 JSON 记为失败；
- token、延迟和费用计入成本。

混合 Agent：

- 数值控制器生成合法候选和收益；
- LLM 只能在收益接近的候选间解释/表达；
- 最终动作仍经过确定性 validator。

---

## 12. 实验设计

### 12.1 E0：资产与协议审计

目的：保证新增实验建立在可复现资产上。

检查：

- 权重和缓存校验和；
- 绝对路径；
- 数据集对象数；
- dish/object 分组；
- 3ch/4ch 增强一致性；
- 现有关键结果复现；
- 每个动作是否具有成对结果。

Gate：

- 关键资产缺失则停止主实验；
- 同实例跨 split 则重建划分；
- 原结果无法在合理容差内复现则先排查。

### 12.2 E1：单工具价值实验

分别测量：

1. RGB → RGB-D；
2. 单视角 → 双视角；
3. 无参照 → 有参照；
4. 无语义 → 有结构化语义；
5. 域外直接输出 → 域外拒答/主动获取。

每个动作报告：

- 点估计误差变化；
- 区间宽度变化；
- interval score 变化；
- 成功率；
- 时间成本；
- 按形态、域和重量分组的收益。

Go/No-Go：

- 若所有动作的 oracle 平均收益均低于 5%，Agent 路由没有必要；
- 若某一动作对超过 90% 样本都最优，固定策略可能足够；
- 只有动作最优性存在显著异质性时，统一 Agent 才有方法学价值。

### 12.3 E2：Agent 与固定协议比较

基线：

1. 初始单图 MM-CFA；
2. Always semantic；
3. Always reference；
4. Always second view；
5. Always depth；
6. Fixed sequence；
7. Random，预算匹配；
8. Uncertainty threshold；
9. Myopic VOI；
10. 两步 CAIA-MM；
11. LLM-only planner；
12. Oracle best action；
13. 如数据量允许，ACO/BRiG-AFA/NM-PPG；
14. FoodAgent 风格的 LLM 专家路由；
15. 最佳固定工具子集的联合 max-score、Bonferroni 和 DCP-MT 类校准；
16. ALMA 的回归化多模态获取版本；
17. C-IP/PIDA 风格的主动问题策略；
18. BCEA 风格的完整策略后重校准；
19. RouteCert 风格的终止路径/模态组认证。

若无法复现某篇近邻工作的完整代码，必须实现其核心决策原则的匹配版本，并在附录列出哪些组件保持一致、哪些因分类到回归的任务差异被替换；不能只用一个弱阈值基线代替全部直接近邻。

关键方法消融：

- CAIA-MM full；
- 去掉因素化不确定性，只输入总区间宽度；
- 去掉 OOD 因素；
- 去掉动作失败模型；
- 去掉成本项；
- 去掉保守 LCB，使用期望收益；
- 一步 vs 两步；
- 路由用真值 morphology 的 oracle 上界 vs 可部署 morphology 预测。

公平性：

- 比较相同平均动作次数；
- 比较相同真实秒数；
- 比较相同设备条件；
- 所有策略使用同一 base 模型；终端校准比较中使用相同协议，全状态校准作为单独安全性维度报告；
- Oracle 只作为上界，不参与排名。

### 12.4 E3：共形有效性

比较：

1. 动作前冻结 q 直接复用；
2. 策略和 q 使用同一校准集；
3. 两阶段/四段隔离；
4. 冻结策略终端校准；
5. 全工具状态 × 五输出联合包络；
6. 路径条件校准；
7. anytime alpha-spending；
8. 五个边际区间、Bonferroni、联合 max-score 与 DCP-MT 类误差预算。

主结论应回答：

- 自适应动作是否导致 naive 区间欠覆盖；
- 终端校准是否恢复覆盖；
- 全状态包络能否在允许换策略/提前停止时维持五输出联合覆盖；
- 恢复覆盖付出多少宽度；
- 终端校准与全状态包络的安全—效率差距有多大；
- 路径条件校准是否在样本足够时提高效率。

### 12.5 E4：跨域与少样本恢复

设置：

1. N5K 校准 → NV-Real 零目标校准；
2. OOD gate + abstention；
3. Agent 主动额外视角；
4. 目标域 calibration \(n=25,50,100,200\)；
5. N5K-only、NV-only、joint base 模型分别测试；
6. 新采集域作为第二个未知域。

报告：

- coverage；
- risk-coverage curve；
- abstention rate；
- catastrophic error recall；
- target calibration sample efficiency；
- 域外动作分布。

不能把“提高到接近 90%”自动解释成理论覆盖，除非对应校准协议满足假设。

### 12.6 E5：LLM 作用消融

比较：

- 无 LLM：固定 ontology 菜单；
- LLM parser；
- LLM planner；
- LLM planner + validator；
- 本地模型 vs 外部模型；
- 真实用户回答 vs oracle 标签；
- 0%/10%/20% 受控错误、含糊回答和拒答；
- 同图自问自答 vs 真正新增用户/传感证据。

指标：

- ontology 解析准确率；
- 无效动作率；
- 数值性能；
- 轨迹复现率；
- token/延迟；
- 用户理解度。

如果无 LLM 与混合 Agent 数值性能相同，应如实写：

> LLM 改善交互自然性，但核心统计收益来自受约束的主动获取策略。

这比强行宣称 LLM 提高重量精度更可信。

### 12.7 E6：用户负担实验

推荐：

- 30–50 名参与者；
- within-subject crossover；
- 每人完成固定协议和 Agent 协议；
- 顺序随机；
- 不收集不必要的健康信息。

指标：

- 完成时间；
- 重拍次数；
- 放弃率；
- 主观负担；
- 问题可理解性；
- 是否愿意日常使用；
- Agent 建议动作的遵循率。

主要目的不是证明临床效果，而是验证“节省交互成本”的系统主张。

### 12.8 E7：失败分析

至少建立以下失败类别：

- OOD 未被检测；
- 参照物不共面；
- 视角重复、没有新增几何信息；
- 食物遮挡；
- 松散堆叠；
- 汤汁/隐藏油；
- food code 匹配错误；
- 用户回答含糊；
- LLM 解析错误；
- 策略错误选择；
- 工具本身失败；
- 校准路径样本不足。

每类给出：

- 数量；
- 误差；
- 选择的动作；
- Oracle 最优动作；
- 是否应拒答；
- 可修复性。

---

## 13. 指标与统计检验

### 13.1 点估计指标

重量：

- MAE；
- median absolute error；
- MAPE：\(\operatorname{mean}(|e_i|/y_i)\)；
- WAPE：\(\sum_i|e_i|/\sum_i y_i = \mathrm{MAE}/\operatorname{mean}(y)\)；
- sMAPE；
- \(R^2\)；
- Spearman \(\rho\)。

正文不再单独使用含义不唯一的 “PMAE”；历史表必须注明它实际对应 MAPE 还是 WAPE。

营养素：

- MAE；
- normalized MAE；
- sMAPE；
- 对接近零营养素单独报告；
- 不使用会被零值放大的原始 MAPE 作为唯一指标。

### 13.2 区间指标

- empirical coverage；
- mean/median width；
- relative width；
- Winkler/interval score；
- normalized Winkler；
- joint coverage；
- conditional coverage audit；
- calibration error。

### 13.3 Agent 指标

- 平均动作数；
- 平均真实耗时；
- 每动作成功率；
- STOP@0 比例；
- ABSTAIN 比例；
- cost-adjusted interval score；
- oracle action regret；
- action routing entropy；
- risk-cost curve；
- area under risk-cost curve；
- 帕累托前沿。

### 13.4 统计方法

主比较应在 episode 层面配对：

1. 配对 bootstrap，至少 10,000 次；
2. episode/dish/object 为重采样单位；
3. 多视角不能按 frame 独立 bootstrap；
4. 覆盖率报告 Clopper-Pearson 或 Wilson CI；
5. 覆盖差异用配对置换或 McNemar 类检验；
6. 连续配对指标用 permutation/Wilcoxon，并同时报告均值差与 CI；
7. 多个次要比较用 Holm 校正；
8. 报告效应量，不只报告 p 值；
9. N5K 以 plate/scan、NV-Real 以 dish、MF3D 以 object、ECUSTFD 以 food instance 为唯一独立单位；
10. 覆盖主张同时报告 episode-level Clopper–Pearson 区间和预注册的单侧 exact binomial 检验；
11. 不能把“经验覆盖约 90%”解释成其 95% 下置信界也达到 90%；若这是验收目标，应提前提高校准目标并做样本量设计。

### 13.5 主终点

建议只设一个主终点：

> 在经验覆盖与名义 90% 一致的前提下，CAIA-MM 相对最佳成本匹配固定策略的 normalized interval score 差异。

次要终点：

- 重量 MAPE 与 WAPE；
- 实际交互时间；
- 域外 catastrophic error；
- 营养素 nWinkler；
- 用户负担。

这样可以减少“挑最好指标”的审稿质疑。

---

## 14. 成功、失败和降级判据

### 14.1 强成功

同时满足：

1. 冻结策略终端校准具有正确协议和覆盖；
2. nWinkler/normalized interval score 相对最佳固定成本基线改善 ≥10%；
3. 平均动作数减少 ≥25% 或平均耗时显著降低；
4. 路由符合可解释规律：刚体偏 reference/depth，制备食物偏 semantic/abstain；
5. 外层测试结果在预注册后一次性得到；
6. LLM 移除后统计核心仍成立；
7. 若把全状态包络列为方法贡献，其五输出联合覆盖成立且区间仍具有实际可操作性。

### 14.2 可发表的中等成功

- 数值精度与 always-acquire 相当；
- 但动作/时间成本显著下降；
- 覆盖保持；
- 能清晰说明哪些样本不需要额外输入。

这仍是有效贡献：Agent 的价值是节省采集，而不一定刷新全输入精度。

### 14.3 Agent 不成立的判据

出现任一情况应停止包装为统一 Agent：

1. Oracle best-action 与最佳固定动作差异 <5%；
2. 全工具 oracle 相对 RGB-only 的区间/误差改善不足预注册阈值；
3. 某一固定动作在 >90% 样本上最优；
4. 策略不优于简单 uncertainty threshold；
5. 动作收益模型在 policy validation 上 \(R^2\le0\)；
6. 终端校准后区间宽度抵消全部动作收益；
7. 用户实际完成时间没有下降；
8. 收益只在真值类别、真值框或无噪声模拟回答下存在；
9. LLM-only 结果高度随机且 validator 后没有额外价值。

若只有全工具状态包络过宽，而冻结策略终端校准仍有效且高效，应删除“策略可变/任意允许状态”的强主张，保留较窄的 \(R^\pi\) 版本。

### 14.4 降级方案

#### 降级 A：Conformal Triage

若多工具路由无优势，只保留：

- 自动输出；
- 请求额外信息；
- 拒答。

研究问题改为“可靠选择性预测和人机升级”。

#### 降级 B：单一动作的自适应触发

若只有 reference 有用：

> 研究何时需要参照物，而不是对所有样本强制放参照物。

若只有语义有用：

> 研究不确定性驱动的最小澄清问题。

#### 降级 C：独立后续论文

若 Agent 需要新数据和大量用户实验，且会使当前 MM-CFA 论文过于庞杂，应将其拆成独立 follow-up，而不是勉强加入已有论文。

---

## 15. 推荐代码目录

建议新增：

```text
mm-cfa-repro/
├── agent_extension/
│   ├── README.md
│   ├── configs/
│   │   ├── paths.yaml
│   │   ├── policy_v1.yaml
│   │   └── costs_v1.yaml
│   ├── preregistration/
│   │   └── caia_mm_prereg_v1.md
│   ├── src/
│   │   ├── schemas.py
│   │   ├── build_group_manifest.py
│   │   ├── episode_store.py
│   │   ├── controller.py
│   │   ├── trace_logger.py
│   │   ├── tools/
│   │   │   ├── base_mm_cfa.py
│   │   │   ├── domain_audit.py
│   │   │   ├── morphology.py
│   │   │   ├── reference_scale.py
│   │   │   ├── multiview.py
│   │   │   ├── depth.py
│   │   │   └── semantic_density.py
│   │   ├── policy/
│   │   │   ├── features.py
│   │   │   ├── gain_model.py
│   │   │   ├── myopic_voi.py
│   │   │   └── two_step.py
│   │   ├── conformal/
│   │   │   ├── terminal.py
│   │   │   ├── all_tool_states.py
│   │   │   ├── path_mondrian.py
│   │   │   ├── anytime.py
│   │   │   └── multioutput.py
│   │   └── llm/
│   │       ├── parser.py
│   │       ├── verbalizer.py
│   │       └── validator.py
│   ├── experiments/
│   │   ├── e00_asset_audit.py
│   │   ├── e01_clean_rgb_rgbd.py
│   │   ├── e02_action_table.py
│   │   ├── e03_tool_value.py
│   │   ├── e04_train_policy.py
│   │   ├── e05_terminal_conformal.py
│   │   ├── e06_all_state_conformal.py
│   │   ├── e07_main_benchmark.py
│   │   ├── e08_cross_domain.py
│   │   ├── e09_llm_ablation.py
│   │   └── e10_make_figures.py
│   ├── tests/
│   ├── results/
│   └── traces/
└── Agent创新点_详细完整实施方案.md
```

### 15.1 配置原则

- 所有路径从 `MM_CFA_ROOT` 推导；
- 数据集版本和 checksum 写入配置；
- 模型 hash 写入结果；
- seed 列表固定；
- 不在代码中硬编码 API key；
- 结果文件包含配置快照；
- 每个 experiment 只负责一个研究问题。

---

## 16. 建议执行顺序

### Phase 0：文献与预注册，2–3 天

1. 系统检索 Agent nutrition、AFA、conformal active perception；
2. 建 related-work spreadsheet；
3. 最终确认不使用绝对“first”或给出严格检索依据；
4. 写预注册；
5. 冻结主终点和 Go/No-Go。

产出：

- `literature_matrix.csv`；
- `caia_mm_prereg_v1.md`。

### Phase 1：资产和干净基线，3–5 天

1. 核验大文件和 checksum；
2. 统一路径；
3. 从原始数据重建 plate/scan/dish/frame manifest；
4. 分开复算 MAPE、WAPE 与 MAE，标记历史污染结果；
5. 修复有限样本 q 与增强 bug；
6. 同协议重训 clean RGB/RGB-D；
7. 建统一推理 API。

Gate：

- clean 对照可复现；
- 所有数据分组无泄漏；
- base 工具稳定。

### Phase 2：公开数据单工具实验，1–2 周

1. N5K depth action；
2. NV-Real view action；
3. ECUSTFD reference action；
4. MetaFood3D morphology/geometry action；
5. semantic oracle 和可部署 semantic tool 分开；
6. 形成 action-value heterogeneity 图。

Gate：

- 至少两种动作在不同子群上各自占优；
- Oracle 路由相对固定动作有 ≥5% 收益；
- 否则降级为单动作触发研究。

### Phase 3：配对数据 pilot，1 周

采 60 个实例：

1. 完整动作采集；
2. 测动作成功率；
3. 测时间成本；
4. 修订 ontology；
5. 估算主指标差值方差；
6. 做正式样本量分析。

Gate：

- 动作成功率 ≥90%；
- 问题理解率 ≥90%；
- 每个主要子群至少有初步样本；
- 无法满足则先改采集协议。

### Phase 4：正式数据采集，2–4 周

目标 400 个独立实例。

同时：

- 自动去 EXIF；
- 实时检查缺失动作；
- 秤重双次确认；
- 每天备份；
- 冻结数据字典；
- 记录排除原因。

### Phase 5：策略和最终校准，1–2 周

1. 建 counterfactual action table；
2. 训练 myopic VOI；
3. 测两步互补；
4. 冻结策略；
5. 在独立 \(D_{\mathrm{cal}}\) 跑完整 Agent；
6. 拟合 terminal conformalizer；
7. 对动作完备 episode 拟合 all-tool-state joint conformalizer；
8. 锁定所有文件 hash。

### Phase 6：一次性主测试，2–3 天

1. 运行外层 test；
2. 不在看到结果后改阈值；
3. 生成主表；
4. 运行预注册统计；
5. 如失败，按预注册解释，不回头选择更有利的口径。

### Phase 7：LLM 和用户消融，1–2 周

1. 无 LLM；
2. 本地 LLM；
3. 商业 LLM；
4. LLM-only planner；
5. 用户 crossover。

### Phase 8：论文整合，1–2 周

重写摘要、贡献、方法、实验、讨论和局限。

总周期：

- 仅公开数据 feasibility 版本：约 4–6 周，不足以单独支撑强方法创新；
- 含正式配对数据和用户实验：约 8–12 周。

---

## 17. 建议命令接口

以下命令是后续应实现的接口，不是当前已经存在的脚本：

```bash
export MM_CFA_ROOT=/home/mm-cfa-repro

python -m agent_extension.experiments.e00_asset_audit \
  --config agent_extension/configs/paths.yaml

python -m agent_extension.experiments.e01_clean_rgb_rgbd \
  --seeds 42 43 44 45 46

python -m agent_extension.experiments.e02_action_table \
  --split policy

python -m agent_extension.experiments.e03_tool_value \
  --bootstrap 10000

python -m agent_extension.experiments.e04_train_policy \
  --policy myopic_voi \
  --max-steps 2

python -m agent_extension.experiments.e05_terminal_conformal \
  --alpha 0.10 \
  --split calibration

python -m agent_extension.experiments.e06_all_state_conformal \
  --alpha 0.10 \
  --joint-outputs weight calories fat carbs protein \
  --split calibration

python -m agent_extension.experiments.e07_main_benchmark \
  --split test \
  --locked

python -m agent_extension.experiments.e08_cross_domain \
  --target-calibration-sizes 0 25 50 100 200

python -m agent_extension.experiments.e09_llm_ablation \
  --modes none parser planner hybrid

python -m agent_extension.experiments.e10_make_figures
```

`--locked` 模式应检查：

- preregistration hash；
- policy hash；
- calibrator hash；
- 数据 split hash；
- model hash；
- 若任一不一致则拒绝运行主测试。

---

## 18. 论文如何整合

### 18.1 不建议简单增加“第九个贡献”

当前论文已有较多贡献和外部分析。再机械增加一个 Agent 模块会让叙事过散。

建议把整篇论文重组为三层：

1. **Estimate**：MM-CFA 点估计和 AEC-S 区间；
2. **Diagnose**：密度、尺度、域偏移和共形失效边界；
3. **Act**：CAIA-MM 主动获取缺失信息并在成本约束下停止。

这样 Agent 是前面诊断结果的必然落点，而不是突然出现的热门附加件。

### 18.2 建议新标题

#### 标题 A

> From Uncertainty Quantification to Active Information Acquisition: A Conformal Agent for Food Weight and Nutrient Estimation

#### 标题 B

> CAIA-MM: Cost-Aware Conformal Active Sensing for Reliable Food Weight and Nutrient Estimation

#### 标题 C

> Beyond Single-Image Food Analysis: A Conformal Agent that Acquires Missing Physical and Semantic Information

#### 标题 D（文献审计后最稳妥）

> Mechanism-Aware Active Food Measurement with Post-Policy Conformal Risk Control

### 18.3 建议贡献重写

可压缩为四项：

1. 提出 MM-CFA 与 AEC-S，对食物重量和营养素进行分层共形预测；
2. 通过跨域、3D、密度和参照物实验识别尺度—密度—域偏移三类信息瓶颈；
3. 提出 CAIA-MM，根据期望信息价值在参照物、视角、深度、语义和拒答间做成本受控路由；
4. 建立两层选择安全校准：冻结策略终端校准追求效率，全工具状态 × 五输出联合包络支持允许状态内换路由/提前停止；系统评估覆盖、效率、用户成本和域外失效。

第 4 项应写成完整系统的有效性与审计贡献，不应声称“策略后重校准”本身是新的共形理论；理论价值需要与连续多输出、物理—语义混合状态和实证 benchmark 共同成立。

### 18.4 建议新增章节

```text
3.X Problem Formulation: Active Food Information Acquisition
3.X CAIA-MM Agent
3.X Cost-Aware Policy Learning
3.X Terminal-Trajectory Conformal Calibration
3.X All-Tool-State Joint Conformal Envelope
4.X Interactive/Paired Dataset Protocol
5.X Tool Value and Action Heterogeneity
5.X Cost–Accuracy–Coverage Benchmark
5.X Cross-Domain Triage and Calibration Recovery
5.X LLM and User-Burden Ablations
6.X What the Agent Acquires and Why
```

### 18.5 建议图

1. CAIA-MM 架构与一次完整轨迹；
2. 不同形态下各动作的真实收益；
3. risk-cost 帕累托曲线；
4. 固定策略、naive adaptive、terminal conformal、all-state envelope 的 coverage-width 对比；
5. Agent 动作分布按 rigid/prepared/OOD 分层；
6. 目标域校准样本量—覆盖恢复曲线；
7. 用户时间和动作次数对比；
8. 典型成功/失败轨迹。

### 18.6 摘要中可使用的结构

结果未完成前不要填假数字。建议模板：

> Existing food analysis systems passively estimate weight and nutrients from a fixed input, even when the uncertainty is caused by missing physical or semantic information. We introduce CAIA-MM, a cost-aware conformal agent that decides, for each meal, whether to stop, request a reference object, acquire an additional view or depth, ask a structured semantic question, or abstain. The policy is trained independently of calibration. We evaluate both efficient terminal-trajectory calibration and an all-tool-state envelope that jointly audits all predefined states and five outputs under exchangeable episodes. Across [datasets], CAIA-MM achieves [result] at [cost], outperforming fixed and budget-matched acquisition protocols. The learned routing is physically interpretable: [result]. Under domain shift, the agent [result], while correctly withholding coverage claims without target-domain calibration.

### 18.7 投稿方向与所需证据强度

- **方法/融合方向期刊**：需要严格算法形式化、AFA/ALMA/MiCP/DCP-MT 类强基线、终端与全状态联合覆盖证明及跨数据集实验；仅有系统演示不够。
- **应用 AI 期刊**：需要完整系统、消融、复现、成本和失败分析；CAIA-MM 完整版较匹配。
- **医学信息学/数字健康期刊**：除算法外通常还需要真实用户、营养师或临床相关验证；不能只用模拟用户。
- **食品/营养领域期刊**：需要强调称量协议、营养 ground truth、实际使用负担和营养学意义，而不是只强调 Agent 热点。

可能方向可从 Information Fusion、Pattern Recognition、Expert Systems with Applications、Engineering Applications of Artificial Intelligence、IEEE Journal of Biomedical and Health Informatics、Artificial Intelligence in Medicine 等期刊的最新 scope 中筛选。正式选择前应检查当年征稿范围、近期同类论文和数据/伦理要求，不应只按影响因子决定。

---

## 19. 审稿人最可能提出的问题与预先应对

### Q1：这只是把 if-else 包装成 Agent 吗？

应对：

- 明确定义 observation/action/state/cost/transition/stopping；
- 学习实例级动作价值；
- 与固定规则和 threshold 比较；
- 展示同预算优势和 oracle regret。

### Q2：为什么一定需要 LLM？

应对：

- 核心策略不依赖 LLM；
- LLM 只处理语义解析和自然语言交互；
- 提供无 LLM 完整复现；
- 把 LLM 消融作为诚实结果。

### Q3：自适应停止后共形保证还成立吗？

应对：

- 策略在 calibration 前冻结；
- 对完整终端轨迹计算分数；
- 若允许部署时换路由/提前停止，则使用全工具状态 × 五输出最大分数；
- 给出交换性条件和证明；
- 任意用户停止提供 alpha-spending baseline。

### Q4：你是不是又在 calibration 上调策略？

应对：

- 四段隔离；
- preregistration/hash lock；
- \(D_{\mathrm{cal}}\) 只计算最终 q；
- 主 test 只运行一次。

### Q5：各动作来自不同数据集，比较不公平

应对：

- 公开数据只做单工具机制验证；
- 统一路由主结果使用同实例配对动作数据；
- 不做跨数据集动作因果比较。

### Q6：Agent 改善只是因为获得更多信息

应对：

- “always acquire all” 是精度上界基线；
- Agent 的主张是以更低成本接近全信息性能；
- 与相同成本的 fixed/random/threshold 比较。

### Q7：为什么不用普通 active feature acquisition？

应对：

- 加入 ACO、BRiG-AFA 或 NM-PPG 基线；
- 说明异构工具、失败动作、人机成本和终端共形是区别；
- 若通用 AFA 更好，应如实采用其策略并保留领域和校准贡献。

### Q8：90% 只是边际覆盖，困难子群仍可能失效

应对：

- 明确不声称任意条件覆盖；
- 报告形态、域、动作路径和重量区间审计；
- 样本足够时路径 Mondrian；
- 小样本回退全局。

### Q9：域偏移下为什么还谈共形？

应对：

- 明确零目标校准不提供覆盖承诺；
- 报告 OOD、拒答和少样本恢复；
- 使用现有 NV-Real 崩溃结果说明边界，而非隐藏。

### Q10：语义问题是否直接泄露答案？

应对：

- 预先定义允许问题；
- 禁止询问重量/营养数值；
- 把用户答案视为有成本的新特征；
- semantic oracle 与真实 parser 分开报告。

### Q11：参照物只适合 ECUSTFD 刚体食物

应对：

- 正是 Agent 路由的动机；
- 不对 prepared food 强制使用；
- 按形态报告动作收益；
- prepared food 转向 semantic/abstain。

### Q12：400 个新样本是否太小？

应对：

- 主 base 模型使用公开数据；
- 新数据主要训练低容量策略；
- grouped nested cross-fitting；
- 报告覆盖 CI；
- 不做过度细分；
- 如资源允许扩展到 800。

### Q13：LLM/API 会不会不可复现？

应对：

- 核心结果无 LLM；
- 固定 prompt/model snapshot/temperature；
- 保存结构化输入输出；
- 本地模型基线；
- API 结果不参与共形 q 的在线修改。

### Q14：用户动作成本是人为设定的吗？

应对：

- 采集真实秒数和失败率；
- 报告多组 \(\lambda\) 敏感性；
- 展示完整帕累托曲线，不只选一个成本权重。

### Q15：论文已经太多内容，Agent 会不会使主线失焦？

应对：

- 用 Estimate–Diagnose–Act 重构；
- 压缩失败实验到附录；
- 若无法形成完整闭环，则拆为独立 follow-up。

### Q16：FoodAgent、PIDA、C-IP、ALMA、BCEA、RouteCert、MiCP 已覆盖大量相邻思想，新颖性在哪里？

应对：

- 不声称首个营养 Agent、食品主动澄清、共形主动获取、策略后校准或自适应停止；
- 把贡献限定为连续重量—营养五输出、尺度—几何—密度机制、真实物理/语义工具、成本和失败 benchmark；
- 与这些工作的核心策略建立匹配基线，而不是只在 related work 中口头区分；
- 若完整 benchmark 和差异化机制不成立，主动收缩为领域应用贡献。

### Q17：全状态最大分数是否理论正确但区间宽到无用？

应对：

- 同表报告 \(R^\pi\)、\(R^\forall\)、alpha-spending、Bonferroni 和 DCP-MT 类方法；
- 报告联合覆盖、各维宽度、超矩形体积和成本，而不只报告 coverage；
- 预注册实用宽度/interval-score 失败阈值；
- 若包络过宽，删除策略可变的强主张并降级到冻结策略终端校准。

---

## 20. 资源预算

### 20.1 计算

若复用视觉模型：

- 策略训练：CPU 数小时；
- 全动作缓存：视数据量约数小时到 1 天；
- clean 3ch/4ch 重训：预计单卡 RTX 3090 数小时级；
- 多视角简单融合：CPU/GPU 数小时；
- 复杂 3D 不列为第一阶段必需；
- 本地 7B 级 VLM 可选，需单独记录显存和量化。

### 20.2 API

核心结果应可在 0 API 成本下复现。

若使用外部 LLM：

- 每 episode 限制一次解析或解释；
- 保存 token 数和实际账单；
- 不在文档中写死美元价格，因为服务价格会变化；
- 以 token、调用数、延迟作为可复现成本。

### 20.3 人工

正式 400 例数据预计是最大成本：

- 每例 3–5 分钟采集和称重；
- 总原始采集约 20–35 小时；
- 再加质检、标签和重拍；
- 建议分多天、多人并采用标准操作手册。

### 20.4 存储

- 图片和深度数据预计几十 GB；
- 原图与脱敏图分离；
- 轨迹只存结构化信息和文件 hash；
- 训练中间特征可压缩为 `.npz`/parquet；
- 发布前检查数据许可。

---

## 21. 前 72 小时可直接执行的清单

### 第 1 天：资产审计

- [ ] 按 `sha256sum.txt` 核验所有权重和缓存；
- [ ] 确认 `/home/mm-cfa-repro` 与脚本绝对路径关系；
- [ ] 建立 base 模型训练样本 membership 清单，禁止已见样本进入 Agent 主测试；
- [ ] 从原始数据重建 plate/scan/dish/frame 分组 manifest；
- [ ] 分别按 MAPE、WAPE 核对 9.44% 重切分结果与原始 hold-out，停止混用 PMAE；
- [ ] 将 AEC 权重拟合数据与最终 q 校准数据真正分离；
- [ ] 将所有共形 q 改为有限样本顺序统计量并加入单元测试；
- [ ] 统计 N5K dish、NV dish、MF3D object、ECUSTFD instance；
- [ ] 检查多视角是否按实例可配对；
- [ ] 建立 `agent_extension/` 空目录和配置规范；
- [ ] 固化原论文当前结果快照。

### 第 2 天：动作可行性表

- [ ] 定义 5 个动作的输入、输出、失败条件；
- [ ] 确认每个公开数据集能支持哪些动作；
- [ ] 写统一 episode schema；
- [ ] 确定哪些结果是真实工具、哪些只是 oracle；
- [ ] 设计 60 例 pilot 的采集 SOP；
- [ ] 确定标准参照物。

### 第 3 天：预注册与最小原型

- [ ] 冻结 Level A 的主指标和 Go/No-Go；
- [ ] 实现无学习的固定路由基线；
- [ ] 用现有 CSV 构造 reference 与 domain 的第一版 action table；
- [ ] 画每类动作收益分布；
- [ ] 判断是否存在足够 action heterogeneity；
- [ ] 若没有，立即降级为单动作自适应触发，不投入统一 Agent。

---

## 22. 最终决策建议

### 22.1 如果目标是“尽快投稿”

做最小版本：

1. 使用公开数据；
2. 只选择两类动作：`REQUEST_VIEW/REFERENCE` 与 `ABSTAIN`；
3. 使用简单 VOI 策略；
4. 做严格终端轨迹校准；
5. 不强调 LLM；
6. 将 Agent 作为“从诊断到行动”的扩展实验。

预计 4–6 周。

但在 ALMA、C-IP、BCEA、RouteCert 和 PIDA 等直接先例已经存在的情况下，这个版本只能定位为 **feasibility/应用扩展**，不足以单独支撑强方法创新。若当前 MM-CFA 主文已接近投稿，宁可把它留作后续工作，也不要为追求热点而降低整篇论文的严谨性。

### 22.2 如果目标是“显著提高期刊档次”

做完整版本：

1. 采集 400+ 动作完备实例；
2. 支持 reference、view、depth、semantic、abstain；
3. 学习成本敏感策略；
4. 同时比较冻结策略终端共形与全工具状态 × 五输出联合包络；
5. 增加用户负担实验；
6. 与 FoodAgent/Open-KNEAD、ALMA、C-IP/PIDA、BCEA、RouteCert、MiCP 思路、AFA 和 LLM planner 比较；
7. 用 Estimate–Diagnose–Act 重构整篇论文。

预计 8–12 周。

### 22.3 最重要的底线

这项工作的创新不能写成：

> “我们加入了一个 Agent，因此系统更智能。”

应写成：

> “现有不确定性分析定位了实例级缺失信息。我们把这种诊断转化为机制感知的主动食物测量：系统只在预期收益超过真实交互成本时获取额外物理或语义证据，并在独立数据上审计终端策略及预定义工具状态的连续多输出覆盖。”

只有同时做到“物理—语义动作差异、同餐配对数据、真实用户/工具成本、直接近邻基线、独立校准、联合覆盖审计和失败边界”，这个 Agent 创新点才足以支撑期刊论文。

---

## 23. 已核验的重点参考方向

> 以下为方案制定时核验到的相关工作入口。2026 年的多篇工作仍是预印本，投稿前需再次核对正式发表状态、版本和引用格式。

### Agent 与营养估计

1. Coburn et al. **Open-KNEAD: Knowledge-grounded Nutrition Estimation via Agentic Decomposition**. arXiv:2607.12911, 2026.  
   https://arxiv.org/abs/2607.12911
2. Jiang et al. **OmniFood-Bench: Evaluating VLMs for Nutrient Reasoning and Personalized Health Advice**. arXiv:2607.08423, 2026.  
   https://arxiv.org/abs/2607.08423
3. Xu. **A Closed-Loop Multi-Agent System Driven by LLMs for Meal-Level Personalized Nutrition Management**. arXiv:2601.04491, 2026.  
   https://arxiv.org/abs/2601.04491
4. Yao et al. **CaLoRAify: Calorie Estimation with Visual-Text Pairing and LoRA-Driven Visual Language Models**. arXiv:2412.09936, 2024.  
   https://arxiv.org/abs/2412.09936

### 共形预测与 Agent/主动感知

5. Loaiza-Ganem et al. **Conf-Gen: Conformal Uncertainty Quantification for Generative Models**. arXiv:2605.28920, 2026.  
   https://arxiv.org/abs/2605.28920
6. Huang et al. **Conformal Policy Learning for Sensorimotor Control Under Distribution Shifts**. arXiv:2311.01457, 2023.  
   https://arxiv.org/abs/2311.01457
7. Podolinsky et al. **Co-GLANCE: Uncertainty-Aware Active Perception for Heterogeneous Robot Teaming**. arXiv:2606.09919, 2026.  
   https://arxiv.org/abs/2606.09919
8. Gollapudi et al. **Efficient Online Conformal Selection with Limited Feedback**. arXiv:2605.14953, 2026.  
   https://arxiv.org/abs/2605.14953

### 主动特征获取

9. Li and Oliva. **Active Feature Acquisition with Generative Surrogate Models**. arXiv:2010.02433.  
   https://arxiv.org/abs/2010.02433
10. Valancius, Lennon, and Oliva. **Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition**. arXiv:2302.13960.  
    https://arxiv.org/abs/2302.13960
11. Aronsson and Chehreghani. **Non-Myopic Active Feature Acquisition via Pathwise Policy Gradients**. arXiv:2605.05511, 2026.  
    https://arxiv.org/abs/2605.05511
12. Schütz et al. **AFABench: A Generic Framework for Benchmarking Active Feature Acquisition**. arXiv:2508.14734.  
    https://arxiv.org/abs/2508.14734
13. Feng et al. **BRiG-AFA: Bellman Risk-to-Go Learning for Non-Myopic Active Feature Acquisition**. arXiv:2608.02305, 2026.  
    https://arxiv.org/abs/2608.02305

### 食物份量与项目数据

14. Thames et al. **Nutrition5K: Towards Automatic Nutritional Understanding of Generic Food**. CVPR, 2021.
15. Chen et al. **MetaFood3D: Large 3D Food Object Dataset with Nutrition Values**. arXiv:2409.01966, 2024.  
    https://arxiv.org/abs/2409.01966
16. Herse et al. **NutritionVerse-Real: A Fully Annotated Real-World Food Dataset for Nutrition Planning**. arXiv:2401.08598, 2024.  
    https://arxiv.org/abs/2401.08598
17. Ma et al. **MFP3D: Monocular Food Portion Estimation Leveraging 3D Point Clouds**. arXiv:2411.10492.  
    https://arxiv.org/abs/2411.10492
18. Vinod and Zhu. **Food Portion Estimation: From Pixels to Calories**. arXiv:2602.05078, 2026.  
    https://arxiv.org/abs/2602.05078

### 审计后补充的强近邻

19. Zhang et al. **FoodAgent: A Multi-modal Mixture of Experts Reasoning Agent for Divide-and-Conquer Food Nutrition Estimation**. IEEE BSN, 2025.  
    https://openreview.net/forum?id=a7Yl2jzBfw
20. Hoarau et al. **Reducing Aleatoric and Epistemic Uncertainty through Multi-modal Data Acquisition**（ALMA）. Machine Learning, 2026.  
    https://doi.org/10.1007/s10994-026-07042-w
21. Zhou et al. **Adaptive Stopping for Multi-Turn LLM Reasoning**（MiCP）. arXiv:2604.01413, 2026.  
    https://arxiv.org/abs/2604.01413
22. Schlembach, Smirnov, and Winands. **Dynamic Conformal Prediction for Multi-Target Regression: Optimising Informational Efficiency under Joint Validity**（DCP-MT）. PMLR 266, 2025.  
    https://proceedings.mlr.press/v266/schlembach25a.html
23. Liang, Zhu, and Barber. **Conformal Prediction after Data-Dependent Model Selection**. arXiv:2408.07066.  
    https://arxiv.org/abs/2408.07066
24. Nguyen et al. **Conformal Cross-Modal Active Learning**. arXiv:2603.23159, 2026.  
    https://arxiv.org/abs/2603.23159
25. **Conformal Information Pursuit for Interactively Guiding Large Language Models**（C-IP）. NeurIPS, 2025.  
    https://proceedings.neurips.cc/paper_files/paper/2025/hash/823a3d2cf462fb815978314023e48f65-Abstract-Conference.html
26. **Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Models**（BCEA）. arXiv:2606.16667, 2026.  
    https://arxiv.org/abs/2606.16667
27. **Guaranteed Adaptive Modality Acquisition: When the Policy Chooses Its Own Calibration Group**（RouteCert）. arXiv:2608.15520, 2026.  
    https://arxiv.org/abs/2608.15520
28. **PIDA: Proactive Interactive Dietary Assessment from Food Images via Active Clarification**. KSEM conference program；正式年份与版权元数据投稿前复核。  
    https://www.springerprofessional.de/pida-proactive-interactive-dietary-assessment-from-food-images-v/52957762
29. **FoodSage: Addressing Recognition Uncertainty in Automated Dietary Monitoring through Human-Robot Dialogue**. CHI EA, 2025.  
    https://doi.org/10.1145/3706599.3720175
30. **An Agentic Vision-Language Pipeline for Interactive Nutritional Estimation from Food Images**. CVPRW, 2026.  
    https://openaccess.thecvf.com/content/CVPR2026W/MTF/html/Bhatambarekar_An_Agentic_Vision-Language_Pipeline_for_Interactive_Nutritional_Estimation_from_Food_CVPRW_2026_paper.html
31. **Active Feature Acquisition: A Survey**. arXiv:2502.11067.  
    https://arxiv.org/abs/2502.11067

---

## 24. 投稿前最终核对表

- [ ] 没有把普通模块重命名为 Agent；
- [ ] Agent 有明确定义的状态、动作、成本、工具和停止条件；
- [ ] 至少存在两个具有异质收益的动作；
- [ ] 有相同预算的 fixed/random/threshold 基线；
- [ ] 已与 ALMA、C-IP/PIDA、BCEA、RouteCert 的核心思想做匹配比较；
- [ ] 有 oracle best-action 上界；
- [ ] 策略训练与最终 calibration 完全隔离；
- [ ] 冻结策略后对终端轨迹重新共形化；
- [ ] 若声明策略可变或任意允许状态，已验证全工具状态 × 五输出联合包络；
- [ ] q 使用有限样本顺序统计量而非默认线性插值；
- [ ] 明确边际覆盖与条件覆盖的差异；
- [ ] 明确逐输出边际覆盖与五输出联合覆盖的差异；
- [ ] 域外无目标校准时不声称 90% 保证；
- [ ] GT volume、GT category 等 oracle 信息没有伪装成部署工具；
- [ ] 明确区分真实新增观测与模型基于同一张图的自问自答/额外推理；
- [ ] 已发布带 plate/scan/dish/object ID 的分组 manifest；
- [ ] 所有多视角和 incremental scan 按物体/菜品/扫描分组；
- [ ] 3ch/4ch 对照没有混用增强 bug；
- [ ] 正文不再混称 PMAE，MAPE 与 WAPE 的公式和数值均分开；
- [ ] 没有把 empirical shrinkage 或 empirical ceiling 写成未经证明的 Bayesian/information-theoretic 结论；
- [ ] LLM 不直接输出最终克数或共形 q；
- [ ] LLM 模型、prompt、temperature、成本均记录；
- [ ] 近零营养素不只报告 MAPE；
- [ ] 主终点、阈值和统计方法已预注册；
- [ ] 用户交互成本来自真实时间而非只用主观权重；
- [ ] recent preprint 的发表状态在投稿前重新核验；
- [ ] 新颖性不依赖“终端重校准本身首创”或泛化的 Agent first claim；
- [ ] 若统一 Agent 无优势，按预设判据降级，而非事后挑子群；
- [ ] 全部结果、失败和排除样本可审计。

