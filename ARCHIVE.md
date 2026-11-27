# NutriAgent snapshot catalog

本分支只是目录，没有嵌套历史树。要拿回某一刻的完整树，检出对应标签，例如：

```
git fetch origin tag snapshot/icdl
git checkout snapshot/icdl
```

这会把工作区停在该标签的那一次提交上，用来查看那棵树。它不移动 `main`，也不移动 `digi-nutri`。

下面列出打标签时的全部 105 个分支尖端。每个标签名是 `snapshot/` 加上当时的分支名，SHA 是该尖端的完整提交。

`snapshot/main` 是空初始提交 `d43cb084eba7693a45c114dfdf06b258611463f4`。

`snapshot/digi-nutri` 是纳入 #97 之前的尖端 `3c0488a242041ac7d8da6567c0f350c9e9aba63a`。后来的 `digi-nutri` 分支已经向前，同划分标尺的文档记账在分支上，不在这个标签上。

`snapshot/cursor/same-split-dpf-swin-8e07` 是 #97 头 `a89be031c6be18f8e8df8568d64e0e907ebb1897`。这是同划分记录，不是通过。

不要删除这些标签。本目录不删除分支，也不关闭拉取请求。

## 空的 main

当前 `main` 只有空初始提交。检出这个标签得到的是空树，不是 2026-11-13 之前的仓库。

1 个标签。

- `snapshot/main` `d43cb084eba7693a45c114dfdf06b258611463f4` — Reset main to an empty initial commit for the new scheme 空初始提交。main 保持这个提交。旧树不在这里。

## 旧 main / VDR-agent 时期

换到新方案之前的主线和 VDR-agent 工作。`snapshot/archive/main-before-new-scheme-2026-11-13` 与 `snapshot/archive/vdr-old-plan-2026-11-13` 指向同一提交。不要从这些标签继续旧方案。

75 个标签。

- `snapshot/archive/main-before-new-scheme-2026-11-13` `28bd3c9cae9795c822dc0af68f612ec1fd811e15` — Experiment A: dual-view fusion does not stably beat top-view mass (#73) 旧 main。与 snapshot/archive/vdr-old-plan-2026-11-13 同一提交，含实验 A / PR #73。
- `snapshot/archive/v4.0` `ce1dc0616002a2f76fcffcc8a3ef87addd8c8358` — Save the v4.0 plan, code, and results before starting a new scheme. 换方案前保存的 v4.0 计划、代码和结果。不是 digi-nutri 上的 v4 STATUS。
- `snapshot/archive/vdr-old-plan-2026-11-13` `28bd3c9cae9795c822dc0af68f612ec1fd811e15` — Experiment A: dual-view fusion does not stably beat top-view mass (#73) 旧 VDR 方案。与 snapshot/archive/main-before-new-scheme-2026-11-13 同一提交。
- `snapshot/cursor/abstention-calib-aurc-056d` `f3e47713203bb313ff79fd0bee9f9329c52a95de` — Calibrate hold-out AURC with the committed kcal total
- `snapshot/cursor/b1-b2-faithful-scaffold-2b24` `45101b8d24355f434a73eb1e0beba8d0dc77eb39` — Add DietAI24 and FoodAgent baseline scaffolds
- `snapshot/cursor/b1-dish-form-portion-f6db` `6d3b5c3f56a6a7829eea2174bbe54f83efcec0fa` — Harden B1 dish-form binds and dense portions.
- `snapshot/cursor/bc-build-features-e6d2` `5c8262c4a5dc8d0ff949fb612b37e9d75fc934ea` — Add pre-action BC features and a numpy logistic toy
- `snapshot/cursor/bc-mlp-policy-ac66` `219cc4837c4e3cc0bcee84ffb73daa37025da6e9` — Add a fixed numpy MLP behavior-cloning policy
- `snapshot/cursor/cafe2-energy-audit-96a0` `c2f503bd43b5aa15149f55e4119bb49a972a28c5` — Audit Cafe2 held-out energy with food codes held fixed.
- `snapshot/cursor/cafe2-energy-audit-idle-5281` `69358cc0467e9f0f58e6c63e484c822e51b38bbc` — docs: mark the Cafe2 energy audit done and leave coding idle
- `snapshot/cursor/cafe2-energy-audit-unlock-00ba` `5165de9fe9286eec3b9f90ee239ad735b8a0bf6b` — docs: mark gate ③ done and unlock the Cafe2 energy audit
- `snapshot/cursor/cleanup-obsolete-vdr-paths-2866` `fee9cf65a9242dfc311d4c56526c5b20dd2929de` — Remove obsolete weak-step dumps and dead doc paths.
- `snapshot/cursor/closed-loop-bc-20261104` `75f1205f4e93026ec7b7fe0598999c662ca256fd` — Land state-aware closed-loop BC train on weak steps
- `snapshot/cursor/coverage-constrained-docs-f138` `7f7bc6dd14854a72ca8ff39ec16c39238a18fb93` — docs: freeze #53 as a dev checkpoint and set gate ①
- `snapshot/cursor/coverage-early-stop-02a8` `7e6b6c583c0cdb643ff6b28fd8f97fa8535fb77c` — Block STOP while a named component still has no mass write.
- `snapshot/cursor/dagger-round-1-9c23` `0ba473c687ab16963829e57e6d6a626c7d58dc77` — Add one DAgger round using the weak-step teacher
- `snapshot/cursor/docs-first-next-step-1dfd` `609489daa78beb31528f6977272007c78af58dc1` — Put the docs-first rule in the collab protocol and STATUS.
- `snapshot/cursor/docs-gate-2-done-idle-9f17` `2ebe00d63cddd643ca798fb43d2f4ad61e501c83` — docs: mark gate ② done and idle until ③ is unlocked
- `snapshot/cursor/docs-overhead-rgb-eval-bb25` `83179de38fc138ee5f4d9a613803bc98e7be0995` — Document the server run on the overhead RGB test pool.
- `snapshot/cursor/docs-status-fixrate-mae-2b9c` `2b7cce2ce867e30427108bfc6fdde331fda78e50` — docs: fix FixRate script path in STATUS
- `snapshot/cursor/docs-status-fixrate-mae-c99a` `3a9ceeac19180832bc25b3c08275d8b3ce6d23be` — docs: STATUS next FixRate then MAE/AURC
- `snapshot/cursor/docs-status-q44-unlock-p0d-6562` `b7007f7ed563342493af262ff286094a507cc581` — docs(STATUS): Q calibrate #44 done → unlock P0d
- `snapshot/cursor/evening-feasibility-audit-docs-6707` `e1482e465a4f5c9f7efe1a9c5f2e2c6ba95b0995` — docs: record the evening feasibility audit and leave code idle
- `snapshot/cursor/experiment-a-n5k-dual-view-2117` `2d6a3236b7d734aff5e8713803e8e2ae5664ecfd` — Report the Nutrition5k dual-view mass comparison and stop.
- `snapshot/cursor/fixrate-worsenrate-343e` `8a78d3027dff21f0d1842370dfab91b9c86d0b10` — Add hold-out FixRate−WorsenRate and tool cost under K
- `snapshot/cursor/food-id-density-light-84cb` `603a5ce4f40f45a663a2d57d9238413cbfb863e5` — Align multi-component food codes to the mass-estimator calorie total
- `snapshot/cursor/food-id-semantic-audit-a8a6` `940c60cacc20359571044de76cb227f6f42be196` — Audit the frozen #51 food-code swaps for identity and macros.
- `snapshot/cursor/harden-b1-image-path-2e47` `06d85fe916fcbd1d5c9dd51866e9287eb93d94c1` — Harden B1 lookup against duplicate portions and prepared-food codes.
- `snapshot/cursor/heldout-plan-eval-891c` `e3639ee4b4ce0f11c53092ed028de89398853dfc` — Score the frozen #53+#51 path on the cafe2 held-out.
- `snapshot/cursor/mae-aurc-holdout-e69f` `319c8b630674c7d470c167509110518c9fe1cc61` — Add hold-out energy MAE, Coverage, Selective MAE, and AURC
- `snapshot/cursor/massonly-diagnostic-done-idle-d44e` `9b71d3bc01270243828fe2a7abb2e8a330cf6a54` — docs: mark the Mass-only energy-path diagnostic done and leave coding idle
- `snapshot/cursor/massonly-energy-path-4e0a` `4a5bec848f42c907d3b28efec1c9bd8879b1ebd1` — Diagnose the frozen Mass-only energy path on Cafe2.
- `snapshot/cursor/massonly-energy-path-status-ca05` `7104ab0bb1b6281ae4158a9b48c525847ac5619d` — docs: unlock the Mass-only energy-path diagnostic
- `snapshot/cursor/on-policy-rgb-test-de89` `9ee4f3abf2ec94f305aa2b209ae3443ca46882ed` — Report on-policy rgb_test energy with local Qwen3.8-27B-VL
- `snapshot/cursor/ours-old-scaffold-banners-8045` `dca708100b29220168b23cdff31a31137e032ac8` — docs: mark Ours-old scaffolds and sync the PLAN MVP gate
- `snapshot/cursor/overhead-507-b0-b1-analysis-30eb` `5df83e9de074849296a0f4f55f17eda2f65182e6` — [metric] Document the overhead-RGB 507 B0/B1 gap
- `snapshot/cursor/p0a-tool-state-audit-e684` `5ad40cbe71663fa632a61b5a26ae89d8ab94bf6a` — Audit whether closed-loop tools change dish energy and mass.
- `snapshot/cursor/p0b-mass-estimator-fcc8` `32f704c8b5e4104f31f321678e6eef87656bdb43` — Add an RGB-only mass estimator and call it from RE_PORTION.
- `snapshot/cursor/p0c-uncertainty-qphi-1b6b` `f27ca2ed0efd93833bedb1950f0022b3ada3dd25` — Add supervised Q/Cost selection on the U_* error state
- `snapshot/cursor/p0d-error-audit-0749` `ec809f6b7d1deebccab4f698eb8e4bdae222932b` — Audit why P0d cut mass error without beating direct-model energy
- `snapshot/cursor/p0d-same-budget-eval-8e8a` `4b05de956a5397de1702438f76e065681860ac27` — Add the same-budget full-set comparison for mass and calibrated Q/Cost
- `snapshot/cursor/phase1-cross-domain-docs-1f7d` `296a421174fb855ad1406439408798198a43f9f0` — docs: unlock Phase 1 cross-domain diagnostics 旧方案的跨域诊断文档。不是 DiGI-Nutri Phase 1。
- `snapshot/cursor/q-cost-terminal-margin-7f09` `4a9cd6496056045bc3008be3d5ba6187e61d9075` — Calibrate unresolved Q/Cost stops so retrieval and mass run
- `snapshot/cursor/qphi-counterfactual-gate1-e76e` `02aa5fbf3da0f222d48a4348c370c8d7021a8d2b` — Score frozen Q with one-step counterfactuals
- `snapshot/cursor/seed-vdr-agent-3d8c` `e32c5ee7491d986370a4762741c5b48f195e9af9` — Seed the VDR plan, code skeleton, and collab docs.
- `snapshot/cursor/status-b1-b2-landed-66f1` `335a2878fb2bd4d4ae83f0ca9860786f489acd01` — Record B1/B2 scaffold landing and retarget next steps.
- `snapshot/cursor/status-b1-b2-scaffold-f34e` `edb98c450627b3f34ae7e13b53124e15dd069ec8` — Run offline P0 smokes before B1/B2 scaffolding.
- `snapshot/cursor/status-coverage-53-done-b9e2` `0a2bc740723f15dbb1e5c946028cfb826311977c` — STATUS: coverage #53 done; idle until co-decide
- `snapshot/cursor/status-error-audit-48-done-b345` `b92c43ef8cffa318d27aa01692def5ef57624435` — docs(STATUS): error audit #48 done, await co-decision next step
- `snapshot/cursor/status-gate1-done-idle-5cac` `4ece1092d973e4d0a5239174521539aeabfdaf04` — docs: mark gate ① done and keep coding idle
- `snapshot/cursor/status-mae-aurc-next-0367` `b18a94e701bde7597d17aa0f43c93e3fc94b73b4` — docs(STATUS): MAE/AURC #28 done; next step choose A/B/C
- `snapshot/cursor/status-p0-abstention-a-ef13` `d70777474008c2fd1bf2bf47b94c4c99f0687bb5` — docs(STATUS): lock P0=A abstention calib, then P1=B; C parallel
- `snapshot/cursor/status-p0-smoke-066b` `1eb2d12749d38202e3b453bf5579d479c97f7e7f` — Record finished offline P0 smoke and point next work at B1/B2 scaffolds.
- `snapshot/cursor/status-p0a-done-p0b-mass-651f` `8e9edf65b11767c340a3507a72917a6c1374c971` — docs(STATUS): P0a #37 done; next step P0b Mass Estimator
- `snapshot/cursor/status-p0b-done-p0c-next-21a5` `5b61259210cae3249b8d17fae5aabbdfe1a4dd72` — docs(STATUS): P0b #39 done → next P0c U_*+Q_φ
- `snapshot/cursor/status-p0c-done-p0d-gate-b6b6` `aa24403743a61755a8a48b75e25d7b8d013b491a` — docs(STATUS): P0c #41 done; gate A/B before P0d
- `snapshot/cursor/status-p0d-done-error-audit-0469` `b3602714cd30e6dcfa44753147dd60f47b160bc9` — docs(STATUS): P0d #46 done → next error audit (no P1)
- `snapshot/cursor/status-p1-b-after-a-ab3b` `901069f0d884005c197f2154af7773426a97737c` — docs(STATUS): mark P0=A (#31) done; next step P1=B on-policy
- `snapshot/cursor/status-p1b-complete-79ed` `e3e5194a894829ba54278d80c23d1d0826e24d7c` — docs(STATUS): record P1=B #34 complete, next awaits 调研 lock
- `snapshot/cursor/status-p1b-local-qwen-vl-84c3` `64ce9ab88029f58c28f669b6535f05c75543d00c` — docs(STATUS): P1=B names local Qwen3.8-27B-VL for on-policy rollout
- `snapshot/cursor/status-unlock-coverage-early-stop-0038` `0ab022419254244814f5006e487c801f226eedde` — STATUS: food-id #51 done; unlock coverage and early-stop; defer P1
- `snapshot/cursor/status-unlock-food-id-density-dbc5` `e9d7a4e1b1190f440919dc49df1b921845b2f826` — STATUS: unlock food-id/density light fix after Triste co-decide
- `snapshot/cursor/unlock-experiment-a-d481` `cf3f4018ea74d025c86058df0a5631a3d01d106e` — docs: unlock only the Nutrition5k dual-view experiment
- `snapshot/cursor/unlock-gate-2-food-id-audit-1778` `1b7444b5e19c6ca4694469de06899fd9b1d75617` — docs: unlock gate ② food-id semantic audit (③–⑦ still locked)
- `snapshot/cursor/unlock-gate-3-heldout-e4c6` `b7e7752a2ea84c46b23358294822d3776f3fbf85` — docs: unlock gate ③ external held-out planning (④–⑦ still locked)
- `snapshot/cursor/unlock-q-calibrate-before-p0d-4ef6` `e9cf8a07b8b962df00ef78f930696ecadd80c7b7` — docs(STATUS): unlock Q calibrate before P0d (B)
- `snapshot/cursor/vdr-active-specialist-docs-3649` `aa6397e524c618747e2edf84120b9f03f8390c35` — docs: lock VDR active specialist plan; audit-first P0 gate
- `snapshot/cursor/vdr-evidence-aware-plan-d736` `dd74c131d383bc00f64f0ef27607eca1432101e3` — Align PLAN and STATUS with evidence-aware closed-loop.
- `snapshot/cursor/vdr-p0-staging-53a8` `6f6c1cd60d4ee306538a1cfea48e171a951bf3b7` — Land VDR P0 offline eval scripts and an untrained state-feature sketch.
- `snapshot/cursor/weak-steps-fit-all-f588` `b694b5274db9072a738c50356a032a582e082e38` — [smoke] Record train-only weak-step full fit pool
- `snapshot/cursor/weak-steps-holdout-f588` `c4ad6b4db8efbacb1d1b6230bb51fd04f6ef934a` — [smoke] Record train-only weak-step hold-out pool
- `snapshot/cursor/weak-steps-n1024-f588` `d7e08f1cc6c39df46aaecfd844eac5021c966c6b` — [smoke] Record train-only weak-step fit n=1024
- `snapshot/cursor/weak-steps-n256-f588` `fa40fd51f1b39f320f0575c7fe1f9368fe9e45fc` — [smoke] Record train-only weak-step fit n=256
- `snapshot/cursor/weak-steps-n8-smoke-f588` `bd3315e1449ed680bfb037d18fdceca69b9ff448` — [smoke] Record train-only weak-step fit n=8
- `snapshot/cursor/weak-steps-train-565d` `f0f57c93df47151d88c170efeef31c8dab64310e` — [W3] Add a train-only weak step teacher

## icdl

ICDL 线。长寿命分支是 `icdl`。

4 个标签。

- `snapshot/cursor/icdl-phase1-sequence-audit-4baf` `872b66a01b6d795c3d2edf08faf6d41c8ded74d7` — Record the finished incremental-sequence counts and fill per-pair visual rows.
- `snapshot/cursor/icdl-phase2-light-31ad` `ed7813dfbb23bf0dd72f0c656102a9d361b10b61` — Report the ICDL phase 2 light models and stop.
- `snapshot/cursor/unlock-icdl-phase2-docs-3ccb` `03644b7e2d284eb94d7752bf6d2534191311c1cb` — docs: unlock ICDL Phase 2 light experiments on icdl, lock later stages.
- `snapshot/icdl` `6ea6e849b88288b66d76fd063509e9b394daa7b4` — Report ICDL phase 2 light models and stop (#76)

## u1-granularity

U1 粒度线。长寿命分支是 `u1-granularity`。

3 个标签。

- `snapshot/cursor/u1-phase1-data-audit-2142` `9d7822f9e8f3597889d00de76b31b3c40d096170` — Audit Hand annotations against Nutrition5k overhead RGB.
- `snapshot/cursor/u1-phase1-status-95c5` `432f750c1d971d6fa335b881a20fe109027bc19e` — Document the U1 Phase 1 audit unlock.
- `snapshot/u1-granularity` `a733641f16433bea6593abd35ee96031cd79e49a` — Document the U1 Phase 1 audit unlock (#77)

## u2-partial-label

U2 部分标注线。长寿命分支是 `u2-partial-label`。

5 个标签。

- `snapshot/cursor/u2-partial-label-status-0596` `82e4b18159ae7a2ef75cb3a28f2b3aedc9890f50` — docs: unlock only the U2 Phase 1 partial-label audit
- `snapshot/cursor/u2-phase1-partial-label-audit-7453` `e97d2a122ced64395eeaca5377e90567b97df035` — Record the phase-1 audit pull request in the report.
- `snapshot/cursor/u2-phase2-b123-contrasts-2ee6` `bc1da0e22aca57d819f94d633b3f01ad72f7e37a` — Record the B1/B2/B3 holdout contrast and stop.
- `snapshot/cursor/u2-phase2-status-bcdb` `7be53b8f9e2ff08f4f754ce32072337f75d99e3b` — docs: unlock only the U2 Phase 2 B1/B2/B3 contrasts
- `snapshot/u2-partial-label` `ed405bf3d6532e40ec39005612c9179b090b0706` — docs: unlock only the U2 Phase 2 B1/B2/B3 contrasts (#81)

## DiGI-Nutri v1–v3 记录

DiGI-Nutri v1、v2、方案 A、同设置双教师、统一教师和 v3 第一步。这些都是记录。#84、#86、#88、#91、#93、#95 不是通过。

13 个标签。

- `snapshot/cursor/digi-nutri-phase1-abd6` `96843c84091d181b3adca13a3e3e1c60f4604e9d` — Record the single Phase 1 holdout read. #84，未合并的失败记录，不是通过。
- `snapshot/cursor/digi-nutri-phase1-status-54fa` `6a4bb537175d59ed30a5c05739429311872848f3` — docs: put spatial-affinity distillation in the Phase 1 gate v1 Q0 停止后的门禁文本。不是通过。
- `snapshot/cursor/digi-nutri-v2-q0-a40e` `d28b89a58aa82666b0543c0caad63a4a8c1462a4` — Record the v2 Q0 train-gate stop. #86，train_gate_fail_stop，不是通过。
- `snapshot/cursor/digi-nutri-v2-status-6f69` `bb62bc9b533f8fb7725577ed37c41afd17c2db4d` — docs: record the v1 Q0 stop and gate only the v2 teachers 记下 v1 Q0 停止，并只给 v2 教师开门。不是通过。
- `snapshot/cursor/digi-nutri-v3-step1-de25` `8ea7ac399972477ef07dedec7229767b666bbfff` — Record the single v3 step 1 train metrics and official holdout. #95，相对过、绝对没过，q0_fail_stop_wait_triste，不是通过。
- `snapshot/cursor/plan-a-tgeo-9ade` `09e479805bebba25f44633fddc9498def1837246` — Record the Plan A train gate and the one holdout read. #88，留出 q0_fail_stop_wait_triste，不是通过。
- `snapshot/cursor/same-setting-dual-teacher-187d` `5a7ea64f03f41e2836f1131d329303154259c5e9` — Record the same-setting rerun train metrics and the single official holdout. #91 分支尖端。开放拉取请求记录的头是父提交 f92b298ee14ab5e763f6e7be7eef5446a7b6b6b6，比尖端落后 1。留出失败，不是通过。
- `snapshot/cursor/status-dual-teacher-rerun-eec4` `c0e0e78c86cb8ec3714de6ff33749376be0a7a56` — docs: record the Plan A holdout stop and unlock the dual-teacher rerun 记下方案 A 留出停止，并解锁同设置双教师重训。不是通过。
- `snapshot/cursor/status-keep-qwen-gpus-2e22` `664e800a9fbe89aae121dc6a222082acaa3df360` — docs: train on idle GPUs and leave local Qwen running DiGI 时期的文档：空闲卡训练，不停本地千问。不是模型结果。
- `snapshot/cursor/status-plan-a-tgeo-dd8d` `60866b30cec3cf66574c9185c11ef11b98e1cdc7` — docs: record the v2 Q0 train-gate stop and unlock Plan A 记下 v2 训练集门禁失败，并只解锁方案 A。不是通过。
- `snapshot/cursor/status-tmulti-unlock-5cf1` `44a70d2a3ce4c728247e1ca66a639c39e4a6315b` — docs: record the #91 holdout stop and unlock only T_multi 记下 #91 留出停止，并只解锁 T_multi。不是通过。
- `snapshot/cursor/status-v3-step1-b798` `7d7d60804f74fe0d26e15cdaca97d3d7bb76c64a` — docs: record the #93 Q0 failure and unlock only v3 step 1 记下 #93 的 Q0 失败，并只解锁 v3 第一步。不是通过。
- `snapshot/cursor/t-multi-unified-teacher-1389` `484de17e79e32eb85b7ae824928e85d3302c4497` — Record the single official T_multi holdout. #93，留出 q0_fail_stop_wait_triste，不是通过。

## v4 STATUS

打 `snapshot/digi-nutri` 时，`digi-nutri` 停在 v4 STATUS。这个标签必须留在纳入 #97 之前的尖端。它不包含后来的同划分数字记账。

2 个标签。

- `snapshot/cursor/status-v4-same-split-ruler-8e16` `e0aa5abe03c5b3aa8675f0951b96d60174f9b57d` — docs: record #95 Q0 failure and unlock only the v4 same-split ruler 记下 #95 的 Q0 失败，并只解锁同划分标尺。不是 #97 的 507 盘数字记录。
- `snapshot/digi-nutri` `3c0488a242041ac7d8da6567c0f350c9e9aba63a` — docs: record #95 Q0 failure and unlock only v4 same-split reproduction 打标签时的官方 v4 STATUS 尖端，在纳入 #97 之前。标签停在这里。后来的 digi-nutri 分支已经向前，不在这个标签上。

## #97 记录

同划分 DPF-Nutrition / Swin-Nutrition 重训一次、评测一次的记录。不是官方通过门禁，不解锁蒸馏、方案 B 或 v4 教师重训。

1 个标签。

- `snapshot/cursor/same-split-dpf-swin-8e07` `a89be031c6be18f8e8df8568d64e0e907ebb1897` — Record the single official holdout for DPF-Nutrition and Swin-Nutrition. #97 同划分重训一次、评测一次的标尺记录。不是论文公开测试数字，不是通过，也不是新的通过线。

## 其余 cursor/* 分支

没有归进上面任一条研究线的分支。

1 个标签。

- `snapshot/cursor/setup-cloud-dev-environment-9175` `57b9c672eaec576118daaaf62d7901b54d51e0a0` — Add Cloud Agent environment (CPU) for NutriAgent/MM-CFA 早期 Cloud Agent 环境脚手架（NutriAgent/MM-CFA）。不属于后面的研究线。
