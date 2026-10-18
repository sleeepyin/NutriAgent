# From Uncertainty to Action: Conformal Estimation, Information Diagnosis, and Active Acquisition for Food Weight and Nutrient Analysis

中文全文见 `docs/paper_cn.md`。两份正文使用同一套数字。

---

**Abstract**

Food weight and nutrient estimates used in clinical monitoring need finite-sample prediction intervals, not only point values. A single image is missing scale, density, and domain information, and a single global conformal rule is a poor fit to that heterogeneity. This paper organizes the problem into estimation, diagnosis, and action. The estimator, AEC-S, learns a convex combination of base conformal intervals inside each predicted-weight bin and stabilizes sparse bins by empirical shrinkage. Under the honest protocol, coverage on Nutrition5K is 90.4% at a mean width of 74 g (Winkler 181). The per-bin recalibrated variant AEC-S2m reaches 90.4%, 76 g, and Winkler 171 on the same protocol. Diagnosis uses four complementary datasets to separate scale ambiguity, within-class density dispersion, and domain shift: within-class density dispersion is 21.8%, comparable to the 4-channel strict hold-out MAPE of 22.0%; a coin reference brings end-to-end MAPE on rigid foods to 12.9%; leaving the training domain raises dish-level MAPE to 103.7% and drops a frozen nominal 90% interval to 32.8% coverage. The action layer, CAIA-MM, routes among stop, reference, view, depth, a semantic question, and abstention under a cost, then recomputes a conformal quantile on the frozen terminal trajectory. A public-data feasibility run does not show that adaptive routing beats the best fixed policy: myopic value of information has normalized Winkler 4.264 at cost 0.53, while always asking the semantic question has 3.776 at cost 0.91. The third layer is a feasibility study, not a closed accuracy claim.

---

## 1. Introduction

Dietary assessment needs the mass of a serving, its macronutrients, and a statement of how unreliable those numbers are. In clinical use, a point estimate without a coverage guarantee presents measurement error as fact. Conformal prediction supplies a distribution-free finite-sample guarantee when calibration and test points are exchangeable [1, 2]. Food images do not have one residual law: small, typical, and large servings differ, and prepared foods are missing different information from rigid foods.

A well-calibrated interval does not restore the mass variance that a single image never contained. The missing factors are measurable separately: absolute scale, within-class density, and whether the photograph comes from the training domain. Only after those factors are identified does it make sense to ask for a reference object, another view, a depth channel, a preparation label, or to abstain.

The paper claims three contributions, in that order.

1. **Estimate.** AEC-S learns per-bin convex weights over base conformal intervals and stabilizes small bins by empirical shrinkage. Honest-protocol coverage is 90.4% at mean width 74 g. Extended to four macronutrients under strict hold-out, coverage is 89.7–90.6% and normalized Winkler is 2.42.
2. **Diagnose.** Nutrition5K, MetaFood3D, ECUSTFD, and NutritionVerse-Real attribute monocular error to scale, density, and domain. A strong single-image 3D reconstructor does not cross the same empirical ceiling.
3. **Act.** CAIA-MM uses that diagnosis as its action space. The public-data, three-split evaluation is a feasibility study. It does not show that the adaptive policy beats the best fixed policy at matched cost.

Section 4.6 fixes the numerical protocol. The abstract, main tables, and conclusion use the honest intervals and strict hold-out MAPE. The development figure 9.44% is not in the abstract.

## 2. Related work

### 2.1 Food weight and nutrient estimation

Nutrition5K pairs overhead RGB-D captures with scale weights and nutrient labels [7]. Later work estimates portions by recognition, volumetric reconstruction, or direct regression, and usually reports point error only. Here the point predictor is the center of an interval, and the residual is attributed to a physical factor.

### 2.2 Conformal prediction

Split conformal takes a residual quantile on a calibration set [2]. Normalized conformal scales the score by the prediction [3]. Mondrian conformal calibrates inside bins [4]. Weighted conformal reweights scores by a density ratio when covariate shift is detectable [13]; without detectable shift it collapses to ordinary conformal. Adaptive ensemble conformal searches for a convex combination of intervals on the simplex. AEC-S makes those weights depend on predicted weight and shrinks them empirically toward a calibration-set prior. The global weights are not fit on the test set.

### 2.3 Active acquisition

References, extra views, and user questions add information the image does not contain. They are not applied as a fixed capture script. Diagnosis decides which food is missing which factor; a cost-constrained policy decides whether to pay for it. After the policy is frozen, the terminal quantile is fit on a disjoint calibration split.

### 2.4 Closest work

Relative to portion estimators that report only point error, coverage is a primary metric. Relative to one conformal rule, AEC-S may change the rule across predicted-weight bins, but the global weight is a calibration-set quantity. Relative to an appendix of external datasets, each dataset answers one question. Relative to a collected interaction study, CAIA-MM is evaluated on public proxy actions and is described at that scope.

## 3. System

The estimator returns weight, four macronutrients, and intervals at α = 0.10. The diagnostic layer does not train a new deployed model. It uses oracles, leave-one-out lookup, zero-shot 3D reconstruction, and frozen cross-domain calibration to assign residual error to scale, density, or domain. The action layer reads the interval and the factor, then chooses a request or stops.

| Dataset | Role | Question |
|---|---|---|
| Nutrition5K | In-domain system | What the interval system does under controlled RGB-D |
| MetaFood3D | Diagnosis | Whether density dominates, and whether appearance or single-image 3D fills it in |
| ECUSTFD | Constructive check | Whether rigid foods fall below the monocular ceiling once scale is restored |
| NutritionVerse-Real | Stress test | How far points and intervals move once exchangeability fails |

## 4. Estimate

### 4.1 Formulation

$(X, Y)$ is an RGB-D crop and its mass. The base predictor $\hat f$ returns $\hat y$. The interval $C(X)=[L(X), U(X)]$ should satisfy $\mathbb{P}(Y \in C(X)) \ge 1-\alpha$ when the calibration set and the test point are exchangeable. The quantile is an order statistic, not a linearly interpolated quantile. With $n$ calibration scores,

$$
k=\lceil (n+1)(1-\alpha)\rceil,\qquad \hat q = R_{(k)}.
$$

If $k>n$, the quantile is $+\infty$. The same rule is used for nutrients. Published tables were produced with interpolated quantiles. The canonical scripts now use the order statistic. The weight cache required to recompute the tables is not in this repository, so the text keeps the published honest-protocol numbers and names the quantile implementation that produced them.

### 4.2 Base predictor

The base predictor is an equal-weight ensemble of ResNet-18 and EfficientNet-B0, both modified for 4-channel input and trained with log-MSE on 3,260 Nutrition5K RGB-D crops. The strict hold-out uses 2,608 training crops and 652 test crops. On that split the 4-channel MAPE is 22.0%, and 18.4% on items of at least 100 g. An RGB-only retraining reaches 31.4%. The 9.4 percentage-point gap is the modality cost of dropping depth. Cross-domain numbers below use 31.4% as the in-domain RGB baseline.

MAPE 9.44% ± 1.05% (MAE 18.8 g) on ten 50/20/30 resplits is a development figure. Test dishes are drawn from the same pool as training. It is not a deployment accuracy and it is not in the abstract.

MAPE is $\mathrm{mean}(|e_i|/|y_i|)$, matching `sklearn.metrics.mean_absolute_percentage_error` and the MetaFood3D and ECUSTFD scripts. WAPE is $\sum |e| / \sum |y|$. Earlier drafts called both PMAE. This paper does not use PMAE. The figures 22.0%, 18.4%, 31.4%, and 9.44% are MAPE. A WAPE paired with the 22.0% hold-out was not archived, and none is invented here.

### 4.3 AEC-S

Candidate intervals are split, normalized, Mondrian, and a learned score. AEC searches the simplex with Dirichlet samples for a convex weight that keeps empirical coverage at or above $1-\alpha$ and minimizes mean width. The original search used the test set. That is an evaluation leak. The honest protocol fits the global weight on the calibration set.

AEC-S bins by predicted weight into $[0,50)$, $[50,100)$, $[100,200)$, $[200,500)$, and $[500,\infty)$. The local weight in bin $b$ shrinks toward the global weight:

$$
\lambda_b=\frac{n_b}{n_b+\tau},\qquad
\mathbf{w}_b=\lambda_b\mathbf{w}_b^{\mathrm{local}}+(1-\lambda_b)\mathbf{w}_{\mathrm{global}},
$$

with $\tau=30$. If $n_b<20$, the global weight is used. $\lambda_b$ is a sample-size mixing coefficient. It is not the miscoverage level $\alpha$, and it is not a Bayesian posterior. The earlier name “Bayesian shrinkage” is dropped.

Coverage computed by binning on true weight is a coverage audit, not a conditional guarantee. The conditioning variable is predicted weight.

### 4.4 Nutrients

A nutrient is mass times density. Direct regression predicts density and is mixed with a k-NN density at 0.60/0.40. On the ten-split calorie MAPE, the oracle is 7.5%, direct regression is 12.6%, k-NN is 24.6%, and a linear map from weight alone is 34.4%. Direct regression is 63% lower than the weight-only map. The gap from 12.6% to 7.5% is density error. Direct-regression MAPE is 24.3% for fat, 25.3% for carbohydrate, and 22.4% for protein.

Nutrient intervals (AEC-S2m-N) use five candidates: absolute split, relative split, log split, propagation of the weight interval, and weight-binned relative split. The 652 test crops are split evenly into calibration and evaluation, repeated for 10 trials. Point MAPE on this stricter protocol is 22.8%, 36.5%, 39.4%, and 33.7% for calories, fat, carbohydrate, and protein.

### 4.5 Results

**Table 1. Conformal baselines whose weights are not searched on the test set (10 trials, 50/20/30, α=0.10).** AEC and AEC-S are omitted because the historical rows searched global weights on the test set.

| Method | Coverage (%) | Width (g) | Winkler | At 90% |
|---|---|---|---|---|
| Split | 90.1 ± 1.2 | 78 ± 4 | 213 ± 32 | yes |
| Normalized | 90.3 ± 1.3 | 94 ± 7 | 191 ± 31 | yes |
| Mondrian | 90.3 ± 1.6 | 80 ± 7 | 187 ± 30 | yes |
| MLNS | 81.8 ± 5.4 | 77 ± 12 | 192 ± 30 | no |
| MTC | 98.0 ± 0.4 | 315 ± 40 | 352 ± 43 | yes, not usable |

MLNS undercovers because residuals of a strong base predictor are close to noise. MTC’s 315 g width is not usable; that is the only claim made for it.

**Table 2. Honest weight-side variants (10 trials, weights fit on calibration).** These are the numbers in the abstract.

| Variant | Coverage | Width | Winkler |
|---|---|---|---|
| AEC-S, global weights on the test set (leakage control) | 91.1% | 77 g | 183 |
| AEC-S, honest | 90.4% | 74 g | 181 |
| AEC-S2c, honest + global recalibration | 89.9% | 69 g | 172 |
| AEC-S2m, honest + per-bin recalibration | 90.4% | 76 g | 171 |

The leakage control is not identical to the old abstract figure 91.0% ± 1.3% at 76 ± 5 g. Table 2 uses a hold-out calibration/evaluation protocol; the old headline averaged 50/20/30 trials. Neither leaked figure is a main result. The leak is worth about 3 g of width. The deployed weight interval is AEC-S2m. Honest AEC-S at 90.4% and 74 g is the comparison that isolates stratification after the test-set search is removed.

The ablation direction is unchanged: a global ensemble narrows intervals relative to split conformal, and stratification raises coverage. The old ablation’s AEC and AEC-S rows share the test-set search, so “76 g versus 77 g” is not reported as a significant gain. Table 2 is the comparison that stands.

**Table 3. Nutrient intervals, AEC-S2m-N, strict hold-out, 10 trials.** The test split is not used to search weights.

| Nutrient | Coverage | Relative width | Winkler |
|---|---|---|---|
| Calories | 90.5% | 95.6% | 392 |
| Fat | 90.6% | 166.3% | 31 |
| Carbohydrate | 89.7% | 152.1% | 70 |
| Protein | 90.5% | 151.8% | 38 |

Normalized Winkler across the four nutrients is 2.42, below MondrianW (2.50) and Split (2.74). Propagating the weight interval through a point density covers only 63.7% of fat values: density error is not in that interval. Log split reaches a relative width of 359% near zero predictions, and the ensemble down-weights it. Without per-bin recalibration, fat coverage falls to 88.6%.

### 4.6 Protocol

1. **Leakage.** Searching global weights on the test set makes AEC and AEC-S optimistic. Main results are the honest rows of Table 2.
2. **Double-dipping.** Fitting ensemble weights and the recalibration quantile on the same calibration rows biases the quantile downward. Minimum nutrient coverage falls with capacity: 89.7% (5 candidates), 89.4% (7), 89.0% (7 plus nutrient binning), which is 0.3, 0.6, and 1.0 percentage points under the nominal 90%. Splitting the calibration rows restores coverage and widens the intervals. No such configuration beats the 5-candidate score 2.42.
3. **Quantile implementation.** Published numbers use interpolated quantiles. Canonical scripts now use $k=\lceil(n+1)(1-\alpha)\rceil$. Recomputing the tables needs a weight cache that is not in this checkout.
4. **True-weight bins.** Appendix coverage by true weight is an audit, not a conditional guarantee.
5. **MAPE and WAPE.** See Section 4.2. PMAE is not used.

Conformalized quantile regression and weighted conformal are in Appendix C. Conditional quantiles trained on 2,608 crops cover 34.7% of values on average before conformalization (nominal 80%). The best repaired variant has normalized Winkler 2.55, above 2.42. With no detectable in-domain shift, weighted conformal scores 2.54, also above 2.42.

## 5. Diagnose

### 5.1 Three factors

Mass is volume times density. Volume needs absolute scale. Density, beyond class identity, needs preparation and packing. Both error laws depend on whether the image comes from the training domain. The diagnostic claim is that these three factors can be measured separately. If a stronger backbone does not reduce strict hold-out MAPE, the ceiling is informational.

### 5.2 MetaFood3D: density

The cleaned MetaFood3D subset has 625 objects in 108 classes, with scale mass and scanned volume [8]. True volume times true density has error 0. True volume times leave-one-out class-mean density has MAPE 25.0% (23.4% at or above 100 g). Mean $|\log(\rho/\rho_{\mathrm{LOO}})|$ is 21.8%, the same order as the Nutrition5K 4-channel MAPE of 22.0%. Adding ±10% volume noise increases mass MAPE by 2.6 percentage points.

Packed blueberries have density dispersion 2.2%; loose blueberries, 33%. The gap is air between berries. It is not visible.

An appearance-to-density ResNet-18, in object-level 5-fold cross-validation, has density MAPE 30.3%. The global mean is 42.3%. Leave-one-out lookup is 26.2%. The network beats an uninformed prior and loses to class lookup. Appearance carries class-level density and not preparation within the class.

### 5.3 Single-image 3D

SAM 3D Objects is evaluated zero-shot on 130 objects in 30 classes [10]. The median ratio of camera-space volume to true volume is $5.6\times 10^{-5}$, and metric MAPE is 100%. One global scale factor leaves 87.3%. Per-class scale factors reduce median error to 33.3%, with between-class log-log correlation 0.92: the prior ranks almonds below watermelons and is not a ruler. Within-class correlations have median +0.88 and range from −0.98 to +0.99. Avocado, broccoli, and salmon track; loose almonds, blueberries, and bacon do not. Combining the best-case 33.3% volume error with 21.8% density dispersion yields about 40% mass error, worse than the 18.4% monocular MAPE on items of at least 100 g. The failure is missing information.

### 5.4 ECUSTFD: restoring scale

ECUSTFD contains 19 rigid classes, 174 instances, and a 25.0 mm coin in every frame [11]. Across 2,861 images, millimetres per pixel range from 0.084 to 8.33. Log-log correlation of top-view area with volume is 0.59 without calibration and 0.74 with the coin.

The same leave-one-out density is 25.0% MAPE on MetaFood3D and 7.9% on ECUSTFD true volumes. Coin-calibrated area, a leave-one-out height prior, and leave-one-out density give end-to-end mass MAPE 12.9% (n=174), below 18.4%, with no network and no depth.

This is not a win over the Nutrition5K system. ECUSTFD is rigid food; Nutrition5K is mostly prepared food. Where density is stable, one reference crosses the monocular ceiling. Where density is not visible, the geometric signals tested here are not enough.

### 5.5 NutritionVerse-Real: domain shift

NutritionVerse-Real has 889 smartphone images of 251 dishes and no depth [12]. Evaluation is zero-shot. The aligned dish protocol has 880 images. The RGB model’s in-domain MAPE is 31.4%, so degradation is measured from 31.4%, not from the 4-channel 22.0%. Modality costs 9.4 percentage points; the rest is domain.

Zero-shot dish MAPE is 103.7%, MAE is 238.8 g, R² is −0.16, and Spearman ρ is 0.16. Predictive spread shrinks to 0.55 times the spread of the labels. Small dishes are overestimated and large dishes underestimated. MAE is only slightly better than predicting the Nutrition5K median for every dish (245.1 g).

Intervals frozen on Nutrition5K:

| Nominal 90% interval | In-domain coverage | NV-Real coverage | Width |
|---|---|---|---|
| Split | 85.9% | 32.8% | 217 g |
| Normalized | 88.2% | 53.6% | 414 g |
| Mondrian, by weight | 85.4% | 41.5% | 297 g |

Restoring 90% coverage on the target domain moves the split quantile from 108.5 g to 484.3 g (4.46 times). Frozen nutrient intervals fall from 91.1% coverage to 44–56%, and MAPE moves from 22.8/36.5/39.4/33.7% to 64.8/69.5/163.2/95.3% (calories, fat, carbohydrate, protein; labels at least 1 g). Binning does not save coverage. The guarantee requires exchangeability.

A secondary ingredient-level analysis gives rigid-food MAPE 318% and prepared-food MAPE 606%. The direction matches the density factor. The absolute level is not usable.

On the 177-image dish hold-out, Nutrition5K-only training has MAPE 77.4%, not the 103.7% of all 880 images. Joint training reduces that to 23.9%, and the Nutrition5K test MAPE moves from 31.4% to 31.5%. A model trained only on the target domain scores 124.7% MAPE back on Nutrition5K. Multi-domain data buys back the shift, in both directions. The ceiling is coverage of domains, not backbone capacity.

### 5.6 Capacity

Replacing ResNet-18 with DINOv2-small and a new regression head yields calibrated strict hold-out MAPE 31.3%, close to the 3-channel ResNet at 31.4% and worse than the 4-channel ensemble at 22.0%.

### 5.7 Summary

**Table 4. One question per dataset.**

| Dataset | Question | Figure |
|---|---|---|
| Nutrition5K | In-domain point error and intervals | 4-channel MAPE 22.0%; honest AEC-S 90.4% / 74 g |
| MetaFood3D | Does density dominate? | Dispersion 21.8%; lookup 25.0%; CNN 30.3% vs lookup 26.2% |
| MetaFood3D | Can single-image 3D leave? | Metric MAPE 100%; per-class median 33.3% |
| ECUSTFD | Rigid foods after scale is restored | True volume 7.9%; end-to-end 12.9% |
| NV-Real | Leaving the training domain | MAPE 103.7%; split coverage 32.8%; joint training 23.9% |
| Nutrition5K | Is capacity the constraint? | DINOv2 calibrated MAPE 31.3% |

## 6. Act

### 6.1 From diagnosis to actions

Chapter 5 specifies the action space. Rigid foods are missing scale, so the reference action is justified by the ECUSTFD end-to-end MAPE of 12.9%. Prepared foods have density that appearance does not determine, so a semantic question is justified by the 33% loose-blueberry dispersion and by the CNN losing to lookup. A frozen interval at 32.8% coverage justifies abstention rather than another 90% claim. Depth and an extra view are worth their cost only when geometry still carries information.

### 6.2 Formulation

An episode has a state, an action, a cost, and a stopping rule. The state includes the current point, the current interval, morphology (rigid or prepared), and cost already paid. Actions are stop, request depth, request a view, request a reference, ask a semantic question, and abstain. The policy chooses before paying. Stop or abstain ends the episode. A handwritten rule of the form “ask if the interval is wide” is not this policy: it is not frozen on held-out data, and it does not refit the quantile after freezing.

### 6.3 Cost and three-way isolation

Public datasets do not contain a new photograph taken on request. The feasibility protocol treats measurements that already exist as proxy rewards: the Nutrition5K depth channel, the ECUSTFD coin, MetaFood3D density lookup, and the NV-Real shift. These are proxies, not paired interactions.

Rows are split three ways and not reused. The policy is fit on the policy split. The terminal quantile is fit on the calibration split after the policy is frozen. The test split is scored once. Coverage on the calibration split is an audit.

### 6.4 Feasibility

The public-data run on 2026-09-04 wrote 1,648 episodes: ECUSTFD 146, MetaFood3D 625, Nutrition5K 652, NV-Real 225, each split into policy, calibration, and test. At least two actions beat RGB-only by 5 percentage points of MAPE or WAPE on their native sets, so the gate kept a unified policy. Out-of-fold R² of the gain models is negative for every action (depth −0.011, view −0.016, reference −0.012, semantic −0.011). The result directory of that run is not in this checkout. Table 5 is taken from the run log.

**Table 5. Locked test split. Coverage is not a claim that the nominal 90% is met.**

| Policy | Coverage | Normalized Winkler | Cost |
|---|---|---|---|
| Always stop | 0.885 | 4.280 | 0.00 |
| Always depth | 0.882 | 4.288 | 1.18 |
| Always view | 0.887 | 4.304 | 1.21 |
| Always reference | 0.887 | 4.266 | 0.27 |
| Always semantic | 0.930 | 3.776 | 0.91 |
| Always abstain | 0.885 | 4.280 | 1.00 |
| Fixed triage | 0.887 | 4.266 | 3.21 |
| Myopic value of information | 0.887 | 4.264 | 0.53 |

Myopic value of information scores 4.264 at cost 0.53. The best fixed policy is always semantic, at 3.776 and cost 0.91. The preregistered claim of at least a 10% normalized-Winkler reduction against the best cost-matched fixed policy, at coverage consistent with 90%, is not met. The terminal quantile on the calibration split is 7.09 (n=575); that is not a test coverage. Zero-shot coverage on the NV-Real subset is 0.614, and the run does not claim 90%. The default configuration calls no language-model API, so there is no language-model ablation delta.

Diagnosis says which actions should exist. On public proxies, adaptive routing does not beat the best fixed action. Reporting a recovered accuracy for the agent would go beyond the evidence. Paired collection and a user-burden study are still open.

## 7. Discussion

Without the estimator, diagnosis has no uncertainty to attribute. Without diagnosis, actions are a feature list. Without an action layer, diagnosis only names a ceiling. The first two layers are evidenced. The third has a protocol and one honest negative feasibility result.

Under controlled RGB-D, AEC-S2m is the interval to report, together with MAPE. Capture of rigid foods should require a reference. Prepared foods should not be promised a single-image 3D fix for density. A smartphone photo should not display a 90% interval before target-domain calibration; 32.8% coverage is an abstention or a recalibration. Joint training moves the dish-hold-out MAPE on NV-Real from 77.4% to 23.9% with almost no source-domain change.

Bins above 500 g are small, and the leaked audit still sits under 90% coverage; that audit has not been recomputed with honest weights. Nutrient density MAPE on strict hold-out remains 33.7–39.4%, which the interval can report and cannot remove. Public actions are proxies, and the gain model has no predictive R². The order-statistic quantile has not been used to rebuild Tables 1 and 2.

Food images sit next to meals and health information. A reference protocol should not require an identifiable background. Abstention should be a visible state.

## 8. Conclusion

The paper chains food-mass uncertainty from estimation through diagnosis to action. Honest AEC-S covers 90.4% of outcomes at a mean width of 74 g. The weight interval to deploy is AEC-S2m, with Winkler 171. Four datasets assign the remaining error to scale, density, and domain rather than to model capacity. The acquisition agent has an action space aligned with those factors. On public data it does not beat the best fixed policy. The next measurement is a three-split evaluation on paired interactions, not another backbone.

---

## References

[1] Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.

[2] Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002). Inductive confidence machines for regression. *ECML*.

[3] Papadopoulos, H., & Haralambous, H. (2011). Reliable prediction intervals with regression neural networks. *Neural Networks*.

[4] Boström, H., & Johansson, U. (2020). Mondrian conformal regressors. *COPA*.

[5] Adaptive ensemble conformal (AEC). Global weights are fit on the calibration set. Test-set search is reported only as a leakage control.

[6] Multimodal learned nonconformity scores (MLNS). Undercover on this base predictor and not recommended.

[7] Thames, Q., et al. (2021). Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food. *CVPR*.

[8] Chen, Y., et al. (2024). MetaFood3D: Large 3D Food Object Dataset with Nutrition Values. *arXiv:2409.01966*.

[9] Sanatbyek, A., et al. (2025). A Multitask Deep Learning Model for Food Scene Recognition and Portion Estimation. *IEEE Access*, 13, 152033–152045.

[10] Meta AI. (2025). SAM 3D Objects. *arXiv preprint* (facebook/sam-3d-objects).

[11] Liang, Y., & Li, J. (2017). Computer vision-based food calorie estimation: dataset, method, and experiment. *arXiv:1705.07632*.

[12] Herse, S., et al. (2024). NutritionVerse-Real. *arXiv:2401.08598*.

[13] Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.

---

## Appendix A. Honest AEC-S

```
Input: candidate intervals {C_k}, calibration set D_cal, test set D_test, α, τ=30
1. Dirichlet-search w_global on D_cal. Do not use D_test.
2. For each bin b defined by predicted weight:
     if fewer than 20 calibration rows: w_b = w_global
     else search w_local on those rows
     λ_b = n_b / (n_b + τ)
     w_b = λ_b w_local + (1-λ_b) w_global
3. At test time, select w_b by predicted weight only.
4. Any recalibration quantile uses fresh calibration scores and
   k = ceil((n+1)(1-α)), the k-th order statistic.
```

Coverage aggregated by true weight must be labeled as an audit.

## Appendix B. Reproduction

Training matches `Nutri_0811/train_deep_v2_final.py`. Entrypoints:

- Weight intervals: `Nutri_0811/conformal_deep_experiment_final.py` (calibration-set global weights; order-statistic quantiles)
- Honest variants: `Nutri_0811/aec_s_v2_experiment.py`
- Nutrient intervals: `Nutri_0811/nutrient_conformal_final.py`
- Diagnosis: `metafood3d/mf3d_phase1_oracle.py`, `mf3d_phase2_learned_density.py`, `mf3d_oracle6_sam3d.py`, `ecustfd/ecustfd_reference_experiment.py`
- Feasibility agent: `agent_extension/`

Weights, crop caches, and raw datasets are not in this repository. The Nutrition5K pipeline expects `models/deep_crops_cache.npz`. Raw MetaFood3D is about 69 GB, SAM 3D weights about 13 GB, and NV-Real about 1.1 GB. Those files must not be committed. What is stored are the diagnostic CSVs and the summarized log figures. Figures under `Nutri_0811/figures/` still say PMAE and were not regenerated. `make_paper_figures.py` now labels axes as MAPE for the next run that has the cache.

## Appendix C. Moved out of the main text

**True-weight coverage audit from the leaked AEC-S.** Above 500 g, coverage moves from 71.8% (Split) to 84.3% (AEC-S). The 0–50 g audit coverage is 84.2% at width 11 g. The weight pattern is Mondrian-dominated at 0–50 g (67%), Split-dominated at 200–500 g (85%), and MLNS-dominated above 500 g (53%). The table inherits test-set global weights and is only a qualitative direction.

**CQR and weighted conformal (10 trials; minimum coverage / mean relative width / normalized Winkler).** Raw CQR: 22.2% / 38.8% / 4.55, invalid. Best hybrid per-bin variant: 90.5% / 151.5% / 2.55. Weighted conformal: 89.2% / 148.6% / 2.54. AEC-S2m-N: 89.7% / 141.4% / 2.42.

**Figures.** Existing diagnostic and cross-domain plots cover density, blueberries, binned NV-Real coverage, joint training, and SAM 3D scale. The three-layer overview, the AEC-S diagram, an honest coverage–width plot, and an action-layer cost plot should be redrawn when the cache is available. Until then, Table 5 is the action result, so a Pareto plot does not depict a win that the table does not show.
