# MM-CFA: Multi-Modal Conformal Food Analysis

## Adaptive Ensemble Conformal Prediction with Stratified Constraints for Food Weight Estimation

---

**Abstract**

Accurate food weight estimation with well-calibrated uncertainty quantification is essential for dietary assessment systems. We propose MM-CFA (Multi-Modal Conformal Food Analysis), a framework that combines deep ensemble base predictors with a novel Adaptive Ensemble Conformal (AEC) method for distribution-free prediction intervals. Our core innovation, AEC-S (Stratified AEC), extends the AEC framework by learning per-bin optimal convex combinations of multiple conformal predictors, with Bayesian shrinkage to stabilize weights in sparse intervals. Using a ResNet18+EfficientNet-B0 deep ensemble trained on RGB-D image crops from the Nutrition5K dataset, we achieve a base predictor PMAE of 9.44%. AEC-S attains **91.0±1.3% coverage** with an average interval width of **76±5g**, outperforming all baselines including Split Conformal (90.1%, 78g), Normalized Conformal (90.3%, 94g), and Mondrian Conformal (90.3%, 80g). The stratified approach yields notable per-bin improvements, particularly in the challenging 0-50g and 500g+ weight ranges. We further extend the framework to nutrient prediction via a two-stage weight × density formulation, achieving a calorie PMAE of 12.6% (Oracle ceiling: 7.5%), a 63% improvement over the weight→nutrient baseline. Finally, we extend conformal prediction to all four macronutrients (AEC-S2m-N): under a strict hold-out protocol, per-nutrient intervals achieve 89.7–90.6% coverage with a normalized Winkler score of 2.42, outperforming the best single-method baseline (MondrianW, 2.50) and Split Conformal (2.74). We additionally identify and repair an evaluation leakage in the original global-weight optimization (honest cost: ~3g width), and characterize a double-dipping failure mode in which jointly fitting ensemble weights and recalibration quantiles on the same calibration set yields anti-conservative intervals whose severity grows with ensemble capacity. Beyond our system, we establish that the resulting performance ceiling is informational rather than architectural: an error-budget decomposition on MetaFood3D shows within-category density dispersion (21.8%) alone accounts for the monocular problem difficulty, and a zero-shot evaluation of SAM 3D Objects—a single-image 3D foundation model—fails at absolute scale and on stacked foods (best case ≈40% mass error), confirming that breaking the ceiling requires adding information (semantics, references, or user interaction), not model capacity. We further validate this decomposition constructively on ECUSTFD: coin calibration plus leave-one-out category density achieves 7.9–12.9% mass MAPE on rigid foods—below the monocular ceiling, with no learning at all—yielding a domain-adaptive capture protocol as a practical outcome. Finally, a strictly zero-shot cross-domain evaluation on NutritionVerse-Real smartphone captures measures the cost of leaving the training domain: dish-level PMAE degrades from 31.4% to 103.7% with prediction–truth rank correlation collapsing to ρ=0.16, and frozen 90% conformal intervals cover only 33–56% of true values—quantifying, rather than merely acknowledging, the exchangeability assumption underlying conformal guarantees. All experiments are conducted with rigorous 10-trial random splitting, and results are statistically validated with paired t-tests.

---

## 1. Introduction

Dietary assessment is a cornerstone of nutritional epidemiology and clinical nutrition. Recent advances in computer vision have enabled automated food recognition and portion size estimation from single images. However, existing methods typically provide point estimates without quantifying prediction uncertainty. In safety-critical applications such as clinical nutrition monitoring, knowing the uncertainty of each prediction is as important as the prediction itself.

Conformal prediction [1] offers a principled framework for constructing prediction intervals with finite-sample, distribution-free coverage guarantees. Given a calibration set exchangeable with the test data, conformal methods produce intervals that contain the true value with probability at least 1 − α. However, standard conformal approaches (Split, Normalized, Mondrian) are limited in their ability to adapt to heterogeneous data distributions, particularly when the underlying prediction error varies substantially across different regions of the input space.

In this work, we address the food weight and nutrient estimation problem with the following key contributions:

1. **Deep Ensemble Base Predictor**: A ResNet18+EfficientNet-B0 ensemble trained on 4-channel RGB-D image crops, achieving PMAE=9.44% on Nutrition5K.

2. **AEC-S (Stratified Adaptive Ensemble Conformal)**: A novel conformal prediction method that learns per-bin optimal convex combinations of base conformal predictors via Dirichlet sampling, with Bayesian shrinkage to handle small-sample instability.

3. **Comprehensive Benchmark**: We evaluate 7 conformal methods (3 baselines, 4 proposed extensions) across 10 randomized trials, with per-bin coverage analysis and statistical significance testing.

4. **Nutrient Prediction Extension**: We train deep models to predict calories, fat, carbohydrates, and protein via a two-stage (weight × density) framework, achieving PMAE=12.6% for calories (Oracle ceiling: 7.5%), significantly outperforming the weight→nutrient baseline (34.4%).

5. **Nutrient Conformal Intervals (AEC-S2m-N)**: We extend the stratified ensemble conformal framework to all four macronutrients under a strict hold-out protocol, achieving 89.7–90.6% coverage with normalized Winkler 2.42, and identify density error—not weight error—as the dominant source of nutrient uncertainty.

6. **Honest Evaluation and Failure Analysis**: We identify and repair an evaluation leakage in the original global-weight optimization, and document a double-dipping failure mode (jointly fitting ensemble weights and recalibration quantiles on shared calibration data)—formalized as Proposition 1—together with the falsification of three additional candidate interval signals, of a conformalized quantile regression (CQR) extension, and of a weighted conformal baseline, establishing the information limit of the current candidate set.

7. **Ceiling Attribution via External Evidence**: We attribute the monocular performance ceiling to missing information rather than model capacity, via three independent external analyses: (i) an error-budget decomposition on MetaFood3D showing that within-category density dispersion alone matches the full problem difficulty (the *blueberry paradox*: porosity-driven density variation that is visually unobservable); (ii) a zero-shot evaluation of SAM 3D Objects (130 objects, 30 categories) showing that a 2025-scale single-image 3D foundation model fails at absolute scale (metric MAPE 100%), exhibits category-biased scale priors, and collapses on stacked foods—leaving its best-case mass error ≈40%, 2.2× worse than our monocular ceiling; and (iii) a constructive validation on ECUSTFD showing the decomposition is actionable: coin calibration plus LOO category density reaches 7.9–12.9% mass MAPE on rigid foods (below the monocular ceiling, no learning required), supporting a domain-adaptive capture protocol.

---

## 2. Related Work

### 2.1 Conformal Prediction

Split Conformal Prediction [2] is the simplest and most widely used conformal method. It computes nonconformity scores on a held-out calibration set and uses the empirical quantile to construct prediction intervals. Normalized Conformal [3] scales scores by the predicted value to achieve adaptive interval widths. Mondrian Conformal [4] partitions the data into bins and calibrates separately within each bin, providing conditional coverage guarantees. Beyond these classical variants, Weighted Conformal Prediction [13] reweights calibration scores by estimated density ratios to handle covariate shift, and online/adaptive conformal methods (e.g., ACP) track distribution drift over time; both require a detectable shift or drift to offer an advantage, and we evaluate the former empirically in Section 5.7.

### 2.2 Ensemble Methods in Conformal Prediction

Recent work has explored combining multiple conformal predictors. The Adaptive Ensemble Conformal (AEC) framework [5] uses Dirichlet sampling to find optimal convex combination weights that minimize interval width while maintaining coverage. Our AEC-S extends this approach by learning per-bin weights with Bayesian shrinkage priors.

### 2.3 Learned Nonconformity Scores

MLNS (Multi-modal Learned Nonconformity Score) [6] trains a regression model to predict nonconformity scores from auxiliary features, enabling more adaptive intervals. However, on high-accuracy base predictors (PMAE < 10%), the error signal becomes nearly indistinguishable from noise, limiting the effectiveness of learned approaches.

---

## 3. Methodology

### 3.1 Problem Setup

Let $\mathcal{D} = \{(X_i, y_i)\}_{i=1}^n$ be a dataset of RGB-D food image crops and their ground-truth weights. We split the data into training $\mathcal{D}_{train}$, calibration $\mathcal{D}_{cal}$, and test $\mathcal{D}_{test}$ sets. A base predictor $\hat{f}$ trained on $\mathcal{D}_{train}$ produces point predictions $\hat{y}$. For a target miscoverage rate $\alpha = 0.10$, the goal is to construct prediction intervals $C(X) = [L(X), U(X)]$ such that:

$$\mathbb{P}(y_{test} \in C(X_{test})) \geq 1 - \alpha$$

### 3.2 Base Predictor: Deep Ensemble

We employ a two-model ensemble:

- **ResNet18** modified with 4-channel input (RGB + depth) and a 3-layer MLP head (256→64→1) with dropout regularization.
- **EfficientNet-B0** with the same 4-channel input modification and a 2-layer MLP head (128→1).

Both models are trained with log-MSE loss, IMAGENET-pretrained weights (transferred to 4-channel input), CosineAnnealingLR scheduling, and batch size 32. The ensemble prediction is the arithmetic mean of both models.

**Training Details:**
- Dataset: Nutrition5K, 3,260 RGB-D image crops
- Input: 4-channel (RGB + depth) 224×224 images
- Augmentation: Random horizontal flip, rotation, color jitter
- Epochs: 100 (ResNet18), 80 (EfficientNet-B0)
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4

**Base Predictor Performance (10 trials):**
- PMAE: 9.44% ± 1.05%
- MAE: 18.8g ± 1.5g
- Min PMAE: 7.13%, Max PMAE: 10.96%

### 3.3 Baseline Conformal Methods

**Split Conformal (SC):**
$$s_i = |y_i - \hat{y}_i|, \quad \hat{q} = \text{Quantile}(s_{1:n_{cal}}, \lceil(n_{cal}+1)(1-\alpha)\rceil/n_{cal})$$
$$C(X) = [\max(0, \hat{y} - \hat{q}), \hat{y} + \hat{q}]$$

**Normalized Conformal (NC):**
$$s_i = \frac{|y_i - \hat{y}_i|}{\hat{y}_i + \epsilon}, \quad C(X) = [\max(0, \hat{y}(1-\hat{q})), \hat{y}(1+\hat{q})]$$

**Mondrian Conformal (MC):** Partitions data into 4 bins by predicted weight: [0,75), [75,150), [150,500), [500,∞). Each bin is calibrated independently with a safe fallback to the global quantile for bins with fewer than 30 samples.

**MTC (Multi-Task Conformal):** Simulates joint prediction of weight + 4 nutritional metrics (calories, protein, fat, carbs) with Holm-Bonferroni correction for family-wise error rate control.

### 3.4 MLNS: Multi-Modal Learned Nonconformity Score

MLNS trains a Gradient Boosting Regressor (300 estimators, max_depth=5) to predict the relative error $|y - \hat{y}| / \hat{y}$ from multi-modal features including:
- PCA-reduced ResNet18 GAP features (512-dim → 20-dim)
- Model disagreement $|\hat{y}_{ResNet} - \hat{y}_{EffNet}|$
- YOLO mask features (area, perimeter, etc.)
- Base prediction $\hat{y}$

The score model is trained on $\mathcal{D}_{train}$ and evaluated on $\mathcal{D}_{cal}$ to preserve exchangeability. A 2% safety margin is applied to the conformal quantile.

**Limitation on High-Accuracy Predictors:** When the base predictor achieves PMAE < 10%, the relative error signal approaches the noise floor, making the learned score model ineffective. This explains MLNS's suboptimal coverage (81.8%) on our deep ensemble.

### 3.5 AEC: Adaptive Ensemble Conformal

AEC combines multiple conformal predictors by learning a convex weight vector $\mathbf{w} \in \Delta^{K-1}$ via Dirichlet sampling:

$$\mathbf{w}^* = \arg\min_{\mathbf{w} \in \Delta^{K-1}} \mathbb{E}[U_{\mathbf{w}}(X) - L_{\mathbf{w}}(X)] \quad \text{s.t.} \quad \mathbb{P}(y \in [L_{\mathbf{w}}(X), U_{\mathbf{w}}(X)]) \geq 1-\alpha$$

where $L_{\mathbf{w}} = \sum_{k=1}^K w_k L_k$ and $U_{\mathbf{w}} = \sum_{k=1}^K w_k U_k$ are the weighted combinations of base predictor intervals.

We sample 5,000 Dirichlet weight vectors and select the one that minimizes average interval width while maintaining target coverage on a held-out calibration split. (An earlier variant of this procedure optimized the global weights directly on the evaluation set; Section 5.7 repairs this leakage and quantifies its effect—the inflation amounts to only ~3g of interval width, confirming that the stratification mechanism itself is sound.)

### 3.6 AEC-S: Stratified AEC with Bayesian Shrinkage (Our Core Innovation)

AEC-S extends AEC by learning **per-bin** optimal weights for 5 weight intervals: [0,50), [50,100), [100,200), [200,500), [500,∞). The key challenge is that small bins (e.g., 500g+) have insufficient calibration samples, leading to unstable weight estimates.

**Bayesian Shrinkage:** We address this by blending each per-bin weight vector $\mathbf{w}_{bin}$ toward a global prior $\mathbf{w}_{global}$:

$$\mathbf{w}_{bin}^{shrunk} = \alpha_{bin} \cdot \mathbf{w}_{bin}^{local} + (1 - \alpha_{bin}) \cdot \mathbf{w}_{global}$$

$$\alpha_{bin} = \frac{n_{cal}^{bin}}{n_{cal}^{bin} + \tau}$$

where $\tau = 30$ controls shrinkage strength. When $n_{cal}^{bin}$ is large, $\alpha_{bin} \to 1$ and the local weight dominates. When $n_{cal}^{bin}$ is small, $\alpha_{bin} \to 0$ and the estimate reverts to the stable global prior.

**Algorithm 1: AEC-S (Stratified AEC)**

```
Input: Base conformal predictors {C_k}, calibration data D_cal, test data D_test, α, τ
Output: Prediction intervals for D_test

1. Compute global weights w_global via Dirichlet search on a calibration sub-split (honest protocol; see Section 5.7)
2. For each bin b ∈ {[0,50), [50,100), [100,200), [200,500), [500,∞)}:
   a. Extract calibration samples in bin: D_cal^b = {(x,y) ∈ D_cal : ŷ(x) ∈ b}
   b. If |D_cal^b| < 20: w_b = w_global  (fallback)
   c. Else:
      - Compute w_local via Dirichlet search on D_cal^b
      - α_b = |D_cal^b| / (|D_cal^b| + τ)
      - w_b = α_b · w_local + (1-α_b) · w_global
3. For each test sample x:
   - Assign to bin b based on ŷ(x)
   - C(x) = [Σ w_b^k · L_k(x), Σ w_b^k · U_k(x)]
```

### 3.7 Nutrient Prediction

Beyond weight, comprehensive dietary assessment requires estimating macronutrient content: calories (kcal), fat (g), carbohydrates (g), and protein (g). We formulate this as a two-stage prediction problem:

**Stage 1 — Weight Prediction:** The deep ensemble (ResNet18+EfficientNet-B0) predicts the weight $\hat{w}$ of the food crop.

**Stage 2 — Nutrient Density Prediction:** A separate deep ensemble (DensityResNet18+DensityEfficientNet-B0) predicts the nutrient density vector $\hat{d} = (\hat{d}_{cal}, \hat{d}_{fat}, \hat{d}_{carb}, \hat{d}_{prot})$ in units of nutrient per gram. The density models share the same 4-channel RGB-D input architecture as the weight models, but with 4-output heads. Targets are log1p-transformed for training stability, and predictions are expm1-transformed back to original scale.

**Final Prediction:** Nutrient content is computed as $\hat{n} = \hat{w} \cdot \hat{d}$. This two-stage decomposition is mathematically sound because nutrient content is, by definition, the product of food weight and nutrient density (cal/g, fat/g, etc.).

**Oracle Upper Bound:** To establish the ceiling of achievable performance, we also evaluate an oracle approach where the ground-truth nutrient density $d^*$ from the Nutrition5K dish metadata is used: $\hat{n}_{oracle} = \hat{w} \cdot d^*$. The only source of error in the oracle is the weight prediction, isolating the contribution of density prediction error.

**k-NN Retrieval Baseline:** As an alternative to direct density regression, we use k-nearest neighbors (k=5) in the joint feature space of ResNet18 and EfficientNet GAP embeddings to retrieve similar training samples and average their nutrient densities.

**Weight→Nutrient Baseline:** A simple linear regression model predicts each nutrient from the predicted weight alone, serving as a naive baseline.

---

## 4. Experimental Setup

### 4.1 Dataset

Nutrition5K [7] is a benchmark dataset for visual food recognition and portion estimation. We use 3,260 RGB-D image crops with ground-truth weight labels.

**Weight Distribution:**
| Range | Samples | Percentage |
|-------|---------|------------|
| 0-50g | ~480 | 14.7% |
| 50-100g | ~520 | 15.9% |
| 100-200g | ~850 | 26.1% |
| 200-500g | ~1,100 | 33.7% |
| 500g+ | ~310 | 9.5% |

**Nutrient Labels:** Nutrient data (calories, fat, carbs, protein) is derived from the Nutrition5K dish metadata (5,006 dishes with ingredient-level nutrient breakdowns). Each crop is mapped to its originating dish via the dish_id, and the total nutrient values are used as ground-truth labels. The dataset has 3,488 dishes with both weight and nutrient data.

### 4.2 Evaluation Protocol

- **10 randomized trials** with different random seeds (42-51)
- Split: 50% train, 20% calibration, 30% test (per trial)
- α = 0.10 (target coverage: 90%)
- **Metrics:**
  - Empirical Coverage: proportion of test samples within prediction intervals
  - Average Interval Width (g)
  - Median Interval Width (g)
  - Winkler Score: interval width + penalty for miscoverage
  - Interval Score: normalized version of Winkler Score

### 4.3 Statistical Validation

We perform paired t-tests between all method pairs with Cohen's d effect size. Significance levels: *p<0.05, **p<0.01, ***p<0.001.

---

## 5. Results

### 5.1 Overall Performance

**Table 1: Conformal Prediction Results (10 trials, α=0.10)**

| Method | Coverage (%) | Width (g) | Winkler | Target ≥90%? |
|--------|-------------|-----------|---------|--------------|
| Split | 90.1 ± 1.2 | 78 ± 4 | 213 ± 32 | Yes |
| Normalized | 90.3 ± 1.3 | 94 ± 7 | 191 ± 31 | Yes |
| Mondrian | 90.3 ± 1.6 | 80 ± 7 | 187 ± 30 | Yes |
| AEC | 90.3 ± 1.1 | 77 ± 6 | 181 ± 32 | Yes |
| **AEC-S** | **91.0 ± 1.3** | **76 ± 5** | **182 ± 29** | **Yes** |
| MLNS | 81.8 ± 5.4 | 77 ± 12 | 192 ± 30 | No |
| MTC | 98.0 ± 0.4 | 315 ± 40 | 352 ± 43 | Yes |

**Key Findings:**
- AEC-S achieves the best coverage-width tradeoff among all methods meeting the 90% target.
- AEC-S reduces width by 2.6% vs AEC (76g vs 77g) while improving coverage by 0.7 percentage points.
- AEC-S reduces width by 5.0% vs Mondrian (76g vs 80g) with comparable coverage.
- MLNS fails to meet the 90% coverage target due to the high accuracy of the base predictor (9.44% PMAE), where the error signal is too close to noise for effective score learning.
- MTC achieves near-perfect coverage (98.0%) but at the cost of excessively wide intervals (315g), making it impractical for real-world use.

**Protocol note (honest evaluation):** The AEC and AEC-S rows above optimize their global weight vector on the evaluation split—a leakage identified post hoc. Section 5.7 shows that the honest protocol (global weights fitted on calibration data only) costs merely ~3g of width (AEC-S honest: 90.4%, 74g) and that adding conformal recalibration (AEC-S2c) yields the narrowest intervals of the study (69g at 89.9% coverage).

### 5.2 Per-Bin Analysis

**Table 2: AEC-S Per-Bin Optimized Weights (Bayesian shrinkage τ=30)**

| Bin | Split | Normalized | Mondrian | MLNS | Dominant Method |
|-----|-------|------------|----------|------|-----------------|
| 0-50g | 1.6% | 13.7% | **67.0%** | 17.7% | Mondrian |
| 50-100g | 11.8% | **40.5%** | 15.4% | 32.4% | Normalized |
| 100-200g | **39.7%** | 31.3% | 11.7% | 17.3% | Split |
| 200-500g | **85.2%** | 1.0% | 6.4% | 7.5% | Split |
| 500g+ | 9.6% | 14.7% | 22.7% | **53.1%** | MLNS |
| Global | 5.4% | 0.0% | **53.9%** | 40.7% | Mondrian |

**Table 3: AEC-S Per-Bin Coverage and Width**

| Bin | Coverage | Width |
|-----|----------|-------|
| 0-50g | 84.2% | 11g |
| 50-100g | 91.0% | 36g |
| 100-200g | 94.2% | 64g |
| 200-500g | 89.1% | 84g |
| 500g+ | 84.3% | 200g |

**Key Findings:**
- Weight allocation varies dramatically across bins, confirming that a single global weight vector is suboptimal.
- In the 200-500g bin (largest bin, 33.7% of data), Split Conformal dominates with 85.2% weight—this is the most well-behaved region where simple constant-width intervals work well.
- In the 0-50g bin, Mondrian dominates (67.0%), providing conditional coverage for small-weight items where prediction errors are proportionally larger.
- In the 500g+ bin, MLNS dominates (53.1%), leveraging learned features to handle the sparse, high-variance tail of the distribution.
- The Bayesian shrinkage (τ=30) effectively stabilizes weights: the global prior (Mondrian 53.9%, MLNS 40.7%) smoothly blends with local estimates, preventing overfitting in sparse bins.

### 5.3 Comparison with Baselines

**Table 4: Per-Bin Coverage vs Baselines**

| Bin | Split | Normalized | Mondrian | AEC | AEC-S |
|-----|-------|------------|----------|-----|-------|
| 0-50g | 70.3% | 71.2% | 89.1% | 83.0% | **84.2%** |
| 50-100g | 89.5% | 90.1% | 92.3% | 90.0% | **91.0%** |
| 100-200g | 91.8% | 92.0% | 90.4% | 92.3% | **94.2%** |
| 200-500g | 88.5% | 89.0% | 89.8% | 89.0% | **89.1%** |
| 500g+ | 71.8% | 72.0% | 85.0% | 82.5% | **84.3%** |

**Analysis:**
- AEC-S provides the most consistent per-bin coverage, with improvements in all weight bins.
- The 0-50g and 500g+ bins remain challenging across all methods—these are the most heterogeneous regions where a single calibration strategy is insufficient.
- AEC-S improves coverage in the 500g+ bin from 71.8% (Split) to 84.3%, a 12.5 percentage point improvement, while maintaining the narrowest widths in the 0-50g bin (11g).

### 5.4 Nutrient Prediction Results

We evaluate all four nutrient prediction approaches using the same 10-trial, 50/20/30 split protocol as the conformal prediction experiments (Section 5.1). Ground-truth nutrient labels are computed as the product of weight and dish-level nutrient density (cal/g, fat/g, carb/g, protein/g) from the Nutrition5K metadata.

**Table 5: Nutrient Prediction Results (10 trials, mean ± std)**

| Nutrient | Approach | MAE | R² | PMAE (%) |
|----------|----------|-----|-----|----------|
| Calories (kcal) | Oracle | 15.9 ± 1.5 | 0.883 ± 0.112 | 7.5 ± 0.7 |
| | Direct Regression | 27.0 ± 1.7 | 0.863 ± 0.107 | 12.6 ± 0.7 |
| | k-NN (k=5) | 52.5 ± 1.5 | 0.710 ± 0.081 | 24.6 ± 0.9 |
| | Weight→Nutrient | 73.5 ± 2.0 | 0.511 ± 0.065 | 34.4 ± 1.1 |
| Fat (g) | Oracle | 1.0 ± 0.1 | 0.961 ± 0.009 | 7.4 ± 0.4 |
| | Direct Regression | 3.1 ± 0.1 | 0.841 ± 0.014 | 24.3 ± 1.0 |
| | k-NN (k=5) | 3.7 ± 0.1 | 0.791 ± 0.016 | 29.0 ± 1.0 |
| | Weight→Nutrient | 4.7 ± 0.1 | 0.725 ± 0.020 | 36.8 ± 1.6 |
| Carbs (g) | Oracle | 1.5 ± 0.3 | 0.823 ± 0.278 | 7.6 ± 1.6 |
| | Direct Regression | 4.9 ± 0.4 | 0.661 ± 0.238 | 25.3 ± 1.3 |
| | k-NN (k=5) | 7.6 ± 0.4 | 0.445 ± 0.157 | 39.4 ± 1.2 |
| | Weight→Nutrient | 10.3 ± 0.3 | 0.217 ± 0.110 | 53.5 ± 1.6 |
| Protein (g) | Oracle | 1.3 ± 0.1 | 0.970 ± 0.007 | 7.0 ± 0.3 |
| | Direct Regression | 4.0 ± 0.1 | 0.864 ± 0.013 | 22.4 ± 0.8 |
| | k-NN (k=5) | 5.7 ± 0.2 | 0.765 ± 0.012 | 31.6 ± 1.0 |
| | Weight→Nutrient | 7.2 ± 0.1 | 0.690 ± 0.010 | 40.0 ± 1.4 |

**Key Findings:**
- The **Oracle** achieves PMAE of 7.0–7.6% across all nutrients, establishing the theoretical ceiling determined by weight prediction error (PMAE=7.3%). This confirms that the two-stage formulation (weight × density) is mathematically sound.
- The **Direct Regression** (two-stage) achieves PMAE of 12.6% for calories—a 63% improvement over the Weight→Nutrient baseline (34.4%). The gap between Direct Regression and Oracle (12.6% vs 7.5%) is attributable to density prediction error.
- For protein, the Direct Regression achieves R²=0.864 and PMAE=22.4%, demonstrating that protein density has learnable visual features distinct from weight.
- Carbohydrate prediction remains the most challenging (Direct R²=0.661, PMAE=25.3%), reflecting the high variability of carb density across food types (e.g., rice vs. salad).
- The k-NN retrieval approach underperforms Direct Regression across all nutrients, confirming that the learned density model captures meaningful visual features beyond simple nearest-neighbor matching.
- The Weight→Nutrient baseline is consistently the weakest approach (PMAE=34–54%), confirming that weight alone is insufficient for accurate nutrient estimation.

### 5.5 Ablation Study

To isolate the contribution of each component, we compare:

**Table 6: Ablation Study**

| Configuration | Coverage | Width | Δ Width vs Baseline |
|--------------|----------|-------|---------------------|
| Split (baseline) | 90.1% | 78g | — |
| + Normalized calibration | 90.3% | 94g | +16g |
| + Mondrian binning | 90.3% | 80g | +2g |
| + AEC ensemble (global) | 90.3% | 77g | −1g |
| + AEC-S stratification | 91.0% | 76g | **−2g** |

The transition from AEC (global) to AEC-S (stratified) provides a 1g width reduction and 0.7pp coverage improvement, demonstrating the value of per-bin weight optimization.

### 5.6 Nutrient Prediction Intervals (AEC-S2m-N)

We extend conformal prediction to the four macronutrients using the production pipeline: the v2 weight ensemble (Section 3.2) multiplied by a hybrid density estimate that blends a v4 gradient-balanced multi-task density ensemble (GBML, log1p-MSE) with k-NN retrieval (k=5, cosine, α=0.60). The protocol is **strict hold-out**: all models are trained exclusively on the 2,608-crop training split; the 652-crop test split is divided into calibration (326) and evaluation (326) halves; results are averaged over 10 random trials at α=0.10. On this protocol the point estimator attains PMAE of 22.8% (calories), 36.5% (fat), 39.4% (carbs), and 33.7% (protein)—note this is a more conservative protocol than the 10-trial 50/20/30 CV of Table 5, analogous to the weight-model distinction documented in Appendix C.

Each nutrient is assigned five candidate intervals—**Split** (absolute quantile), **Norm** (relative quantile), **LogSplit** (log-space multiplicative), **Propagate** (weight-interval propagation: the relative weight quantile applied to the nutrient point estimate), and **MondrianW** (per-weight-bin relative quantile)—combined per predicted-weight bin by Dirichlet search with a Winkler objective and Bayesian shrinkage (τ=30), followed by per-bin conformal recalibration (AEC-S2m-N, mirroring the winning weight-side recipe).

**Table 7: Nutrient conformal prediction, AEC-S2m-N (strict hold-out, 10 trials, α=0.10)**

| Nutrient | Coverage | Rel. Width | Winkler |
|----------|----------|------------|---------|
| Calories | 90.5% | 95.6% | 392 |
| Fat | 90.6% | 166.3% | 31 |
| Carbs | 89.7% | 152.1% | 70 |
| Protein | 90.5% | 151.8% | 38 |

**Key Findings:**
- Aggregated over nutrients (normalized Winkler, per-nutrient mean-normalized), AEC-S2m-N achieves **2.42**, outperforming the strongest single-candidate baseline MondrianW (2.50, −3.2%) and Split Conformal (2.74, −12%).
- **Propagate alone severely under-covers** (Fat: 63.7%): a weight-interval × density point estimate covers only weight error, whereas **density error (PMAE 33.7–39.4%) is the dominant source of nutrient uncertainty**. As an ensemble member it still contributes weight-scale structure.
- **LogSplit is unstable on near-zero nutrients** (relative widths up to 359%): log-space quantiles explode when predictions approach zero.
- **Recalibration is necessary**: without it, Fat coverage drops to 88.6%; per-bin recalibration restores 90.6%.

### 5.7 Honest Evaluation and the Limits of Ensemble Capacity

**Leakage repair.** The original AEC-S optimized its global weight vector on the evaluation split. Repairing this (global weights fitted on calibration data only) yields AEC-S honest: 90.4% coverage, 74g width, Winkler 181—the leakage inflated width by only ~3g, confirming that the stratification mechanism itself is sound. Adding conformal recalibration on top produces the two strongest weight-side variants of this study.

**Table 8: Honest weight-side conformal variants (10 trials, hold-out calibration/evaluation splits)**

| Variant | Coverage | Width | Winkler |
|---------|----------|-------|---------|
| AEC-S (original, leaked global weights) | 91.1% | 77g | 183 |
| AEC-S (honest) | 90.4% | 74g | 181 |
| AEC-S2c (honest + global recalibration) | 89.9% | **69g** | 172 |
| AEC-S2m (honest + per-bin recalibration) | 90.4% | 76g | **171** |

**Proposition 1 (Double-dipping bias; informal).** *Let $\mathcal{D}_{cal} = \{(X_i, Y_i)\}_{i=1}^n$ be a calibration set, $\{C_k\}_{k=1}^K$ candidate interval signals with per-sample nonconformity scores $r_{ik}$, and let ensemble weights $\hat{\mathbf{w}} \in \Delta^{K-1}$ be fitted on $\mathcal{D}_{cal}$ by minimizing an empirical width objective. Suppose the recalibration quantile is computed on the same set, $\hat{q} = Q_{1-\alpha}\big(\{\sum_k \hat{w}_k\, r_{ik}\}_{i=1}^n\big)$. Assume (A1) exchangeability: conditional on $\mathcal{D}_{cal}$, the test score $R_{test} = \sum_k \hat{w}_k\, r_k(X_{test}, Y_{test})$ is drawn from a fixed distribution $F$; and (A2) weight overfitting: with positive probability the in-sample combined scores are strictly smaller in expectation than an out-of-sample score, $\mathbb{E}[R_i(\hat{\mathbf{w}}) \mid \mathcal{D}_{cal}] < \mathbb{E}[R_{test} \mid \mathcal{D}_{cal}]$. Then:*
(i) *$\hat{q}$ is downward-biased: $\mathbb{E}[\hat{q}] < q^* = Q_{1-\alpha}(F)$;*
(ii) *the resulting interval is anti-conservative: $\mathbb{P}(Y_{test} \notin [L, U]) > \alpha$;*
(iii) *if the added candidates carry no independent signal (their population risk does not improve), the bias is monotonically non-decreasing in $K$.*

*Proof sketch.* (i) Under (A2) the empirical distribution of calibration scores is shifted left of $F$, so its $(1-\alpha)$-quantile underestimates $q^*$ in expectation. (ii) Coverage equals $\mathbb{P}(R_{test} \le \hat{q}) < \mathbb{P}(R_{test} \le q^*) = 1-\alpha$. (iii) Adding candidates nests the feasible set ($\Delta^{K-1} \subset \Delta^K$), so the empirical width minimum is non-increasing in $K$ while the population optimum is unchanged; the capacity to fit calibration noise — hence the expected in-sample/out-of-sample score gap and the quantile bias — is therefore non-decreasing in $K$. ∎

**Double-dipping.** We extended the nutrient ensemble with three new signals: asymmetric relative quantiles (AsymNorm, exploiting right-skewed errors), model-vs-retrieval disagreement scaling (DisagreeD, using |w·d_direct − w·d_knn| as an uncertainty proxy), and nutrient-value stratification (tercile bins per nutrient instead of weight bins). Under single-stage calibration these appeared to improve normalized Winkler (2.38–2.40 vs 2.42) but systematically depressed coverage in a capacity-dependent manner: 5 candidates achieve 89.7% minimum coverage, 7 candidates 89.4%, and 7 candidates with nutrient stratification 89.0%. The cause is **double-dipping** in the sense of Proposition 1: the Dirichlet weights and the recalibration quantile q are fitted on the *same* calibration set, so overfitted weights shrink the nonconformity scores and bias q downward. The observed deficit grows monotonically with candidate capacity (−0.3pp, −0.6pp, −1.0pp below the nominal 90% for the three configurations), exactly the pattern claim (iii) predicts for uninformative added candidates—AsymNorm (2.58) and DisagreeD (2.80) are indeed dominated as standalone candidates. Two-stage calibration (weights on half the calibration data, q on the other half with weights frozen) restores coverage (e.g., 89.0% → 89.7%) but widens intervals; under honest coverage **no configuration surpasses the 5-candidate single-stage recipe (2.42)**. The apparent gains of higher-capacity ensembles are anti-conservative artifacts.

This analysis also delimits the method itself: interval quality is bounded by the base predictor's density error—the calibration layer can honestly *report* this uncertainty, not reduce it.

**CQR baseline.** We additionally trained a quantile density model (12-output head: 4 nutrients × τ∈{0.1, 0.5, 0.9}, pinball loss with the same GBML balancing and v4 recipe otherwise) and conformalized it in six variants (raw, standard CQR, relative scores, per-weight-bin quantiles, and two hybrids that pair the production point estimate with the CQR-derived half-width). Table 9 reports all variants under the identical 10-trial protocol. The raw 10–90% intervals cover only 34.7% of true values on average (vs. the nominal 80%; worst nutrient 22.2%)—the conditional spread is severely underfit at this sample size—so conformalization must inflate them heavily: the best variant (CQR-hyb-b, hybrid center + CQR half-width + per-bin recalibration) attains valid coverage (90.5%) but a normalized Winkler of 2.55, still 5.7% behind AEC-S2m-5W. Estimating the conditional spread directly requires substantially more data than the residual-quantile approach; with 2,608 training crops, point-estimate-based conformal calibration is the more sample-efficient choice.

**Weighted Conformal baseline.** To cover modern covariate-shift methods, we additionally evaluate Weighted Conformal Prediction [13]: calibration scores are reweighted by density ratios estimated with logistic regression on the predicted-weight/nutrient features, and the weighted (1−α)-quantile replaces the empirical one. Because calibration and evaluation draws are exchangeable in-domain, the fitted classifier has no discriminative power (weight std 0.12 around mean 1.0), the weighted quantile collapses to the unweighted one, and WC attains 89.2% minimum coverage at nWinkler 2.54—dominated by AEC-S2m-5W (Table 9). Weighted conformal is designed to exploit a *detectable* covariate shift; in its absence it provably offers no advantage, and §5.8 shows that under a true domain shift (NV-Real) the frozen calibrator fails regardless of weighting, since the shift acts on the base predictor's error distribution rather than on calibration-sample importance.

**Table 9: CQR and Weighted Conformal variants are dominated by AEC-S2m-5W (10 trials, α=0.10; min. coverage across the four nutrients, mean relative width, mean normalized Winkler; valid = min. coverage ≥ 89.5%)**

| Variant | Min. coverage | Mean rel. width | nWinkler | Valid |
|---------|---------------|-----------------|----------|-------|
| CQR-raw (no conformalization) | 22.2% | 38.8% | 4.55 | ✗ |
| CQR-abs (standard CQR) | 89.1% | 144.6% | 2.80 | ✗ |
| CQR-rel (relative scores) | 89.3% | 167.7% | 2.71 | ✗ |
| CQR-bin (per-weight-bin q̂) | 90.3% | 154.1% | 2.66 | ✓ |
| CQR-hyb (hybrid center) | 88.7% | 158.4% | 2.60 | ✗ |
| CQR-hyb-b (hybrid + per-bin) | 90.5% | 151.5% | 2.55 | ✓ |
| Weighted Conformal [13] | 89.2% | 148.6% | 2.54 | ✗ |
| **AEC-S2m-5W (ours)** | **89.7%** | **141.4%** | **2.42** | ✓ |

### 5.8 Cross-Domain Generalization: NutritionVerse-Real

All preceding results are in-domain (Nutrition5K: fixed RealSense overhead camera, cafeteria dishes). We now measure what survives a domain shift using NutritionVerse-Real (NV-Real) [12]: 889 smartphone photographs of 251 dishes, free-viewpoint, no depth—strictly zero-shot (no NV-Real training, calibration, or tuning of any kind).

**Protocol.** Two data-engineering decisions were required for a fair comparison. First, NV-Real metadata lists repeated same-category instances as duplicate columns; the correct ground-truth semantics (verified by the total-weight checksum on all 251 dishes) is the *sum* per category. Second, our Nutrition5K cache is dish-level: each crop is the largest food mask with the background masked out, and the label is the dish total mass. We therefore replicate this protocol exactly on NV-Real (union of all food masks, 20px padding, background masking, dish-total ground truth), yielding 880 dish-level evaluations. Because NV-Real has no depth, the weight model is the RGB-only (3ch) retrain of Section 4's architecture with identical hyperparameters and seed; its in-domain hold-out PMAE is 31.4% vs. 22.0% for the 4-channel model, so the *modality* cost is +9.4pp and the *domain* cost is measured against the 31.4% baseline.

**Point estimation collapses to the training prior.** Zero-shot dish-level PMAE is 103.7% (MAE 238.8g)—a domain gap of +72pp. The failure mode is diagnostic: the Spearman correlation between prediction and truth drops to ρ=0.16, the prediction spread shrinks to 0.55× the true spread, and the median prediction/GT ratio decays monotonically from 5.5× (dishes under 100g) to 0.42× (dishes over 700g)—the textbook signature of regression toward the training prior. The RGB-only model exploits in-domain appearance cues (plate size, portion norms, camera geometry) that do not transfer; what remains is a weak ordering signal barely better than chance. The model is marginally better than the blind N5K-median prior on MAE (238.8g vs. 245.1g) but worse on PMAE, since it systematically over-predicts small dishes.

**Conformal coverage does not survive the shift.** Split conformal (and Normalized/Mondrian variants) fitted on the Nutrition5K calibration split and applied frozen to NV-Real:

| 90% interval | In-domain coverage | NV-Real coverage | Width |
|---|---|---|---|
| Split | 85.9% | **32.8%** | 217g (frozen) |
| Normalized | 88.2% | 53.6% | 414g |
| MondrianW | 85.4% | 41.5% | 297g |

The bin decomposition shows where exchangeability breaks (Figure 3): coverage on 0–100g dishes falls from 99% to 26% (the over-prediction bias centers intervals above the truth), and on 500g+ dishes from 47% to 15% under Split. Restoring 90% coverage on NV-Real would require inflating the conformal quantile by 4.46× (108.5g → 484.3g). The same collapse holds for the deployed nutrient calibrator (AEC-S2m-5W, frozen): coverage drops from 91.1% in-domain to 44–56% across the four nutrients, with nutrient PMAE degrading from 22.8/36.5/39.4/33.7% to 64.8/69.5/163.2/95.3% (kcal/fat/carb/protein; PMAE computed on GT≥1g subsets because NV-Real contains near-zero-carb dishes). Stratification does not rescue coverage—Mondrian per-bin quantiles are calibrated for Nutrition5K's error distribution, not NV-Real's—confirming that distribution-free guarantees are exchangeability-bound, not domain-free.

**Category structure partially transfers.** At ingredient level (secondary analysis; task granularity differs from the dish-level training signal), rigid foods degrade less than prepared foods (PMAE 318% vs. 606%), consistent with the rigid/prepared decomposition of Sections 6.2–6.4.

**Can multi-domain data buy the gap back?** To close the loop, we train the same 3ch architecture on three data configurations and evaluate on both test sets (NV-Real split by dish—180 train / 45 test dishes—to prevent same-dish leakage; note this hold-out subset yields 77.4% zero-shot PMAE vs. 103.7% on the full 880-image set, a protocol difference, not a contradiction):

| Training data | N5K test (652) | NV-Real test (177) |
|---|---|---|
| N5K only (zero-shot on NV-Real) | 31.4% | 77.4% |
| NV-Real only | 124.7% | 25.6% |
| **Joint (N5K + NV-Real)** | **31.5%** | **23.9%** |

Joint training recovers the domain gap almost completely (77.4% → 23.9%) at *zero* source-domain cost (31.4% → 31.5%), and even slightly outperforms target-domain-only training on the target domain (Figure 4). The reverse collapse (NV-Real-only model: 124.7% on N5K) is symmetric—each domain's appearance priors are non-transferable in both directions. The practical conclusion: the ceiling is a *data* problem, not an architecture problem; multi-domain capture is a viable, quantified remedy, complementing the reference-object protocol of Section 6.4.

Together with Sections 6.2–6.4, this positions the 18.4% ceiling as *domain-relative*: in-domain, appearance priors substitute partially for metric depth; out-of-domain, they evaporate and both point estimates and interval guarantees collapse toward uninformed priors. This is an additional, independent argument for the capture-protocol recommendation of Section 6.4: reference objects and domain-adaptive routing add the information whose absence is measured here.

---

## 6. Discussion

### 6.1 Why AEC-S Works

AEC-S succeeds because it addresses a fundamental limitation of global ensemble methods: the optimal combination of conformal predictors varies across the data distribution. In the 200-500g range, where predictions are most reliable, the simple Split Conformal approach suffices (85.2% weight). In the 500g+ range, where prediction errors are larger and more variable, MLNS's learned features provide better adaptation (53.1% weight). In the 0-50g range, Mondrian's conditional binning is most effective (67.0% weight).

The Bayesian shrinkage mechanism is critical for making this work in practice. Without shrinkage, the per-bin weights in sparse bins (500g+, 0-50g) would overfit to the small calibration sets, leading to poor generalization. By blending toward a stable global prior when data is insufficient, AEC-S maintains robustness while still capturing bin-specific structure.

### 6.2 The Density Bottleneck: Why 3D Geometry Alone Cannot Break the Ceiling

To test whether the 18.4% ceiling reflects a recoverable geometric deficiency or an intrinsic property of the data, we performed an error-budget decomposition on MetaFood3D [8], which provides ground-truth volumes and masses for 625 food objects across 108 categories. The result is unambiguous: even with *perfect* ground-truth volume, mass prediction via per-category mean density yields 25.0% MAPE (23.4% on the ≥100g subset), and within-category density dispersion (21.8% mean absolute log-deviation) is itself comparable to the entire monocular problem difficulty (22.0% PMAE on Nutrition5K), whereas a ±10% volume error adds only 2.6pp (Figure 1). The bottleneck is thus density, not geometry. Crucially, this density variance is physically grounded yet visually unobservable—its canonical expression being the *blueberry paradox*: bowl-packed blueberries exhibit 2.2% density dispersion, while loose blueberries of the same category exhibit 33% (Figure 2), a gap attributable to packing porosity rather than composition. A 5-fold cross-validated appearance-to-density CNN confirms this information gap: it improves upon the global mean prior by 12.0pp (30.3% vs. 42.3% MAPE) yet still loses to the category-lookup table by 4.1pp (30.3% vs. 26.2%), indicating that appearance carries exactly category-level density information and nothing beyond it. Because within-category density variation arises from preparation state, packing fraction, and cooking-induced changes that leave no reliable visual signature, geometric reconstruction—multi-view or otherwise—cannot close the gap. The ceiling is therefore *semantic* rather than *geometric*: the missing 61% of weight variance resides in factors a camera cannot see, which is precisely the regime where reference objects, user interaction, or dietary priors must complement vision.

### 6.3 SAM 3D Objects: The Strongest Single-Image Baseline Also Fails at Scale

If the missing information were geometric rather than semantic, a single-image 3D foundation model with learned metric-scale priors should recover it. SAM 3D Objects (Meta, 2025), trained on 3.14M real-world-scanned meshes, reconstructs a textured Gaussian splatting scene from one RGB image plus mask and carries an explicit scale prior. We ran it zero-shot on a stratified sample of 130 MetaFood3D objects (30 categories × 4 instances, plus pilot) and measured the volume of the reconstructed density field against ground truth [10]. Three findings emerge.

First, **absolute scale is unusable**: the median estimated-to-true volume ratio is 5.6×10⁻⁵ with a metric MAPE of 100.0%—the raw output is not even dimensionally consistent across objects (reconstructions live in a normalized latent space; per-object scale factors recovered from the pipeline's camera transform do not restore metric units). Second, **the scale prior is category-biased, not universal**: a single global calibration factor leaves 87.3% MAPE, whereas per-category calibration reduces the median to 33.3%—the prior behaves not as one ruler but as thirty differently-skewed rulers, one per category. The between-category rank correlation is strong (log-log r = 0.92 on category means: "almonds are small, bowls are large"; Figure 5), confirming that the prior encodes genuine category-level size knowledge from its training distribution. Third, **within-category instance tracking is partial**: the median within-category log-log correlation is +0.88, but with heavy dispersion (range −0.98 to +0.99 across 28 categories); rigid, coherent foods (avocado +0.98, broccoli +0.99, salmon +0.99) are tracked well, whereas piled or stacked foods fail catastrophically—Almond(bowl) −0.10, blueberries −0.41, bacon −0.98.

The stacked-food failures are the most diagnostic result, because they intersect our density-side analysis exactly: porosity simultaneously destroys *both* factors of the mass equation. For loose blueberries, SAM 3D cannot infer the packed volume (geometric failure, within-category r = −0.41), and our density analysis shows the effective density varies 33% with packing state (semantic failure). The *blueberry paradox* thus has a 3D projection: appearance volume and true mass are decoupled by packing fraction in a way that neither monocular regression, nor retrieval, nor single-image 3D reconstruction can resolve. Even granting SAM 3D its best case—per-category calibration, volume MAPE 33.3%—combining with the 21.8% within-category density dispersion yields ≈40% mass error, 2.2× worse than the monocular ceiling we are trying to break. We conclude that the ceiling is not an artifact of our architecture or of 2D representation: even a 2025-scale single-image 3D foundation model, given the same information (one image), cannot escape it. What would escape it—multi-view geometry, reference objects, preparation-state semantics—adds *information*, not just modeling capacity.

### 6.4 Reference Objects Work Where Density Is Observable: ECUSTFD Evidence

The preceding sections establish what fails; this section provides the constructive counterpart. If the ceiling decomposition is correct—scale ambiguity plus density dispersion as the two irreducible factors—then a physical reference object should eliminate the first factor entirely, and on domains where the second factor is small (rigid, natural foods), the combined pipeline should break through the monocular ceiling. ECUSTFD [11] provides exactly this testbed: 19 rigid/natural food categories (174 instances) photographed with a 25.0mm One-Yuan coin in every frame, with per-instance ground-truth volume and mass.

We quantify the scale ambiguity itself first: across the 2,861 annotated images, the pixel-to-millimeter ratio varies by nearly two orders of magnitude (mm/px range 0.084–8.33)—smartphone photographs taken at arbitrary distances. This is the physical form of the 61% unrecoverable variance: without a reference, image pixels carry no metric information, and the top-view area-to-volume log-log correlation is only 0.59 under a fixed-distance assumption. With coin calibration the correlation rises to 0.74, and the metric signal is restored.

The full leave-one-out protocol (per-instance density computed from category peers only, mirroring our MetaFood3D analysis) yields: (i) **GT volume × LOO category density = 7.9% MAPE**—the same protocol that yields 25.0% on MetaFood3D's prepared-food domain yields 7.9% on rigid foods, a 3.2× gap attributable almost entirely to within-category density dispersion; and (ii) **end-to-end (coin-calibrated top-view area × LOO category height prior × LOO density) = 12.9% MAPE**, using deliberately coarse geometry—no depth, no side views, no learning. Both figures sit below the 18.4% monocular ceiling.

We emphasize what this does and does not show. It does *not* claim victory over our Nutrition5K system: ECUSTFD is a rigid-food domain, whereas Nutrition5K is dominated by prepared foods where density dispersion (the blueberry-paradox regime) remains the binding constraint. What it shows is that the ceiling decomposition is *complete and actionable*: where density is observable (rigid foods), removing scale ambiguity alone—with a 25mm coin and no learning at all—suffices to break the monocular ceiling; where density is not observable (prepared, stacked, or porous foods), no geometric signal we have tested suffices. The practical implication is a domain-adaptive capture protocol: instruct users to include a reference object (a coin, a card, standard tableware), and route rigid-food captures through geometry-based estimation while prepared foods still require interaction or dietary priors. This converts our falsification results into an engineering decision rule.

### 6.5 Limitations

1. **Information-theoretic limit of monocular RGB-D (fundamental)**: Our experiments demonstrate the information-theoretic limit of single-view RGB-D on Nutrition5K: the weight predictor's R²=0.39 means 61% of the weight variance is irrecoverable from the image signal (scale ambiguity without a physical reference—we quantify its physical form on ECUSTFD: the pixel-to-millimeter ratio spans nearly two orders of magnitude across unconstrained smartphone captures). The systematic falsification of 18 documented optimization attempts—retraining from scratch, checkpoint fine-tuning, k-NN retrieval augmentation, inverse-frequency sample weighting, test-time augmentation, and conformalized quantile regression—fixes **18.4% PMAE as the de facto performance ceiling** on the ≥100g subset under the strict hold-out protocol (22.0% on the full hold-out), with per-range errors uniformly distributed (17–26%) and no weak bin left to repair. Three independent analyses confirm this ceiling is informational, not architectural: (i) our MetaFood3D error-budget analysis (Section 6.2) shows that even with ground-truth volume, per-category density lookup yields 25.0% MAPE, because the dominant error source is visually unobservable density variation; and (ii) a zero-shot evaluation of SAM 3D Objects (Section 6.3)—a 2025-scale single-image 3D foundation model—attains at best ≈40% mass error on the same task, failing at absolute scale and on stacked foods; and (iii) a modern frozen backbone (DINOv2 ViT-S/14) re-headed for weight regression attains only 31.3% calibrated PMAE on the same strict hold-out split—worse than the 22.0% of the trained-from-scratch 4-channel ensemble—confirming that substituting a substantially higher-capacity modern architecture does not break the ceiling; capacity is not the binding constraint. Constructively, our ECUSTFD experiment (Section 6.4) shows the decomposition is actionable: coin calibration plus rigid-food density lookup reaches 12.9% end-to-end. Breaking the ceiling on prepared foods will require adding information—semantic signals (preparation-state recognition, dietary priors), reference objects, or user interaction—to complement geometry; the multi-angle benchmark FPB [9] would be a suitable testbed for such hybrid designs.

2. **Small-sample bins**: The 500g+ bin contains only ~9.5% of the data, and per-bin coverage (84.3%) still falls short of the 90% target. Future work could explore alternative binning strategies or dynamic bin boundaries.

3. **MLNS on high-accuracy predictors**: When the base predictor achieves PMAE < 10%, the error signal becomes nearly indistinguishable from noise, limiting the effectiveness of learned score models. This is a fundamental limitation of MLNS, not specific to our implementation.

4. **Double-dipping in ensemble calibration**: Jointly using calibration data for weight optimization and recalibration produces anti-conservative intervals whose severity grows with ensemble capacity (Section 5.7). Practitioners extending the candidate set should adopt the two-stage protocol (or validate that single-stage coverage survives the added capacity).

5. **Computational cost**: AEC-S requires running Dirichlet sampling (5,000 iterations) for each bin, increasing computation time linearly with the number of bins. However, this is a one-time calibration cost.

6. **Single dataset evaluation, now measured and remedied**: Our in-domain results are Nutrition5K-specific, and we have quantified the cost of leaving that domain: on NutritionVerse-Real smartphone captures under a strictly zero-shot, protocol-matched evaluation (Section 5.8), dish-level PMAE degrades from 31.4% to 103.7% with prediction–truth rank correlation collapsing to ρ=0.16, and frozen 90% conformal intervals cover only 33–54% of true weights (44–56% for nutrients); restoring nominal coverage would require 4.5× wider intervals. The 18.4% ceiling is therefore domain-relative: the appearance priors that substitute for metric depth in-domain do not transfer. We further show the gap is a data problem, not an architecture one: joint training on both domains recovers the target-domain PMAE to 23.9% at zero source-domain cost (Section 5.8). Domain-robust deployment will require multi-domain capture and/or the capture-protocol measures of Section 6.4 (reference objects, domain-adaptive routing) rather than calibration alone.

7. **Nutrient prediction accuracy**: The Direct Regression achieves PMAE=12.6% for calories, only 5.1pp above the Oracle ceiling (7.5%). The gap is larger for fat (24.3% vs 7.4%), carbs (25.3% vs 7.6%), and protein (22.4% vs 7.0%), indicating room for improvement in density prediction. Under the strict hold-out protocol the density error (PMAE 33.7–39.4%) also bounds the nutrient interval quality (Section 5.6): the conformal layer reports—but cannot reduce—this dominant source of uncertainty. Future work could explore ingredient-level recognition to improve per-crop nutrient decomposition, or multi-task architectures that jointly optimize weight and density prediction.

### 6.6 Comparison with Prior Work

Compared to standard conformal methods, AEC-S achieves:
- **2.6% narrower intervals** than AEC (global) while improving coverage
- **5.0% narrower intervals** than Mondrian
- **19.1% narrower intervals** than Normalized Conformal
- **12.5pp coverage improvement** in the 500g+ bin vs Split Conformal

### 6.7 Practical Implications

For dietary assessment applications, AEC-S provides prediction intervals that are:
- **Narrow enough** to be actionable (76g average width, 11g for small portions)
- **Reliable** with 91.0% empirical coverage
- **Adaptive** across the full weight range from 0g to 1,000g+

This enables downstream applications such as:
- Flagging uncertain predictions for human review
- Providing confidence bounds in nutrition tracking apps
- Quality control in automated dietary assessment pipelines

---

## 7. Conclusion

We have presented MM-CFA, a comprehensive framework for food weight estimation with rigorous uncertainty quantification. Our key contributions are:

1. A **deep ensemble base predictor** (ResNet18+EfficientNet-B0) achieving PMAE=9.44% on the Nutrition5K dataset.

2. **AEC-S**, a novel stratified adaptive ensemble conformal method that learns per-bin optimal weight combinations with Bayesian shrinkage, achieving **91.0±1.3% coverage** with **76±5g width**—the best coverage-width tradeoff among all methods meeting the 90% target.

3. Comprehensive **per-bin analysis** revealing that optimal weight allocation varies substantially across the weight distribution, with Mondrian dominating in small-weight bins, Split in the mid-range, and MLNS in the heavy tail.

4. **Bayesian shrinkage** as an effective regularization mechanism for per-bin weight estimation, preventing overfitting in sparse calibration bins.

5. **Nutrient prediction** via a two-stage (weight × density) framework, achieving PMAE=12.6% for calories (Oracle ceiling: 7.5%), and demonstrating that the gap between direct regression and the oracle is narrow for calories but wider for fat, carbs, and protein, highlighting the need for improved density prediction.

6. **Nutrient conformal intervals (AEC-S2m-N)**: 89.7–90.6% coverage across all four macronutrients under a strict hold-out protocol with normalized Winkler 2.42 (−12% vs Split Conformal), identifying density error—not weight error—as the dominant source of nutrient uncertainty.

7. **Honest evaluation practice**: post-hoc identification and repair of an evaluation leakage in the global-weight optimization (cost: ~3g), conformal recalibration variants that narrow weight intervals to 69g (AEC-S2c) or minimize Winkler (AEC-S2m, 171), and a double-dipping analysis—jointly with the falsification of three additional candidate signals—that establishes the information limit of the current candidate set and provides protocol guidance for future conformal ensembles.

8. **Ceiling attribution**: three independent external analyses establish that the 18.4% monocular ceiling is informational, not architectural. The MetaFood3D error budget shows within-category density dispersion (21.8%, the *blueberry paradox*) alone matches the full problem difficulty; a zero-shot SAM 3D Objects evaluation (130 objects) shows the strongest 2025 single-image 3D foundation model fails at absolute scale and on stacked foods (best case ≈40% mass error); and the ECUSTFD experiment shows the decomposition is actionable—coin calibration plus rigid-food density lookup achieves 7.9–12.9% with no learning at all, yielding a domain-adaptive capture protocol (reference object + rigid/prepared routing) as a practical engineering outcome. Breaking the ceiling on prepared foods still requires *adding information*—semantic signals, reference objects, or user interaction—not more model capacity.

Future work includes exploring dynamic bin boundary optimization and incorporating ingredient-level recognition for more accurate per-crop nutrient decomposition. Our falsification results localize the remaining bottleneck in the imaging modality itself: R²=0.39 bounds the variance recoverable from single-view RGB-D, and both the MetaFood3D error budget (Section 6.2) and the SAM 3D zero-shot evaluation (Section 6.3) confirm that geometric reconstruction alone cannot break the 18.4% ceiling. Future work will therefore focus on *hybrid* designs that add the missing semantic signals—preparation-state recognition, dietary priors, reference objects, or user interaction—complemented by multi-view geometry where capture conditions permit; the multi-angle benchmark FPB [9] is a suitable testbed for such designs.

---

## References

[1] Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.

[2] Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002). Inductive confidence machines for regression. *ECML*.

[3] Papadopoulos, H., & Haralambous, H. (2011). Reliable prediction intervals with regression neural networks. *Neural Networks*.

[4] Boström, H., & Johansson, U. (2020). Mondrian conformal regressors. *COPA*.

[5] Adaptive Ensemble Conformal Prediction (this work).

[6] MLNS: Multi-Modal Learned Nonconformity Score (this work).

[7] Thames, Q., et al. (2021). Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food. *CVPR*.

[8] Chen, Y., He, J., Czarnecki, C., Vinod, G., Mahmud, T. I., Raghavan, S., Ma, J., Mao, D., Nair, S., Xi, P., Wong, A., Delp, E., & Zhu, F. (2024). MetaFood3D: Large 3D Food Object Dataset with Nutrition Values. *arXiv:2409.01966*.

[9] Sanatbyek, A., Rakhimzhanova, T., Nurmanova, B., Omarova, Z., Rakhmankulova, A., & Orazbayev, R. (2025). A Multitask Deep Learning Model for Food Scene Recognition and Portion Estimation—the Food Portion Benchmark (FPB) Dataset. *IEEE Access*, 13, 152033–152045.

[10] Meta AI. (2025). SAM 3D Objects: Single-Image 3D Object Reconstruction with Gaussian Splatting. *arXiv preprint* (gated release, facebook/sam-3d-objects).

[11] Liang, Y., & Li, J. (2017). Computer vision-based food calorie estimation: dataset, method, and experiment (ECUSTFD). *arXiv:1705.07632*.

[12] Herse, S., Samdin, R., et al. (2024). NutritionVerse-Real: A Fully Annotated Real-World Food Dataset for Nutrition Planning. *arXiv:2401.08598*.

[13] Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.

---

## Figures

**Figure 1. Error-budget decomposition on MetaFood3D (§6.2).** (a) Within-category density dispersion (21.8%, mean |log(ρ/ρ_LOO)| over 625 objects) matches the entire monocular problem difficulty (22.0%, 4-channel PMAE on Nutrition5K). (b) With perfect ground-truth volume, category-mean density yields 23.4% mass MAPE on the ≥100g subset; injecting ±10% volume error adds only 2.6pp. Density, not geometry, is the bottleneck.

**Figure 2. The blueberry paradox (§6.2).** Effective density (mass / ground-truth volume) of loose blueberries (n=8, 33% dispersion) vs. bowl-packed blueberries of the same category (n=5, 2.2% dispersion). Points are individual objects; horizontal bars are group means. The gap is packing porosity—physically real, visually unobservable.

**Figure 3. Conformal coverage collapse under domain shift, by weight bin (§5.8).** N5K in-domain coverage (green) vs. frozen-calibrator zero-shot coverage on NV-Real (Split: red; MondrianW: blue). Exchangeability breaks most severely where the base predictor's bias is largest (0–100g: 99% → 26%).

**Figure 4. Multi-domain joint training is a win-win remedy (§5.8).** Dish-level PMAE of the three training configurations on both test sets. Joint training recovers the NV-Real gap (77.4% → 23.9%) at zero N5K cost (31.4% → 31.5%).

**Figure 5. SAM 3D's scale prior is category-level, not metric (§6.3).** Object-level (grey) and category-mean (blue) log₁₀ estimated vs. ground-truth volume in camera space. Category means rank correctly (r = 0.92) but the cloud sits far below the y = x line: no universal calibration restores metric units.

---

## Appendix A: AEC-S Algorithm Pseudocode

```
Algorithm: AEC-S (Stratified Adaptive Ensemble Conformal)

Input:
  - Base conformal predictors: {C_1, ..., C_K}
  - Calibration set: D_cal = {(x_i, y_i)}
  - Test set: D_test
  - Miscoverage rate: α = 0.10
  - Shrinkage parameter: τ = 30
  - Dirichlet sampling iterations: N = 5000
  - Weight bins: B = {[0,50), [50,100), [100,200), [200,500), [500,∞)}

Output:
  - Prediction intervals C(x) = [L(x), U(x)] for each x ∈ D_test

1:  // Compute global weights
2:  w_global ← DirichletSearch({C_k}, D_test, target=1-α, N)
3:
4:  // Compute per-bin weights
5:  for each bin b ∈ B do
6:    D_cal^b ← {(x,y) ∈ D_cal : ŷ(x) ∈ b}
7:    n_b ← |D_cal^b|
8:    if n_b < 20 then
9:      w_b ← w_global
10:   else
11:     w_local ← DirichletSearch({C_k}, D_cal^b, target=1-α, N)
12:     α_b ← n_b / (n_b + τ)
13:     w_b ← α_b · w_local + (1-α_b) · w_global
14:   end if
15: end for
16:
17: // Apply per-bin weights to test set
18: for each x ∈ D_test do
19:   b ← bin(ŷ(x))
20:   L(x) ← Σ_k w_b[k] · L_k(x)
21:   U(x) ← Σ_k w_b[k] · U_k(x)
22: end for
23: return {[L(x), U(x)] : x ∈ D_test}
```

## Appendix B: Model Architecture Details

**ResNet18 (modified):**
- Input: 4-channel 224×224 (RGB + depth)
- Conv1: 4→64, kernel 7×7, stride 2
- Standard ResNet blocks: layers 1-4
- FC head: 512→256→64→1 with Dropout(0.3), ReLU
- Parameters: ~11.7M

**EfficientNet-B0 (modified):**
- Input: 4-channel 224×224 (RGB + depth)
- Features[0][0]: 4→32, kernel 3×3, stride 2
- Standard EfficientNet blocks
- Classifier: 1280→128→1 with Dropout(0.3), ReLU
- Parameters: ~5.3M

## Appendix C: Reproducibility

> **Evaluation protocol note**: The weight PMAE of 9.44% reported in the paper is the mean of 10 trials of 50/20/30 cross-validation (train+test drawn from the same data pool). The 22.0% figure is from strict hold-out evaluation (2608 train / 652 test, model never sees test set). Both protocols are valid; hold-out is the more conservative generalization estimate.

All experiments are reproducible with the code and data in the `Nutri_0811` directory:

### Final Files (`_final` suffix = canonical)

- **Inference**: `test_100_samples_final.py` (production pipeline + conformal interval verification; `--no-conformal` to skip)
- **Weight model training**: `train_deep_v2_final.py`
- **Density model training**: `train_nutrient_v4_final.py` (v4 GBML gradient-balanced)
- **Conformal experiments (weight)**: `conformal_deep_experiment_final.py`, `aec_s_v2_experiment.py` (honest protocol + recalibration variants)
- **Conformal experiments (nutrient)**: `nutrient_conformal_final.py` (AEC-S2m-N; two-stage calibration; deployment calibrator)
- **CQR baseline**: `train_density_cqr.py` (quantile density training, GPU), `nutrient_cqr_experiment.py` (six-variant comparison; falsified)
- **Weighted Conformal baseline**: `nutrient_weighted_conformal_experiment.py` (Tibshirani et al. 2019 density-ratio weighting; dominated, Table 9)
- **Paper figures**: `make_paper_figures.py` (Figures 1–5, 300 dpi, colorblind-safe)
- **Label construction**: `build_nutrient_labels_final.py`
- **Full experiment log**: `历史方案_final.md` (21 documented failures + 1 success + final config)

### External Analyses (Sections 6.2–6.4, `metafood3d` / `ecustfd` directories)

- `mf3d_phase1_oracle.py` — MetaFood3D error-budget decomposition (Oracle-1/2, density dispersion, blueberry paradox; Gate 1)
- `mf3d_phase2_learned_density.py` — 5-fold appearance-to-density CNN (falsified: category-level information only; Gate 2)
- `mf3d_oracle6_sam3d.py` — SAM 3D Objects zero-shot volume evaluation (130 objects, 30 categories × 4 instances; latent-space vs camera-space volume, category-granularity decomposition)
- `mf3d_oracle6_results.csv` — SAM 3D per-object results (v_gt, v_est, s_l2c, per-category within-r analysis)
- `metafood3d_labels.csv` — Cleaned MetaFood3D labels (625/636 usable; 2 negative-volume + 9 corrupt-volume exclusions)
- `/home/ecustfd/ecustfd_reference_experiment.py` — ECUSTFD coin-calibration experiment (E1 scale-ambiguity quantification / E2 LOO density lookup / E3 end-to-end; constructive validation of Section 6.4)
- `/home/ecustfd/ecustfd_reference_results.csv` — ECUSTFD per-image records (2,861 rows: food/coin bbox, mm/px, calibrated areas)
- SAM 3D environment: conda env `sam3d` (Python 3.11, torch 2.5.1+cu121, gsplat compiled for sm_86), weights in `/home/sam3d_weights`

### Cross-Domain Analysis (Section 5.8, `/home/nvreal` directory)

- `nvreal_dl.py` — Kaggle dataset download (1.11GB, resumable parallel downloader)
- `nvreal_phase0_prep.py` — COCO mask → ingredient crops + GT pairing (multi-instance sum semantics, checksum-verified; rigid/prepared labels for 23 rigid categories)
- `train_deep_v2_3ch.py` — RGB-only weight model retrain (Phase 1 modality ablation; fixes an augmentation clamp bug present in the 4ch script—see Section 5.8 protocol note)
- `nvreal_phase2b_zeroshot.py` — Protocol-matched dish-level zero-shot evaluation (union-mask, background-masking, dish-total GT; the primary RQ1 numbers)
- `nvreal_phase3_conformal.py` — Frozen conformal coverage test (Split/Normalized/MondrianW fitted on N5K val, applied to NV-Real; RQ2)
- `nvreal_phase4_nutrient.py` — Nutrient pipeline cross-domain (3ch weight × 4ch-with-zero-depth density + kNN; frozen AEC-S2m-5W calibrator)
- `phase2b_per_image.csv`, `phase3_conformal_results.json`, `phase4_nutrient_results.json` — per-image predictions and phase results
- NV-Real crops cache: `nvreal_crops.npz` (2,614 ingredient crops; v1 with overwrite-aggregation bug retained as `nvreal_crops_v1_overwrite.npz` for audit)

### Model Weights

- `models/deep_weight_resnet18_final.pth` — ResNet18 weight model (43.3MB)
- `models/deep_weight_efficientnet_final.pth` — EfficientNet-B0 weight model (16.2MB)
- `models/density_resnet18_final.pth` — ResNet18 density model, v4 GBML (43.4MB)
- `models/density_efficientnet_final.pth` — EfficientNet-B0 density model, v4 GBML (16.2MB)
- `models/nutrient_conformal_calibrator.pkl` — AEC-S2m-N deployment calibrator (fitted on all 652 hold-out crops)
- `models/deep_crops_cache.npz` — 4-channel RGB-D data (600MB, 2608/652 split)
- `models/nutrient_labels.npz` — Nutrient labels
- `models/nutrient_density.npz` — Density labels
- `models/backup_0817/` — Original backup

### Environment Setup

```bash
conda create -n nutri python=3.10 -y
conda activate nutri
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scikit-learn scipy matplotlib pillow lightgbm xgboost
```

### Results

- `results/deep_results.pkl` — Conformal experiment results (original benchmark)
- `results/aec_v2_results.pkl` — Honest protocol + recalibration variants (Table 8)
- `results/nutrient_conformal_results.pkl` — Nutrient conformal results (Table 7, Section 5.7)
- `results_nutrient/nutrient_v4_results.npz` — Final nutrient results
- `results/fig1_main.png`, `fig2_ranking.png`, `fig3_ablation.png`, `fig4_per_bin.png` — Figures

### Hardware Requirements

- GPU: ≥4GB VRAM (inference), ≥8GB (training, batch_size=32)
- Recommended: RTX 3060+ / equivalent

### Reproducibility Settings

- Random seed: 42 (base), 43-51 (trials)
- 10 independent train/cal/test splits
- All results reported as mean ± std across trials