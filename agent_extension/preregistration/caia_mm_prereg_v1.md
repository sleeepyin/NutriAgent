# CAIA-MM preregistration v1

Date: 2026-09-04
Protocol: Protocol A (frozen public base + X_test / external / new data only) + Level A public-data feasibility
Level B MM-CFA-Interact is out of scope for this automated run (physical collection cannot be executed by the script).

## Frozen claims

We will NOT claim: first nutrition Agent; first conformal Agent; first conformal active multimodal acquire; first conformal food questioning; first post-policy recalibration; first adaptive stopping; first AFA.

Main sentence: systematic benchmark of mechanism-aware, cost-sensitive physical and semantic information acquisition with joint conformal auditing for continuous food weight and nutrient estimation.

## Hypotheses

- H1: Under empirical coverage consistent with nominal 90%, CAIA-MM reduces normalized Winkler vs the best cost-matched fixed policy by >=10%.
- H2: Reference/depth helps rigid more than prepared; semantic helps prepared more than rigid.
- H3: Reusing pre-action q, or fitting policy and q on the same cal set, undercovers; terminal calibration on a frozen policy restores marginal coverage.
- H4: With zero target-domain cal we do not claim 90% coverage; coverage recovers monotonically with n=25,50,100,200 target cal.
- H5: LLM-only planner has higher invalid-action rate; numerical gain comes from constrained tool choice.

## Primary endpoint

Normalized interval score (Winkler / mean |y|) of CAIA-MM vs best cost-matched fixed policy, on episodes that are never used for policy or q, while empirical coverage is consistent with 90% (Clopper-Pearson + pre-registered one-sided exact binomial).

## Secondary endpoints

MAPE, WAPE (never PMAE), real action count / prior cost, OOD catastrophic-error recall, nutrient nWinkler on labeled subsets, abstain rate, selective risk.

## Actions and costs (prior units)

STOP 0, ASK_SEMANTIC 1, REQUEST_VIEW 2, REQUEST_REFERENCE 3, REQUEST_DEPTH 3, ABSTAIN 1.
T_max = 2.

## Calibration

Finite-sample order statistic: k=ceil((n+1)(1-alpha)), q=+inf if k>n. No np.quantile interpolation.
Four-way isolation: D_base / D_policy / D_cal / D_test.
Terminal R^pi is the efficiency main result.
All-tool-state R^forall is the safety result and is allowed to be conservative.
Path Mondrian only if group n>=50, else global q.
Anytime alpha-spending is a safety baseline.

## Go / No-Go (E1)

- If every action oracle mean gain <5% vs RGB-only: do not package a unified Agent (degrade A: conformal triage).
- If one fixed action is best on >90% of episodes: degrade B (single-action trigger).
- Unified Agent requires heterogeneous winners across subgroups.

## Failure of the Agent (do not spin)

Oracle best-action vs best fixed <5%; policy no better than uncertainty threshold; gain-model policy-val R2<=0; terminal intervals wipe out action gains; gains exist only with GT category / GT box / noiseless answers.

## Statistics

Episode-level paired bootstrap (default 10000, fallback 2000 if time-bound).
Coverage: Clopper-Pearson 95% + one-sided exact binomial vs 0.90.
Holm correction on secondary comparisons.
Independent unit: N5K cache-index (limitation recorded), NV dish, MF3D object, ECUSTFD instance.

## Locked hashes

This file is hashed before E7 locked test. Changing it after seeing test results voids the confirmatory claim.
