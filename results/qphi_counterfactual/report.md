# Gate ① — leakage audit and counterfactual Q

One-step branches from frozen rgb_train hold-out weak-step states. Those dishes did not drive the #51 food-code swap or the #53 coverage guard (both were read off the rgb_test development 505). They were also not the fit rows used to train Q or to set τ. A_legal drops STOP and ABSTAIN while a component is open and has no mass-estimator write. Each legal action is executed once from that same state. The oracle minimizes post-action absolute energy error. Mass error is logged and is not the selection objective. Q is the frozen checkpoint plus the existing terminal-margin fence and coverage guard. τ is not refit. No reinforcement learning. rgb_test labels are not read.

## Leakage audit

Conclusion: `no_dish_leakage`. `leakage_exists=false`.

Fit is 3679 dishes / 25864 steps (3678 dishes have more than one step). Hold-out is 380 dishes / 2771 steps (380 dishes have more than one step).

Dishes present in both files: 0. Rows on those dishes: 0. Dishes whose file disagrees with `pool_of_dish`: 0.

The cut is `sha256(dish_id)` modulo 10, not a shuffle of decision steps. Q training consumes the fit file and scores the hold-out file. No weights were updated after this audit.

## Sample

Population id: `qphi_counterfactual_onestep_holdout_v1`. Smoke run: False.

Hold-out dishes scored: 380. Decision states branched: 2771. Coverage-choice states (|A_legal|≥2 and STOP/ABSTAIN illegal): 2391. Headline keeps the 1914 of those whose dish has a frozen RGB mass forward, so RE_PORTION is a real specialist call. Of the headline states, 1877 move post energy error by more than 1 kcal.

Development dishes in `#53` paired file: 505. Overlap with this sample: 0. rgb_test ids in the sample: 0.

Frozen τ = 0.001645429361972572. Q sha256 prefix `a59ba6e49cf45220`. Matches calibration prefix: True. Mass weights, Q weights, and τ were not edited. Fresh mass forwards (cache misses only): 0. RE_PORTION cache hits / misses during branches: 919 / 226.

Portions from `RE_PORTION` use the frozen dish-level mass mean split equally across components, because these states have no vision gram hints. Ingredient grams are not hints. The #51 same-head swap is not applied inside a one-step branch: it runs on a finished multi-component plate, and the earlier components in these logs were written by the privileged teacher.

## Coverage-choice with a frozen mass forward (headline)

n = 1914. Identification is membership in the energy-optimal set. Regret is post energy error minus the oracle post energy error. Improvement is pre error minus post error, on states where both energies exist.

| Policy | Identification | Exact tie-break | Energy improvement (kcal) | Energy regret (kcal) | Mass improvement (g) | Mass regret (g) | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 1.0 | 1.0 | 82.19 | 0.0 | 27.8669 | 9.2806 | 1.7555 |
| Q_φ (frozen + fence) | 0.6165 | 0.6066 | 24.1441 | 49.9586 | 24.2622 | 12.383 | 1.6243 |
| Rule Utility | 0.6102 | 0.6003 | 23.034 | 50.914 | 25.2972 | 11.4923 | 1.7074 |
| Random (expectation) | 0.4338 | 0.38 | 4.038 | 78.152 | 12.1196 | 26.5302 | 2.1399 |
| Fixed Pipeline | 0.5914 | 0.5914 | 20.62 | 52.9916 | 18.8706 | 17.0234 | 1.5199 |

Reading. Route A scores energy. Mass is the parallel log of the energy-chosen action, including its regret against the mass oracle.

This table is not the #53 development MAE. #53 is coverage-constrained selection on the rgb_test 505, which is a development and diagnostic set. Those dishes are not in this sample. The #53 energy drop is the coverage constraint, not evidence that Q got smarter.

On this one-step hold-out, frozen Q_φ energy regret is 49.9586 kcal, rule utility 50.914, fixed pipeline 52.9916, random 78.152. Q is ahead of those three. The margin over the rule is 0.9554 kcal, while both remain about 49.9586 kcal behind the oracle. Identification is 0.6165 for Q and 0.6102 for the rule.

States whose energy-best legal action is the RE_DECOMPOSE no-op: 241 of 1914. On those states the no-op keeps the current error and every real tool increases it. Q regret there is 258.5335 kcal and the rule is 258.3722 kcal. Frozen Q cannot emit RE_DECOMPOSE: that head was masked because the fit weak steps never logged it. On the other states, Q regret is 19.6957 kcal and the rule is 20.8132 kcal.

Removing that no-op from A_legal and rescoring the same states gives the sensitivity table below. This run does not treat the no-op as a reason to edit Q. #53 remains coverage-constrained selection on the development 505, not a claim that Q got smarter.


### Chosen-action counts (headline, no-op included)

- oracle: `{"CORRECT_CODE": 361, "RETRIEVE": 964, "RE_DECOMPOSE": 241, "RE_PORTION": 348}`
- q_phi: `{"CORRECT_CODE": 87, "RETRIEVE": 1195, "RE_PORTION": 632}`
- rule_utility: `{"CORRECT_CODE": 101, "RETRIEVE": 1354, "RE_PORTION": 459}`
- random: `{"EXPECTED": 1914}`
- fixed_pipeline: `{"RETRIEVE": 995, "RE_PORTION": 919}`

## Sensitivity: A_legal without the RE_DECOMPOSE no-op

Same headline states. RE_DECOMPOSE is removed before the oracle, the rule, random, and the fixed pipeline are scored. Q is not re-argmaxed. Its frozen action is kept when it is still legal, which it is, because the masked head does not select RE_DECOMPOSE.

| Policy | Identification | Exact tie-break | Energy improvement (kcal) | Energy regret (kcal) | Mass improvement (g) | Mass regret (g) | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 1.0 | 1.0 | 44.2749 | 0.0 | 30.894 | 4.3104 | 1.6139 |
| Q_φ (frozen + fence) | 0.7356 | 0.7252 | 24.1441 | 17.3261 | 24.2622 | 10.0182 | 1.6243 |
| Rule Utility | 0.7299 | 0.72 | 23.034 | 18.2815 | 25.2972 | 9.1274 | 1.7074 |
| Random (expectation) | 0.7034 | 0.6799 | 2.7817 | 35.712 | 26.0287 | 8.4977 | 1.6799 |
| Fixed Pipeline | 0.7064 | 0.7064 | 20.62 | 20.3591 | 18.8706 | 14.6586 | 1.5199 |

### Chosen-action counts (no-op removed)

- oracle: `{"CORRECT_CODE": 382, "RETRIEVE": 1175, "RE_PORTION": 357}`
- q_phi: `{"CORRECT_CODE": 87, "RETRIEVE": 1195, "RE_PORTION": 632}`
- rule_utility: `{"CORRECT_CODE": 101, "RETRIEVE": 1354, "RE_PORTION": 459}`
- random: `{"EXPECTED": 1914}`
- fixed_pipeline: `{"RETRIEVE": 995, "RE_PORTION": 919}`

## All coverage-choice states, including dishes with no overhead RGB

RE_PORTION on a dish with no frozen forward fails and does not invent grams. Those states stay out of the headline so a missing image is not scored as a specialist call.

| Policy | Identification | Exact tie-break | Energy improvement (kcal) | Energy regret (kcal) | Mass improvement (g) | Mass regret (g) | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 1.0 | 1.0 | 71.9559 | 0.0 | 25.6428 | 7.925 | 1.7248 |
| Q_φ (frozen + fence) | 0.6428 | 0.6244 | 19.9588 | 43.4476 | 22.667 | 10.4115 | 1.6048 |
| Rule Utility | 0.634 | 0.6156 | 18.6913 | 44.5067 | 23.5196 | 9.699 | 1.6713 |
| Random (expectation) | 0.4555 | 0.3803 | 2.586 | 69.3699 | 11.324 | 23.8032 | 2.1408 |
| Fixed Pipeline | 0.6253 | 0.6148 | 17.1712 | 45.7768 | 18.2251 | 14.123 | 1.5211 |

## Decisive energy slice

Headline states whose legal actions disagree by more than 1 kcal in post energy error. The 1 kcal cutoff is the existing FixRate ε. It was not refit.

| Policy | Identification | Exact tie-break | Energy improvement (kcal) | Energy regret (kcal) | Mass improvement (g) | Mass regret (g) | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 1.0 | 1.0 | 83.4639 | 0.0 | 28.2992 | 9.2991 | 1.7517 |
| Q_φ (frozen + fence) | 0.6121 | 0.602 | 24.5201 | 50.622 | 24.522 | 12.5431 | 1.6196 |
| Rule Utility | 0.6058 | 0.5956 | 23.3928 | 51.5901 | 25.573 | 11.6405 | 1.7043 |
| Random (expectation) | 0.4246 | 0.3783 | 4.1013 | 79.3626 | 12.2487 | 26.8784 | 2.1348 |
| Fixed Pipeline | 0.5866 | 0.5866 | 20.9413 | 53.6955 | 19.0467 | 17.2453 | 1.5131 |

## Closed-component states

STOP is legal because the component is not open. This is not the coverage-constrained choice. Shown so the terminal states are not dropped silently.

| Policy | Identification | Exact tie-break | Energy improvement (kcal) | Energy regret (kcal) | Mass improvement (g) | Mass regret (g) | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| Q_φ (frozen + fence) | 1.0 | 0.9763 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| Rule Utility | 1.0 | 0.9368 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| Random (expectation) | 1.0 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| Fixed Pipeline | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |

## Budget curves

A cap keeps actions with cost ≤ B. Oracle, rule utility, random, and the fixed pipeline re-select inside that set. Q_φ does not re-argmax and τ does not change. If the frozen Q action costs more than B, that state is idle (the pre-action error is kept). Caps are the three action costs already in the controller (1, 2, 3), not a search.

The curve is not monotone. At cost ≤ 1, a state whose only legal tools are RETRIEVE (cost 2) and RE_DECOMPOSE (cost 3) cannot act, so the error stays put. At cost ≤ 2 that state must retrieve, and retrieval can increase energy error. The no-op that would have kept the error costs 3, so it enters only at the last cap.

### Cost ≤ 1

n = 1914. Idle Q states: 1195.

| Policy | Identification | Exact tie-break | Energy improvement (kcal) | Energy regret (kcal) | Mass improvement (g) | Mass regret (g) | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 1.0 | 1.0 | 48.3357 | 0.0 | 0.1966 | 6.0977 | 0.4801 |
| Q_φ (frozen + fence) | 0.7288 | 0.7288 | 31.6858 | 16.6499 | -5.3456 | 11.6399 | 0.3757 |
| Rule Utility | 0.7597 | 0.7597 | 29.6298 | 18.7059 | -9.308 | 15.6023 | 0.4801 |
| Random (expectation) | 0.7599 | 0.7599 | 14.6242 | 33.7114 | -5.3686 | 11.6629 | 0.4801 |
| Fixed Pipeline | 0.7466 | 0.7466 | 28.4268 | 19.9089 | -10.7372 | 17.0315 | 0.4801 |

### Cost ≤ 2

n = 1914. Idle Q states: 0.

| Policy | Identification | Exact tie-break | Energy improvement (kcal) | Energy regret (kcal) | Mass improvement (g) | Mass regret (g) | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 1.0 | 1.0 | 44.2749 | 0.0 | 30.894 | 4.3104 | 1.6139 |
| Q_φ (frozen + fence) | 0.7356 | 0.7252 | 24.1441 | 17.3261 | 24.2622 | 10.0182 | 1.6243 |
| Rule Utility | 0.7299 | 0.72 | 23.034 | 18.2815 | 25.2972 | 9.1274 | 1.7074 |
| Random (expectation) | 0.7034 | 0.6799 | 2.7817 | 35.712 | 26.0287 | 8.4977 | 1.6799 |
| Fixed Pipeline | 0.7064 | 0.7064 | 20.62 | 20.3591 | 18.8706 | 14.6586 | 1.5199 |

### Cost ≤ 3

n = 1914. Idle Q states: 0.

| Policy | Identification | Exact tie-break | Energy improvement (kcal) | Energy regret (kcal) | Mass improvement (g) | Mass regret (g) | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 1.0 | 1.0 | 82.19 | 0.0 | 27.8669 | 9.2806 | 1.7555 |
| Q_φ (frozen + fence) | 0.6165 | 0.6066 | 24.1441 | 49.9586 | 24.2622 | 12.383 | 1.6243 |
| Rule Utility | 0.6102 | 0.6003 | 23.034 | 50.914 | 25.2972 | 11.4923 | 1.7074 |
| Random (expectation) | 0.4338 | 0.38 | 4.038 | 78.152 | 12.1196 | 26.5302 | 2.1399 |
| Fixed Pipeline | 0.5914 | 0.5914 | 20.62 | 52.9916 | 18.8706 | 17.0234 | 1.5199 |

## Value calibration

n action-values = 3475. MAE(Q, ΔL) = 0.0226. Pearson(Q, ΔL) = 0.7471. Pearson(Q, kcal improvement) = 0.7471. Slope of realized ΔL on predicted Q = 0.8406 (intercept 0.0039).

The slope is a description. It was not written back into the fence or the weights.

| Q bin | n | Mean predicted Q | Mean realized ΔL | Mean realized kcal |
|---|---:|---:|---:|---:|
| -0.3591–-0.007284 | 348 | -0.081641 | -0.056795 | -141.9881 |
| -0.007284–-0.003142 | 343 | -0.004399 | -0.00439 | -10.9755 |
| -0.003142–-0.002566 | 352 | -0.002798 | 0.000303 | 0.7582 |
| -0.002566–-0.002136 | 283 | -0.002351 | 0.000397 | 0.9913 |
| -0.002136–-0.00147 | 411 | -0.001858 | 0.000478 | 1.1943 |
| -0.00147–-0.001064 | 348 | -0.001242 | -0.002678 | -6.6955 |
| -0.001064–-0.00026 | 347 | -0.000814 | 0.001461 | 3.6533 |
| -0.00026–0.00604 | 348 | 0.002315 | 0.00497 | 12.4244 |
| 0.00604–0.024313 | 347 | 0.012079 | 0.014193 | 35.483 |
| 0.024313–0.312639 | 348 | 0.087416 | 0.087109 | 217.7725 |

### By action

- RETRIEVE: n=1637, MAE=0.0235, mean Q=-0.012, mean ΔL=-0.0021, Pearson=0.7545
- RE_PORTION: n=919, MAE=0.0258, mean Q=0.0245, mean ΔL=0.0203, Pearson=0.805
- RE_DECOMPOSE: n=0, MAE=None, mean Q=None, mean ΔL=None, Pearson=None
- ABSTAIN: n=0, MAE=None, mean Q=None, mean ΔL=None, Pearson=None
- STOP: n=0, MAE=None, mean Q=None, mean ΔL=None, Pearson=None
- CORRECT_CODE: n=919, MAE=0.0177, mean Q=-0.0007, mean ΔL=0.0006, Pearson=0.0163

## Logged teacher transition vs live tool

Mean absolute energy-delta gap on the logged action name: 42.389 kcal (n=1616). Logged actions outside A_legal: 0.

- CORRECT_CODE: n=130, mean |Δkcal teacher − live| = 77.8519
- RETRIEVE: n=732, mean |Δkcal teacher − live| = 4.6077
- RE_PORTION: n=754, mean |Δkcal teacher − live| = 72.9537

## Freeze and bans

Frozen hashes unchanged: True.
No Mass retrain. No Q retrain. No τ retune. No rgb_test label was a scoring target or a threshold. No reinforcement learning. No P1. Gates ②–⑦ were not started. RE_PORTION does not fall back to the gram grid or to ingredient grams when the mass forward is missing.

## Blockers

- RE_PORTION had no frozen mass forward on 226 branches, all outside the mass-available headline. Grams were not invented.
- One-step branches, not a full-horizon rollout. Budget curves are one-step cost caps.
- Weak-step states have no vision gram hints, so RE_PORTION uses the frozen equal split of the dish mass.
- The #51 plate-level code swap is not inside the one-step branch.
