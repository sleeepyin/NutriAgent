# Weak-step fit / hold-out leakage audit

Conclusion: `no_dish_leakage`.

`leakage_exists` = false.

Are weak-step fit and hold-out grouped by dish_id, or do decision steps of the same dish fall on both sides?

Rule: `int(sha256(dish_id).hexdigest()[:8], 16) % 10 == 0 → holdout, else fit`.

Fit dishes 3679, rows 25864, multi-step dishes 3678.
Hold-out dishes 380, rows 2771, multi-step dishes 380.
Dishes in both splits: 0.
Misfiled against the hash: 0.

Q training fits every supervised row of the fit file and scores the hold-out file. There is no random decision-step split in vdr.controller.q_data or scripts/qphi_train.py. The mass-estimator probe-val residue (sha256 % 10 == 1) is a second dish-grouped slice inside fit. It is not this cut. This audit does not refit Q, Mass, or τ.

Q, Mass, and τ were not retrained.
