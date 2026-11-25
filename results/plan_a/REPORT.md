# Plan A geometry teacher

This run retrains only the geometry teacher. Separate encoders and a zero-initialized
residual are a BatchNorm repair. They are not a new method, and predicted mass times
predicted density is the same product as the frozen run.

## D_table audit

The count below was computed with the current height code, before that code was changed.
The border median included missing-depth zeros. Height is then
`clip(D_table - D_raw, 0, 150 mm) / 150`, followed by one 3×3 closing.

Kept train plates: 2754 of 2758.
`D_table == 0`: 65 / 2754 = 0.023602033406.
All-zero height maps: 65 / 2754 = 0.023602033406.
Those two sets are the same 65 plates.

Every kept plate has at least one zero on the 1-pixel border (2754 / 2754). The median fraction of border zeros is 0.295170. The median table depth on the full kept set is 403.1000 mm.

65 of 2754 kept training plates have D_table exactly 0, and the height map is all zeros on exactly those plates. A blank height map on 65 plates is not a rare leftover, so missing-depth zeros are excluded from the border median before Plan A training. Desktop subtraction, the 0-15 cm clip, and one 3x3 closing stay unchanged. This judgment is frozen with the audit. It is not a percentage cutoff and it is not revised after train or holdout error.

Training therefore sets `apply_d_table_zero_exclusion` to true.

## Training

Plan A `T_geo` is training. Measured parameter count 56562054, under the 60M cap. Batch size 32. RGB weights were loaded from the frozen B1 checkpoint, and the epoch-0 forward matched that checkpoint. The train gate has not been evaluated.
