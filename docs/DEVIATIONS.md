# Deviations recorded before training

This file is the pre-training record for the official 2754/507 same-split retrain of DPF-Nutrition and Swin-Nutrition. It was written before either training run. The frozen numbers live in `configs/repro_frozen.json`.

No official author training repository was found for either paper. DPF-Nutrition's data-availability statement says the corresponding author can provide material on request. Swin-Nutrition's paper does not link code. This reproduction therefore follows the papers. Third-party nutrition checkpoints are not loaded.

The local file `/tmp/papers/dpf.pdf` is the 9-page arXiv preprint (2310.11702), not the published MDPI article. The published article is Foods 2023, 12(23), 4293. Its HTML was read. Where the published methods paragraph and the preprint equation disagree, the published paragraph is followed. Learning-rate exponents are missing from the HTML text extract because they were superscripts. The arXiv PDF prints them as 1×10^−5, 1×10^−6, and 5×10^−5. Those are the values used.

## Shared protocol

- Train on the 2754 kept official depth-train plates. The only train-only exclusions are `dish_1558109714`, `dish_1564159636`, `dish_1557862384`, and `dish_1558109511`. The 507-plate `official_depth_test` list is not shrunk.
- One training run per method. The checkpoint is the last epoch. There is no validation split and no holdout model selection.
- Labels are plate-level calories, mass, fat, carbohydrate, and protein from Nutrition5k cafe metadata. Per-ingredient grams are not model inputs.
- Reported dataset PMAE is this repository's definition, `100 × sum(|error|) / sum(|ground truth|)`, for energy, mass, fat, carb, and protein, plus the arithmetic mean of those five. MAE and plate PMAE are supporting numbers. Plate PMAE skips plates whose absolute ground truth is 0.
- These runs do not claim the published DPF numbers or the published Swin-Nutrition numbers. Swin-Nutrition's public numbers are from cleaned RGB frames (train 207,576 / test 34,934). This run places that structure on the 2754/507 overhead still-frame split.
- DiGI-Nutri DropPath, ColorJitter, Mixup, EMA, bidirectional gating, and `0.1 × SmoothL1` residual Atwater are not added. The frozen DiGI split, `L_task`, `ŷ = M̂ × ρ̂`, and height formula are not modified and are not the forward path of either reproduction.
- No B1, `T_multi`, student, distillation, or Plan B run is started.
- Seed `20261125` is a reproducibility seed. It was not chosen on the holdout.
- Adam betas are the library defaults `(0.9, 0.999)`. Neither paper states betas.

## DPF-Nutrition

Published setting that is implemented:

- Depth module: DPT-Hybrid. The encoder is a ViT-B/ResNet-50. The decoder is the official DPT RefineNet head from isl-org/DPT, randomly initialized. Vendor note: `third_party/dpt/NOTICE`.
- Depth training: long side 384, random 384 crop, Adam, cosine from 1×10^−5 to 1×10^−6, 60 epochs, batch 8.
- Fusion: separate ResNet-101 streams for RGB and depth, CAB at each of five stages, a third ResNet-101 fused from shallow maps to deep maps, then a 2048-d fully connected layer and five 1-d heads.
- Fusion training: resize to 336×448 (height × width, the 3:4 frame), random horizontal flip, Food2K initialization, Adam 5×10^−5, exponential decay 0.98 per epoch, 150 epochs, batch 8.
- Test-time depth is the trained DPT prediction. Ground-truth depth is not the fusion input. This is the paper's predicted-depth setting, not the ground-truth-depth row of its ablation table.
- Nutrition loss is the geometric mean of five L1 terms, equation (8) in the preprint: the fifth root of the product.

Deviations, fixed before training:

1. **Encoder checkpoint.** The paper says ImageNet-1K initialization. The public DPT-Hybrid encoder file is `jx_vit_base_resnet50_384-9fd3c705.pth` (`vit_base_r50_s16_384.orig_in21k_ft_in1k`): ImageNet-21k pretraining followed by ImageNet-1k fine-tuning. A pure ImageNet-1k hybrid file was not found. MiDaS depth weights are not loaded. The decoder is random. Training aborts if the encoder state does not match this timm module. The release file is 395854632 bytes. A shorter download is incomplete and is not loaded.
2. **timm name.** isl-org/DPT calls `vit_base_resnet50_384`. timm 1.0.31 names that module `vit_base_r50_s16_384`. The factory call is the only edit inside the vendored DPT code.
3. **Depth loss.** DPF does not write a depth-loss equation. It points to the DPT paper for the module. DPT's mixed-dataset loss is scale-and-shift invariant, which removes the absolute scale this nutrition model needs for portion. Supervision is masked L1 on metric meters (`depth_raw` uint16 × 1e-4), ignoring raw zeros and the padding used to make the 384 crop. The official DPT head is non-negative. Output is metric depth, not inverse depth.
4. **Depth normalization.** DPT's own preprocessing uses mean 0.5 and std 0.5. That is used for the depth module. The paper does not restate it.
5. **No extra depth-module flip.** The depth-module paragraph names the long-side resize and the 384 crop only. Horizontal flip is not added there.
6. **Padding implements the 384 crop.** A 640×480 frame becomes 384×288 when the long side is 384, so a 384 square crop needs padding. Training draws a random pad offset. Evaluation uses a centered pad. The content region is unpadded before the depth map is resampled for the fusion network.
7. **Fusion resolution alignment.** The two modules are specified at different resolutions. Predicted depth is resampled with bilinear interpolation from the unpadded DPT content to 336×448. Per-image min-max normalization is not used, because it would remove portion scale.
8. **Depth as three channels.** The depth ResNet is a Food2K/ImageNet 3-channel stem. Metric depth is clipped by a fixed 2.0 m, replicated to three channels, and ImageNet-normalized. The paper does not state this conversion. The scale is global, not per image.
9. **Center crop.** The fusion paragraph says the frames are resized to 336×448, maintaining aspect ratio, and that center crop and random horizontal flip are used. 336×448 is exactly the 3:4 aspect of the 480×640 overhead frames, so the center crop of that resized frame is the frame itself. No smaller crop size is stated.
10. **CAB width.** Equations (3)–(5) use one 1×1 convolution, batch norm, and ReLU, then sigmoid, for channel attention, and a 3×3 convolution, batch norm, and ReLU, then sigmoid, for spatial attention. No channel-reduction ratio is stated, so the 1×1 keeps the stage width. The concat 1×1 returns the stage width so the residual add in the fusion network is defined. ReLU sits before sigmoid, as written, even though that keeps the gate at or above 0.5.
11. **Shallow-to-deep chain.** The preprint equation writes `F_i = C_i + Res_i(C_{i-1})`. The published methods paragraph says the fused feature is refined by the next ResNet block, combined with the next CAB map, and that this is repeated from shallow layers to deep layers inside the third ResNet. The implementation is `F_0 = C_0` and `F_i = C_i + Res_i(F_{i-1})`, using that ResNet's max-pool and `layer1`–`layer4`. Five maps are conv1, layer1, layer2, layer3, and layer4. The third ResNet's conv stem is unused because the inputs are CAB feature maps. Its `layer1`–`layer4` still come from the Food2K checkpoint.
12. **Head activation.** The paper says one 2048-d fully connected layer and five 1-d heads. It does not put a ReLU between them. None is added.
13. **Food2K file.** The authors' Food2K release lists ResNet-101 behind a Baidu share code, not a stable file URL. The file used here was downloaded from the Hugging Face mirror of `T0MYYY/dpf-nutrition`, filename `food2k_resnet101.pth`. Before training it was checked and accepted as a classification backbone: `fc` weight `[2000, 2048]`, `conv1` weight `[64, 3, 7, 7]`, 624 tensors, sha256 `1ade087a459cf00ac6cbd4e91b91b7d4346189b4a551a2c0ae802fbc3c31ef2c`. It loaded into the three ResNet-101 streams. It is not a DPF nutrition checkpoint. If that check had failed, fusion would not be trained and ImageNet would not be substituted. The depth module still uses the hybrid encoder above, not Food2K.
14. **Weight decay.** DPF does not state Adam weight decay. It is 0.
15. **Geometric-mean epsilon.** The product is evaluated as `exp(mean(log(L1 + 1e-6)))` so a zero term does not collapse the backward pass. This is the same geometric mean up to that epsilon.
16. **Schedule step.** Cosine and the 0.98 decay are applied once per epoch, which matches how the papers state epoch counts and decay rates.

## Swin-Nutrition

Published setting that is implemented:

- Backbone Swin-B. Four stages, patch size 4. ImageNet-1k initialization. The paper's implementation paragraph names Swin-B and ImageNet-1k.
- Feature fusion is top-down upsampling plus lateral 1×1 convolutions, in the feature-pyramid form the paper describes. Each of the four fused maps is global-average-pooled.
- Loss equations (1) and (2): each subtask term is the sum of absolute errors divided by the sum of ground truth, and the total is `sum_τ [L_τ / (2 c_τ²) + ln(1 + c_τ²)]` with learnable `c_τ`.
- Adam, 40 epochs, batch 64, initial learning rate 1×10^−5, exponential decay 0.9 per epoch.

Deviations, fixed before training:

1. **Input size.** The paper does not state height and width. Swin-B ImageNet-1k is 224×224. Training and the nutrition evaluation use a direct resize to 224×224. That changes the 4:3 frame to a square. No other crop size is stated.
2. **Weight decay.** The sentence "the weights decayed to 1×10^−5" is read as Adam weight decay 1×10^−5. The learning-rate schedule is already the stated exponential decay from the same 1×10^−5 initial rate, so the second 1×10^−5 is not a second learning rate.
3. **Augmentation.** The paper says data augmentation and regularization were used and does not name the operators. The only geometric augmentation used here is random horizontal flip with probability 0.5. Mixup, ColorJitter, EMA, and stochastic depth are not added. `drop_path_rate` is 0 even though some Swin-B fine-tunes use a nonzero rate. The paper does not give a DropPath rate, and the locked DiGI DropPath is not imported.
4. **FFM width.** The 1×1 lateral convolution's output width is not stated. All four maps are projected to 256 channels, then added to the upsampled deeper map. 256 is the channel width of Swin-B stage 2. Fusion is addition, which is the feature-pyramid lateral connection the paper describes. No extra 3×3 smoothing convolution is added.
5. **Prediction head.** Pooled vectors from the four maps are concatenated (1024-d) and one linear layer maps them to the five nutrients. The multi-task behavior is the uncertainty loss. The paper's "uniform predicted header" ablation is a different head, not this run.
6. **Subtask denominator.** Equation (2) is a ratio of sums. The denominator used in each step is the fixed training-set sum scale implemented as batch MAE divided by the training-set mean of that nutrient. That is the MAE-to-average-of-all-ground-truth reading in the paragraph under equation (2), and it does not depend on which plates fall in the batch. Ground truth on this split is non-negative.
7. **Uncertainty initialization.** `c_τ` is initialized to 1. The paper does not state the initialization. The denominator uses `c_τ² + 1e-8` so a zero parameter cannot divide by zero.
8. **ImageNet normalization.** Mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)` are the standard ImageNet-1k Swin-B normalization. The paper does not restate them.
9. **Checkpoint.** The file is the official Swin-B ImageNet-1k release `swin_base_patch4_window7_224.pth`, not the ImageNet-22k-to-1k release. The classification head is dropped.

## FPS ruler

FPS is measured on this machine for both methods with the same batch and the same square resolution, after the nutrition evaluation. It is a ruler measurement, not a claim that one method is faster. The shared resolution can differ from 224×224 and from 336×448 because Swin-B window 7 needs each stage map to be divisible by 7, which 384×384 is not. The configured ruler is batch 8 at 448×448. DPF's timed forward includes the DPT depth module and the fusion module. A memory failure at that ruler is reported as a failed measurement, not replaced by a quieter setting after the holdout exists. timm's Swin-B patch embed defaults to a fixed 224 check. That check is turned off (`strict_img_size=False`, `dynamic_img_size=True`) so the shared 448 ruler can run. Nutrition training and the holdout read stay at 224×224.

## What this file does not do

It does not set a new numeric gate. It does not authorize a second holdout evaluation. It does not authorize teacher training.
