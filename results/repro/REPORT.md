# Same-split retrain report

DPF-Nutrition and Swin-Nutrition were each trained once on the 2754 kept official training plates and read once on `official_depth_test`.
These numbers are not the papers' published test-set numbers, and they are not a new pass line.

Holdout plates: 507. Groups: 507. Train/holdout overlap: 0.

## swin_nutrition

| Item | Dataset PMAE | MAE | Plate PMAE |
|---|---:|---:|---:|
| energy | 28.7965 | 73.5656 | 70.1299 |
| mass | 28.6120 | 56.7975 | 50.1471 |
| fat | 33.7579 | 4.3314 | 170.3972 |
| carb | 31.6087 | 6.2217 | 103.7700 |
| protein | 33.5169 | 5.8583 | 67.4561 |
| five-item mean | 31.2584 |  |  |

Train-set dataset PMAE, eval mode, no augmentation: energy 21.9855, mass 24.9513, fat 20.2229, carb 22.3317, protein 21.6103, mean 22.2203.

## dpf_nutrition

| Item | Dataset PMAE | MAE | Plate PMAE |
|---|---:|---:|---:|
| energy | 18.3073 | 46.7692 | 22.9088 |
| mass | 14.4209 | 28.6267 | 19.7245 |
| fat | 25.8270 | 3.3138 | 74.8088 |
| carb | 24.0962 | 4.7430 | 48.3024 |
| protein | 24.8154 | 4.3374 | 40.8938 |
| five-item mean | 21.4934 |  |  |

Train-set dataset PMAE, eval mode, no augmentation: energy 5.7872, mass 10.2139, fat 4.4334, carb 8.4856, protein 5.5836, mean 6.9007.

## FPS ruler

Hardware: NVIDIA GeForce RTX 4090 D. Physical GPU 1. Batch 8. Resolution 448×448.
The timed region is model forward only, inputs already on GPU, eval mode, no grad, after warmup.
DPF's forward includes the depth module and the fusion module. Predicted depth is clipped, replicated, and ImageNet-normalized with the fusion helper before the fusion module. This is not a speed claim.

- swin_nutrition: 157.674 images/s
- dpf_nutrition: measurement failed: CUDA out of memory. Tried to allocate 26.00 MiB. GPU 0 has a total capacity of 23.53 GiB of which 39.62 MiB is free. Including non-PyTorch memory, this process has 23.48 GiB memory in use. Of the allocated memory 23.06 GiB is allocated by PyTorch, and 10.08 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

## Pretrained weight load

- swin_nutrition: 325/325 tensors. Recount with the same ImageNet loader, without restarting training. train_swin.py printed 'swin imagenet tensors 325'.
- dpf_depth_encoder: 308/308 tensors. Every encoder tensor except num_batches_tracked is present with a matching shape.

Deviations fixed before training are in `docs/DEVIATIONS.md`.
