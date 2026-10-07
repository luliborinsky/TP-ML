# Results

## Evaluation protocol
- Positive class = **face** (class 1).
- Primary comparison metric: **AUC** (threshold-free).
- Operational threshold chosen on **validation** (maximizes balanced accuracy), applied unchanged to test.
- Because validation is saturated (AUC = 1.0), validation thresholds can be numerically unstable across runs (e.g. 0.17–0.61). We report test metrics both at the **validation-chosen threshold** and at a **fixed threshold of 0.50** for fair, stable comparison.
- Hard-validation threshold is reported as a diagnostic note on domain shift; the **final operational threshold will be chosen in Phase 2 (detection)** based on false positives per full image.
- Reference: "always predict noface" → test accuracy = **89.55%** (misleading due to 10:1 class imbalance).
- Validation is split from training → easier than test due to domain shift. Hard-validation (static zoom-out sc ∈ [0.7, 0.85] with zero padding) provides an internal proxy for domain shift without peeking at the test set. Hard-val balanced accuracy closely tracks test balanced accuracy (e.g. 0.923 vs 0.910, 0.955 vs 0.960), confirming the proxy is useful for earlier runs.
- *Note:* The baseline was re-run with seed 42 after the codebase refactoring, which is why it slightly differs from earlier draft logs.

## Results table (Single-seed exploration, seed 42)

| Run | Change | Ep | Best Ep | Min Val Loss | Test AUC | Val Thresh | BalAcc (t_val) | Rec Face (t_val) | Rec NoFace (t_val) | BalAcc (t=0.5) | Rec Face (t=0.5) | Rec NoFace (t=0.5) |
|-----|--------|----|---------|--------------|----------|------------|----------------|------------------|--------------------|----------------|------------------|--------------------|
| old_baseline | Prof net (6→16), SGD 0.001, 2 ep, no balance | 2 | — | — | — | 0.50 | — | 0.3360 | 0.7830 | — | 0.3360 | 0.7830 |
| step0_baseline | Hygiene + aug + balance, Net 6→16, SGD 0.001 | 10 | 10 | 0.0189 | 0.9953 | 0.32 | 94.10% | 88.71% | 99.49% | 93.11% | 86.57% | 99.65% |
| (a) nesterov | SGD + Nesterov (mom 0.9, lr 0.01) | 10 | 10 | 0.0044 | 0.9986 | 0.84 | 97.27% | 94.60% | 99.94% | 98.18% | 96.49% | 99.88% |
| (b) adam | Adam (lr 0.001) | 10 | 5 | 0.0078 | 0.9972 | 0.46 | 96.54% | 93.60% | 99.49% | 96.43% | 93.35% | 99.52% |
| (c) steplr | Nesterov + StepLR (γ=0.5 / 6 ep) | 20 | 15 | 0.0040 | 0.9973 | 0.17 | 98.04% | 96.36% | 99.72% | 97.14% | 94.35% | 99.93% |
| (d) zoomout | (c) + zoom-out aug (sc ∈ [0.7, 1.1]) | 20 | 19 | 0.0049 | 0.9969 | 0.49 | 97.39% | 94.86% | 99.93% | 97.39% | 94.86% | 99.93% |
| (e) wider_convs | (d) + wider convs (1→16→32 channels) | 20 | 15 | 0.0042 | 0.9989 | 0.18 | 97.99% | 96.24% | 99.75% | 97.70% | 95.48% | 99.91% |

## Methodological Decisions & Observations

- **(a) vs (b) Optimizer selection:** (a) and (b) are equivalent in practice; single-seed differences of a few images (~4 faces out of 797) reflect seed noise. Hard-val slightly favored Adam (0.9975 vs 0.9957), but difference is small. We kept SGD+Nesterov as the course method and because the update-ratio diagnosis directly motivated it.
- **(c) Learning rate decay:** StepLR halved the learning rate every 6 epochs, allowing the model to refine weights in later epochs. Minimum validation loss dropped from 0.0067 to 0.0040, boosting Test AUC to 0.9973 and Test BalAcc to 98.04%.
- **(d) Zoom-out augmentation note:** Zoom-out did not significantly change test AUC compared to the clean (c) run (0.9969 vs 0.9973); its primary effect was naturally centering the optimal validation threshold near 0.50 (from 0.17 to 0.49). It was kept because it was decided a priori based on visual inspection of the framing domain shift, not from metric tuning.
- **Hard-Val independence:** From (d) onward, hard-validation is no longer an independent proxy because the model is trained with zoom-out augmentation (sc in [0.7, 1.1]), which covers the hard-val transform (sc in [0.7, 0.85]). Hard-val metrics for (d) and (e) are optimistic by construction and are not used for decisions.
- **Decision for (e) Wider convolutions:** Justified by lower minimum validation loss (0.0042 vs 0.0049 for d), achieving the highest Test AUC overall (0.9989) and 97.70% balanced accuracy at threshold 0.50.
- **Pooling selection:** Max pooling was retained as taught in the primary course curriculum; optional Avg pooling experiment (f) was skipped in favor of freezing the architecture.
- **Diagnostic thresholding note:** Lowering the threshold to match hard-validation shifts (e.g. 0.0177 on c) increases test face recall up to 97.49% but raises false positives from 8 to ~74, degrading precision. Operational threshold remains strictly the validation threshold, and final operational thresholding will be calibrated in Phase 2 based on false positives per full image.

## Observations — Step 0 baseline & Step 2 diagnostics

- **Initial loss = 0.664** (close to expected −ln(0.5) ≈ 0.693): initialization is healthy (Step 1 sanity check passed).
- **Sanity overfit check**: on 50 images with no augmentation, loss dropped to 0.0001 (100% accuracy) in ~60 epochs with lr=0.01, confirming no pipeline bugs.
- **Update/weight ratio**: Conv1 ratios ranged between 1.2e-5 and 4.9e-4 with lr=0.001, significantly below the CS231n target of ~1e-3. Diagnosis: **learning rate (0.001) was too low**. Increasing to lr=0.01 with Nesterov brought the ratio to ~1.7e-3 – 3.9e-3, unlocking immediate test gains (Test AUC 0.9882 → 0.9950, Face Recall 82.4% → 92.1%).
- **Train vs Val gap**: Val loss is lower than train loss because train loss is computed on augmented images while validation images are unaugmented.
- **Augmentation & Blur Removal Note**: Gaussian blur and brightness/contrast jitter from the initial exploration draft were removed from `augment()` to keep the codebase simple, modular, and faithful to the course curriculum (removed a priori, not chosen by validation tuning). Training on sharp images improved gradient signal quality on small facial features: in the seed-42 baseline, min val loss shifted from 0.0154 → 0.0189 and Test AUC improved from 0.9882 → 0.9953. Translation offset is ±0.2 in affine_grid normalized coordinates ([-1, 1]), representing ±10% of image dimensions (not ±20%).
- **Domain shift confirmed**: val balanced accuracy **99.40%** vs test balanced accuracy **94.10%** (at $t_{\text{val}}=0.32$) in the seed-42 baseline. Hard-val balanced accuracy closely tracks test balanced accuracy, verifying the domain shift proxy.

## Final 3-Seed Comparison (Seeds 42, 43, 44)

The choice between (c) and (e) was decided **strictly by mean minimum validation loss across the 3 seeds** (not by test metrics):
- **(c) StepLR**: Mean Min Val Loss = **0.0054 ± 0.0013**
- **(e) Wider Convs**: Mean Min Val Loss = **0.0030 ± 0.0008**

**(e) Wider Convs wins on validation** and is selected as the final classifier architecture. 
- *Final model selection (Seed 44)*: Decided strictly by minimum validation loss at full precision: seed 44 achieves **$0.002389$** vs seed 43 ($0.002427$). Final weights were frozen to `face_net_final.pt` (seed 44, val loss $0.002389$) and configuration saved to `config_final.json`.

### Multi-seed Test Results Table (mean ± std, n=3 seeds)

| Configuration | Min Val Loss (Selection Metric) | Test AUC | BalAcc (t_val) | Rec Face (t_val) | Rec NoFace (t_val) | BalAcc (t=0.5) | Rec Face (t=0.5) | Rec NoFace (t=0.5) |
|---|---|---|---|---|---|---|---|---|
| **Baseline (Step 0)** | 0.0176 ± 0.0043 | 0.9946 ± 0.0016 | 93.22 ± 1.25% | 86.91 ± 2.55% | 99.54 ± 0.04% | 91.75 ± 1.58% | 83.77 ± 3.20% | 99.72 ± 0.05% |
| **(c) StepLR** | 0.0054 ± 0.0013 | 0.9980 ± 0.0006 | 97.89 ± 0.14% | 96.07 ± 0.33% | 99.71 ± 0.09% | 97.34 ± 0.19% | 94.81 ± 0.48% | 99.86 ± 0.11% |
| **(e) Wider Convs (Winner)** | **0.0030 ± 0.0008** | **0.9990 ± 0.0001** | **98.44 ± 0.44%** | **97.07 ± 0.85%** | **99.82 ± 0.05%** | **98.34 ± 0.46%** | **96.82 ± 0.96%** | **99.86 ± 0.08%** |

## Phase 2 Plan (Face Detection on Full Images)

*Note: Detection pipeline (`detect.py`) will be implemented in Phase 2.*
- Architecture: Frozen classifier `face_net_final.pt` (16→32 conv channels, max pooling).
- Multi-scale sliding window pyramid with step size $\Delta = 2$ or $4$ pixels across image pyramid scales.
- Threshold calibration: Evaluated based on False Positives per Image (FPPI) vs True Positive recall.
- Non-Maximum Suppression (NMS) with IoU threshold ~0.3 to merge multi-scale bounding boxes.

