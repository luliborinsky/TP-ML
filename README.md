# Face / Non-Face CNN Classifier (Phase 1)

Binary CNN classifier for 36×36 face vs. non-face image patches, designed as Phase 1 of the INSA Lyon IF 5 / OT2 Computer Vision & Deep Learning project. Phase 2 will implement multi-scale sliding-window face detection on full-resolution scenes.

---

## 1. Environment Setup

Requirements: Python 3.9+ and PyTorch. Compatible with Apple Silicon (`mps`), NVIDIA CUDA, and CPU.

```bash
# Clone the repository
git clone <repo-url>
cd "TP ML"

# Create and activate virtual environment
python3 -m venv ml-env
source ml-env/bin/activate

# Install dependencies
pip install torch torchvision numpy pillow matplotlib
```

---

## 2. Project Structure

```text
TP ML/
├── README.md               # Project guide and instructions
├── CLAUDE.md               # Development roadmap and guidelines
├── results.md              # Experimental results, ablation table, and methodological notes
├── .gitignore              # Ignores local caches and temp weights (preserves models/face_net_final.pt)
├── cnn.py                  # Main classifier training script (course-aligned architecture)
├── diagnostics.py          # Metrics, threshold optimization, hard validation, visualizations
├── eval_cnn.py             # One-command evaluation of final model on test images (no training)
├── models/
│   ├── face_net_final.pt   # Final frozen classifier weights (Wider Convs, Seed 44)
│   └── config_final.json   # Final architecture and hyperparameter configuration
├── experiments/
│   ├── run_3seed_suite.py  # 3-seed comparison suite (Baseline vs StepLR vs Wider Convs)
│   ├── run_experiments.py  # Single-seed step-by-step runner (experiments a–e)
│   ├── run_final_comparison.py
│   └── run_step3.py
├── logs/                   # Per-run JSON training history logs
├── figures/                # Training curves and filter visualizations
├── torchsampler/           # Early project code (CNN.py, CNN_early_stopping.py, imbalanced.py)
└── legacy/                 # Original course template files (kept for reference)
```

---

## 3. How to Train and Evaluate

### Quick Evaluation (Test Detection Rates without Training)
To evaluate the final model ([`models/face_net_final.pt`](./models/face_net_final.pt)) on the test set in ~2 seconds:
```bash
python eval_cnn.py
```
This prints the exact detection percentages:
- **Face Recall**: % of test faces correctly detected
- **No-Face Recall**: % of non-face background patches correctly rejected
- **Balanced Accuracy & AUC**

### Full Training (Main Training Script)
Run the primary training routine with default hyperparameters (or customized CLI flags):
```bash
python cnn.py --optimizer nesterov --lr 0.01 --epochs 20 --scheduler steplr --c1 16 --c2 32 --min-scale 0.7
```

### Reproducing Step-by-Step Experiments
Run individual ablation steps with fixed seed 42:
```bash
python experiments/run_experiments.py step0   # Baseline (SGD lr=0.001)
python experiments/run_experiments.py a       # + Nesterov momentum
python experiments/run_experiments.py b       # Adam optimizer comparison
python experiments/run_experiments.py c       # + StepLR schedule
python experiments/run_experiments.py d       # + Zoom-out data augmentation
python experiments/run_experiments.py e       # + Wider convolutions (16->32)
```

### Reproducing the 3-Seed Comparison
Run the multi-seed evaluation (seeds 42, 43, 44) comparing Baseline vs. (c) StepLR vs. (e) Wider Convs:
```bash
python experiments/run_3seed_suite.py
```

---

## 4. Results Summary

Detailed metric tables, confusion matrices, and ablation analysis are documented in [`results.md`](./results.md).

- **Final Model Selection**: Architecture **(e) Wider Convs** won strictly on validation loss ($0.0030 \pm 0.0008$ vs. $0.0054 \pm 0.0013$ for StepLR).
- **Frozen Checkpoint**: Selected from **Seed 44** with minimum validation loss $0.002389$.
- **Test Performance (3-seed mean ± std)**:
  - **Test AUC**: $0.9990 \pm 0.0001$
  - **Balanced Accuracy ($t=0.50$)**: $98.34 \pm 0.46\%$ (Face Recall: $96.82\%$, Non-Face Recall: $99.86\%$)
  - **Balanced Accuracy ($t_{\text{val}}$)**: $98.44 \pm 0.44\%$
