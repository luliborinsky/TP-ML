"""
run_step3.py — Step 3 Optimizer & Learning Rate Experiments

Runs 3 seeds for each configuration and reports mean ± std across seeds:
  Part a: SGD + Nesterov, lr 0.01, 10 epochs
  Part b: Adam, lr 0.001, 10 epochs
  Part c: Best of a/b + Cosine LR decay, 20 epochs
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cnn import CONFIG, run_experiment


def run_experiment_group(tag_prefix, opt_name, lr, scheduler, epochs, seeds=[42, 43, 44]):
    results = []
    print(f"\n=======================================================")
    print(f"Running Group: {tag_prefix} ({opt_name}, lr={lr}, sched={scheduler}, ep={epochs})")
    print(f"=======================================================")

    for seed in seeds:
        cfg = CONFIG.copy()
        cfg['optimizer'] = opt_name
        cfg['lr'] = lr
        cfg['epochs'] = epochs
        cfg['scheduler'] = scheduler
        cfg['seed'] = seed
        cfg['tag'] = f"{tag_prefix}_s{seed}"
        cfg['checkpoint_path'] = f"./models/face_net_{tag_prefix}_s{seed}.pt"
        cfg['log_path'] = f"./logs/training_log_{tag_prefix}_s{seed}.json"
        cfg['plot_path'] = f"./figures/training_curves_{tag_prefix}_s{seed}.png"

        res = run_experiment(cfg)
        results.append(res)

    # Compute summary statistics
    val_aucs = [r['val_m']['auc'] for r in results]
    val_bas  = [r['val_m']['balanced_acc'] for r in results]
    hard_aucs = [r['hard_m']['auc'] for r in results]
    hard_bas  = [r['hard_m']['balanced_acc'] for r in results]
    test_aucs = [r['test_m']['auc'] for r in results]
    test_bas  = [r['test_m']['balanced_acc'] for r in results]
    test_rec_face = [r['test_m']['recall_face'] for r in results]
    test_rec_noface = [r['test_m']['recall_noface'] for r in results]

    print("\n" + "="*70)
    print(f"SUMMARY FOR {tag_prefix} across {len(seeds)} seeds:")
    print(f"Val AUC:         {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    print(f"Val BalAcc:      {np.mean(val_bas):.4f} ± {np.std(val_bas):.4f}")
    print(f"Hard Val AUC:    {np.mean(hard_aucs):.4f} ± {np.std(hard_aucs):.4f}")
    print(f"Hard Val BalAcc: {np.mean(hard_bas):.4f} ± {np.std(hard_bas):.4f}")
    print(f"Test AUC:        {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print(f"Test BalAcc:     {np.mean(test_bas):.4f} ± {np.std(test_bas):.4f}")
    print(f"Test Rec(Face):  {np.mean(test_rec_face):.4f} ± {np.std(test_rec_face):.4f}")
    print(f"Test Rec(NoFace):{np.mean(test_rec_noface):.4f} ± {np.std(test_rec_noface):.4f}")
    print("="*70 + "\n")

    return results


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'a'
    if part == 'a':
        run_experiment_group('step3_nesterov', 'nesterov', 0.01, 'none', 10)
    elif part == 'b':
        run_experiment_group('step3_adam', 'adam', 0.001, 'none', 10)
    elif part == 'c_nesterov':
        run_experiment_group('step3_c_nesterov_cos', 'nesterov', 0.01, 'cosine', 20)
    elif part == 'c_adam':
        run_experiment_group('step3_c_adam_cos', 'adam', 0.001, 'cosine', 20)
    else:
        print(f"Unknown part: {part}")
