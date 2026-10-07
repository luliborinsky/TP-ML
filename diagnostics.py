
#diagnostics.py — Diagnostic tools, evaluation metrics, and plotting for CNN training


import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def auc_score(y, p):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.0
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def compute_metrics(y_true, y_prob, threshold=0.5):
    pred = (y_prob > threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ba   = (rec0 + rec1) / 2.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    auc  = auc_score(y_true, y_prob)

    return {
        'auc':              round(auc, 4),
        'recall_face':      round(rec1, 4),
        'recall_noface':    round(rec0, 4),
        'balanced_acc':     round(ba, 4),
        'precision_face':   round(prec, 4),
        'confusion':        {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
    }


def find_best_threshold(y_true, y_prob):
    candidates = np.unique(np.concatenate([
        np.geomspace(0.001, 0.05, 50),
        np.linspace(0.05, 0.95, 91),
        1.0 - np.geomspace(0.001, 0.05, 50),
    ]))
    best_thresh, best_ba = 0.5, 0.0
    for t in candidates:
        pred = (y_prob > t).astype(int)
        rec1 = (pred[y_true == 1] == 1).mean() if (y_true == 1).sum() > 0 else 0
        rec0 = (pred[y_true == 0] == 0).mean() if (y_true == 0).sum() > 0 else 0
        ba = (rec0 + rec1) / 2
        if ba > best_ba:
            best_ba = ba
            best_thresh = t
    return round(float(best_thresh), 4)


def build_hard_val_loader(x_val_raw, y_val, norm_fn, seed=123):
    """Synthetic 'hard validation' set to monitor robustness to domain shift.

    Applies zoom-out (sc in [0.7, 0.85], padding_mode='zeros') with fixed seed.
    NOTE: Multiplicative brightness was omitted because per-image standardization
    (mean 0, std 1) analytically cancels any global scaling factor.
    """
    g = torch.Generator().manual_seed(seed)
    N = len(x_val_raw)

    sc = 0.7 + 0.15 * torch.rand(N, generator=g)
    ang = (torch.rand(N, generator=g) * 2 - 1) * math.radians(5)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    tx = (torch.rand(N, generator=g) * 2 - 1) * 0.05
    ty = (torch.rand(N, generator=g) * 2 - 1) * 0.05
    theta = torch.stack([torch.stack([cos, -sin, tx], dim=1),
                         torch.stack([sin,  cos, ty], dim=1)], dim=1)
    grid = F.affine_grid(theta, x_val_raw.shape, align_corners=False)
    x_zoomed = F.grid_sample(x_val_raw, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=False)

    x_hard = norm_fn(x_zoomed)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_hard, y_val),
        batch_size=1024, shuffle=False, num_workers=0)


def plot_training_curves(history, save_path):
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Train vs Val loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-o', label='Train loss')
    ax.plot(epochs, val_loss, 'r-s', label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train vs Val Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss in log scale
    ax = axes[0, 1]
    ax.plot(epochs, train_loss, 'b-o', label='Train loss')
    ax.plot(epochs, val_loss, 'r-s', label='Val loss')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Loss (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation & Hard Validation metrics
    ax = axes[1, 0]
    ax.plot(epochs, [h['val_auc'] for h in history], 'g-o', label='Val AUC')
    ax.plot(epochs, [h['val_balanced_acc'] for h in history], 'r-s', label='Val Bal Acc')
    if 'hard_val_auc' in history[0]:
        ax.plot(epochs, [h['hard_val_auc'] for h in history], 'g--', label='Hard Val AUC')
        ax.plot(epochs, [h['hard_val_balanced_acc'] for h in history], 'r--', label='Hard Val Bal Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Metrics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)

    # Update/weight ratios per layer
    ax = axes[1, 1]
    layer_names = [k for k in history[0].get('update_ratios', {}).keys()]
    for name in layer_names:
        vals = [h['update_ratios'].get(name, 0) for h in history]
        ax.plot(epochs, vals, '-o', label=name, markersize=4)
    ax.axhline(y=1e-3, color='k', linestyle='--', alpha=0.5, label='target 1e-3')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||update|| / ||weight||')
    ax.set_title('Update/Weight Ratio per Step (target ≈ 1e-3)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_filters(model, save_path):
    filters = model.conv1.weight.data.cpu()
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
    for i, ax in enumerate(axes):
        f = filters[i, 0] 
        ax.imshow(f, cmap='gray', interpolation='nearest')
        ax.set_title(f'Filter {i}')
        ax.axis('off')
    fig.suptitle('Conv1 Learned Filters (5x5)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Filter visualization saved to {save_path}")
