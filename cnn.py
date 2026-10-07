
# cnn.py — Face / non-face CNN classifier

#Structure mirrors the course template:
# 1. Data loading & DataLoader (with balanced sampling and GPU-friendly augmentation)
# 2. CNN architecture: class Net(nn.Module)
# 3. Training loop with validation checkpointing
# 4. Final evaluation on the test set

#Diagnostics and plotting are in diagnostics.py to keep this file concise.


import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data.sampler import WeightedRandomSampler

from diagnostics import (
    compute_metrics,
    find_best_threshold,
    build_hard_val_loader,
    plot_training_curves,
    plot_filters,
)

# Configuration

CONFIG = {
    'train_dir':       './train_images',
    'test_dir':        './test_images',
    'cache_train':     './cache_train.pt',
    'cache_test':      './cache_test.pt',
    'valid_size':      0.2,
    'batch_size':      32,
    'epochs':          10,
    'lr':              0.01,
    'momentum':        0.9,
    'optimizer':       'nesterov',   # 'sgd', 'nesterov', 'adam'
    'scheduler':       'none',       # 'none', 'steplr', 'cosine'
    'seed':            42,
    'tag':             'step3_nesterov',
    'checkpoint_path': './face_net_step3_nesterov.pt',
    'log_path':        './training_log_step3_nesterov.json',
    'plot_path':       './training_curves_step3_nesterov.png',
}

CLASSES = ('noface', 'face')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# Data Loading and Transforms
def load_all(folder, cache_file):
    if os.path.exists(cache_file):
        d = torch.load(cache_file, weights_only=True)
        return d['x'], d['y']
    ds = torchvision.datasets.ImageFolder(folder)
    arr = np.stack([np.array(Image.open(path).convert('L'))
                    for path, _ in ds.samples])
    x = torch.from_numpy(arr).unsqueeze(1)  
    y = torch.tensor(ds.targets)
    torch.save({'x': x, 'y': y}, cache_file)
    return x, y


def norm(x):
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True)
    return (x - m) / (s + 1e-6)


def augment(x, min_scale=0.9):
    B, _, H, W = x.shape

    # Random horizontal flip
    flip = (torch.rand(B) < 0.5).view(B, 1, 1, 1)
    x = torch.where(flip, x.flip(3), x)

    # Affine: rotation ±10°, scale [min_scale, 1.1], translation ±0.2 (±10% of image size in [-1, 1] coords)
    ang = (torch.rand(B) * 2 - 1) * (10.0 * np.pi / 180.0)
    sc  = min_scale + (1.1 - min_scale) * torch.rand(B)
    tx  = (torch.rand(B) * 2 - 1) * 0.2
    ty  = (torch.rand(B) * 2 - 1) * 0.2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], dim=1),
                         torch.stack([sin,  cos, ty], dim=1)], dim=1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros',
                       align_corners=False)

    return x


def prep_train(x_uint8, min_scale=0.9):
    return norm(augment(x_uint8.float() / 255.0, min_scale=min_scale))


def build_dataloaders(cfg):
    train_x, train_y = load_all(cfg['train_dir'], cfg['cache_train'])
    test_x,  test_y  = load_all(cfg['test_dir'],  cfg['cache_test'])

    num_train = len(train_y)
    indices = list(range(num_train))
    # Fixed random state for split so all runs share the same validation set
    np.random.RandomState(42).shuffle(indices)
    split = int(np.floor(cfg['valid_size'] * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    valid_idx_t = torch.tensor(valid_idx)
    x_val_raw   = train_x[valid_idx_t].float() / 255.0
    y_val       = train_y[valid_idx_t]
    X_val       = norm(x_val_raw)
    X_test      = norm(test_x.float() / 255.0)

    # Hard validation
    hard_val_loader = build_hard_val_loader(x_val_raw, y_val, norm_fn=norm, seed=123)

    # Balanced sampling for training
    counts  = torch.bincount(train_y[train_idx])
    weights = torch.zeros(len(train_y), dtype=torch.double)
    weights[train_idx] = 1.0 / counts[train_y[train_idx]].double()
    sampler = WeightedRandomSampler(weights, num_samples=len(train_idx),
                                    replacement=True)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=cfg['batch_size'], sampler=sampler, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=1024, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, test_y),
        batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

    print(f"Train: {len(train_idx)} | Valid: {len(valid_idx)} | Test: {len(test_y)}")
    return train_loader, valid_loader, hard_val_loader, test_loader


#  Model Architecture

class Net(nn.Module):

    def __init__(self, c1=6, c2=16, pooling='max'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 5)
        self.pool  = nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.fc1   = nn.Linear(c2 * 6 * 6, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def build_model(device, c1=6, c2=16, pooling='max'):
    return Net(c1=c1, c2=c2, pooling=pooling).to(device)


# Training & Inference

def predict_proba(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prob = F.softmax(model(x), dim=1)[:, 1]
            ps.append(prob.cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def compute_val_loss(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total_loss += criterion(model(x), y).item()
            n_batches += 1
    return total_loss / n_batches


def train_one_epoch(model, loader, criterion, optimizer, device, min_scale=0.9):
    model.train()
    total_loss, n_batches = 0.0, 0
    for inputs, labels in loader:
        inputs = prep_train(inputs, min_scale=min_scale).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train(model, train_loader, valid_loader, hard_val_loader, cfg, device):
    criterion = nn.CrossEntropyLoss()

    opt_name = cfg.get('optimizer', 'sgd').lower()
    lr = cfg['lr']
    if opt_name == 'nesterov':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=cfg.get('momentum', 0.9), nesterov=True)
    elif opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=cfg.get('momentum', 0.9))

    sched_name = cfg.get('scheduler', 'none').lower()
    if sched_name == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, cfg['epochs'] // 3), gamma=0.5)
    elif sched_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs'], eta_min=lr * 0.01)
    else:
        scheduler = None

    history = []
    best_val_loss = float('inf')
    best_epoch = 0

    min_scale = cfg.get('min_scale', 0.9)

    for epoch in range(1, cfg['epochs'] + 1):
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, min_scale=min_scale)

        if scheduler is not None:
            scheduler.step()

        val_loss = compute_val_loss(model, valid_loader, criterion, device)
        val_y, val_p = predict_proba(model, valid_loader, device)
        val_m = compute_metrics(val_y, val_p)

        hard_y, hard_p = predict_proba(model, hard_val_loader, device)
        hard_m = compute_metrics(hard_y, hard_p)

        curr_lr = optimizer.param_groups[0]['lr']
        history.append({
            'epoch':                 epoch,
            'train_loss':            round(avg_loss, 4),
            'val_loss':              round(val_loss, 4),
            'lr':                    curr_lr,
            'val_auc':               val_m['auc'],
            'val_balanced_acc':      val_m['balanced_acc'],
            'hard_val_auc':          hard_m['auc'],
            'hard_val_balanced_acc': hard_m['balanced_acc'],
        })

        print(f"Epoch {epoch:2d}/{cfg['epochs']} | "
              f"train {avg_loss:.4f}  val {val_loss:.4f} | "
              f"val AUC {val_m['auc']:.4f}  bal_acc {val_m['balanced_acc']:.4f} | "
              f"hard_val AUC {hard_m['auc']:.4f}")

        # Checkpoint on lowest validation loss (early stopping from class)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), cfg['checkpoint_path'])
            print(f"  -> Checkpoint saved (best val loss: {best_val_loss:.4f})")

    print(f"\nTraining completed. Best epoch: {best_epoch} (val loss: {best_val_loss:.4f})")
    return history, best_epoch


#  Experiment Runner

def run_experiment(cfg):
    set_seed(cfg['seed'])
    device = get_device()
    tag = cfg.get('tag', 'run')
    print(f"\n{'='*65}\nExperiment: {tag} (seed={cfg['seed']}) | Device: {device}\n{'='*65}")

    train_loader, valid_loader, hard_val_loader, test_loader = build_dataloaders(cfg)

    c1 = cfg.get('c1', 6)
    c2 = cfg.get('c2', 16)
    pooling = cfg.get('pooling', 'max')
    model = build_model(device, c1=c1, c2=c2, pooling=pooling)

    history, best_epoch = train(model, train_loader, valid_loader, hard_val_loader, cfg, device)

    # Save training log
    if 'log_path' in cfg and cfg['log_path']:
        with open(cfg['log_path'], 'w') as f:
            json.dump({'config': cfg, 'history': history, 'best_epoch': best_epoch}, f, indent=2)

    # Load best checkpoint
    model.load_state_dict(
        torch.load(cfg['checkpoint_path'], weights_only=True, map_location=device))
    model.to(device)

    # 1. Threshold chosen on normal validation
    val_y, val_p = predict_proba(model, valid_loader, device)
    thresh_val = find_best_threshold(val_y, val_p)
    val_m = compute_metrics(val_y, val_p, thresh_val)

    # 2. Threshold chosen on hard validation
    hard_y, hard_p = predict_proba(model, hard_val_loader, device)
    thresh_hard = find_best_threshold(hard_y, hard_p)
    hard_m = compute_metrics(hard_y, hard_p, thresh_hard)

    # 3. Test evaluated once with both thresholds and fixed 0.50
    test_y, test_p = predict_proba(model, test_loader, device)
    test_m_val = compute_metrics(test_y, test_p, thresh_val)
    test_m_hard = compute_metrics(test_y, test_p, thresh_hard)
    test_m_half = compute_metrics(test_y, test_p, 0.5)

    print("\n" + "=" * 60)
    print(f"FINAL TEST EVALUATION [{tag}] — epoch {best_epoch}")
    print("=" * 60)
    print(f"Validation threshold ({thresh_val}): Test AUC={test_m_val['auc']} | BalAcc={test_m_val['balanced_acc']} | Rec(Face)={test_m_val['recall_face']} | Rec(NoFace)={test_m_val['recall_noface']}")
    print(f"Fixed threshold 0.50 : Test AUC={test_m_half['auc']} | BalAcc={test_m_half['balanced_acc']} | Rec(Face)={test_m_half['recall_face']} | Rec(NoFace)={test_m_half['recall_noface']}")
    print(f"Hard-Val threshold   ({thresh_hard}): Test AUC={test_m_hard['auc']} | BalAcc={test_m_hard['balanced_acc']} | Rec(Face)={test_m_hard['recall_face']} | Rec(NoFace)={test_m_hard['recall_noface']}")
    print("=" * 60)

    return {
        'history': history,
        'best_epoch': best_epoch,
        'val_m': val_m,
        'hard_m': hard_m,
        'test_m_val': test_m_val,
        'test_m_hard': test_m_hard,
        'test_m_half': test_m_half,
        'thresh_val': thresh_val,
        'thresh_hard': thresh_hard,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default=CONFIG['optimizer'], choices=['sgd', 'nesterov', 'adam'])
    parser.add_argument('--lr', type=float, default=CONFIG['lr'])
    parser.add_argument('--momentum', type=float, default=CONFIG['momentum'])
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--scheduler', default=CONFIG['scheduler'], choices=['none', 'steplr', 'cosine'])
    parser.add_argument('--seed', type=int, default=CONFIG['seed'])
    parser.add_argument('--min-scale', type=float, default=0.9)
    parser.add_argument('--c1', type=int, default=6)
    parser.add_argument('--c2', type=int, default=16)
    parser.add_argument('--pooling', default='max', choices=['max', 'avg'])
    parser.add_argument('--tag', type=str, default='step3_nesterov')
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg['optimizer'] = args.optimizer
    cfg['lr'] = args.lr
    cfg['momentum'] = args.momentum
    cfg['epochs'] = args.epochs
    cfg['scheduler'] = args.scheduler
    cfg['seed'] = args.seed
    cfg['min_scale'] = args.min_scale
    cfg['c1'] = args.c1
    cfg['c2'] = args.c2
    cfg['pooling'] = args.pooling
    cfg['tag'] = args.tag
    cfg['checkpoint_path'] = f"./models/face_net_{args.tag}.pt"
    cfg['log_path'] = f"./logs/training_log_{args.tag}.json"
    cfg['plot_path'] = f"./figures/training_curves_{args.tag}.png"

    run_experiment(cfg)


if __name__ == '__main__':
    main()
