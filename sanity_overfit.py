"""
sanity_overfit.py: overfit a tiny subset (no augmentation)

If the model can memorize ~50 images and loss → ~0, the training pipeline
has no bugs. If loss stays high, something is broken.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Reuse data loading and model from cnn.py
from cnn import load_all, norm, Net, get_device, set_seed

SEED = 42
SUBSET_SIZE = 50   # ~25 face + ~25 noface
EPOCHS = 500
LR = 0.01


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    # Load full training data
    train_x, train_y = load_all('./train_images', './cache_train.pt')

    # Pick a balanced tiny subset
    idx_0 = (train_y == 0).nonzero(as_tuple=True)[0]
    idx_1 = (train_y == 1).nonzero(as_tuple=True)[0]
    perm_0 = idx_0[torch.randperm(len(idx_0))[:SUBSET_SIZE // 2]]
    perm_1 = idx_1[torch.randperm(len(idx_1))[:SUBSET_SIZE // 2]]
    subset_idx = torch.cat([perm_0, perm_1])

    X = norm(train_x[subset_idx].float() / 255).to(device)
    y = train_y[subset_idx].to(device)
    print(f"Subset: {len(X)} images ({(y == 0).sum()} noface, {(y == 1).sum()} face)")

    # Fresh model, NO augmentation, NO regularization
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    # Train on the same tiny batch repeatedly
    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # Check accuracy on the same subset (should reach 100%)
        _, pred = torch.max(outputs, 1)
        acc = (pred == y).float().mean().item()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3} | loss {loss.item():.6f} | acc {acc:.1%}")

        if loss.item() < 1e-4:
            print(f"\n✓ Loss reached {loss.item():.6f} at epoch {epoch} — "
                  f"pipeline is correct!")
            break
    else:
        if loss.item() < 0.01:
            print(f"\n✓ Loss = {loss.item():.6f} after {EPOCHS} epochs — OK")
        else:
            print(f"\n✗ Loss = {loss.item():.4f} after {EPOCHS} epochs — "
                  f"something may be wrong!")


if __name__ == '__main__':
    main()
