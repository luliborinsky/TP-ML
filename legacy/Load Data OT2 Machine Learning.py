import os
from collections import Counter

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# ---------------------------------------------------------------------------
# Device: GPU de Apple (MPS) si está disponible, si no CUDA, si no CPU
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
train_dir = './train_images'    # folder containing training images
test_dir = './test_images'      # folder containing test images

for d in (train_dir, test_dir):
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"No encuentro la carpeta '{d}'. Tiene que estar al lado de este script, "
            f"con subcarpetas 0/ (no caras) y 1/ (caras)."
        )

transform = transforms.Compose(
    [transforms.Grayscale(),   # transforms to gray-scale (1 input channel)
     transforms.ToTensor(),    # transforms to Torch tensor (needed for PyTorch)
     transforms.Normalize(mean=(0.5,), std=(0.5,))])  # (x - 0.5) / 0.5 -> values in (-1, +1)

# Define two pytorch datasets (train/test)
train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

valid_size = 0.2   # proportion of validation set (80% train, 20% validation)
batch_size = 32

# Define randomly the indices of examples to use for training and for validation
num_train = len(train_data)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

# Define two "samplers" that will randomly pick examples from the training and validation set
train_sampler = SubsetRandomSampler(train_new_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Dataloaders (take care of loading the data from disk, batch by batch, during training)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

classes = ('noface', 'face')  # "1" means "face" and "0" non-face (only used for display)


# ---------------------------------------------------------------------------
# Verificación del setup
# ---------------------------------------------------------------------------
def check_setup():
    print("PyTorch:", torch.__version__)
    print("Device:", device)
    print()
    print("Train total:", len(train_data), "| Test total:", len(test_data))
    print("Split: train", len(train_new_idx), "| valid", len(valid_idx))
    print("Mapeo carpetas -> clase:", train_data.class_to_idx)
    print("Imágenes por clase (train):", Counter(train_data.targets))
    print("Imágenes por clase (test): ", Counter(test_data.targets))
    print()
    images, labels = next(iter(train_loader))
    print("Shape de un batch:", tuple(images.shape))   # esperado: (32, 1, 36, 36)
    print("Rango de valores:", round(images.min().item(), 3), "a", round(images.max().item(), 3))
    print("Labels del batch:", labels.tolist())

    # Prueba rápida de que el device funciona
    x = images.to(device)
    print("Batch movido a", x.device, "OK")


def main():
    check_setup()

    # Training (lo armamos en el siguiente paso)
    # loop over epochs: one epoch = one pass through the whole training dataset
    # for epoch in range(1, n_epochs+1):
    #   loop over iterations: one iteration = 1 batch of examples
    #   for data, target in train_loader:


# Obligatorio en macOS por los DataLoaders con num_workers > 0
if __name__ == "__main__":
    main()
