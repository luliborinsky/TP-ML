
from pathlib import Path
import random
import shutil

ROOT = Path("detection_data/wider_yolo")
TRAIN_IMAGES = ROOT / "train" / "images"
TRAIN_LABELS = ROOT / "train" / "labels"
VAL_IMAGES = ROOT / "val" / "images"
VAL_LABELS = ROOT / "val" / "labels"

random.seed(42)

images = sorted(
    p for p in TRAIN_IMAGES.rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if len(images) == 0:
    raise RuntimeError("No training images found.")

if VAL_IMAGES.exists() and any(VAL_IMAGES.rglob("*")):
    raise RuntimeError(
        "Validation folder is not empty. "
        "Stopping to avoid moving files twice."
    )

random.shuffle(images)
n_val = round(len(images) * 0.10)
val_images = images[:n_val]

VAL_IMAGES.mkdir(parents=True, exist_ok=True)
VAL_LABELS.mkdir(parents=True, exist_ok=True)

for image in val_images:
    relative = image.relative_to(TRAIN_IMAGES)
    label = (TRAIN_LABELS / relative).with_suffix(".txt")

    if not label.is_file():
        raise FileNotFoundError(f"Missing label: {label}")

    destination_image = VAL_IMAGES / relative
    destination_label = VAL_LABELS / relative.with_suffix(".txt")

    destination_image.parent.mkdir(parents=True, exist_ok=True)
    destination_label.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(image), str(destination_image))
    shutil.move(str(label), str(destination_label))

print(f"Images remaining for training: {len(images) - n_val}")
print(f"Images reserved for validation: {n_val}")
print("Split completed.")
