from pathlib import Path
from ultralytics import YOLO



PROJECT_DIR = Path(__file__).resolve().parent


DATASET_DIR = PROJECT_DIR / "detection_data" / "yolo_dataset"


TRAIN_IMAGES = DATASET_DIR / "train" / "images"
VAL_IMAGES = DATASET_DIR / "val" / "images"


if not TRAIN_IMAGES.exists():
    raise FileNotFoundError(f"Training images not found: {TRAIN_IMAGES}")

if not VAL_IMAGES.exists():
    raise FileNotFoundError(f"Validation images not found: {VAL_IMAGES}")


DATA_YAML = DATASET_DIR / "data_generated.yaml"

DATA_YAML.write_text(
    f"""path: {DATASET_DIR.as_posix()}

train: {TRAIN_IMAGES.as_posix()}
val: {VAL_IMAGES.as_posix()}

names:
  0: face
""",
    encoding="utf-8"
)



model = YOLO("yolo26n.pt")


# Train
model.train(
    data=str(DATA_YAML),
    epochs=30,
    imgsz=640,
    batch=2,
    device="cpu",
    workers=0
)