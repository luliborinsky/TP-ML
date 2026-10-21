
from pathlib import Path
from ultralytics import YOLO

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "runs" / "detect" / "wider_face_test" / "weights" / "best.pt"
IMAGE_DIR = BASE_DIR / "detection_data" / "wider_yolo" / "val" / "images"
OUTPUT_DIR = BASE_DIR / "runs" / "detect"

def main():
    model = YOLO(str(MODEL_PATH))

    # Find images inside all subdirectories
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        p for p in IMAGE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    print(f"Images found: {len(images)}")

    if not images:
        print("ERROR: No images found.")
        return

    # Predict in batches and save annotated images
    model.predict(
        source=[str(p) for p in images],
        imgsz=640,
        conf=0.20,
        iou=0.60,
        device="cpu",
        save=True,
        project=str(OUTPUT_DIR),
        name="wider_face_predictions",
        exist_ok=True,
        stream=False,
        verbose=False,
    )

    print("Prediction finished.")

if __name__ == "__main__":
    main()
