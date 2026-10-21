
from pathlib import Path
import shutil

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent

IMAGES_DIR = PROJECT_DIR / "WIDER_train" / "images"
ANNOTATIONS_FILE = PROJECT_DIR / "wider_face_train_bbx_gt.txt"

OUTPUT_DIR = PROJECT_DIR / "detection_data" / "wider_yolo" / "train"
OUTPUT_IMAGES = OUTPUT_DIR / "images"
OUTPUT_LABELS = OUTPUT_DIR / "labels"


def main():
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(
            f"Images folder not found: {IMAGES_DIR}"
        )

    if not ANNOTATIONS_FILE.is_file():
        raise FileNotFoundError(
            f"Annotations file not found: {ANNOTATIONS_FILE}"
        )

    

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

    # Parse the WIDER FACE annotation file.

    with ANNOTATIONS_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        lines = iter(
            line.strip()
            for line in file
            if line.strip()
        )


        image_count = 0
        missing_count = 0
        face_count = 0
        invalid_count = 0

        while True:
            try:
                relative_image = next(lines).strip()
            except StopIteration:
                break

            
            if not relative_image.lower().endswith(".jpg"):
                if relative_image == "0 0 0 0 0 0 0 0 0 0":
                    print("Skipping stray zero line.")
                    invalid_count += 1
                    continue

                print("Unexpected line:", repr(relative_image))
                invalid_count += 1
                break


            try:
                count_line = next(lines).strip()
                number_of_faces = int(count_line)
            except (StopIteration, ValueError):
                print("Invalid annotation near:", repr(relative_image))
                print("Line interpreted as face count:", repr(count_line))
                print("Stopping conversion to avoid incorrect labels.")
                invalid_count += 1
                break

            boxes = []

            for _ in range(number_of_faces):
                try:
                    values = next(lines).strip().split()
                except StopIteration:
                    invalid_count += 1
                    break

                if len(values) < 4:
                    invalid_count += 1
                    continue

                x, y, width, height = map(float, values[:4])

                if width <= 0 or height <= 0:
                    continue

                boxes.append((x, y, width, height))

            source_image = IMAGES_DIR / Path(relative_image)

            if not source_image.is_file():
                missing_count += 1
                continue

            # Read the real image dimensions.
            try:
                from PIL import Image

                with Image.open(source_image) as image:
                    image_width, image_height = image.size
            except Exception as error:
                print(f"Could not read {source_image}: {error}")
                invalid_count += 1
                continue

            label_lines = []

            for x, y, width, height in boxes:
                # Clip boxes to image boundaries.
                x1 = max(0.0, x)
                y1 = max(0.0, y)
                x2 = min(float(image_width), x + width)
                y2 = min(float(image_height), y + height)

                box_width = x2 - x1
                box_height = y2 - y1

                if box_width <= 0 or box_height <= 0:
                    continue

                center_x = (x1 + x2) / 2 / image_width
                center_y = (y1 + y2) / 2 / image_height
                normalized_width = box_width / image_width
                normalized_height = box_height / image_height

                label_lines.append(
                    f"0 {center_x:.6f} {center_y:.6f} "
                    f"{normalized_width:.6f} {normalized_height:.6f}"
                )

            # Preserve the relative folder structure to avoid name collisions.
            relative_path = Path(relative_image)
            destination_image = OUTPUT_IMAGES / relative_path
            destination_label = (
                OUTPUT_LABELS / relative_path.with_suffix(".txt")
            )

            destination_image.parent.mkdir(
                parents=True, exist_ok=True
            )
            destination_label.parent.mkdir(
                parents=True, exist_ok=True
            )

            shutil.copy2(source_image, destination_image)
            destination_label.write_text(
                "\n".join(label_lines),
                encoding="utf-8"
            )

            image_count += 1
            face_count += len(label_lines)

            if image_count % 1000 == 0:
                print(f"Processed {image_count} images...")

    print("\nConversion finished.")
    print(f"Images processed: {image_count}")
    print(f"Face boxes written: {face_count}")
    print(f"Images missing: {missing_count}")
    print(f"Invalid annotations/images: {invalid_count}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
