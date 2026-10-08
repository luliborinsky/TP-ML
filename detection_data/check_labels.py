import os
from PIL import Image, ImageDraw

# Folders
images_dir = "detection_data/Images"
labels_dir = "detection_data/Labels"

# Go through all images
for filename in sorted(os.listdir(images_dir)):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(images_dir, filename)

    # Corresponding label file
    name = os.path.splitext(filename)[0]
    label_path = os.path.join(labels_dir, name + ".txt")

    if not os.path.exists(label_path):
        print(f"No label found for {filename}")
        continue

    # Open image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size

    # Read YOLO labels
    with open(label_path, "r") as f:
        for line in f:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id, x_center, y_center, box_width, box_height = map(
                float, values
            )

            # Convert normalized YOLO coordinates to pixels
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            # Calculate corners
            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2

            # Draw bounding box
            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

            # Write class name
            draw.text(
                (x1, y1 - 15),
                "face",
                fill="red"
            )

    # Show image
    print(f"Showing: {filename}")
    image.show()