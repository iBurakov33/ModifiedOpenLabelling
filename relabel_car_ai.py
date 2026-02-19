import os
from PIL import Image
from transformers import pipeline

DATASET_ROOT = "D:/VS_code/Praktika/ModifiedOpenLabelling"

IMAGES_ROOT = os.path.join(DATASET_ROOT, "images/train")
LABELS_ROOT = os.path.join(DATASET_ROOT, "labels/train")
OUTPUT_LABELS_ROOT = os.path.join(DATASET_ROOT, "labels_new/train")

OLD_CAR_LABEL = 2

NEW_LABELS = {
    "car": 2,
    "bus": 6,
    "truck": 7
}

MILITARY_KEYWORDS = [
    "tank",
    "armored",
    "armoured",
    "military",
    "apc",
    "artillery",
    "missile",
    "launcher",
    "howitzer"
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224"
)

def yolo_to_pixels(bbox, img_w, img_h):
    x_c, y_c, w, h = bbox
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

def classify_crop(crop):
    preds = classifier(crop)

    for p in preds:
        label = p["label"].lower()

        for kw in MILITARY_KEYWORDS:
            if kw in label:
                return "truck"

        if "bus" in label:
            return "bus"

        if "truck" in label or "lorry" in label:
            return "truck"

        if "car" in label:
            return "car"

    return "truck"

for root, _, files in os.walk(IMAGES_ROOT):
    for file in files:
        if not file.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(root, file)

        rel_path = os.path.relpath(image_path, IMAGES_ROOT)
        rel_dir = os.path.dirname(rel_path)

        label_path = os.path.join(
            LABELS_ROOT,
            rel_dir,
            os.path.splitext(file)[0] + ".txt"
        )

        if not os.path.exists(label_path):
            continue

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                continue
            parts = line.strip().split()
            cls = int(parts[0])
            bbox = list(map(float, parts[1:]))

            if cls != OLD_CAR_LABEL:
                new_lines.append(line.strip())
                continue

            x1, y1, x2, y2 = yolo_to_pixels(bbox, img_w, img_h)
            crop = image.crop((x1, y1, x2, y2))

            new_class_name = classify_crop(crop)
            new_class_id = NEW_LABELS[new_class_name]

            new_line = " ".join(
                [str(new_class_id)] + [f"{v:.6f}" for v in bbox]
            )
            new_lines.append(new_line)

            print(new_class_name, "←", rel_path)

        output_dir = os.path.join(OUTPUT_LABELS_ROOT, rel_dir)
        os.makedirs(output_dir, exist_ok=True)

        output_label_path = os.path.join(
            output_dir,
            os.path.splitext(file)[0] + ".txt"
        )

        with open(output_label_path, "w") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")
            else:
                f.write("\n")

        print(f"✔ processed {rel_path}")
