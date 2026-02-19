from pathlib import Path

from PIL import Image
from transformers import pipeline


# Путь к датасету(вставьте свой)
ROOT = Path(r"C:\Users\mason\OneDrive\Рабочий стол\ModifiedOpenLabelling-main")
IMAGES_DIR = ROOT / "images" / "train"
LABELS_DIR = ROOT / "labels" / "train"

# ID классов в current class_list.txt:
# 0: Person
# 1: Bike
# 2: Car  <- исходный класс, который хотим уточнить как car / bus / truck
# 3: Small Animal
# 4: Large Animal
# 5: Bird
CAR_CLASS_ID = 2

# Новые классы. Предполагается, что вы добавите их В КОНЕЦ файла class_list.txt,
# чтобы они получили индексы 6 и 7, не меняя существующую разметку.
# 6: Bus
# 7: Truck
# 8,9,10 <- Классы для кораблей
LABEL_TO_ID = {"car": 2, "bus": 6, "truck": 7}
ZS_LABELS = list(LABEL_TO_ID.keys())


def find_image_for_label(label_path: Path) -> Path | None:
    rel = label_path.relative_to(LABELS_DIR)
    # тот же подкаталог и имя файла, только расширение картинки
    stem = rel.with_suffix("").name
    subdir = rel.parent

    for ext in (".jpg", ".jpeg", ".png"):
        candidate = IMAGES_DIR / subdir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def relabel_file(label_path: Path, zs_classifier) -> None:
    img_path = find_image_for_label(label_path)
    if img_path is None:
        print(f"[WARN] can t find image for {label_path}")
        return

    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    new_lines: list[str] = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        return

    print(f"\nLabel file: {label_path}")

    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            new_lines.append(line)
            continue

        cls = int(parts[0])
        x_center, y_center, bw, bh = map(float, parts[1:5])

        if cls != CAR_CLASS_ID:
            new_lines.append(line)
            continue

        xc = x_center * w
        yc = y_center * h
        bw_px = bw * w
        bh_px = bh * h
        x1 = max(0, int(xc - bw_px / 2))
        y1 = max(0, int(yc - bh_px / 2))
        x2 = min(w, int(xc + bw_px / 2))
        y2 = min(h, int(yc + bh_px / 2))

        if x2 <= x1 or y2 <= y1:
            new_lines.append(line)
            continue

        roi = img.crop((x1, y1, x2, y2))
        zs_outputs = zs_classifier(roi, candidate_labels=ZS_LABELS)
        best = max(zs_outputs, key=lambda o: o["score"])
        best_label = best["label"]
        best_score = best["score"]

        new_cls = LABEL_TO_ID.get(best_label, CAR_CLASS_ID)

        print(
            f"  obj#{idx}: car -> {best_label} (score={best_score:.3f}), "
            f"bbox=({x1},{y1},{x2},{y2})"
        )
        parts[0] = str(new_cls)
        new_lines.append(" ".join(map(str, parts)))
    new_content = "\n".join(new_lines) + "\n"
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def main():
    print("init (CLIP)...")
    zs_classifier = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
    )

    txt_files = sorted(LABELS_DIR.rglob("*.txt"))
    print(f"Найдено файлов меток: {len(txt_files)}")

    for label_path in txt_files:
        relabel_file(label_path, zs_classifier)

    print("Done.")

if __name__ == "__main__":
    main()

