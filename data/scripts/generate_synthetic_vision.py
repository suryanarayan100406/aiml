"""
Generate synthetic vision training data for YOLOv8 desktop distraction detector.
Creates synthetic screenshot-like images with annotated objects and YOLO-format labels.

Detection classes:
  0: tab_bar      — browser tab bar region
  1: phone        — mobile phone on desk
  2: distraction  — non-work items (cups, snacks, papers)
  3: work_tool    — primary work application/IDE

Output:
  data/processed/vision_dataset/
    images/train/  — synthetic training images
    images/val/    — synthetic validation images
    labels/train/  — YOLO format label files
    labels/val/    — YOLO format label files
    data.yaml      — YOLOv8 dataset config
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── Configuration ─────────────────────────────────────────────────────────────

NUM_TRAIN = 400
NUM_VAL = 100
IMG_SIZE = 640
CLASSES = ["tab_bar", "phone", "distraction", "work_tool"]

# Color palettes for different screen regions
BG_COLORS = [
    (30, 30, 40),    # Dark IDE
    (240, 240, 245),  # Light browser
    (20, 20, 30),    # Terminal
    (45, 45, 55),    # Code editor dark
    (250, 250, 252),  # Document editor
    (35, 40, 50),    # Dark mode browser
]

TAB_BAR_COLORS = [
    (50, 50, 60),    # Chrome dark
    (222, 225, 230), # Chrome light
    (60, 55, 68),    # Firefox dark
    (235, 235, 240), # Safari light
    (40, 42, 54),    # VS Code tab bar
]

PHONE_COLORS = [
    (15, 15, 15),    # Black phone
    (200, 200, 210), # Silver phone
    (40, 40, 50),    # Dark phone with screen
]

DISTRACTION_COLORS = [
    (139, 90, 43),   # Coffee cup
    (200, 50, 50),   # Red snack wrapper
    (180, 180, 170), # Paper
    (60, 120, 60),   # Plant
    (100, 100, 180), # Water bottle
]


def draw_tab_bar(draw: ImageDraw.Draw, num_tabs: int) -> list:
    """Draw a browser tab bar at the top of the image. Returns YOLO annotations."""
    annotations = []
    bar_height = random.randint(28, 45)
    bar_color = random.choice(TAB_BAR_COLORS)

    # Draw tab bar background
    draw.rectangle([0, 0, IMG_SIZE, bar_height], fill=bar_color)

    # Draw individual tabs
    tab_width = min(IMG_SIZE // max(num_tabs, 1), 180)
    for i in range(min(num_tabs, IMG_SIZE // 20)):
        x1 = i * tab_width
        x2 = x1 + tab_width - 2
        shade = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in bar_color)
        draw.rectangle([x1, 2, x2, bar_height - 2], fill=shade)
        # Tab separator
        draw.line([x2, 2, x2, bar_height - 2], fill=(100, 100, 100), width=1)

    # YOLO annotation: class_id x_center y_center width height (normalized)
    x_center = 0.5
    y_center = (bar_height / 2) / IMG_SIZE
    width = 1.0
    height = bar_height / IMG_SIZE
    annotations.append((0, x_center, y_center, width, height))  # class 0 = tab_bar

    return annotations


def draw_phone(draw: ImageDraw.Draw) -> list:
    """Draw a phone shape on the desk area. Returns YOLO annotations."""
    annotations = []

    # Phone dimensions and position (lower portion of screen = "desk")
    pw = random.randint(40, 70)
    ph = random.randint(70, 130)
    px = random.randint(10, IMG_SIZE - pw - 10)
    py = random.randint(IMG_SIZE // 2, IMG_SIZE - ph - 10)

    phone_color = random.choice(PHONE_COLORS)
    # Draw phone body
    draw.rounded_rectangle([px, py, px + pw, py + ph], radius=8, fill=phone_color)
    # Draw screen area
    screen_margin = 4
    screen_color = tuple(min(255, c + 40) for c in phone_color)
    draw.rounded_rectangle(
        [px + screen_margin, py + screen_margin + 5,
         px + pw - screen_margin, py + ph - screen_margin - 5],
        radius=4, fill=screen_color
    )
    # Small home button or notch
    draw.ellipse(
        [px + pw // 2 - 4, py + ph - 12, px + pw // 2 + 4, py + ph - 4],
        fill=(80, 80, 80)
    )

    # YOLO annotation
    x_center = (px + pw / 2) / IMG_SIZE
    y_center = (py + ph / 2) / IMG_SIZE
    width = pw / IMG_SIZE
    height = ph / IMG_SIZE
    annotations.append((1, x_center, y_center, width, height))  # class 1 = phone

    return annotations


def draw_distraction(draw: ImageDraw.Draw) -> list:
    """Draw a distraction object (cup, paper, etc.). Returns YOLO annotations."""
    annotations = []
    obj_type = random.choice(["cup", "paper", "snack"])
    color = random.choice(DISTRACTION_COLORS)

    if obj_type == "cup":
        w, h = random.randint(25, 40), random.randint(35, 55)
        x = random.randint(10, IMG_SIZE - w - 10)
        y = random.randint(IMG_SIZE // 2, IMG_SIZE - h - 10)
        # Cup body
        draw.rounded_rectangle([x, y, x + w, y + h], radius=4, fill=color)
        # Handle
        draw.arc([x + w - 5, y + h // 4, x + w + 12, y + 3 * h // 4],
                 start=-90, end=90, fill=color, width=3)
    elif obj_type == "paper":
        w, h = random.randint(50, 90), random.randint(60, 100)
        x = random.randint(10, IMG_SIZE - w - 10)
        y = random.randint(IMG_SIZE // 3, IMG_SIZE - h - 10)
        draw.rectangle([x, y, x + w, y + h], fill=color)
        # Text lines
        for line_y in range(y + 8, y + h - 8, 8):
            line_w = random.randint(w // 2, w - 10)
            draw.line([x + 5, line_y, x + 5 + line_w, line_y],
                      fill=(60, 60, 60), width=1)
    else:  # snack
        w, h = random.randint(30, 50), random.randint(20, 35)
        x = random.randint(10, IMG_SIZE - w - 10)
        y = random.randint(IMG_SIZE // 2, IMG_SIZE - h - 10)
        draw.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=color)

    x_center = (x + w / 2) / IMG_SIZE
    y_center = (y + h / 2) / IMG_SIZE
    width_norm = w / IMG_SIZE
    height_norm = h / IMG_SIZE
    annotations.append((2, x_center, y_center, width_norm, height_norm))

    return annotations


def draw_work_tool(draw: ImageDraw.Draw, bg_color: tuple) -> list:
    """Draw a main work application area. Returns YOLO annotations."""
    annotations = []

    # Work area takes up most of the screen
    margin = random.randint(5, 40)
    top_offset = random.randint(35, 55)  # Below tab bar
    x1, y1 = margin, top_offset
    x2, y2 = IMG_SIZE - margin, IMG_SIZE - random.randint(30, 80)

    # Slightly different shade from background
    work_color = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in bg_color)
    draw.rectangle([x1, y1, x2, y2], fill=work_color)

    # Simulate code/text lines
    line_colors = [
        (120, 180, 120), (180, 140, 100), (100, 150, 200),
        (200, 200, 200), (150, 120, 180), (180, 180, 100)
    ]
    for line_y in range(y1 + 10, y2 - 10, random.randint(14, 20)):
        indent = random.randint(10, 80)
        line_w = random.randint(100, x2 - x1 - indent - 20)
        line_color = random.choice(line_colors)
        draw.line([x1 + indent, line_y, x1 + indent + line_w, line_y],
                  fill=line_color, width=2)

    w = x2 - x1
    h = y2 - y1
    x_center = (x1 + w / 2) / IMG_SIZE
    y_center = (y1 + h / 2) / IMG_SIZE
    width_norm = w / IMG_SIZE
    height_norm = h / IMG_SIZE
    annotations.append((3, x_center, y_center, width_norm, height_norm))

    return annotations


def generate_image(rng: random.Random) -> tuple:
    """Generate a single synthetic screenshot with annotations."""
    bg_color = rng.choice(BG_COLORS)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg_color)
    draw = ImageDraw.Draw(img)

    all_annotations = []

    # Always draw work tool background
    all_annotations.extend(draw_work_tool(draw, bg_color))

    # Tab bar — always present, varying tab counts
    num_tabs = rng.choice([3, 5, 8, 10, 12, 15, 18, 20, 25, 30])
    all_annotations.extend(draw_tab_bar(draw, num_tabs))

    # Phone — 40% chance
    if rng.random() < 0.4:
        all_annotations.extend(draw_phone(draw))

    # Distractions — 0–3 objects
    num_distractions = rng.choices([0, 1, 2, 3], weights=[0.3, 0.35, 0.25, 0.1])[0]
    for _ in range(num_distractions):
        all_annotations.extend(draw_distraction(draw))

    # Add some noise for realism
    arr = np.array(img)
    noise = np.random.normal(0, 3, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Store metadata for the fused dataset
    metadata = {
        "tab_count": num_tabs,
        "phone_visible": 1 if any(a[0] == 1 for a in all_annotations) else 0,
        "distraction_count": sum(1 for a in all_annotations if a[0] == 2),
        "focus_ratio": next(
            (a[3] * a[4] for a in all_annotations if a[0] == 3), 0.5
        ),
    }

    return img, all_annotations, metadata


def main():
    rng = random.Random(42)
    base_dir = os.path.join(os.path.dirname(__file__), "..", "processed", "vision_dataset")

    # Create directory structure
    for split in ["train", "val"]:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)

    # Write data.yaml for YOLOv8
    data_yaml = f"""path: {os.path.abspath(base_dir)}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(os.path.join(base_dir, "data.yaml"), "w") as f:
        f.write(data_yaml)

    # Generate images
    all_metadata = []
    for split, num_images in [("train", NUM_TRAIN), ("val", NUM_VAL)]:
        for i in range(num_images):
            img, annotations, metadata = generate_image(rng)

            # Save image
            img_path = os.path.join(base_dir, "images", split, f"screen_{i:04d}.png")
            img.save(img_path)

            # Save YOLO labels
            label_path = os.path.join(base_dir, "labels", split, f"screen_{i:04d}.txt")
            with open(label_path, "w") as f:
                for cls, xc, yc, w, h in annotations:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            metadata["image"] = f"screen_{i:04d}.png"
            metadata["split"] = split
            all_metadata.append(metadata)

    # Save metadata CSV for fused dataset generation
    import csv
    meta_path = os.path.join(base_dir, "vision_metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "tab_count",
                                                "phone_visible", "distraction_count",
                                                "focus_ratio"])
        writer.writeheader()
        writer.writerows(all_metadata)

    print(f"✅ Generated {NUM_TRAIN + NUM_VAL} synthetic screenshots → {base_dir}")
    print(f"   Train: {NUM_TRAIN}, Val: {NUM_VAL}")
    print(f"   Classes: {CLASSES}")
    print(f"   YOLOv8 config: {os.path.join(base_dir, 'data.yaml')}")


if __name__ == "__main__":
    main()
