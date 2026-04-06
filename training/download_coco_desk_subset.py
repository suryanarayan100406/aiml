"""
Download COCO dataset subset for desk/workspace distraction detection.
Downloads only images containing desk-relevant objects:
  - cell phone (COCO class 77) → phone distraction
  - laptop (COCO class 73) → workspace monitor
  - keyboard (COCO class 76) → work tool
  - mouse (COCO class 74) → work tool
  - tv/monitor (COCO class 72) → workspace screen
  - remote (COCO class 75) → distraction
  - book (COCO class 84) → can be distraction
  - cup (COCO class 47) → neutral
  - bottle (COCO class 44) → neutral

Organizes into YOLO format structure ready for training.
Output: data/coco_desk_subset/

Usage:
    python training/download_coco_desk_subset.py
    python training/download_coco_desk_subset.py --max-images 500
"""

import os
import sys
import io

# Force UTF-8 for Windows PowerShell to prevent UnicodeEncodeError
if isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import shutil
import random
import argparse
import urllib.request
from pathlib import Path
from collections import defaultdict

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "coco_desk_subset"

# ─── COCO category IDs we care about ──────────────────────────
# These are the COCO annotation category IDs (NOT the class indices)
COCO_DESK_CATEGORIES = {
    # COCO_cat_id: (our_class_id, our_class_name)
    77: (0, "phone"),       # cell phone → primary distraction
    73: (1, "laptop"),      # laptop → workspace
    72: (1, "monitor"),     # tv/monitor → workspace (same class as laptop)
    76: (2, "keyboard"),    # keyboard → work tool
    74: (2, "mouse"),       # mouse → work tool (same class as keyboard)
    75: (3, "distraction"), # remote → distraction
    84: (3, "distraction"), # book → distraction (when not working)
    47: (3, "distraction"), # cup → neutral but counts for distraction_count
    44: (3, "distraction"), # bottle → neutral but counts for distraction_count
}

OUR_CLASS_NAMES = ["phone", "monitor", "work_tool", "distraction"]

# ─── COCO URLs ─────────────────────────────────────────────────
COCO_ANNO_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"

# We'll use the val set (5K images, ~1GB) — much faster than train (118K images, 18GB)
# For a demo, we only need the annotations + download individual images via URL
COCO_IMG_BASE = "http://images.cocodataset.org"


def download_file(url, dest, desc=""):
    """Download a file with progress bar"""
    if os.path.exists(dest):
        print(f"  ✓ Already exists: {dest}")
        return

    print(f"  ↓ Downloading {desc or url}...")
    
    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size if total_size > 0 else 0
        sys.stdout.write(f'\r    {pct:.1f}% ({count * block_size // 1024 // 1024}MB)')
        sys.stdout.flush()

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\n  ✓ Saved: {dest}")


def download_annotations():
    """Download COCO 2017 annotations (val set)"""
    anno_zip = DATASET_DIR / "annotations_trainval2017.zip"
    anno_dir = DATASET_DIR / "annotations"

    if (anno_dir / "instances_val2017.json").exists():
        print("✓ Annotations already downloaded")
        return anno_dir / "instances_val2017.json"

    download_file(COCO_ANNO_URL, str(anno_zip), "COCO 2017 annotations (~252MB)")

    # Extract
    import zipfile
    print("  ↻ Extracting annotations...")
    with zipfile.ZipFile(str(anno_zip), 'r') as z:
        z.extractall(str(DATASET_DIR))
    
    print(f"  ✓ Extracted to {anno_dir}")
    return anno_dir / "instances_val2017.json"


def parse_annotations(anno_path, max_images=300):
    """Parse COCO annotations and filter for desk-relevant categories"""
    print(f"\n📋 Parsing annotations from {anno_path.name}...")

    with open(anno_path, 'r') as f:
        coco = json.load(f)

    # Build image lookup
    images_by_id = {img['id']: img for img in coco['images']}

    # Find annotations with our categories
    relevant_annos = defaultdict(list)  # image_id -> [annotations]
    cat_counts = defaultdict(int)

    for anno in coco['annotations']:
        cat_id = anno['category_id']
        if cat_id in COCO_DESK_CATEGORIES:
            img_id = anno['image_id']
            our_class_id, our_class_name = COCO_DESK_CATEGORIES[cat_id]
            relevant_annos[img_id].append({
                'bbox': anno['bbox'],  # [x, y, width, height] in pixels
                'class_id': our_class_id,
                'class_name': our_class_name,
                'coco_cat_id': cat_id,
                'area': anno['area'],
            })
            cat_counts[our_class_name] += 1

    print(f"  Found {len(relevant_annos)} images with desk-relevant objects")
    print(f"  Category distribution:")
    for name, count in sorted(cat_counts.items()):
        print(f"    {name}: {count} annotations")

    # Prioritize images with phones (our key detection target)
    phone_images = [img_id for img_id, annos in relevant_annos.items()
                    if any(a['class_id'] == 0 for a in annos)]
    other_images = [img_id for img_id in relevant_annos if img_id not in phone_images]

    # Select images: prioritize phone images, fill rest with others
    selected = phone_images[:max_images // 2]
    remaining = max_images - len(selected)
    random.shuffle(other_images)
    selected += other_images[:remaining]

    print(f"  Selected {len(selected)} images (phone priority: {min(len(phone_images), max_images // 2)})")

    return selected, relevant_annos, images_by_id


def download_images(image_ids, images_by_id, split="val2017"):
    """Download selected COCO images"""
    img_dir = DATASET_DIR / "images" / split
    os.makedirs(img_dir, exist_ok=True)

    print(f"\n🖼️  Downloading {len(image_ids)} images...")
    downloaded = 0
    failed = 0

    for i, img_id in enumerate(image_ids):
        img_info = images_by_id[img_id]
        filename = img_info['file_name']
        dest = img_dir / filename

        if dest.exists():
            downloaded += 1
            continue

        url = f"{COCO_IMG_BASE}/{split}/{filename}"
        try:
            urllib.request.urlretrieve(url, str(dest))
            downloaded += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ Failed: {filename} ({e})")

        # Progress
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(image_ids)}] Downloaded {downloaded}, Failed {failed}")

    print(f"  ✓ Downloaded {downloaded} images, {failed} failed")
    return img_dir


def create_yolo_labels(image_ids, relevant_annos, images_by_id, img_dir):
    """Convert COCO annotations to YOLO format labels"""
    label_dir = DATASET_DIR / "labels" / img_dir.name
    os.makedirs(label_dir, exist_ok=True)

    print(f"\n📝 Creating YOLO labels...")
    count = 0

    for img_id in image_ids:
        img_info = images_by_id[img_id]
        img_w = img_info['width']
        img_h = img_info['height']
        filename = Path(img_info['file_name']).stem + '.txt'

        annos = relevant_annos[img_id]
        lines = []

        for a in annos:
            # COCO bbox: [x, y, width, height] (top-left corner)
            # YOLO format: class x_center y_center width height (all normalized 0-1)
            bx, by, bw, bh = a['bbox']
            x_center = (bx + bw / 2) / img_w
            y_center = (by + bh / 2) / img_h
            w_norm = bw / img_w
            h_norm = bh / img_h

            # Clamp to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            if w_norm > 0.01 and h_norm > 0.01:  # Skip tiny annotations
                lines.append(f"{a['class_id']} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if lines:
            with open(label_dir / filename, 'w') as f:
                f.write('\n'.join(lines))
            count += 1

    print(f"  ✓ Created {count} label files")
    return label_dir


def create_dataset_yaml(img_dir):
    """Create data.yaml for YOLOv8 training"""
    # Split into train/val (80/20)
    all_images = sorted(list((DATASET_DIR / "images" / img_dir.name).glob("*.jpg")))
    random.shuffle(all_images)

    split_idx = int(len(all_images) * 0.8)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    # Create train/val split directories
    for split in ['train', 'val']:
        os.makedirs(DATASET_DIR / "images" / split, exist_ok=True)
        os.makedirs(DATASET_DIR / "labels" / split, exist_ok=True)

    # Move/copy files to train/val splits
    label_src = DATASET_DIR / "labels" / img_dir.name

    for img_path in train_imgs:
        shutil.copy2(img_path, DATASET_DIR / "images" / "train" / img_path.name)
        label_file = label_src / (img_path.stem + '.txt')
        if label_file.exists():
            shutil.copy2(label_file, DATASET_DIR / "labels" / "train" / label_file.name)

    for img_path in val_imgs:
        shutil.copy2(img_path, DATASET_DIR / "images" / "val" / img_path.name)
        label_file = label_src / (img_path.stem + '.txt')
        if label_file.exists():
            shutil.copy2(label_file, DATASET_DIR / "labels" / "val" / label_file.name)

    print(f"\n📁 Dataset split: {len(train_imgs)} train / {len(val_imgs)} val")

    # Create data.yaml
    yaml_content = f"""# ANI Flow Optimizer — Desk Distraction Dataset
# Source: COCO 2017 subset (desk-relevant classes)
# Generated by download_coco_desk_subset.py

path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val

nc: {len(OUR_CLASS_NAMES)}
names: {OUR_CLASS_NAMES}

# Class mapping from COCO:
#   phone (COCO: cell_phone 77) → distraction indicator
#   monitor (COCO: tv 72, laptop 73) → workspace presence
#   work_tool (COCO: keyboard 76, mouse 74) → active work indicator
#   distraction (COCO: remote 75, book 84, cup 47, bottle 44) → distraction count
"""

    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"  ✓ Created {yaml_path}")
    return yaml_path


def print_dataset_summary():
    """Print final dataset summary"""
    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY")
    print("=" * 60)
    print(f"  Location: {DATASET_DIR}")
    print(f"  Classes: {OUR_CLASS_NAMES}")

    for split in ['train', 'val']:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        n_imgs = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {split}: {n_imgs} images, {n_lbls} labels")

    yaml_path = DATASET_DIR / "data.yaml"
    print(f"\n  data.yaml: {yaml_path}")
    print(f"\n  ✅ Ready for training! Run:")
    print(f"     python training/train_vision_yolov8.py --data {yaml_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Download COCO subset for desk detection")
    parser.add_argument("--max-images", type=int, default=300,
                        help="Max images to download (default: 300)")
    parser.add_argument("--use-train", action="store_true",
                        help="Download from train2017 instead of val2017 (larger, slower)")
    args = parser.parse_args()

    print("=" * 60)
    print("📦 COCO Desk Subset Downloader")
    print("   Downloading desk-relevant images for distraction detection")
    print("=" * 60)

    # Step 1: Download annotations
    anno_path = download_annotations()

    # Step 2: Parse and filter
    split = "train2017" if args.use_train else "val2017"
    anno_file = DATASET_DIR / "annotations" / f"instances_{split}.json"
    if not anno_file.exists():
        anno_file = DATASET_DIR / "annotations" / "instances_val2017.json"
        split = "val2017"

    selected_ids, relevant_annos, images_by_id = parse_annotations(
        anno_file, max_images=args.max_images
    )

    # Step 3: Download images
    img_dir = download_images(selected_ids, images_by_id, split)

    # Step 4: Create YOLO labels
    create_yolo_labels(selected_ids, relevant_annos, images_by_id, img_dir)

    # Step 5: Split and create data.yaml
    create_dataset_yaml(img_dir)

    # Summary
    print_dataset_summary()


if __name__ == "__main__":
    main()
