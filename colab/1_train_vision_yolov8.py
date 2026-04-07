"""
============================================================================
  ANI Flow Optimizer — Vision Model Training (Google Colab)
  Model: YOLOv8-nano fine-tuned on COCO desk-distraction subset
  
  HOW TO USE:
    1. Open Google Colab (colab.research.google.com)
    2. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
    3. Paste this entire script into a cell and run
    4. Download the output files from /content/ani_models/
  
  OUTPUT FILES:
    - desk_distraction_v1.onnx (YOLOv8n fine-tuned, ~12MB)
    - vision_class_mapping.json
    - vision_metrics.json
============================================================================
"""

# ──────────────────────────────────────────────────────────────
# Step 0: Install dependencies
# ──────────────────────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("ultralytics>=8.0.0")
install("onnx>=1.15.0")
install("onnxruntime>=1.17.0")

import os, json, shutil, random, time
import urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np

# ──────────────────────────────────────────────────────────────
# Step 1: Configuration
# ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/content/ani_models")
DATASET_DIR = Path("/content/coco_desk_subset")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_IMAGES = 400  # Number of COCO images to download
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640

# COCO category IDs → Our 4 desk classes
COCO_DESK_CATEGORIES = {
    77: (0, "phone"),       # cell phone → primary distraction
    73: (1, "monitor"),     # laptop → workspace
    72: (1, "monitor"),     # tv/monitor → workspace
    76: (2, "work_tool"),   # keyboard → work tool
    74: (2, "work_tool"),   # mouse → work tool
    75: (3, "distraction"), # remote → distraction
    84: (3, "distraction"), # book → distraction
    47: (3, "distraction"), # cup → neutral/distraction
    44: (3, "distraction"), # bottle → neutral/distraction
}
CLASS_NAMES = ["phone", "monitor", "work_tool", "distraction"]

print("=" * 60)
print("🖼️  ANI Vision Model — YOLOv8-nano Desk Distraction Detector")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# Step 2: Download COCO 2017 Annotations
# ──────────────────────────────────────────────────────────────
ANNO_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNO_ZIP = DATASET_DIR / "annotations_trainval2017.zip"
ANNO_FILE = DATASET_DIR / "annotations" / "instances_val2017.json"

DATASET_DIR.mkdir(parents=True, exist_ok=True)

if not ANNO_FILE.exists():
    print("\n📥 Downloading COCO 2017 annotations (~252MB)...")
    urllib.request.urlretrieve(ANNO_URL, str(ANNO_ZIP))
    print("   Extracting...")
    import zipfile
    with zipfile.ZipFile(str(ANNO_ZIP), 'r') as z:
        z.extractall(str(DATASET_DIR))
    print(f"   ✅ Annotations extracted to {ANNO_FILE}")
else:
    print(f"✅ Annotations already exist: {ANNO_FILE}")

# ──────────────────────────────────────────────────────────────
# Step 3: Parse & Filter for Desk-Relevant Images
# ──────────────────────────────────────────────────────────────
print(f"\n🔍 Parsing COCO annotations for desk-relevant objects...")

with open(ANNO_FILE, 'r') as f:
    coco = json.load(f)

images_by_id = {img['id']: img for img in coco['images']}

# Find all annotations with our categories
relevant_annos = defaultdict(list)
cat_counts = defaultdict(int)

for anno in coco['annotations']:
    cat_id = anno['category_id']
    if cat_id in COCO_DESK_CATEGORIES:
        img_id = anno['image_id']
        our_class_id, our_class_name = COCO_DESK_CATEGORIES[cat_id]
        relevant_annos[img_id].append({
            'bbox': anno['bbox'],  # [x, y, width, height]
            'class_id': our_class_id,
            'class_name': our_class_name,
            'area': anno['area'],
        })
        cat_counts[our_class_name] += 1

print(f"   Found {len(relevant_annos)} images with desk-relevant objects")
print(f"   Category distribution:")
for name, count in sorted(cat_counts.items()):
    print(f"     {name}: {count} annotations")

# Prioritize images with phones (key detection target)
phone_images = [iid for iid, annos in relevant_annos.items()
                if any(a['class_id'] == 0 for a in annos)]
other_images = [iid for iid in relevant_annos if iid not in phone_images]
random.seed(42)
random.shuffle(other_images)

selected = phone_images[:MAX_IMAGES // 2]
selected += other_images[:MAX_IMAGES - len(selected)]
print(f"   Selected {len(selected)} images (phone priority: {min(len(phone_images), MAX_IMAGES // 2)})")

# ──────────────────────────────────────────────────────────────
# Step 4: Download Selected Images
# ──────────────────────────────────────────────────────────────
COCO_IMG_BASE = "http://images.cocodataset.org"
SPLIT = "val2017"

img_dir = DATASET_DIR / "images" / SPLIT
img_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📥 Downloading {len(selected)} images...")
downloaded = 0
failed = 0

for i, img_id in enumerate(selected):
    img_info = images_by_id[img_id]
    filename = img_info['file_name']
    dest = img_dir / filename

    if dest.exists():
        downloaded += 1
        continue

    url = f"{COCO_IMG_BASE}/{SPLIT}/{filename}"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, str(dest))
            downloaded += 1
            break
        except Exception:
            if attempt == 2:
                failed += 1
            time.sleep(0.5)

    if (i + 1) % 50 == 0:
        print(f"   [{i+1}/{len(selected)}] Downloaded {downloaded}, Failed {failed}")

print(f"   ✅ Done: {downloaded} images, {failed} failed")

# ──────────────────────────────────────────────────────────────
# Step 5: Create YOLO Format Labels + Train/Val Split
# ──────────────────────────────────────────────────────────────
print(f"\n📝 Creating YOLO-format labels and train/val split...")

# Create split directories
for split in ['train', 'val']:
    (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# Create label files for each image
label_count = 0
all_image_paths = []

for img_id in selected:
    img_info = images_by_id[img_id]
    filename = img_info['file_name']
    img_w, img_h = img_info['width'], img_info['height']
    img_path = img_dir / filename

    if not img_path.exists():
        continue

    annos = relevant_annos[img_id]
    lines = []
    for a in annos:
        bx, by, bw, bh = a['bbox']
        x_center = max(0, min(1, (bx + bw / 2) / img_w))
        y_center = max(0, min(1, (by + bh / 2) / img_h))
        w_norm = max(0, min(1, bw / img_w))
        h_norm = max(0, min(1, bh / img_h))
        if w_norm > 0.01 and h_norm > 0.01:
            lines.append(f"{a['class_id']} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    if lines:
        all_image_paths.append((img_path, lines, filename))
        label_count += 1

# 80/20 split
random.shuffle(all_image_paths)
split_idx = int(len(all_image_paths) * 0.8)
train_set = all_image_paths[:split_idx]
val_set = all_image_paths[split_idx:]

for split_name, split_data in [("train", train_set), ("val", val_set)]:
    for img_path, label_lines, filename in split_data:
        shutil.copy2(str(img_path), str(DATASET_DIR / "images" / split_name / filename))
        label_file = DATASET_DIR / "labels" / split_name / (Path(filename).stem + ".txt")
        with open(label_file, 'w') as f:
            f.write('\n'.join(label_lines))

print(f"   ✅ {label_count} labeled images → {len(train_set)} train / {len(val_set)} val")

# Create data.yaml
data_yaml_content = f"""# ANI Desk Distraction Dataset (COCO subset)
path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""

data_yaml_path = DATASET_DIR / "data.yaml"
with open(data_yaml_path, 'w') as f:
    f.write(data_yaml_content)

print(f"   ✅ data.yaml created: {data_yaml_path}")

# ──────────────────────────────────────────────────────────────
# Step 6: Fine-tune YOLOv8-nano
# ──────────────────────────────────────────────────────────────
print(f"\n🚀 Starting YOLOv8-nano fine-tuning...")
print(f"   Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, ImgSize: {IMG_SIZE}")
print(f"   Dataset: {data_yaml_path}")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
print(f"   ✅ Loaded YOLOv8n pretrained on COCO")

results = model.train(
    data=str(data_yaml_path),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name="desk_distraction_v1",
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    augment=True,
    patience=15,
    save=True,
    save_period=10,
    plots=True,
    verbose=True,
    project="/content/yolo_runs",
    exist_ok=True,
)

# Print results
print("\n📊 Training Results:")
for key, val in results.results_dict.items():
    print(f"   {key}: {val}")

# ──────────────────────────────────────────────────────────────
# Step 7: Export Best Model to ONNX
# ──────────────────────────────────────────────────────────────
print(f"\n📦 Exporting best model to ONNX...")

best_path = Path("/content/yolo_runs/desk_distraction_v1/weights/best.pt")
if best_path.exists():
    best_model = YOLO(str(best_path))
    print(f"   Using best.pt weights")
else:
    best_model = model
    print(f"   Using last weights (best.pt not found)")

onnx_path = best_model.export(
    format="onnx",
    opset=12,
    simplify=True,
    imgsz=IMG_SIZE,
)

dest = OUTPUT_DIR / "desk_distraction_v1.onnx"
if onnx_path and os.path.exists(onnx_path):
    shutil.copy2(onnx_path, str(dest))
    size_mb = os.path.getsize(str(dest)) / 1024 / 1024
    print(f"   ✅ ONNX model: {dest} ({size_mb:.1f} MB)")
else:
    print(f"   ❌ ONNX export failed!")

# Save class mapping
mapping = {
    "mode": "finetuned_coco_subset",
    "onnx_model": "desk_distraction_v1.onnx",
    "num_classes": len(CLASS_NAMES),
    "class_names": CLASS_NAMES,
    "input_shape": [1, 3, IMG_SIZE, IMG_SIZE],
    "training": {
        "epochs": EPOCHS,
        "dataset_size": label_count,
        "train_size": len(train_set),
        "val_size": len(val_set),
    },
    "note": "Fine-tuned on COCO desk subset. 4 custom classes (phone, monitor, work_tool, distraction)."
}
mapping_path = OUTPUT_DIR / "vision_class_mapping.json"
with open(mapping_path, 'w') as f:
    json.dump(mapping, f, indent=2)

# ──────────────────────────────────────────────────────────────
# Step 8: Validation
# ──────────────────────────────────────────────────────────────
print(f"\n📋 Running validation on the fine-tuned model...")
val_results = best_model.val(data=str(data_yaml_path))

metrics = {
    "mAP50": float(val_results.results_dict.get("metrics/mAP50(B)", 0)),
    "mAP50_95": float(val_results.results_dict.get("metrics/mAP50-95(B)", 0)),
    "precision": float(val_results.results_dict.get("metrics/precision(B)", 0)),
    "recall": float(val_results.results_dict.get("metrics/recall(B)", 0)),
    "epochs": EPOCHS,
    "dataset_images": label_count,
    "class_names": CLASS_NAMES,
}

metrics_path = OUTPUT_DIR / "vision_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ VISION MODEL TRAINING COMPLETE!")
print(f"   mAP@0.5:     {metrics['mAP50']:.4f}")
print(f"   mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
print(f"   Precision:    {metrics['precision']:.4f}")
print(f"   Recall:       {metrics['recall']:.4f}")
print(f"\n   Output files in: {OUTPUT_DIR}")
print(f"   - desk_distraction_v1.onnx")
print(f"   - vision_class_mapping.json")
print(f"   - vision_metrics.json")

# ──────────────────────────────────────────────────────────────
# Step 9: Quick ONNX Inference Test
# ──────────────────────────────────────────────────────────────
print(f"\n🧪 Quick ONNX inference test...")
import onnxruntime as ort

session = ort.InferenceSession(str(dest))
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"   Input: {input_name} {input_shape}")

dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
output = session.run(None, {input_name: dummy})
print(f"   Output shapes: {[o.shape for o in output]}")
print(f"   ✅ ONNX inference works!")

print("\n" + "=" * 60)
print("🎉 Vision model ready! Download files from /content/ani_models/")
print("=" * 60)
