"""
Train YOLOv8-nano for desk distraction detection.

Two modes:
  1. Fine-tune on COCO desk subset (better accuracy, needs dataset)
  2. Export pretrained COCO model directly (fastest, good enough for demo)

Classes: phone, monitor, work_tool, distraction
Output: models/desk_distraction_v1.onnx

Usage:
    # Option 1: Fine-tune on downloaded COCO subset
    python training/train_vision_yolov8.py --data data/coco_desk_subset/data.yaml --epochs 50

    # Option 2: Just export pretrained model (no training needed)
    python training/train_vision_yolov8.py --export-pretrained

    # Option 3: Quick test with COCO8 mini-dataset
    python training/train_vision_yolov8.py --data coco8.yaml --epochs 10
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def export_pretrained():
    """
    Export pretrained YOLOv8n (COCO 80 classes) directly to ONNX.
    This already detects cell_phone, laptop, tv, keyboard, mouse, etc.
    The frontend maps COCO classes to our 4 desk categories.
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("🚀 Exporting pretrained YOLOv8n (COCO 80 classes) to ONNX")
    print("=" * 60)

    model = YOLO("yolov8n.pt")
    print("✅ Loaded YOLOv8n pretrained on COCO (80 classes)")
    print("   Includes: cell_phone, laptop, tv, keyboard, mouse, remote, book, cup, bottle")

    # Export to ONNX
    os.makedirs(MODELS_DIR, exist_ok=True)
    onnx_path = model.export(
        format="onnx",
        opset=12,
        simplify=True,
        imgsz=640,
    )

    if onnx_path and os.path.exists(onnx_path):
        dest = MODELS_DIR / "desk_distraction_v1.onnx"
        shutil.copy2(onnx_path, str(dest))
        size_mb = os.path.getsize(str(dest)) / 1024 / 1024
        print(f"\n✅ ONNX model exported: {dest}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Input:  [1, 3, 640, 640] float32")
        print(f"   Output: YOLO detection format")

        # Write a class mapping file for the frontend
        mapping = {
            "mode": "pretrained_coco",
            "onnx_model": "desk_distraction_v1.onnx",
            "coco_to_desk": {
                "67": {"class_id": 0, "name": "phone", "type": "distraction"},
                "63": {"class_id": 1, "name": "laptop", "type": "workspace"},
                "62": {"class_id": 1, "name": "monitor", "type": "workspace"},
                "66": {"class_id": 2, "name": "keyboard", "type": "work_tool"},
                "64": {"class_id": 2, "name": "mouse", "type": "work_tool"},
                "65": {"class_id": 3, "name": "remote", "type": "distraction"},
                "73": {"class_id": 3, "name": "book", "type": "neutral"},
                "41": {"class_id": 3, "name": "cup", "type": "neutral"},
                "39": {"class_id": 3, "name": "bottle", "type": "neutral"},
            },
            "desk_classes": ["phone", "monitor", "work_tool", "distraction"],
            "note": "COCO class IDs (0-indexed): cell_phone=67, laptop=63, tv=62, keyboard=66, mouse=64"
        }
        mapping_path = MODELS_DIR / "vision_class_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"   Class mapping: {mapping_path}")
    else:
        print("❌ ONNX export failed!")
        sys.exit(1)

    return str(dest)


def train_finetune(data_yaml, epochs, batch, imgsz):
    """Fine-tune YOLOv8n on the COCO desk subset"""
    from ultralytics import YOLO

    print("=" * 60)
    print("🖼️  YOLOv8-nano Desk Distraction Detector — Fine-tuning")
    print("=" * 60)

    if not os.path.exists(data_yaml):
        print(f"❌ Dataset not found: {data_yaml}")
        print("   Run first: python training/download_coco_desk_subset.py")
        sys.exit(1)

    # Load pretrained YOLOv8-nano (COCO weights)
    model = YOLO("yolov8n.pt")
    print(f"✅ Loaded YOLOv8n pretrained model")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epochs: {epochs}, Batch: {batch}, ImgSize: {imgsz}")

    # Fine-tune on desk distraction dataset
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="desk_distraction_v1",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        patience=15,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        project=str(MODELS_DIR / "vision_runs"),
    )

    print("\n📊 Training Results:")
    print(f"   mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    # Export best model to ONNX
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load best weights
    best_path = MODELS_DIR / "vision_runs" / "desk_distraction_v1" / "weights" / "best.pt"
    if best_path.exists():
        best_model = YOLO(str(best_path))
    else:
        best_model = model

    onnx_path = best_model.export(format="onnx", opset=12, simplify=True, imgsz=imgsz)
    print(f"\n✅ ONNX model exported: {onnx_path}")

    # Copy to models directory
    dest = MODELS_DIR / "desk_distraction_v1.onnx"
    if onnx_path and os.path.exists(onnx_path):
        shutil.copy2(onnx_path, str(dest))
        print(f"   Copied to: {dest}")

    # Write class mapping
    mapping = {
        "mode": "finetuned",
        "onnx_model": "desk_distraction_v1.onnx",
        "desk_classes": ["phone", "monitor", "work_tool", "distraction"],
        "note": "Fine-tuned on COCO desk subset. 4 custom classes."
    }
    mapping_path = MODELS_DIR / "vision_class_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"   Class mapping: {mapping_path}")

    # Validation
    print("\n📋 Running validation...")
    val_results = model.val(data=data_yaml)
    print(f"   Validation mAP@0.5: {val_results.results_dict.get('metrics/mAP50(B)', 'N/A')}")

    print("\n✅ Vision model training complete!")
    return str(dest)


def main():
    parser = argparse.ArgumentParser(description="Train/Export YOLOv8 vision model")
    parser.add_argument("--data", default=None, help="Path to data.yaml for fine-tuning")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--export-pretrained", action="store_true",
                        help="Skip training, just export pretrained COCO model to ONNX")
    args = parser.parse_args()

    if args.export_pretrained:
        export_pretrained()
    else:
        if args.data is None:
            # Default dataset path
            args.data = str(PROJECT_ROOT / "data" / "coco_desk_subset" / "data.yaml")

        train_finetune(args.data, args.epochs, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
