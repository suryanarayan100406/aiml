"""
Train YOLOv8-nano on desktop distraction detection.
Fine-tunes pretrained COCO weights on custom screenshot dataset.

Classes: tab_bar, phone, distraction, work_tool
Output: models/desk_distraction_v1.onnx
"""
import os, sys, argparse

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 vision model")
    parser.add_argument("--data", default=None, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--export-onnx", action="store_true", default=True)
    args = parser.parse_args()

    # Default data path
    if args.data is None:
        args.data = os.path.join(os.path.dirname(__file__), "..",
                                 "data", "processed", "vision_dataset", "data.yaml")

    if not os.path.exists(args.data):
        print(f"❌ Dataset not found: {args.data}")
        print("   Run: python data/scripts/generate_synthetic_vision.py")
        sys.exit(1)

    print("=" * 60)
    print("🖼️  YOLOv8-nano Desktop Distraction Detector Training")
    print("=" * 60)

    from ultralytics import YOLO

    # Load pretrained YOLOv8-nano (trained on COCO)
    model = YOLO('yolov8n.pt')
    print(f"✅ Loaded YOLOv8n pretrained model")
    print(f"   Dataset: {args.data}")
    print(f"   Epochs: {args.epochs}, Batch: {args.batch}, ImgSize: {args.imgsz}")

    # Fine-tune on custom desktop distraction dataset
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name='desk_distraction_v1',
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        augment=True,
        patience=15,       # Early stopping
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        project=os.path.join(os.path.dirname(__file__), "..", "models", "vision_runs"),
    )

    print("\n📊 Training Results:")
    print(f"   mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    # Export to ONNX
    if args.export_onnx:
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)

        best = model  # best weights are loaded automatically
        onnx_path = best.export(format='onnx', opset=12, simplify=True)
        print(f"\n✅ ONNX model exported: {onnx_path}")

        # Copy to models directory
        import shutil
        dest = os.path.join(model_dir, "desk_distraction_v1.onnx")
        if os.path.exists(onnx_path):
            shutil.copy2(onnx_path, dest)
            print(f"   Copied to: {dest}")

    # Validation
    print("\n📋 Running validation...")
    val_results = model.val(data=args.data)
    print(f"   Validation mAP@0.5: {val_results.results_dict.get('metrics/mAP50(B)', 'N/A')}")

    print("\n✅ Vision model training complete!")

if __name__ == "__main__":
    main()
