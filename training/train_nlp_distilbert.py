"""
Fine-tune DistilBERT for task type classification.

Classes: DEEP_WORK(0), SHALLOW_WORK(1), CREATIVE(2), ADMINISTRATIVE(3), COMMUNICATION(4)
Output: models/task_nlp_classifier.onnx
"""
import os, sys, argparse, json
import numpy as np
import torch
from torch import nn

CLASS_NAMES = ["DEEP_WORK", "SHALLOW_WORK", "CREATIVE", "ADMINISTRATIVE", "COMMUNICATION"]

def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT NLP model")
    parser.add_argument("--data", default=None, help="Path to labeled_tasks.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--export-onnx", action="store_true", default=True)
    args = parser.parse_args()

    proc_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    if args.data is None:
        args.data = os.path.join(proc_dir, "labeled_tasks.csv")

    if not os.path.exists(args.data):
        print(f"❌ Not found: {args.data}")
        print("   Run: python data/scripts/generate_synthetic_tasks.py")
        sys.exit(1)

    print("=" * 60)
    print("📝 DistilBERT Task Classifier Training")
    print("=" * 60)

    from transformers import (
        DistilBertTokenizer, DistilBertForSequenceClassification,
        TrainingArguments, Trainer
    )
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score

    # Load dataset
    dataset = Dataset.from_csv(args.data)
    print(f"✅ Loaded {len(dataset)} task descriptions")

    # Tokenize
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_fn(batch):
        return tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=args.max_length
        )

    dataset = dataset.map(tokenize_fn, batched=True, batch_size=64)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Split
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = split['train']
    eval_ds = split['test']
    print(f"   Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=5
    )

    # Freeze all layers except classifier and last transformer block
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'transformer.layer.5' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='macro'),
        }

    # Training
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    output_dir = os.path.join(model_dir, "nlp_training_output")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=args.lr,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    print("\n🚀 Starting training...")
    train_result = trainer.train()
    print(f"\n   Training loss: {train_result.training_loss:.4f}")

    # Evaluate
    print("\n📊 Evaluation Results:")
    eval_result = trainer.evaluate()
    print(f"   Accuracy: {eval_result.get('eval_accuracy', 'N/A')}")
    print(f"   F1 (macro): {eval_result.get('eval_f1', 'N/A')}")

    # Confusion matrix
    preds = trainer.predict(eval_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids

    from sklearn.metrics import classification_report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Save model
    model.save_pretrained(os.path.join(model_dir, "task_nlp_model"))
    tokenizer.save_pretrained(os.path.join(model_dir, "task_nlp_model"))

    # ONNX export
    if args.export_onnx:
        print("\n📦 Exporting to ONNX...")
        model.eval()
        dummy_input_ids = torch.zeros(1, args.max_length, dtype=torch.long)
        dummy_attention_mask = torch.ones(1, args.max_length, dtype=torch.long)

        onnx_path = os.path.join(model_dir, "task_nlp_classifier.onnx")
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            opset_version=14,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch'},
                'attention_mask': {0: 'batch'}
            },
        )
        print(f"   ✅ ONNX exported: {onnx_path}")

    # Save metrics
    metrics = {
        "accuracy": float(eval_result.get('eval_accuracy', 0)),
        "f1_macro": float(eval_result.get('eval_f1', 0)),
        "class_names": CLASS_NAMES,
        "epochs": args.epochs,
    }
    with open(os.path.join(model_dir, "nlp_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save vocab for browser tokenizer
    vocab_path = os.path.join(model_dir, "task_nlp_model", "vocab.txt")
    if os.path.exists(vocab_path):
        import shutil
        shutil.copy2(vocab_path, os.path.join(model_dir, "vocab.txt"))

    print("\n✅ NLP model training complete!")

if __name__ == "__main__":
    main()
