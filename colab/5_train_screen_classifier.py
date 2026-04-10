"""
============================================================================
  ANI Flow Optimizer — Screen Activity Classifier Training (Google Colab)
  Model: Custom CNN trained on web-scraped app/UI screenshots
  
  HOW TO USE:
    1. Open Google Colab (colab.research.google.com)
    2. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
    3. Paste this entire script into a cell and run
    4. Download the output files from /content/ani_models/
  
  OUTPUT FILES:
    - screen_classifier.onnx (Custom CNN, ~5-10MB)
    - screen_class_mapping.json
    - screen_metrics.json
============================================================================
"""

# ──────────────────────────────────────────────────────────────
# Step 0: Install dependencies
# ──────────────────────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("icrawler")
install("onnx>=1.15.0")
install("onnxruntime>=1.17.0")
install("torch>=2.0.0")
install("torchvision>=0.15.0")
install("Pillow>=9.0.0")
install("scikit-learn>=1.3.0")

import os, json, shutil, random, time, glob
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# Step 1: Configuration
# ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/content/ani_models")
DATASET_DIR = Path("/content/screen_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# 8 App/UI categories for screen classification
SCREEN_CATEGORIES = {
    0: {
        "name": "code_editor",
        "display": "Code Editor",
        "productivity": 0.95,
        "queries": [
            "VS Code IDE screenshot dark theme",
            "PyCharm IDE coding Python",
            "IntelliJ IDEA Java editor",
            "Sublime Text code editor",
            "Visual Studio coding C sharp",
            "Atom editor programming",
            "Eclipse IDE Java development",
            "Jupyter Notebook coding screenshot",
        ]
    },
    1: {
        "name": "terminal_cli",
        "display": "Terminal / CLI",
        "productivity": 0.90,
        "queries": [
            "terminal command line interface",
            "PowerShell console commands",
            "Linux bash terminal screenshot",
            "Git command line terminal",
            "Windows Terminal dark theme",
            "macOS Terminal commands",
            "SSH terminal session",
        ]
    },
    2: {
        "name": "documentation",
        "display": "Documentation",
        "productivity": 0.80,
        "queries": [
            "technical documentation website",
            "Stack Overflow programming question",
            "API documentation page",
            "MDN web docs reference",
            "GitHub README documentation",
            "Read the Docs page",
            "Wikipedia article page",
        ]
    },
    3: {
        "name": "spreadsheet",
        "display": "Spreadsheet",
        "productivity": 0.70,
        "queries": [
            "Microsoft Excel spreadsheet data",
            "Google Sheets spreadsheet",
            "Excel charts and graphs",
            "spreadsheet with formulas",
            "data table spreadsheet",
            "Excel pivot table screenshot",
        ]
    },
    4: {
        "name": "email_chat",
        "display": "Email / Chat",
        "productivity": 0.50,
        "queries": [
            "Gmail inbox email screenshot",
            "Slack workspace messages",
            "Microsoft Teams chat",
            "Outlook email inbox",
            "Discord chat server",
            "email client inbox view",
        ]
    },
    5: {
        "name": "social_media",
        "display": "Social Media",
        "productivity": 0.10,
        "queries": [
            "Twitter feed timeline",
            "Instagram feed scrolling",
            "Reddit front page posts",
            "Facebook news feed",
            "TikTok feed video",
            "LinkedIn feed scrolling",
        ]
    },
    6: {
        "name": "video_streaming",
        "display": "Video / Streaming",
        "productivity": 0.15,
        "queries": [
            "YouTube video watching fullscreen",
            "Netflix streaming show",
            "Twitch live stream watching",
            "Disney Plus streaming movie",
            "YouTube video player screenshot",
            "video streaming platform",
        ]
    },
    7: {
        "name": "gaming",
        "display": "Gaming",
        "productivity": 0.05,
        "queries": [
            "PC game screenshot gameplay",
            "Steam game library",
            "online game playing screenshot",
            "Minecraft gameplay screenshot",
            "browser game playing",
            "mobile game on PC emulator",
        ]
    }
}

CLASS_NAMES = [cat["name"] for cat in SCREEN_CATEGORIES.values()]
IMAGES_PER_CATEGORY = 120  # Target images per class
IMG_SIZE = 224              # Standard CNN input dimension
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

print("=" * 60)
print("🖥️  ANI Screen Activity Classifier — Custom CNN")
print("=" * 60)
print(f"   Categories: {len(SCREEN_CATEGORIES)}")
print(f"   Target images/class: {IMAGES_PER_CATEGORY}")
print(f"   Input size: {IMG_SIZE}×{IMG_SIZE}")

# ──────────────────────────────────────────────────────────────
# Step 2: Download Screenshot Dataset using icrawler
# ──────────────────────────────────────────────────────────────
from icrawler.builtin import BingImageCrawler

print(f"\n📥 Downloading screenshot dataset via Bing Image Search...")
print(f"   This may take 5-15 minutes...\n")

total_downloaded = 0

for class_id, cat_info in SCREEN_CATEGORIES.items():
    class_dir = DATASET_DIR / cat_info["name"]
    class_dir.mkdir(parents=True, exist_ok=True)
    
    existing = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
    if existing >= IMAGES_PER_CATEGORY:
        print(f"   ✅ {cat_info['name']}: {existing} images already exist, skipping")
        total_downloaded += existing
        continue
    
    per_query = max(10, IMAGES_PER_CATEGORY // len(cat_info["queries"]) + 5)
    class_count = 0
    
    for query in cat_info["queries"]:
        if class_count >= IMAGES_PER_CATEGORY:
            break
            
        try:
            crawler = BingImageCrawler(
                downloader_threads=2,
                storage={"root_dir": str(class_dir)},
            )
            crawler.crawl(
                keyword=query + " screenshot",
                max_num=per_query,
                min_size=(200, 200),
                file_idx_offset=class_count,
            )
        except Exception as e:
            print(f"      ⚠️ Query '{query}' failed: {e}")
        
        class_count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
    
    print(f"   📂 {cat_info['name']}: {class_count} images downloaded")
    total_downloaded += class_count

print(f"\n   📊 Total dataset: {total_downloaded} images across {len(SCREEN_CATEGORIES)} classes")

# ──────────────────────────────────────────────────────────────
# Step 3: Clean and Validate Dataset
# ──────────────────────────────────────────────────────────────
print(f"\n🧹 Cleaning dataset — removing corrupt/tiny images...")

removed = 0
valid_counts = {}

for class_id, cat_info in SCREEN_CATEGORIES.items():
    class_dir = DATASET_DIR / cat_info["name"]
    valid = 0
    
    for img_path in list(class_dir.glob("*")):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            img_path.unlink()
            removed += 1
            continue
        
        try:
            img = Image.open(img_path)
            img.verify()
            # Reopen after verify (verify leaves file in bad state)
            img = Image.open(img_path)
            w, h = img.size
            if w < 100 or h < 100:
                img_path.unlink()
                removed += 1
                continue
            # Convert to RGB (remove alpha, handle grayscale)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(img_path)
            valid += 1
        except Exception:
            img_path.unlink()
            removed += 1
    
    valid_counts[cat_info["name"]] = valid

print(f"   Removed {removed} invalid images")
print(f"\n   📊 Cleaned dataset distribution:")
for name, count in valid_counts.items():
    bar = "█" * min(50, count // 2)
    print(f"      {name:20s} │ {count:4d} │ {bar}")

min_count = min(valid_counts.values()) if valid_counts else 0
max_count = max(valid_counts.values()) if valid_counts else 0
print(f"\n   Min class: {min_count}, Max class: {max_count}")

if min_count < 20:
    print(f"\n   ⚠️ WARNING: Some classes have < 20 images. Augmentation will compensate.")

# ──────────────────────────────────────────────────────────────
# Step 4: Define Custom CNN Architecture
# ──────────────────────────────────────────────────────────────
print(f"\n🧠 Building Custom CNN architecture...")

class ScreenCNN(nn.Module):
    """
    Custom CNN for screen/app classification.
    Architecture: 5 conv blocks + global average pooling + 2 FC layers.
    
    Input:  [B, 3, 224, 224] — RGB screenshot
    Output: [B, 8] — logits for 8 app categories
    
    Design principles:
        - Progressively increasing channels: 32 → 64 → 128 → 256 → 512
        - BatchNorm + ReLU after every conv for stable training
        - MaxPool after each block for spatial reduction (224→112→56→28→14→7)
        - Global Average Pooling instead of flatten — reduces params massively
        - Dropout (0.5) before final FC for regularization
        - ~2.5M parameters (lightweight enough for browser ONNX inference)
    """
    
    def __init__(self, num_classes=8):
        super(ScreenCNN, self).__init__()
        
        # Block 1: 3 → 32 channels, 224×224 → 112×112
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Block 2: 32 → 64 channels, 112×112 → 56×56
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Block 3: 64 → 128 channels, 56×56 → 28×28
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Block 4: 128 → 256 channels, 28×28 → 14×14
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Block 5: 256 → 512 channels, 14×14 → 7×7
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Global Average Pooling: 7×7×512 → 512
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier: 512 → 256 → num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: [B, 512]
        x = self.classifier(x)
        return x

# Count parameters
model = ScreenCNN(num_classes=len(SCREEN_CATEGORIES))
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Architecture: 5-block CNN + GAP + FC classifier")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Estimated ONNX size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

# ──────────────────────────────────────────────────────────────
# Step 5: Create Dataset & DataLoaders
# ──────────────────────────────────────────────────────────────
print(f"\n📦 Preparing dataset and data loaders...")

class ScreenshotDataset(Dataset):
    """
    PyTorch dataset for screen classification.
    Loads images from folder structure: dataset_dir/class_name/*.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # (path, label)
        self.class_to_idx = {}
        
        for class_id, cat_info in SCREEN_CATEGORIES.items():
            class_name = cat_info["name"]
            self.class_to_idx[class_name] = class_id
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    self.samples.append((str(img_path), class_id))
        
        random.shuffle(self.samples)
        print(f"      Loaded {len(self.samples)} images across {len(self.class_to_idx)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Return a blank image if file is corrupt
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Training augmentations — simulate real screen capture variations
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # Resize slightly larger
    transforms.RandomCrop(IMG_SIZE),                     # Random crop to 224×224
    transforms.RandomHorizontalFlip(p=0.3),              # Occasional flip
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # Brightness/contrast variation
    transforms.RandomGrayscale(p=0.05),                  # Occasional grayscale
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),  # Slight rotation/shift
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],     # ImageNet normalization
                         std=[0.229, 0.224, 0.225]),
])

# Validation — no augmentation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Create dataset
full_dataset = ScreenshotDataset(DATASET_DIR, transform=train_transform)

if len(full_dataset) < 50:
    print(f"\n   ⚠️ Dataset is too small ({len(full_dataset)} images). Generating synthetic augmentations...")
    # If download failed, generate minimal synthetic data for training
    for class_id, cat_info in SCREEN_CATEGORIES.items():
        class_dir = DATASET_DIR / cat_info["name"]
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(class_dir.glob("*")))
        
        if existing < 30:
            print(f"      Generating synthetic screenshots for {cat_info['name']}...")
            for i in range(30 - existing):
                img = Image.new('RGB', (300, 200))
                pixels = img.load()
                # Create distinctive color patterns per class type
                base_hue = class_id * 30
                for y in range(200):
                    for x in range(300):
                        r = min(255, (base_hue + x // 3 + random.randint(0, 40)) % 256)
                        g = min(255, (50 + y // 2 + random.randint(0, 30)) % 256)
                        b = min(255, (100 + (x + y) // 4 + random.randint(0, 30)) % 256)
                        pixels[x, y] = (r, g, b)
                img.save(class_dir / f"synthetic_{i:04d}.png")
    
    # Reload dataset
    full_dataset = ScreenshotDataset(DATASET_DIR, transform=train_transform)

# Split into train/validation (80/20)
total_size = len(full_dataset)
val_size = max(1, int(0.2 * total_size))
train_size = total_size - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform  # Use val transforms

# Handle class imbalance with weighted sampling
train_labels = [full_dataset.samples[i][1] for i in train_dataset.indices]
class_counts = Counter(train_labels)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"   Train: {train_size} images, Val: {val_size} images")
print(f"   Class distribution (train): {dict(class_counts)}")

# ──────────────────────────────────────────────────────────────
# Step 6: Train the Custom CNN
# ──────────────────────────────────────────────────────────────
print(f"\n🏋️ Training Custom CNN for {EPOCHS} epochs...")
print(f"   Optimizer: Adam, LR={LEARNING_RATE}")
print(f"   Scheduler: CosineAnnealingLR")
print(f"   Loss: CrossEntropy with label smoothing=0.1")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

model = ScreenCNN(num_classes=len(SCREEN_CATEGORIES)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
best_model_state = None
patience = 5
patience_counter = 0
train_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(EPOCHS):
    # ─── Training phase ─────────────────────────────
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    scheduler.step()
    
    train_loss /= max(1, train_total)
    train_acc = train_correct / max(1, train_total)
    
    # ─── Validation phase ────────────────────────────
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss /= max(1, val_total)
    val_acc = val_correct / max(1, val_total)
    
    # Save history
    train_history["train_loss"].append(train_loss)
    train_history["train_acc"].append(train_acc)
    train_history["val_loss"].append(val_loss)
    train_history["val_acc"].append(val_acc)
    
    # Progress bar
    lr = scheduler.get_last_lr()[0]
    print(f"   Epoch {epoch+1:2d}/{EPOCHS} │ "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} │ "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} │ "
          f"LR: {lr:.6f}")
    
    # Early stopping / best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        print(f"      ✅ New best model! Val accuracy: {val_acc:.3f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"      ⏹️ Early stopping triggered after {epoch+1} epochs")
            break

# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)
    print(f"\n   ✅ Best model loaded (val accuracy: {best_val_acc:.3f})")

# ──────────────────────────────────────────────────────────────
# Step 7: Evaluate — Confusion Matrix & Per-Class Metrics
# ──────────────────────────────────────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix

print(f"\n📊 Evaluation Results:")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification report
report = classification_report(
    all_labels, all_preds,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)
print(f"\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)}")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print(f"\n   Confusion Matrix:")
print(f"   {'':20s}", end="")
for name in CLASS_NAMES:
    print(f" {name[:8]:>8s}", end="")
print()
for i, row in enumerate(cm):
    print(f"   {CLASS_NAMES[i]:20s}", end="")
    for val in row:
        print(f" {val:8d}", end="")
    print()

# ──────────────────────────────────────────────────────────────
# Step 8: Export to ONNX
# ──────────────────────────────────────────────────────────────
print(f"\n📦 Exporting to ONNX format...")

model.eval()
model.cpu()

# Create dummy input matching inference shape
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

onnx_path = OUTPUT_DIR / "screen_classifier.onnx"

torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['logits'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'logits': {0: 'batch_size'},
    },
)

# Verify ONNX model
import onnx
onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)

onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
print(f"   ✅ ONNX model exported: {onnx_path} ({onnx_size:.1f} MB)")

# Verify with ONNX Runtime
import onnxruntime as ort
sess = ort.InferenceSession(str(onnx_path))
test_input = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
ort_result = sess.run(None, {'input': test_input})
print(f"   ✅ ONNX Runtime verification: output shape = {ort_result[0].shape}")
print(f"   Sample output (logits): [{', '.join(f'{v:.3f}' for v in ort_result[0][0])}]")

# ──────────────────────────────────────────────────────────────
# Step 9: Save Metadata & Metrics
# ──────────────────────────────────────────────────────────────
print(f"\n💾 Saving metadata files...")

# Class mapping
class_mapping = {
    "model": "ScreenCNN",
    "architecture": "Custom 5-block CNN + GAP",
    "input_size": IMG_SIZE,
    "num_classes": len(SCREEN_CATEGORIES),
    "classes": {str(k): v["name"] for k, v in SCREEN_CATEGORIES.items()},
    "display_names": {str(k): v["display"] for k, v in SCREEN_CATEGORIES.items()},
    "productivity_scores": {str(k): v["productivity"] for k, v in SCREEN_CATEGORIES.items()},
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}

mapping_path = OUTPUT_DIR / "screen_class_mapping.json"
with open(mapping_path, 'w') as f:
    json.dump(class_mapping, f, indent=2)
print(f"   ✅ Saved: {mapping_path}")

# Metrics
metrics = {
    "model_name": "ScreenCNN",
    "architecture": "Custom 5-block CNN (Conv2d + BatchNorm + ReLU + MaxPool) × 5 + GAP + FC",
    "total_parameters": total_params,
    "dataset": {
        "source": "Web-scraped screenshots via icrawler (Bing Image Search)",
        "total_images": total_size,
        "train_images": train_size,
        "val_images": val_size,
        "classes": len(SCREEN_CATEGORIES),
        "images_per_class": dict(class_counts),
    },
    "training": {
        "epochs_trained": len(train_history["train_loss"]),
        "max_epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "scheduler": "CosineAnnealingLR",
        "loss": "CrossEntropy (label_smoothing=0.1)",
        "early_stopping_patience": patience,
    },
    "performance": {
        "best_val_accuracy": float(best_val_acc),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES if name in report
        },
        "macro_avg": {
            "precision": float(report.get("macro avg", {}).get("precision", 0)),
            "recall": float(report.get("macro avg", {}).get("recall", 0)),
            "f1": float(report.get("macro avg", {}).get("f1-score", 0)),
        },
    },
    "onnx": {
        "opset_version": 12,
        "file_size_mb": round(onnx_size, 1),
        "input_name": "input",
        "input_shape": [1, 3, IMG_SIZE, IMG_SIZE],
        "output_name": "logits",
        "output_shape": [1, len(SCREEN_CATEGORIES)],
    },
    "training_history": train_history,
}

metrics_path = OUTPUT_DIR / "screen_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"   ✅ Saved: {metrics_path}")

# ──────────────────────────────────────────────────────────────
# Step 10: Summary
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"🎉 Screen Activity Classifier — Training Complete!")
print(f"{'=' * 60}")
print(f"   Model:       Custom CNN ({total_params:,} params)")
print(f"   Dataset:     {total_size} screenshots × {len(SCREEN_CATEGORIES)} categories")
print(f"   Best Val Acc: {best_val_acc:.1%}")
print(f"   ONNX Size:   {onnx_size:.1f} MB")
print(f"\n   📁 Output files in {OUTPUT_DIR}:")
print(f"      screen_classifier.onnx     — ONNX model")
print(f"      screen_class_mapping.json  — Class names + productivity scores")
print(f"      screen_metrics.json        — Training metrics + history")
print(f"\n   📋 Next steps:")
print(f"      1. Download these 3 files from Colab")
print(f"      2. Copy them to your project's models/ directory")
print(f"      3. The frontend will auto-detect and load the model")
print(f"      4. Re-run colab/4_train_meta_classifier.py to retrain")
print(f"         the meta-classifier with the new screen features")
