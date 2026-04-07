"""
============================================================================
  ANI Flow Optimizer — NLP Model Training (Google Colab)
  Model: DistilBERT fine-tuned for task-type classification
  
  HOW TO USE:
    1. Open Google Colab (colab.research.google.com)
    2. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
    3. Paste this entire script into a cell and run
    4. Download the output files from /content/ani_models/
  
  APPROACH:
    - Generates 2000 base task descriptions from templates (400 per class)
    - Augments 3x with word dropout, synonym replacement, typos → ~8000 samples
    - Fine-tunes DistilBERT with last transformer layer + classifier unfrozen
    - Exports to ONNX for browser inference
  
  OUTPUT FILES:
    - task_nlp_classifier.onnx (~260MB)
    - vocab.txt (DistilBERT WordPiece vocabulary)
    - nlp_metrics.json
============================================================================
"""

# ──────────────────────────────────────────────────────────────
# Step 0: Install dependencies
# ──────────────────────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("transformers>=4.38.0")
install("datasets>=2.18.0")
install("torch>=2.2.0")
install("scikit-learn>=1.4.0")
install("onnx>=1.15.0")
install("onnxruntime>=1.17.0")
install("onnxscript>=0.1.0")
install("nltk>=3.8.0")

import os, json, csv, random, shutil
import numpy as np
import torch
from pathlib import Path

OUTPUT_DIR = Path("/content/ani_models")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("📝 ANI NLP Model — DistilBERT Task Classifier")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# Step 1: Generate Base Task Descriptions
# ──────────────────────────────────────────────────────────────
print("\n📋 Generating task description training data...")

# ─── Templates ─────────────────────────────────────────────────
DEEP_WORK_TEMPLATES = [
    "Implement {algo} algorithm for the {component} module",
    "Debug the {issue} crash in the {component} service",
    "Refactor the {component} codebase to use {pattern} pattern",
    "Write unit tests for the {component} engine covering edge cases",
    "Optimize {component} query performance reducing latency by 50%",
    "Design database schema for {feature} with normalization",
    "Analyze {data_type} dataset and build regression model",
    "Write technical specification for {feature} architecture",
    "Implement {protocol} authentication flow with token refresh",
    "Build data pipeline for processing {data_type} in real-time",
    "Migrate {component} from monolith to microservice architecture",
    "Create machine learning feature extraction for {data_type} analysis",
    "Write compiler pass for {algo} optimization in the build system",
    "Implement distributed {algo} consensus protocol for {component}",
    "Design and implement caching strategy for {component} reducing DB load",
    "Profile and fix memory leak in {component} under high concurrency",
    "Build real-time {data_type} processing pipeline with exactly-once semantics",
    "Implement custom {algo} solver for the constraint optimization engine",
    "Design fault-tolerant {component} with automatic failover and recovery",
    "Develop end-to-end encryption module for {component} with key rotation",
    "Architect a sharding strategy for the {component} database layer",
    "Implement WebSocket connection pooling for the {component} real-time service",
    "Build a custom query optimizer for the {component} analytics engine",
    "Create automated performance benchmarking suite for {component}",
]

SHALLOW_WORK_TEMPLATES = [
    "Update {component} dependency versions in package.json",
    "Fix typo in {doc} documentation page",
    "Add logging to {component} endpoint",
    "Update README with new {feature} setup instructions",
    "Rename {old_name} variable to {new_name} across codebase",
    "Add input validation for {field} field in {component} form",
    "Update {config} configuration for staging environment",
    "Move {file} to the {component} directory",
    "Add environment variable for {config} setting",
    "Run linter and fix formatting issues in {component}",
    "Update API version number to {version}",
    "Add missing type annotations to {component} module",
    "Clean up unused imports in {component} files",
    "Update changelog for version {version} release",
    "Pin {dependency} to specific version for stability",
    "Add default value for {field} in {component} model",
    "Update CI pipeline to use Node {version}",
    "Fix broken link in {doc} documentation",
    "Add .env.example file with required variables",
    "Bump version number for {component} hotfix release",
    "Sort CSS properties alphabetically in {component} stylesheet",
    "Remove deprecated API endpoint from {component} router",
    "Update copyright year in all license headers",
    "Fix indentation inconsistency in {component} config files",
]

CREATIVE_TEMPLATES = [
    "Design new onboarding flow for first-time {user_type} users",
    "Create visual identity for {brand} product launch",
    "Brainstorm innovative solutions for {problem} user pain point",
    "Design interactive {component} visualization with animations",
    "Write compelling copy for {page} landing page",
    "Prototype new {feature} experience using Figma",
    "Create motion design for {component} state transitions",
    "Design gamification system for {feature} user engagement",
    "Sketch wireframes for {feature} mobile experience",
    "Create illustration set for {doc} help center articles",
    "Design data visualization dashboard for {data_type} metrics",
    "Compose original background music for {feature} meditation mode",
    "Create brand storytelling narrative for {brand} campaign",
    "Design micro-interactions for {component} hover and focus states",
    "Build generative art system for user profile avatars",
    "Create typography system for {brand} design language",
    "Design immersive {feature} experience with parallax scrolling",
    "Storyboard tutorial video for {feature} walkthrough",
    "Create responsive illustration that adapts to {component} viewport",
    "Design award-worthy UI for {feature} settings panel",
    "Concept exploration for a new {brand} product packaging design",
    "Create a mood board for the {feature} redesign project",
    "Design accessible color palette for {brand} following WCAG 2.1",
    "Illustrate technical architecture diagram for {component} documentation",
]

ADMINISTRATIVE_TEMPLATES = [
    "Review and approve {count} pending pull requests",
    "Update {doc} JIRA tickets with current sprint status",
    "Organize team standup notes from this week",
    "Schedule {meeting_type} meeting with {team} team",
    "Process expense reports for {month} purchases",
    "Update project timeline in {tool} for Q{quarter} milestones",
    "File quarterly {report_type} compliance report",
    "Review and update team access permissions in {tool}",
    "Create onboarding checklist for new {role} hire",
    "Audit {component} service uptime logs for last month",
    "Prepare slide deck for {meeting_type} stakeholder presentation",
    "Update team roster and contact information in HR system",
    "Review and categorize incoming support tickets for triage",
    "Reconcile {month} budget allocation across departments",
    "Document standard operating procedures for {process} workflow",
    "Archive completed {component} project files and close tickets",
    "Compile weekly status report for {team} management review",
    "Coordinate vendor contract renewal for {tool} licenses",
    "Update inventory of development hardware and software assets",
    "Plan and book travel for upcoming {meeting_type} conference",
    "Generate monthly KPI dashboard for {team} leadership review",
    "Organize shared drive folder structure for {component} project",
    "Complete mandatory annual security training certification",
    "Submit timesheet corrections for the past pay period",
]

COMMUNICATION_TEMPLATES = [
    "Draft email to {team} team about {topic} deadline change",
    "Prepare presentation for {meeting_type} quarterly review",
    "Write blog post about our {feature} technical architecture",
    "Reply to client feedback about {component} performance issues",
    "Create internal FAQ document for {feature} rollout",
    "Record demo video showing {feature} new capabilities",
    "Write release notes for {component} version {version}",
    "Draft proposal for {feature} partnership opportunity",
    "Compose newsletter update about {topic} progress this quarter",
    "Create tutorial walkthrough for {feature} API integration",
    "Write incident postmortem for the {component} outage last week",
    "Prepare talking points for {meeting_type} customer call",
    "Draft SOW document for {feature} consulting engagement",
    "Write technical blog comparing {algo} vs alternative approaches",
    "Create onboarding documentation for {component} SDK users",
    "Record podcast episode discussing {topic} industry trends",
    "Draft press release for {feature} product announcement",
    "Write RFP response for {component} enterprise contract",
    "Create knowledge base article for {feature} troubleshooting",
    "Compose apology communication regarding {component} service disruption",
    "Write user research summary for the {feature} usability study",
    "Create slide deck comparing competitive {component} solutions",
    "Draft executive summary of {topic} for the board meeting",
    "Produce a screencast tutorial for the new {feature} workflow",
]

FILL_VALUES = {
    "algo": ["binary search", "A*", "gradient descent", "Dijkstra", "quicksort",
             "backpropagation", "dynamic programming", "BFS", "Monte Carlo",
             "simulated annealing", "genetic", "k-means", "random forest",
             "transformer", "attention mechanism", "beam search"],
    "component": ["payment", "auth", "search", "notification", "analytics",
                  "dashboard", "user-profile", "inventory", "messaging",
                  "billing", "scheduling", "reporting", "cache", "gateway",
                  "recommendation", "streaming", "workflow"],
    "issue": ["null pointer", "race condition", "memory leak", "timeout",
             "deadlock", "stack overflow", "segfault", "OOM", "CORS", "infinite loop"],
    "pattern": ["observer", "strategy", "factory", "singleton", "decorator",
               "repository", "CQRS", "event-driven", "hexagonal", "mediator"],
    "feature": ["dark mode", "real-time sync", "multi-tenant", "offline-first",
               "push notification", "two-factor auth", "auto-save", "undo-redo",
               "collaborative editing", "version history", "export", "SSO"],
    "data_type": ["time-series", "geospatial", "clickstream", "log",
                 "transaction", "sensor", "genomic", "NLP corpus", "image"],
    "protocol": ["OAuth 2.0", "JWT", "SAML", "OpenID Connect", "mTLS", "WebAuthn"],
    "doc": ["API reference", "getting started", "deployment", "architecture",
           "contributing", "security", "migration", "troubleshooting"],
    "old_name": ["userData", "tempVal", "processItem", "handleEvent", "dataList"],
    "new_name": ["userProfile", "intermediateValue", "transformItem", "onEvent", "dataCollection"],
    "field": ["email", "phone_number", "address", "date_of_birth", "username", "password"],
    "config": ["database", "redis", "S3", "CDN", "logging", "feature-flag", "monitoring"],
    "file": ["utils.py", "helpers.js", "constants.ts", "types.d.ts", "config.yaml"],
    "version": ["3.2.1", "4.0.0", "2.8.0", "5.1.0", "1.12.0", "6.0.0-rc.1"],
    "dependency": ["lodash", "axios", "moment", "webpack", "prisma", "react-query"],
    "user_type": ["enterprise", "developer", "student", "creator", "analyst", "designer"],
    "brand": ["NovaTech", "Luminary", "AuraSync", "FlowState", "Zenith", "Nexus"],
    "problem": ["onboarding drop-off", "feature discoverability", "retention",
               "mobile performance", "accessibility", "search relevance"],
    "page": ["homepage", "pricing", "product tour", "signup", "features", "about"],
    "count": ["12", "8", "15", "6", "20", "3"],
    "meeting_type": ["sprint planning", "retrospective", "all-hands",
                    "1-on-1", "design review", "architecture", "stakeholder"],
    "team": ["engineering", "product", "design", "QA", "DevOps", "marketing", "sales"],
    "month": ["January", "February", "March", "October", "November", "December"],
    "tool": ["Jira", "Confluence", "Notion", "Linear", "Asana", "GitHub", "Slack"],
    "quarter": ["1", "2", "3", "4"],
    "report_type": ["SOC2", "GDPR", "accessibility", "security", "financial"],
    "role": ["frontend engineer", "backend engineer", "designer", "PM", "QA", "SRE"],
    "process": ["deployment", "incident response", "code review", "release", "onboarding"],
    "topic": ["Q1 roadmap", "infrastructure migration", "team restructuring",
             "product launch", "security audit", "performance optimization"],
}


def fill_template(template):
    result = template
    for key, values in FILL_VALUES.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


# ──────────────────────────────────────────────────────────────
# Step 2: Data Augmentation
# ──────────────────────────────────────────────────────────────
print("🔄 Applying data augmentation...")

# Download NLTK data for synonyms
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import wordnet

def get_synonyms(word):
    """Get synonyms from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                synonyms.add(name)
    return list(synonyms)

def augment_word_dropout(text, p=0.1):
    """Randomly drop words with probability p."""
    words = text.split()
    if len(words) <= 3:
        return text
    kept = [w for w in words if random.random() > p]
    return ' '.join(kept) if len(kept) > 2 else text

def augment_synonym_replace(text, p=0.15):
    """Replace words with synonyms with probability p."""
    words = text.split()
    new_words = []
    for w in words:
        if random.random() < p and len(w) > 3:
            syns = get_synonyms(w.lower())
            if syns:
                new_words.append(random.choice(syns))
            else:
                new_words.append(w)
        else:
            new_words.append(w)
    return ' '.join(new_words)

def augment_typos(text, p=0.02):
    """Add random character-level typos."""
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < p and chars[i].isalpha():
            op = random.choice(['swap', 'delete', 'insert', 'replace'])
            if op == 'swap' and i < len(chars) - 1:
                chars[i], chars[i+1] = chars[i+1], chars[i]
            elif op == 'delete':
                chars[i] = ''
            elif op == 'insert':
                chars[i] = chars[i] + random.choice('abcdefghijklmnopqrstuvwxyz')
            elif op == 'replace':
                chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
    return ''.join(chars)

def augment_word_swap(text):
    """Randomly swap two adjacent words."""
    words = text.split()
    if len(words) < 4:
        return text
    idx = random.randint(1, len(words) - 2)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return ' '.join(words)


# Generate base + augmented dataset
random.seed(42)
CATEGORIES = [
    (DEEP_WORK_TEMPLATES, 0, "DEEP_WORK"),
    (SHALLOW_WORK_TEMPLATES, 1, "SHALLOW_WORK"),
    (CREATIVE_TEMPLATES, 2, "CREATIVE"),
    (ADMINISTRATIVE_TEMPLATES, 3, "ADMINISTRATIVE"),
    (COMMUNICATION_TEMPLATES, 4, "COMMUNICATION"),
]

DEMAND_PROFILES = {
    0: (0.85, 0.08),
    1: (0.25, 0.10),
    2: (0.70, 0.12),
    3: (0.35, 0.10),
    4: (0.50, 0.12),
}

NUM_BASE_PER_CLASS = 400
all_tasks = []

for templates, label, label_name in CATEGORIES:
    for i in range(NUM_BASE_PER_CLASS):
        template = random.choice(templates)
        text = fill_template(template)
        mean, std = DEMAND_PROFILES[label]
        demand = max(0.0, min(1.0, random.gauss(mean, std)))
        
        # Original
        all_tasks.append({"text": text, "label": label, "label_name": label_name, "cognitive_demand": round(demand, 3)})
        
        # Augmentation 1: Word dropout
        aug1 = augment_word_dropout(text, p=0.1)
        if aug1 != text:
            all_tasks.append({"text": aug1, "label": label, "label_name": label_name, "cognitive_demand": round(demand, 3)})
        
        # Augmentation 2: Synonym replacement
        aug2 = augment_synonym_replace(text, p=0.15)
        if aug2 != text:
            all_tasks.append({"text": aug2, "label": label, "label_name": label_name, "cognitive_demand": round(demand, 3)})
        
        # Augmentation 3: Typos + word swap (every 3rd sample)
        if i % 3 == 0:
            aug3 = augment_typos(augment_word_swap(text), p=0.02)
            all_tasks.append({"text": aug3, "label": label, "label_name": label_name, "cognitive_demand": round(demand, 3)})

# Add ~5% cross-category noise (hard negatives)
noise_count = int(len(all_tasks) * 0.05)
for _ in range(noise_count):
    templates, label, label_name = random.choice(CATEGORIES)
    wrong_label = random.choice([l for l in range(5) if l != label])
    text = fill_template(random.choice(templates))
    demand_m, demand_s = DEMAND_PROFILES[wrong_label]
    demand = max(0.0, min(1.0, random.gauss(demand_m, demand_s)))
    all_tasks.append({"text": text, "label": wrong_label, "label_name": CATEGORIES[wrong_label][2], "cognitive_demand": round(demand, 3)})

random.shuffle(all_tasks)

from collections import Counter
dist = Counter(t["label_name"] for t in all_tasks)
print(f"   ✅ Generated {len(all_tasks)} task descriptions (with augmentation)")
for label, count in sorted(dist.items()):
    print(f"     {label}: {count}")

# Save to CSV
csv_path = OUTPUT_DIR / "labeled_tasks_augmented.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label", "label_name", "cognitive_demand"])
    writer.writeheader()
    writer.writerows(all_tasks)
print(f"   ✅ Saved: {csv_path}")

# ──────────────────────────────────────────────────────────────
# Step 3: Fine-tune DistilBERT
# ──────────────────────────────────────────────────────────────
print(f"\n🚀 Fine-tuning DistilBERT for task classification...")

from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

CLASS_NAMES = ["DEEP_WORK", "SHALLOW_WORK", "CREATIVE", "ADMINISTRATIVE", "COMMUNICATION"]
MAX_LENGTH = 128
EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# Load dataset
dataset = Dataset.from_csv(str(csv_path))
print(f"   ✅ Loaded {len(dataset)} samples")

# Tokenize
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_fn(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )

dataset = dataset.map(tokenize_fn, batched=True, batch_size=64)

# Cast label to ClassLabel so stratified split works
from datasets import ClassLabel as CL
dataset = dataset.cast_column('label', CL(names=CLASS_NAMES))

dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Train/test split
split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
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

# Metrics callback
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='macro'),
    }

# Training
training_args = TrainingArguments(
    output_dir="/content/nlp_training_output",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=32,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    warmup_steps=200,
    weight_decay=0.01,
    learning_rate=LEARNING_RATE,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to='none',
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print(f"\n🚀 Starting training ({EPOCHS} epochs)...")
train_result = trainer.train()
print(f"   Training loss: {train_result.training_loss:.4f}")

# ──────────────────────────────────────────────────────────────
# Step 4: Evaluate
# ──────────────────────────────────────────────────────────────
print(f"\n📊 Evaluation Results:")
eval_result = trainer.evaluate()
print(f"   Accuracy: {eval_result.get('eval_accuracy', 'N/A')}")
print(f"   F1 (macro): {eval_result.get('eval_f1', 'N/A')}")

preds = trainer.predict(eval_ds)
y_pred = np.argmax(preds.predictions, axis=-1)
y_true = preds.label_ids

print(f"\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ──────────────────────────────────────────────────────────────
# Step 5: Save Model & Export ONNX
# ──────────────────────────────────────────────────────────────
print(f"\n💾 Saving model...")

model_save_dir = OUTPUT_DIR / "task_nlp_model"
model.save_pretrained(str(model_save_dir))
tokenizer.save_pretrained(str(model_save_dir))
print(f"   ✅ Model saved: {model_save_dir}")

# Copy vocab.txt to output dir for browser tokenizer
vocab_src = model_save_dir / "vocab.txt"
vocab_dest = OUTPUT_DIR / "vocab.txt"
if vocab_src.exists():
    shutil.copy2(str(vocab_src), str(vocab_dest))
    print(f"   ✅ vocab.txt copied to {vocab_dest}")

# ONNX export — produce a SINGLE file (required for ONNX Runtime Web)
print(f"\n📦 Exporting to ONNX...")
model.eval()
device = torch.device('cpu')
model = model.to(device)

dummy_input_ids = torch.zeros(1, MAX_LENGTH, dtype=torch.long).to(device)
dummy_attention_mask = torch.ones(1, MAX_LENGTH, dtype=torch.long).to(device)

onnx_path = str(OUTPUT_DIR / "task_nlp_classifier.onnx")
onnx_temp_path = str(OUTPUT_DIR / "_temp_nlp.onnx")

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    onnx_temp_path,
    opset_version=14,
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch'},
        'attention_mask': {0: 'batch'}
    },
    do_constant_folding=True,
)

# Merge external data into single file (ONNX Runtime Web requires single file)
import onnx
onnx_model = onnx.load(onnx_temp_path, load_external_data=True)
onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)

# Clean up temp files
for f in OUTPUT_DIR.glob("_temp_nlp*"):
    f.unlink(missing_ok=True)
for f in OUTPUT_DIR.glob("task_nlp_classifier.onnx.data"):
    f.unlink(missing_ok=True)

onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
print(f"   ✅ ONNX exported (single file): {onnx_path} ({onnx_size:.1f} MB)")

# ──────────────────────────────────────────────────────────────
# Step 6: ONNX Inference Test
# ──────────────────────────────────────────────────────────────
print(f"\n🧪 Quick ONNX inference test...")
import onnxruntime as ort

session = ort.InferenceSession(onnx_path)
inputs = {
    'input_ids': np.zeros((1, MAX_LENGTH), dtype=np.int64),
    'attention_mask': np.ones((1, MAX_LENGTH), dtype=np.int64),
}
output = session.run(None, inputs)
print(f"   Output logits shape: {output[0].shape}")
print(f"   ✅ ONNX inference works!")

# Test with a real sentence
test_sentences = [
    "Implement gradient descent algorithm for the payment module",
    "Fix typo in API reference documentation page",
    "Design new onboarding flow for first-time developer users",
    "Review and approve 12 pending pull requests",
    "Draft email to engineering team about Q1 roadmap deadline change",
]

print(f"\n📝 Test predictions:")
for sentence in test_sentences:
    tokens = tokenizer(sentence, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='np')
    output = session.run(None, {
        'input_ids': tokens['input_ids'].astype(np.int64),
        'attention_mask': tokens['attention_mask'].astype(np.int64),
    })
    logits = output[0][0]
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    pred = np.argmax(probs)
    print(f"   [{CLASS_NAMES[pred]:15s} {probs[pred]*100:5.1f}%] {sentence[:60]}")

# ──────────────────────────────────────────────────────────────
# Step 7: Save Metrics
# ──────────────────────────────────────────────────────────────
metrics = {
    "accuracy": float(eval_result.get('eval_accuracy', 0)),
    "f1_macro": float(eval_result.get('eval_f1', 0)),
    "training_loss": float(train_result.training_loss),
    "epochs": EPOCHS,
    "total_samples": len(all_tasks),
    "train_size": len(train_ds),
    "eval_size": len(eval_ds),
    "augmentation": "word_dropout + synonym_replace + typos + cross_category_noise",
    "class_names": CLASS_NAMES,
    "max_length": MAX_LENGTH,
    "model_size_mb": onnx_size,
}

metrics_path = OUTPUT_DIR / "nlp_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ NLP MODEL TRAINING COMPLETE!")
print(f"   Accuracy: {metrics['accuracy']:.4f}")
print(f"   F1 (macro): {metrics['f1_macro']:.4f}")
print(f"\n   Output files in: {OUTPUT_DIR}")
print(f"   - task_nlp_classifier.onnx ({onnx_size:.1f} MB)")
print(f"   - vocab.txt")
print(f"   - nlp_metrics.json")
print(f"   - task_nlp_model/ (full model for further fine-tuning)")

print("\n" + "=" * 60)
print("🎉 NLP model ready! Download files from /content/ani_models/")
print("=" * 60)
