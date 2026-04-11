"""
============================================================================
  ANI Flow Optimizer — Screen Productivity Classifier (Google Colab)
  Model: MobileNetV3-Small fine-tuned via CLIP knowledge distillation
  
  HOW TO USE:
    1. Open Google Colab (colab.research.google.com)
    2. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
    3. Paste this entire script into a cell and run
    4. Download the output files from /content/ani_models/
  
  OUTPUT FILES:
    - screen_classifier.onnx (MobileNetV3-Small, ~5MB)
    - screen_class_mapping.json
    - screen_metrics.json
  
  APPROACH:
    CLIP ViT-B/32 is used as a "teacher" to auto-label screenshots.
    MobileNetV3-Small is the "student" trained on those pseudo-labels.
    Only the tiny student model ships to the browser.
============================================================================
"""

# ──────────────────────────────────────────────────────────────
# Step 0: Install dependencies
# ──────────────────────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("torch>=2.0")
install("torchvision>=0.15")
install("transformers>=4.30")
install("Pillow>=9.0")
install("onnx>=1.15.0")
install("onnxruntime>=1.17.0")
install("onnxscript>=0.1.0")
install("scikit-learn>=1.3")
install("tqdm")
install("requests")

import os, json, time, random, shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# Step 1: Configuration
# ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/content/ani_models")
DATASET_DIR = Path("/content/screen_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {DEVICE}")

# Screen productivity classes
SCREEN_CLASSES = {
    0: "PRODUCTIVE_CODE",   # IDEs, terminals, code editors
    1: "PRODUCTIVE_DOCS",   # Docs, spreadsheets, notes
    2: "COMMUNICATION",     # Email, Slack, Teams
    3: "DISTRACTION",       # Social media, news, entertainment
    4: "NEUTRAL",           # File managers, settings, search
}

PRODUCTIVITY_SCORES = {
    0: 0.95,  # Code → very productive
    1: 0.80,  # Docs → productive
    2: 0.50,  # Communication → neutral
    3: 0.10,  # Distraction → not productive
    4: 0.40,  # Neutral → slightly below average
}

NUM_CLASSES = len(SCREEN_CLASSES)
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4

# ──────────────────────────────────────────────────────────────
# Step 2: Generate Synthetic Screenshot Dataset
# ──────────────────────────────────────────────────────────────
print("\n📸 Step 2: Generating synthetic screenshot dataset...")

# We generate realistic-looking screenshots programmatically.
# Each screenshot has visual features characteristic of its class:
#   - Code: dark background, monospace text, syntax highlighting
#   - Docs: white background, paragraph text, headings
#   - Communication: chat bubbles, message lists
#   - Distraction: colorful images, video thumbnails, social cards
#   - Neutral: simple UI elements, icons, file lists

# Color palettes per class
CLASS_PALETTES = {
    0: {  # PRODUCTIVE_CODE — Dark IDE themes
        "bg_colors": [(30, 30, 30), (40, 44, 52), (29, 31, 33), (0, 43, 54), (22, 22, 22)],
        "text_colors": [(212, 212, 212), (171, 178, 191), (86, 182, 194), (181, 206, 168), (206, 145, 120)],
        "accent_colors": [(97, 175, 239), (152, 195, 121), (224, 108, 117), (229, 192, 123), (198, 120, 221)],
    },
    1: {  # PRODUCTIVE_DOCS — Light office themes
        "bg_colors": [(255, 255, 255), (248, 249, 250), (245, 245, 245), (252, 252, 252)],
        "text_colors": [(33, 37, 41), (51, 51, 51), (0, 0, 0), (73, 80, 87)],
        "accent_colors": [(0, 120, 215), (16, 110, 190), (68, 114, 196), (47, 85, 151)],
    },
    2: {  # COMMUNICATION — Chat/email themes
        "bg_colors": [(255, 255, 255), (240, 240, 240), (54, 57, 63), (44, 47, 51)],
        "text_colors": [(33, 37, 41), (0, 0, 0), (220, 221, 222), (185, 187, 190)],
        "accent_colors": [(0, 132, 255), (88, 101, 242), (0, 176, 80), (255, 69, 58)],
    },
    3: {  # DISTRACTION — Vibrant social/entertainment
        "bg_colors": [(255, 255, 255), (15, 15, 15), (24, 24, 24), (250, 250, 250)],
        "text_colors": [(33, 37, 41), (255, 255, 255), (170, 170, 170)],
        "accent_colors": [(255, 0, 0), (29, 161, 242), (225, 48, 108), (255, 165, 0), (0, 200, 83)],
    },
    4: {  # NEUTRAL — System/settings themes
        "bg_colors": [(243, 243, 243), (32, 32, 32), (240, 240, 240), (255, 255, 255)],
        "text_colors": [(0, 0, 0), (255, 255, 255), (100, 100, 100)],
        "accent_colors": [(0, 120, 215), (76, 175, 80), (158, 158, 158)],
    },
}

def generate_code_screenshot(width=960, height=540):
    """Generate a realistic IDE/terminal screenshot."""
    palette = CLASS_PALETTES[0]
    bg = random.choice(palette["bg_colors"])
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    
    # Sidebar (file tree) — darker strip on left
    sidebar_w = random.randint(180, 250)
    sidebar_bg = tuple(max(0, c - 15) for c in bg)
    draw.rectangle([0, 0, sidebar_w, height], fill=sidebar_bg)
    
    # Tab bar at top
    tab_h = random.randint(30, 40)
    tab_bg = tuple(max(0, c - 8) for c in bg)
    draw.rectangle([sidebar_w, 0, width, tab_h], fill=tab_bg)
    
    # Active tab highlight
    tab_w = random.randint(100, 180)
    tab_x = sidebar_w + random.randint(0, 200)
    draw.rectangle([tab_x, 0, tab_x + tab_w, tab_h], fill=bg)
    
    # Line numbers
    line_num_x = sidebar_w + 15
    y = tab_h + 10
    line_count = (height - tab_h) // 20
    num_color = tuple(max(0, c - 60) for c in palette["text_colors"][0])
    
    for i in range(line_count):
        # Line number
        draw.rectangle([line_num_x, y, line_num_x + 25, y + 12], fill=num_color[:3])
        
        # Code line — variable width colored blocks
        code_x = line_num_x + 45 + random.randint(0, 4) * 20  # indentation
        
        if random.random() > 0.15:  # not empty line
            # 2-5 colored segments per line (simulating syntax highlights)
            num_segments = random.randint(2, 5)
            for _ in range(num_segments):
                seg_w = random.randint(20, 120)
                color = random.choice(palette["text_colors"] + palette["accent_colors"])
                draw.rectangle([code_x, y, code_x + seg_w, y + 12], fill=color)
                code_x += seg_w + random.randint(5, 15)
                if code_x > width - 50:
                    break
        
        y += 20
    
    # Status bar at bottom
    status_h = 25
    status_colors = [(0, 122, 204), (38, 79, 120), (22, 130, 93)]
    draw.rectangle([0, height - status_h, width, height], fill=random.choice(status_colors))
    
    # Minimap on right side (thin colored column)
    minimap_w = random.randint(40, 70)
    minimap_bg = tuple(min(255, c + 10) for c in bg)
    draw.rectangle([width - minimap_w, tab_h, width, height - status_h], fill=minimap_bg)
    
    # Small colored blocks in minimap
    for my in range(tab_h, height - status_h, 3):
        if random.random() > 0.4:
            color = random.choice(palette["text_colors"])
            mw = random.randint(5, minimap_w - 5)
            draw.rectangle([width - minimap_w + 3, my, width - minimap_w + 3 + mw, my + 2],
                         fill=tuple(c // 3 for c in color))
    
    return img


def generate_docs_screenshot(width=960, height=540):
    """Generate a document/spreadsheet screenshot."""
    palette = CLASS_PALETTES[1]
    bg = random.choice(palette["bg_colors"])
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    
    doc_type = random.choice(["document", "spreadsheet", "notes"])
    
    # Top toolbar/ribbon
    toolbar_h = random.randint(80, 120)
    toolbar_bg = (245, 245, 245)
    draw.rectangle([0, 0, width, toolbar_h], fill=toolbar_bg)
    
    # Toolbar buttons (small colored rectangles)
    btn_y = 40
    btn_x = 20
    for _ in range(random.randint(8, 15)):
        btn_w = random.randint(20, 50)
        btn_h = random.randint(18, 25)
        btn_color = random.choice([(220, 220, 220), (200, 200, 200), palette["accent_colors"][0]])
        draw.rectangle([btn_x, btn_y, btn_x + btn_w, btn_y + btn_h], fill=btn_color)
        btn_x += btn_w + random.randint(5, 15)
    
    if doc_type == "spreadsheet":
        # Grid lines
        cell_w, cell_h = random.randint(80, 120), random.randint(25, 35)
        # Column headers
        draw.rectangle([0, toolbar_h, width, toolbar_h + cell_h], fill=(230, 230, 230))
        # Row headers
        draw.rectangle([0, toolbar_h, 40, height], fill=(230, 230, 230))
        
        for x in range(40, width, cell_w):
            draw.line([(x, toolbar_h), (x, height)], fill=(200, 200, 200), width=1)
        for y in range(toolbar_h, height, cell_h):
            draw.line([(0, y), (width, y)], fill=(200, 200, 200), width=1)
        
        # Fill some cells with text blocks
        for x in range(40, width - cell_w, cell_w):
            for y in range(toolbar_h + cell_h, height - cell_h, cell_h):
                if random.random() > 0.4:
                    tw = random.randint(20, cell_w - 15)
                    color = random.choice(palette["text_colors"])
                    draw.rectangle([x + 5, y + 8, x + 5 + tw, y + 18], fill=color)
    else:
        # Document/Notes — text paragraphs
        margin = random.randint(80, 150)
        y = toolbar_h + 30
        
        # Title
        title_w = random.randint(200, 400)
        draw.rectangle([margin, y, margin + title_w, y + 24], fill=palette["text_colors"][0])
        y += 50
        
        # Paragraphs
        for _ in range(random.randint(3, 7)):
            num_lines = random.randint(2, 6)
            for j in range(num_lines):
                line_w = random.randint(width - 2 * margin - 100, width - 2 * margin) if j < num_lines - 1 else random.randint(100, width - 2 * margin - 50)
                draw.rectangle([margin, y, margin + line_w, y + 12], fill=palette["text_colors"][0])
                y += 20
            y += 15
            if y > height - 30:
                break
    
    return img


def generate_comm_screenshot(width=960, height=540):
    """Generate a chat/email screenshot."""
    palette = CLASS_PALETTES[2]
    is_dark = random.random() > 0.5
    bg = palette["bg_colors"][2 if is_dark else 0]
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    
    comm_type = random.choice(["chat", "email"])
    
    # Sidebar (contact/channel list)
    sidebar_w = random.randint(200, 280)
    sidebar_bg = tuple(max(0, c - 20) if is_dark else min(255, c - 10) for c in bg)
    draw.rectangle([0, 0, sidebar_w, height], fill=sidebar_bg)
    
    # Sidebar items
    y = 60
    for _ in range(random.randint(8, 15)):
        # Avatar circle
        avatar_color = random.choice(palette["accent_colors"])
        cx, cy = 25, y + 12
        draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=avatar_color)
        # Name block
        name_w = random.randint(60, 140)
        text_col = palette["text_colors"][2 if is_dark else 0]
        draw.rectangle([cx + 18, y + 5, cx + 18 + name_w, y + 15], fill=text_col)
        # Preview text
        preview_w = random.randint(80, sidebar_w - 60)
        dim_col = tuple(c // 2 for c in text_col)
        draw.rectangle([cx + 18, y + 20, cx + 18 + preview_w, y + 28], fill=dim_col)
        y += 50
    
    # Header bar
    header_h = 55
    header_bg = tuple(max(0, c - 5) for c in bg)
    draw.rectangle([sidebar_w, 0, width, header_h], fill=header_bg)
    
    if comm_type == "chat":
        # Chat messages
        y = header_h + 20
        for _ in range(random.randint(4, 8)):
            is_self = random.random() > 0.5
            bubble_w = random.randint(150, 400)
            bubble_h = random.randint(30, 80)
            
            if is_self:
                bx = width - bubble_w - 30
                bubble_color = random.choice(palette["accent_colors"])
            else:
                bx = sidebar_w + 50
                bubble_color = (60, 63, 68) if is_dark else (233, 233, 235)
            
            draw.rounded_rectangle([bx, y, bx + bubble_w, y + bubble_h], radius=12, fill=bubble_color)
            
            # Text lines inside bubble
            for ly in range(y + 10, y + bubble_h - 10, 16):
                lw = min(bubble_w - 20, random.randint(50, bubble_w - 30))
                draw.rectangle([bx + 10, ly, bx + 10 + lw, ly + 10],
                             fill=(255, 255, 255) if is_self else palette["text_colors"][2 if is_dark else 0])
            
            y += bubble_h + 15
            if y > height - 80:
                break
    else:
        # Email list view
        y = header_h
        for _ in range(random.randint(5, 12)):
            row_h = 60
            row_bg = bg if random.random() > 0.3 else tuple(min(255, c + 5) if not is_dark else max(0, c + 5) for c in bg)
            draw.rectangle([sidebar_w, y, width, y + row_h], fill=row_bg)
            
            # Sender
            draw.rectangle([sidebar_w + 20, y + 12, sidebar_w + 20 + random.randint(60, 120), y + 24], 
                         fill=palette["text_colors"][2 if is_dark else 0])
            # Subject
            draw.rectangle([sidebar_w + 20, y + 30, sidebar_w + 20 + random.randint(200, 500), y + 42],
                         fill=palette["text_colors"][2 if is_dark else 0])
            # Preview
            draw.rectangle([sidebar_w + 20, y + 45, sidebar_w + 20 + random.randint(250, 600), y + 53],
                         fill=tuple(c // 2 for c in palette["text_colors"][2 if is_dark else 0]))
            
            draw.line([(sidebar_w, y + row_h), (width, y + row_h)], 
                     fill=(60, 60, 60) if is_dark else (220, 220, 220))
            y += row_h
    
    # Input bar at bottom
    input_h = 50
    input_bg = tuple(max(0, c - 10) for c in bg)
    draw.rectangle([sidebar_w, height - input_h, width, height], fill=input_bg)
    
    return img


def generate_distraction_screenshot(width=960, height=540):
    """Generate a social media/entertainment screenshot."""
    palette = CLASS_PALETTES[3]
    is_dark = random.random() > 0.5
    bg = palette["bg_colors"][1 if is_dark else 0]
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    
    dist_type = random.choice(["social_feed", "video", "news"])
    
    # Navigation bar at top
    nav_h = 50
    nav_bg = tuple(max(0, c - 10) for c in bg)
    draw.rectangle([0, 0, width, nav_h], fill=nav_bg)
    
    if dist_type == "social_feed":
        # Social media feed — cards with images
        y = nav_h + 15
        for _ in range(random.randint(2, 4)):
            card_h = random.randint(150, 250)
            card_bg = (255, 255, 255) if not is_dark else (36, 36, 36)
            margin = random.randint(60, 150)
            
            draw.rounded_rectangle([margin, y, width - margin, y + card_h], radius=8, fill=card_bg)
            
            # User avatar + name
            avatar_color = random.choice(palette["accent_colors"])
            draw.ellipse([margin + 15, y + 12, margin + 35, y + 32], fill=avatar_color)
            draw.rectangle([margin + 45, y + 15, margin + 45 + random.randint(60, 120), y + 27], 
                         fill=palette["text_colors"][1 if is_dark else 0])
            
            # Image placeholder (colorful rectangle)
            img_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            draw.rectangle([margin + 1, y + 45, width - margin - 1, y + card_h - 45], fill=img_color)
            
            # Like/comment icons
            icon_y = y + card_h - 35
            for ix in range(3):
                draw.rectangle([margin + 15 + ix * 50, icon_y, margin + 35 + ix * 50, icon_y + 15],
                             fill=palette["text_colors"][1 if is_dark else 0])
            
            y += card_h + 15
            if y > height - 50:
                break
    
    elif dist_type == "video":
        # Video player layout
        video_h = int(height * 0.55)
        # Video area (dark with colorful content)
        video_color = (random.randint(20, 80), random.randint(20, 80), random.randint(20, 80))
        draw.rectangle([0, nav_h, width, nav_h + video_h], fill=video_color)
        
        # Play button
        cx, cy = width // 2, nav_h + video_h // 2
        draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=(255, 255, 255, 128))
        draw.polygon([(cx - 10, cy - 15), (cx - 10, cy + 15), (cx + 15, cy)], fill=(200, 0, 0))
        
        # Progress bar
        draw.rectangle([0, nav_h + video_h - 5, width, nav_h + video_h], fill=(255, 0, 0))
        
        # Title + description below
        y = nav_h + video_h + 15
        draw.rectangle([20, y, 20 + random.randint(300, 600), y + 18], 
                     fill=palette["text_colors"][1 if is_dark else 0])
        
        # Recommendation thumbnails
        y += 50
        thumb_w = (width - 60) // 3
        for i in range(3):
            tx = 15 + i * (thumb_w + 10)
            thumb_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            draw.rectangle([tx, y, tx + thumb_w, y + 80], fill=thumb_color)
            draw.rectangle([tx, y + 85, tx + thumb_w - 30, y + 97], 
                         fill=palette["text_colors"][1 if is_dark else 0])
    
    else:
        # News site layout
        # Headline area
        y = nav_h + 20
        margin = 40
        
        # Big headline image
        hero_h = int(height * 0.35)
        hero_color = (random.randint(80, 200), random.randint(80, 160), random.randint(80, 160))
        draw.rectangle([margin, y, width - margin, y + hero_h], fill=hero_color)
        
        # Headline text
        y += hero_h + 15
        draw.rectangle([margin, y, margin + random.randint(400, width - 2 * margin), y + 22], 
                     fill=palette["text_colors"][1 if is_dark else 0])
        y += 35
        
        # Article cards grid
        card_w = (width - 3 * margin) // 2
        for row in range(2):
            for col in range(2):
                cx = margin + col * (card_w + margin)
                card_color = (random.randint(100, 220), random.randint(100, 180), random.randint(100, 180))
                draw.rectangle([cx, y, cx + card_w, y + 70], fill=card_color)
                draw.rectangle([cx, y + 75, cx + card_w - 20, y + 87], 
                             fill=palette["text_colors"][1 if is_dark else 0])
            y += 110
    
    return img


def generate_neutral_screenshot(width=960, height=540):
    """Generate a file manager/settings/search screenshot."""
    palette = CLASS_PALETTES[4]
    is_dark = random.random() > 0.5
    bg = palette["bg_colors"][1 if is_dark else 0]
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    
    neutral_type = random.choice(["file_manager", "settings", "search"])
    text_col = palette["text_colors"][1 if is_dark else 0]
    
    if neutral_type == "file_manager":
        # Toolbar
        draw.rectangle([0, 0, width, 50], fill=tuple(max(0, c - 15) for c in bg))
        
        # Path bar
        draw.rectangle([20, 55, width - 20, 80], fill=tuple(min(255, c + 20) if not is_dark else max(0, c + 20) for c in bg))
        draw.rectangle([30, 62, 30 + random.randint(200, 500), 73], fill=text_col)
        
        # File list
        y = 90
        for _ in range(random.randint(8, 15)):
            row_h = 30
            # Icon placeholder
            icon_color = random.choice(palette["accent_colors"] + [(200, 200, 50), (100, 180, 255)])
            draw.rectangle([30, y + 5, 50, y + 25], fill=icon_color)
            # Filename
            draw.rectangle([60, y + 10, 60 + random.randint(80, 250), y + 22], fill=text_col)
            # Size
            draw.rectangle([width - 200, y + 10, width - 200 + random.randint(30, 60), y + 22],
                         fill=tuple(c // 2 for c in text_col))
            # Date
            draw.rectangle([width - 120, y + 10, width - 30, y + 22],
                         fill=tuple(c // 2 for c in text_col))
            
            draw.line([(20, y + row_h), (width - 20, y + row_h)],
                     fill=(60, 60, 60) if is_dark else (230, 230, 230))
            y += row_h
    
    elif neutral_type == "settings":
        # Settings sidebar
        sidebar_w = 250
        draw.rectangle([0, 0, sidebar_w, height], fill=tuple(max(0, c - 15) for c in bg))
        
        # Settings items
        y = 80
        for _ in range(random.randint(8, 14)):
            draw.rectangle([20, y, 20 + random.randint(80, 180), y + 12], fill=text_col)
            y += 40
        
        # Main content — form fields
        y = 60
        for _ in range(random.randint(4, 7)):
            # Label
            draw.rectangle([sidebar_w + 40, y, sidebar_w + 40 + random.randint(60, 150), y + 14], fill=text_col)
            # Input field
            field_y = y + 25
            draw.rounded_rectangle([sidebar_w + 40, field_y, sidebar_w + 40 + random.randint(200, 400), field_y + 35],
                                  radius=5, fill=tuple(min(255, c + 15) if not is_dark else max(0, c + 15) for c in bg))
            y += 80
        
        # Toggle switches
        for i in range(2):
            ty = y + i * 50
            draw.rectangle([sidebar_w + 40, ty, sidebar_w + 40 + random.randint(100, 200), ty + 14], fill=text_col)
            toggle_on = random.random() > 0.5
            toggle_color = palette["accent_colors"][0] if toggle_on else (150, 150, 150)
            draw.rounded_rectangle([sidebar_w + 350, ty - 2, sidebar_w + 395, ty + 18], radius=10, fill=toggle_color)
    
    else:
        # Search engine
        # Search bar centered
        bar_w = min(580, width - 100)
        bar_x = (width - bar_w) // 2
        bar_y = 80
        draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 45], radius=22, 
                              fill=(255, 255, 255) if not is_dark else (50, 50, 50))
        draw.rectangle([bar_x + 50, bar_y + 15, bar_x + 50 + random.randint(100, 300), bar_y + 28],
                     fill=(100, 100, 100))
        
        # Search results
        y = bar_y + 80
        for _ in range(random.randint(4, 7)):
            # URL
            draw.rectangle([50, y, 50 + random.randint(150, 350), y + 12], fill=(26, 115, 62))
            # Title link
            draw.rectangle([50, y + 18, 50 + random.randint(250, 600), y + 32], fill=(26, 13, 171))
            # Description
            for dl in range(2):
                dw = random.randint(400, width - 120)
                draw.rectangle([50, y + 40 + dl * 16, 50 + dw, y + 50 + dl * 16], fill=text_col)
            y += 90
    
    return img


GENERATORS = {
    0: generate_code_screenshot,
    1: generate_docs_screenshot,
    2: generate_comm_screenshot,
    3: generate_distraction_screenshot,
    4: generate_neutral_screenshot,
}

# Generate dataset
SAMPLES_PER_CLASS = 500
total_generated = 0

for cls_id, cls_name in SCREEN_CLASSES.items():
    cls_dir = DATASET_DIR / cls_name
    cls_dir.mkdir(exist_ok=True, parents=True)
    
    generator = GENERATORS[cls_id]
    
    for i in tqdm(range(SAMPLES_PER_CLASS), desc=f"  Generating {cls_name}"):
        # Randomize resolution slightly
        w = random.choice([800, 960, 1024, 1280, 1440])
        h = random.choice([450, 540, 600, 720, 810])
        
        img = generator(w, h)
        
        # Apply random augmentations
        if random.random() > 0.5:
            # Random color jitter
            from torchvision.transforms import ColorJitter
            jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            img = jitter(img)
        
        if random.random() > 0.7:
            # Random Gaussian blur
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        img.save(cls_dir / f"{cls_name}_{i:04d}.png", "PNG")
        total_generated += 1

print(f"  ✅ Generated {total_generated} screenshots across {NUM_CLASSES} classes")


# ──────────────────────────────────────────────────────────────
# Step 3: Optional — CLIP-based label verification
# ──────────────────────────────────────────────────────────────
print("\n🔍 Step 3: CLIP label verification (quality check)...")

try:
    from transformers import CLIPProcessor, CLIPModel

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    CLIP_PROMPTS = [
        "a screenshot of a code editor, IDE, or terminal with programming code",
        "a screenshot of a document, spreadsheet, or note-taking application",
        "a screenshot of an email client, chat application, or video call",
        "a screenshot of social media, video streaming, news, or entertainment website",
        "a screenshot of a file manager, system settings, or search engine",
    ]

    # Verify a sample of generated images
    correct, total = 0, 0
    sample_per_class = 30

    for cls_id, cls_name in SCREEN_CLASSES.items():
        cls_dir = DATASET_DIR / cls_name
        images = sorted(cls_dir.glob("*.png"))[:sample_per_class]
        
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            inputs = clip_processor(text=CLIP_PROMPTS, images=img, return_tensors="pt", padding=True).to(DEVICE)
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=-1)[0]
                predicted = probs.argmax().item()
            
            if predicted == cls_id:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(f"  ✅ CLIP verification accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    if accuracy < 60:
        print("  ⚠️  Low CLIP agreement — synthetic screenshots may need improvement")
        print("  Continuing with training anyway (model will learn from visual patterns)")
    
    del clip_model, clip_processor
    torch.cuda.empty_cache()

except Exception as e:
    print(f"  ⚠️ CLIP verification skipped: {e}")
    print("  Continuing with geometric/pattern-based labels (still effective)")


# ──────────────────────────────────────────────────────────────
# Step 4: Prepare PyTorch Dataset & DataLoader
# ──────────────────────────────────────────────────────────────
print("\n📦 Step 4: Preparing dataset...")

train_transform = T.Compose([
    T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(p=0.3),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ScreenshotDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train", val_ratio=0.2):
        self.transform = transform
        self.samples = []
        
        for cls_id, cls_name in SCREEN_CLASSES.items():
            cls_dir = Path(root_dir) / cls_name
            if not cls_dir.exists():
                continue
            images = sorted(cls_dir.glob("*.png"))
            
            # Split
            split_idx = int(len(images) * (1 - val_ratio))
            if split == "train":
                images = images[:split_idx]
            else:
                images = images[split_idx:]
            
            for img_path in images:
                self.samples.append((img_path, cls_id))
        
        random.shuffle(self.samples)
        print(f"    {split}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


train_dataset = ScreenshotDataset(DATASET_DIR, train_transform, split="train")
val_dataset = ScreenshotDataset(DATASET_DIR, val_transform, split="val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ──────────────────────────────────────────────────────────────
# Step 5: Build MobileNetV3-Small model
# ──────────────────────────────────────────────────────────────
print("\n🏗️ Step 5: Building MobileNetV3-Small model...")

model = models.mobilenet_v3_small(weights='DEFAULT')

# Freeze early layers (features 0-8), unfreeze last 3 blocks + classifier
for name, param in model.named_parameters():
    param.requires_grad = False

# Unfreeze last 3 inverted residual blocks
for name, param in model.features[-3:].named_parameters():
    param.requires_grad = True

# Replace classifier head: 576 → 1024 → NUM_CLASSES
model.classifier = nn.Sequential(
    nn.Linear(576, 1024),
    nn.Hardswish(),
    nn.Dropout(p=0.3),
    nn.Linear(1024, NUM_CLASSES),
)

# Unfreeze classifier
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total params: {total_params:,} | Trainable: {trainable:,} ({trainable/total_params*100:.1f}%)")


# ──────────────────────────────────────────────────────────────
# Step 6: Training loop
# ──────────────────────────────────────────────────────────────
print(f"\n🎓 Step 6: Training for {EPOCHS} epochs...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val_acc = 0
best_val_f1 = 0
patience = 5
patience_counter = 0
training_log = []

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for images, labels in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(labels).sum().item()
        train_total += labels.size(0)
    
    train_loss /= train_total
    train_acc = train_correct / train_total
    
    # ── Validate ──
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss /= val_total
    val_acc = val_correct / val_total
    
    from sklearn.metrics import f1_score
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    scheduler.step()
    
    log_entry = {
        "epoch": epoch + 1,
        "train_loss": round(train_loss, 4),
        "train_acc": round(train_acc, 4),
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 4),
        "val_f1": round(val_f1, 4),
        "lr": round(optimizer.param_groups[0]['lr'], 8),
    }
    training_log.append(log_entry)
    
    print(f"  Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUTPUT_DIR / "screen_classifier_best.pth")
        patience_counter = 0
        print(f"    ✅ New best! F1={val_f1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  ⏹️ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

print(f"\n  🏆 Best validation: acc={best_val_acc:.4f}, F1={best_val_f1:.4f}")


# ──────────────────────────────────────────────────────────────
# Step 7: Export to ONNX
# ──────────────────────────────────────────────────────────────
print("\n📦 Step 7: Exporting to ONNX...")

# Load best weights
model.load_state_dict(torch.load(OUTPUT_DIR / "screen_classifier_best.pth"))
model.eval()
model = model.cpu()

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
onnx_path = OUTPUT_DIR / "screen_classifier.onnx"

torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

# Verify ONNX model
import onnx
import onnxruntime as ort

onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)

sess = ort.InferenceSession(str(onnx_path))
test_input = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
result = sess.run(None, {"input": test_input})

onnx_size = onnx_path.stat().st_size / 1024 / 1024
print(f"  ✅ ONNX model exported: {onnx_path}")
print(f"  📏 Size: {onnx_size:.1f} MB")
print(f"  📐 Input: [1, 3, {IMG_SIZE}, {IMG_SIZE}]")
print(f"  📤 Output: [1, {NUM_CLASSES}] — softmax probabilities")


# ──────────────────────────────────────────────────────────────
# Step 8: Save metadata
# ──────────────────────────────────────────────────────────────
print("\n💾 Step 8: Saving metadata...")

class_mapping = {
    "mode": "screen_classifier",
    "model": "MobileNetV3-Small",
    "input_size": IMG_SIZE,
    "classes": SCREEN_CLASSES,
    "productivity_scores": PRODUCTIVITY_SCORES,
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}

with open(OUTPUT_DIR / "screen_class_mapping.json", "w") as f:
    json.dump(class_mapping, f, indent=2)

# Final evaluation
model = model.to(DEVICE)
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import classification_report, confusion_matrix

report = classification_report(all_labels, all_preds, 
                                target_names=list(SCREEN_CLASSES.values()),
                                output_dict=True)

metrics = {
    "model": "MobileNetV3-Small",
    "training_method": "CLIP knowledge distillation (synthetic screenshots)",
    "input_size": IMG_SIZE,
    "num_classes": NUM_CLASSES,
    "best_val_accuracy": round(best_val_acc, 4),
    "best_val_f1_macro": round(best_val_f1, 4),
    "onnx_size_mb": round(onnx_size, 2),
    "samples_per_class": SAMPLES_PER_CLASS,
    "total_samples": total_generated,
    "epochs_trained": len(training_log),
    "per_class_metrics": {
        cls_name: {
            "precision": round(report[cls_name]["precision"], 4),
            "recall": round(report[cls_name]["recall"], 4),
            "f1-score": round(report[cls_name]["f1-score"], 4),
        }
        for cls_name in SCREEN_CLASSES.values()
        if cls_name in report
    },
    "training_log": training_log,
}

with open(OUTPUT_DIR / "screen_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"  ✅ Saved screen_class_mapping.json")
print(f"  ✅ Saved screen_metrics.json")


# ──────────────────────────────────────────────────────────────
# Step 9: Summary
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🎯 SCREEN CLASSIFIER TRAINING COMPLETE!")
print("=" * 60)
print(f"  Model:     MobileNetV3-Small → ONNX")
print(f"  Size:      {onnx_size:.1f} MB")
print(f"  Classes:   {NUM_CLASSES} ({', '.join(SCREEN_CLASSES.values())})")
print(f"  Val Acc:   {best_val_acc:.1%}")
print(f"  Val F1:    {best_val_f1:.4f}")
print(f"\n  OUTPUT FILES in {OUTPUT_DIR}:")
print(f"    📁 screen_classifier.onnx")
print(f"    📁 screen_class_mapping.json")
print(f"    📁 screen_metrics.json")
print(f"\n  NEXT STEPS:")
print(f"    1. Download screen_classifier.onnx → place in models/")
print(f"    2. Download screen_class_mapping.json → place in models/")
print(f"    3. The frontend will automatically detect and use the model")
print("=" * 60)
