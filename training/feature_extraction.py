"""
Shared feature extraction pipeline for all modalities.
Provides reusable extraction functions for audio (Librosa), vision (YOLO), and NLP (DistilBERT).
"""
import numpy as np


# ─── Audio Feature Extraction (52 features) ───────────────────────────────────

def extract_audio_features(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Extract 52-dimensional feature vector from an audio file.
    Requires librosa.
    
    Features:
      [0-12]  MFCC means (13)
      [13-25] MFCC stds (13)
      [26-38] MFCC delta means (13)
      [39-41] MFCC delta-delta means (3)
      [42]    Spectral centroid
      [43]    Spectral rolloff
      [44]    Zero crossing rate
      [45]    RMS energy
      [46]    Pitch mean (F0)
      [47]    Pitch variance (F0)
      [48]    Tempo (BPM)
      [49]    WPM estimate (based on voiced segments)
      [50]    WPM variance estimate
      [51]    Silence ratio
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=sr)
    duration = len(y) / sr

    features = np.zeros(52, dtype=np.float32)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features[0:13] = np.mean(mfcc, axis=1)
    features[13:26] = np.std(mfcc, axis=1)
    features[26:39] = np.mean(mfcc_delta, axis=1)
    features[39:42] = np.mean(mfcc_delta2[:3], axis=1)

    # Spectral features
    features[42] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features[43] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features[44] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    features[45] = float(np.mean(librosa.feature.rms(y=y)))

    # Pitch (F0) using pyin
    f0, voiced_flag, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
    f0_clean = f0[voiced_flag] if voiced_flag is not None else np.array([])
    features[46] = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
    features[47] = float(np.var(f0_clean)) if len(f0_clean) > 0 else 0.0

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features[48] = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    # Approximate WPM from voiced segment analysis
    voiced_ratio = np.sum(voiced_flag) / len(voiced_flag) if voiced_flag is not None and len(voiced_flag) > 0 else 0.5
    estimated_wpm = voiced_ratio * 150  # rough approximation
    features[49] = estimated_wpm
    features[50] = abs(estimated_wpm - 130) * 2  # variance proxy
    features[51] = 1.0 - voiced_ratio  # silence ratio

    return features


# ─── Vision Feature Extraction (4 features) ───────────────────────────────────

def extract_vision_features(yolo_results) -> np.ndarray:
    """
    Extract 4-dimensional feature vector from YOLO detection results.
    
    Features:
      [0] tab_count_norm     - normalized tab count (0-1)
      [1] phone_visible      - binary (0 or 1)
      [2] distraction_count  - normalized distraction count (0-1)
      [3] focus_ratio        - ratio of focus area to screen (0-1)
    
    Args:
        yolo_results: Ultralytics YOLO results object
    """
    features = np.zeros(4, dtype=np.float32)

    # Class mapping: 0=tab_bar, 1=phone, 2=distraction, 3=work_tool
    CLASS_TAB = 0
    CLASS_PHONE = 1
    CLASS_DISTRACTION = 2
    CLASS_WORK = 3

    if yolo_results is None or len(yolo_results) == 0:
        return features

    boxes = yolo_results[0].boxes
    if boxes is None:
        return features

    classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.array(boxes.cls)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.array(boxes.conf)
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)

    # Tab count: count tab_bar detections, normalize to [0, 1]
    tab_detections = np.sum(classes == CLASS_TAB)
    features[0] = min(tab_detections / 30.0, 1.0)

    # Phone visible: any phone detection with confidence > 0.5
    phone_mask = (classes == CLASS_PHONE) & (confs > 0.5)
    features[1] = 1.0 if np.any(phone_mask) else 0.0

    # Distraction count: normalized
    distraction_count = np.sum(classes == CLASS_DISTRACTION)
    features[2] = min(distraction_count / 5.0, 1.0)

    # Focus ratio: area of work_tool detections / total image area
    img_area = 640 * 640  # assuming 640x640 input
    work_mask = classes == CLASS_WORK
    if np.any(work_mask):
        work_boxes = xyxy[work_mask]
        work_area = sum((b[2]-b[0]) * (b[3]-b[1]) for b in work_boxes)
        features[3] = min(work_area / img_area, 1.0)
    else:
        features[3] = 0.5  # default

    return features


# ─── NLP Feature Extraction (3 features) ──────────────────────────────────────

def extract_nlp_features(logits: np.ndarray) -> np.ndarray:
    """
    Extract 3-dimensional feature vector from DistilBERT classification logits.
    
    Features:
      [0] task_class_encoded      - predicted class (0-4)
      [1] cognitive_demand_score  - proxy for cognitive demand (0-1)
      [2] task_confidence         - max softmax probability (0-1)
    """
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    features = np.zeros(3, dtype=np.float32)
    features[0] = float(np.argmax(probs))
    features[2] = float(np.max(probs))

    # Cognitive demand mapping: DEEP=0.9, SHALLOW=0.2, CREATIVE=0.7, ADMIN=0.3, COMM=0.5
    demand_map = {0: 0.9, 1: 0.2, 2: 0.7, 3: 0.3, 4: 0.5}
    predicted_class = int(features[0])
    features[1] = demand_map.get(predicted_class, 0.5)

    return features


# ─── Feature Fusion ───────────────────────────────────────────────────────────

def fuse_features(vision_feats: np.ndarray, audio_feats: np.ndarray,
                  nlp_feats: np.ndarray) -> np.ndarray:
    """
    Fuse features from all 3 modalities into an 11-dimensional vector
    for the meta-classifier.
    
    Vision (4) + Audio-derived (4) + NLP (3) = 11 features
    """
    # From audio 52-dim vector, extract the 4 meta-features
    audio_meta = np.zeros(4, dtype=np.float32)
    audio_meta[0] = 2.0  # speech_class placeholder (set by audio model)
    audio_meta[1] = 0.8  # speech_confidence placeholder
    audio_meta[2] = audio_feats[49] / 220.0  # wpm_norm
    voiced = 1.0 - audio_feats[51]  # fluency = inverse of silence
    audio_meta[3] = voiced

    return np.concatenate([vision_feats, audio_meta, nlp_feats]).astype(np.float32)
