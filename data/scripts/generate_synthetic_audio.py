"""
Generate synthetic audio features for speech cognitive load model training.
Creates .npy arrays of 52-dimensional feature vectors matching the Librosa extraction pipeline.

Classes:
  0: ERRATIC_SPEECH   — high WPM variance, many pauses, low fluency
  1: SLOW_LABORED     — < 80 WPM, high pause rate
  2: NORMAL_FOCUSED   — 100–150 WPM, low variance
  3: FAST_ENERGIZED   — 160–220 WPM, low variance
  4: RAPID_SCATTERED  — > 220 WPM or extremely high variance

Feature vector (52 features):
  [0-12]  MFCC means (13 coefficients)
  [13-25] MFCC stds (13 coefficients)
  [26-38] MFCC delta means (13 coefficients)
  [39-51] Actually mapped to:
    [39-41]  MFCC delta-delta means (first 3, rest zeros for simplicity in synthetic)
    [42]     Spectral centroid
    [43]     Spectral rolloff
    [44]     Zero crossing rate
    [45]     RMS energy
    [46]     Pitch mean (F0)
    [47]     Pitch variance
    [48]     Tempo (BPM of speech rhythm)
    [49]     WPM mean
    [50]     WPM variance
    [51]     Silence ratio
"""

import numpy as np
import os


# ─── Class-Specific Feature Distributions ──────────────────────────────────────

# Each class has characteristic audio signatures.
# We model each feature dimension as (mean, std) per class.

CLASS_PROFILES = {
    0: {  # ERRATIC_SPEECH
        "name": "ERRATIC_SPEECH",
        "mfcc_mean_base": -200, "mfcc_mean_spread": 80,
        "mfcc_std_base": 60, "mfcc_std_spread": 30,
        "spectral_centroid": (2800, 600),
        "spectral_rolloff": (5500, 1000),
        "zcr": (0.12, 0.04),
        "rms": (0.06, 0.025),
        "pitch_mean": (180, 50),
        "pitch_var": (1200, 400),      # HIGH variance = erratic
        "tempo": (90, 30),
        "wpm_mean": (130, 40),
        "wpm_var": (900, 300),         # HIGH WPM variance
        "silence_ratio": (0.35, 0.10), # Many pauses
    },
    1: {  # SLOW_LABORED
        "name": "SLOW_LABORED",
        "mfcc_mean_base": -250, "mfcc_mean_spread": 60,
        "mfcc_std_base": 40, "mfcc_std_spread": 20,
        "spectral_centroid": (1800, 400),
        "spectral_rolloff": (3800, 800),
        "zcr": (0.06, 0.02),
        "rms": (0.03, 0.01),
        "pitch_mean": (140, 30),
        "pitch_var": (200, 100),
        "tempo": (60, 15),
        "wpm_mean": (65, 15),          # SLOW
        "wpm_var": (100, 50),
        "silence_ratio": (0.50, 0.12), # Lots of pauses
    },
    2: {  # NORMAL_FOCUSED
        "name": "NORMAL_FOCUSED",
        "mfcc_mean_base": -180, "mfcc_mean_spread": 50,
        "mfcc_std_base": 45, "mfcc_std_spread": 15,
        "spectral_centroid": (2200, 300),
        "spectral_rolloff": (4500, 600),
        "zcr": (0.08, 0.02),
        "rms": (0.05, 0.015),
        "pitch_mean": (170, 25),
        "pitch_var": (300, 100),       # LOW variance = steady
        "tempo": (110, 15),
        "wpm_mean": (125, 15),         # Normal range
        "wpm_var": (80, 30),           # LOW WPM variance = consistent
        "silence_ratio": (0.18, 0.05),
    },
    3: {  # FAST_ENERGIZED
        "name": "FAST_ENERGIZED",
        "mfcc_mean_base": -160, "mfcc_mean_spread": 55,
        "mfcc_std_base": 55, "mfcc_std_spread": 20,
        "spectral_centroid": (3000, 400),
        "spectral_rolloff": (5800, 700),
        "zcr": (0.10, 0.03),
        "rms": (0.08, 0.02),
        "pitch_mean": (210, 35),
        "pitch_var": (400, 150),
        "tempo": (140, 20),
        "wpm_mean": (185, 20),         # Fast
        "wpm_var": (100, 40),          # Low variance = consistent speed
        "silence_ratio": (0.10, 0.04),
    },
    4: {  # RAPID_SCATTERED
        "name": "RAPID_SCATTERED",
        "mfcc_mean_base": -150, "mfcc_mean_spread": 70,
        "mfcc_std_base": 65, "mfcc_std_spread": 25,
        "spectral_centroid": (3200, 500),
        "spectral_rolloff": (6200, 800),
        "zcr": (0.14, 0.04),
        "rms": (0.09, 0.03),
        "pitch_mean": (230, 45),
        "pitch_var": (1500, 500),      # HIGH variance
        "tempo": (160, 25),
        "wpm_mean": (240, 30),         # Very fast
        "wpm_var": (1100, 350),        # HIGH WPM variance
        "silence_ratio": (0.08, 0.03),
    },
}


def generate_sample(class_id: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a single 52-dimensional feature vector for a given class."""
    profile = CLASS_PROFILES[class_id]
    features = np.zeros(52, dtype=np.float32)

    # MFCCs means (features 0-12) — 13 coefficients
    base = profile["mfcc_mean_base"]
    spread = profile["mfcc_mean_spread"]
    for i in range(13):
        # Higher-order MFCCs tend toward zero
        coeff_scale = 1.0 / (1.0 + 0.3 * i)
        features[i] = rng.normal(base * coeff_scale, spread * coeff_scale)

    # MFCC stds (features 13-25)
    std_base = profile["mfcc_std_base"]
    std_spread = profile["mfcc_std_spread"]
    for i in range(13):
        coeff_scale = 1.0 / (1.0 + 0.2 * i)
        features[13 + i] = abs(rng.normal(std_base * coeff_scale, std_spread * coeff_scale))

    # MFCC delta means (features 26-38)
    for i in range(13):
        features[26 + i] = rng.normal(0, spread * 0.3 / (1.0 + 0.3 * i))

    # MFCC delta-delta means (features 39-41, rest are other features)
    for i in range(3):
        features[39 + i] = rng.normal(0, spread * 0.15)

    # Spectral centroid (42)
    m, s = profile["spectral_centroid"]
    features[42] = max(0, rng.normal(m, s))

    # Spectral rolloff (43)
    m, s = profile["spectral_rolloff"]
    features[43] = max(0, rng.normal(m, s))

    # Zero crossing rate (44)
    m, s = profile["zcr"]
    features[44] = max(0, rng.normal(m, s))

    # RMS energy (45)
    m, s = profile["rms"]
    features[45] = max(0, rng.normal(m, s))

    # Pitch mean (46)
    m, s = profile["pitch_mean"]
    features[46] = max(50, rng.normal(m, s))

    # Pitch variance (47)
    m, s = profile["pitch_var"]
    features[47] = max(0, rng.normal(m, s))

    # Tempo (48)
    m, s = profile["tempo"]
    features[48] = max(30, rng.normal(m, s))

    # WPM mean (49)
    m, s = profile["wpm_mean"]
    features[49] = max(20, rng.normal(m, s))

    # WPM variance (50)
    m, s = profile["wpm_var"]
    features[50] = max(0, rng.normal(m, s))

    # Silence ratio (51)
    m, s = profile["silence_ratio"]
    features[51] = np.clip(rng.normal(m, s), 0, 1)

    return features


def main():
    rng = np.random.default_rng(42)
    samples_per_class = 300
    num_classes = 5

    total = samples_per_class * num_classes
    X = np.zeros((total, 52), dtype=np.float32)
    y = np.zeros(total, dtype=np.int64)

    idx = 0
    for class_id in range(num_classes):
        for _ in range(samples_per_class):
            X[idx] = generate_sample(class_id, rng)
            y[idx] = class_id
            idx += 1

    # Shuffle
    perm = rng.permutation(total)
    X = X[perm]
    y = y[perm]

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "..", "processed")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "audio_features.npy"), X)
    np.save(os.path.join(output_dir, "audio_labels.npy"), y)

    print(f"✅ Generated {total} audio feature vectors (52-dim) → {output_dir}")
    print(f"   Shape: X={X.shape}, y={y.shape}")
    for class_id in range(num_classes):
        name = CLASS_PROFILES[class_id]["name"]
        count = np.sum(y == class_id)
        print(f"   Class {class_id} ({name}): {count} samples")


if __name__ == "__main__":
    main()
