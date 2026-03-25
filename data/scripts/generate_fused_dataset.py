"""
Generate fused dataset for the meta-classifier.
Combines vision, audio, and NLP model outputs into 11-feature vectors
with simulated self-reported flow state labels.

Flow State Labels:
  0: PSEUDO_WORKING  1: TASK_SWITCHING  2: DISTRACTED  3: SOFT_FLOW  4: DEEP_FLOW
"""
import csv, os, random

FEATURE_COLS = [
    "tab_count_norm", "phone_visible", "distraction_count_norm", "focus_ratio",
    "speech_class", "speech_confidence", "wpm_norm", "fluency_score",
    "task_class_encoded", "cognitive_demand_score", "task_confidence"
]

# (mean, std) for each feature per flow state
PROFILES = {
    0: {"name":"PSEUDO_WORKING",  "p":[(0.7,0.15),(0.5,0.3),(0.4,0.15),(0.5,0.1),(0,1.2),(0.5,0.15),(0.5,0.2),(0.4,0.15),(1,1.0),(0.3,0.15),(0.4,0.15)]},
    1: {"name":"TASK_SWITCHING",   "p":[(0.85,0.1),(0.3,0.25),(0.3,0.1),(0.4,0.12),(3,1.0),(0.6,0.15),(0.75,0.15),(0.5,0.15),(3,1.2),(0.45,0.15),(0.5,0.15)]},
    2: {"name":"DISTRACTED",       "p":[(0.6,0.2),(0.8,0.15),(0.7,0.15),(0.3,0.1),(0,1.5),(0.4,0.15),(0.4,0.2),(0.3,0.15),(4,1.0),(0.2,0.1),(0.3,0.15)]},
    3: {"name":"SOFT_FLOW",        "p":[(0.4,0.12),(0.1,0.15),(0.15,0.1),(0.7,0.1),(2,0.8),(0.75,0.12),(0.6,0.1),(0.7,0.1),(0,1.0),(0.65,0.12),(0.7,0.12)]},
    4: {"name":"DEEP_FLOW",        "p":[(0.2,0.1),(0.02,0.05),(0.05,0.05),(0.85,0.08),(2,0.5),(0.85,0.08),(0.55,0.08),(0.85,0.08),(0,0.5),(0.85,0.08),(0.85,0.08)]},
}

def gen_sample(fs, rng):
    profile = PROFILES[fs]
    sample = {}
    for i, feat in enumerate(FEATURE_COLS):
        m, s = profile["p"][i]
        v = rng.gauss(m, s)
        if feat == "phone_visible":
            v = 1 if v > 0.5 else 0
        elif feat in ("speech_class", "task_class_encoded"):
            v = int(max(0, min(4, round(v))))
        else:
            v = round(max(0.0, min(1.0, v)), 4)
        sample[feat] = v
    sample["flow_state_label"] = fs
    sample["flow_state_name"] = profile["name"]
    return sample

def main():
    rng = random.Random(42)
    samples = []
    for fs in range(5):
        for _ in range(200):
            samples.append(gen_sample(fs, rng))
    rng.shuffle(samples)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fused_flow_dataset.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_COLS+["flow_state_label","flow_state_name"])
        w.writeheader()
        w.writerows(samples)

    print(f"Generated {len(samples)} fused samples -> {out_path}")
    from collections import Counter
    for lbl, cnt in sorted(Counter(s["flow_state_name"] for s in samples).items()):
        print(f"  {lbl}: {cnt}")

if __name__ == "__main__":
    main()
