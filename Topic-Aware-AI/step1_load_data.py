"""
STEP 1 — Load Real Dataset (IMDB)
==================================
Load 2000 training samples from the IMDB movie review dataset.
Saves texts and labels to disk for reuse in subsequent steps.
"""

import os
import json
from datasets import load_dataset

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
NUM_SAMPLES = 2000
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────
print("📦 Loading IMDB dataset...")
dataset = load_dataset("imdb")

texts  = dataset["train"]["text"][:NUM_SAMPLES]
labels = dataset["train"]["label"][:NUM_SAMPLES]    # 0 = negative, 1 = positive

print(f"✅ Loaded {len(texts)} samples")
print(f"   Positive: {sum(labels)}  |  Negative: {NUM_SAMPLES - sum(labels)}")

# ─────────────────────────────────────────────
# Preview
# ─────────────────────────────────────────────
print("\n📝 Sample review (first 300 chars):")
print(texts[0][:300])
print(f"\n🏷  Label: {'Positive ✅' if labels[0] == 1 else 'Negative ❌'}")

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "raw_data.json"), "w", encoding="utf-8") as f:
    json.dump({"texts": list(texts), "labels": list(labels)}, f, ensure_ascii=False, indent=2)

print(f"\n💾 Data saved → {OUTPUT_DIR}/raw_data.json")
