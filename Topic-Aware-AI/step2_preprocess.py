"""
STEP 2 — Preprocess Text (for Gensim / LDA)
=============================================
Clean and tokenize the raw texts using Gensim's simple_preprocess.
Removes stop-words and short tokens, then saves processed tokens.
"""

import os
import json
import nltk
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from tqdm import tqdm

nltk.download("stopwords", quiet=True)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load raw data
# ─────────────────────────────────────────────
print("📂 Loading raw data...")
with open(os.path.join(OUTPUT_DIR, "raw_data.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

texts  = data["texts"]
labels = data["labels"]

# ─────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────
CUSTOM_STOP = STOPWORDS.union({"film", "movie", "br", "one", "would", "could", "like"})

def preprocess(text: str) -> list[str]:
    """Tokenise and remove stop-words."""
    return [
        token for token in simple_preprocess(text, deacc=True)
        if token not in CUSTOM_STOP and len(token) > 2
    ]

print("🧹 Preprocessing texts...")
processed = [preprocess(t) for t in tqdm(texts, desc="Tokenising")]

# ─────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────
avg_len = sum(len(p) for p in processed) / len(processed)
print(f"\n📊 Average tokens per doc : {avg_len:.1f}")
print(f"📝 Sample tokens (doc 0)  : {processed[0][:15]}")

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "processed_tokens.json"), "w", encoding="utf-8") as f:
    json.dump(processed, f)

print(f"\n💾 Processed tokens saved → {OUTPUT_DIR}/processed_tokens.json")
