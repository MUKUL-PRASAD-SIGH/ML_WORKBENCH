"""
STEP 5 — Combine BERT + LDA Features (Hybrid Representation)
==============================================================
Concatenates BERT semantic embeddings with LDA topic distributions to
create a unified hybrid feature vector per document.

Final vector = [BERT(768) | LDA(5)] = 773-dimensional hybrid representation.
"""

import os
import json
import numpy as np

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load both feature sources
# ─────────────────────────────────────────────
print("📂 Loading BERT embeddings...")
bert_embeddings = np.load(os.path.join(OUTPUT_DIR, "bert_embeddings.npy"))

print("📂 Loading LDA topic distributions...")
with open(os.path.join(OUTPUT_DIR, "topic_distributions.json"), "r") as f:
    topic_dists = json.load(f)
lda_vectors = np.array(topic_dists)

print("📂 Loading labels...")
with open(os.path.join(OUTPUT_DIR, "raw_data.json"), "r") as f:
    data = json.load(f)
labels = np.array(data["labels"])

# ─────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────
assert bert_embeddings.shape[0] == lda_vectors.shape[0], "Sample count mismatch!"
print(f"\n📐 BERT embeddings shape : {bert_embeddings.shape}")
print(f"📐 LDA vectors shape     : {lda_vectors.shape}")

# ─────────────────────────────────────────────
# Normalize before concatenation
# ─────────────────────────────────────────────
def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms

bert_norm = l2_normalize(bert_embeddings)
lda_norm  = l2_normalize(lda_vectors)      # already sums-to-1, but normalise for scale

# ─────────────────────────────────────────────
# Concatenate → Hybrid features
# ─────────────────────────────────────────────
hybrid_features = np.concatenate([bert_norm, lda_norm], axis=1)
print(f"\n🔗 Hybrid feature vector : {hybrid_features.shape}")
print(   "   = BERT({}) + LDA({}) = {}".format(
    bert_norm.shape[1], lda_norm.shape[1], hybrid_features.shape[1]))

# ─────────────────────────────────────────────
# Similarity demo: top-3 most similar to doc 0
# ─────────────────────────────────────────────
from sklearn.metrics.pairwise import cosine_similarity

print("\n🔍 Similarity demo — top-3 most similar docs to doc[0]:")
query = hybrid_features[0:1]
all_sims = cosine_similarity(query, hybrid_features)[0]
top_idx  = np.argsort(all_sims)[::-1][1:4]   # skip self (idx 0)
for rank, idx in enumerate(top_idx, 1):
    print(f"  Rank {rank} → doc[{idx}]  sim={all_sims[idx]:.4f}  label={labels[idx]}")

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, "hybrid_features.npy"), hybrid_features)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"),          labels)

print(f"\n💾 Hybrid features → {OUTPUT_DIR}/hybrid_features.npy")
print(f"💾 Labels          → {OUTPUT_DIR}/labels.npy")
print(f"\n✅ Combining step complete! Hybrid dim = {hybrid_features.shape[1]}")
