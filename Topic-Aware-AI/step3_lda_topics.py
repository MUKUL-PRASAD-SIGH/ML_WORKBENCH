"""
STEP 3 — Build LDA Topic Model (Gensim)
=========================================
Trains a Latent Dirichlet Allocation model on the preprocessed tokens.
Saves the dictionary, corpus, model, and per-document topic distributions.
"""

import os
import json
import pickle
import numpy as np
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from tqdm import tqdm

NUM_TOPICS  = 5
PASSES      = 10
RANDOM_SEED = 42
OUTPUT_DIR  = "data"
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load preprocessed tokens
# ─────────────────────────────────────────────
print("📂 Loading preprocessed tokens...")
with open(os.path.join(OUTPUT_DIR, "processed_tokens.json"), "r") as f:
    processed = json.load(f)

# ─────────────────────────────────────────────
# Build Dictionary & Corpus
# ─────────────────────────────────────────────
print("📚 Building dictionary and BoW corpus...")
dictionary = corpora.Dictionary(processed)
dictionary.filter_extremes(no_below=5, no_above=0.7)   # filter rare & very common words
corpus = [dictionary.doc2bow(doc) for doc in processed]

print(f"   Dictionary size : {len(dictionary)} unique tokens")
print(f"   Corpus size     : {len(corpus)} documents")

# ─────────────────────────────────────────────
# Train LDA
# ─────────────────────────────────────────────
print(f"\n🧠 Training LDA with {NUM_TOPICS} topics ({PASSES} passes)...")
lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    passes=PASSES,
    random_state=RANDOM_SEED,
    alpha="auto",
    eta="auto",
)

# ─────────────────────────────────────────────
# Coherence Score
# ─────────────────────────────────────────────
coherence_model = CoherenceModel(model=lda, texts=processed, dictionary=dictionary, coherence="c_v")
score = coherence_model.get_coherence()
print(f"\n📈 Coherence Score (c_v): {score:.4f}  (higher = better topics)")

# ─────────────────────────────────────────────
# Print Topics
# ─────────────────────────────────────────────
print("\n🗂  Discovered Topics:")
print("─" * 60)
for idx, topic in lda.print_topics(num_words=10):
    print(f"  Topic {idx}: {topic}\n")

# ─────────────────────────────────────────────
# Per-document topic distributions
# ─────────────────────────────────────────────
print("🔢 Computing topic distributions for every document...")
topic_distributions = []
for bow in tqdm(corpus, desc="Getting topic vectors"):
    dist = lda.get_document_topics(bow, minimum_probability=0.0)
    vec  = np.array([prob for _, prob in sorted(dist, key=lambda x: x[0])])
    topic_distributions.append(vec.tolist())

# ─────────────────────────────────────────────
# Save everything
# ─────────────────────────────────────────────
dictionary.save(os.path.join(MODELS_DIR, "lda_dictionary.gensim"))
lda.save(os.path.join(MODELS_DIR, "lda_model.gensim"))

with open(os.path.join(OUTPUT_DIR, "topic_distributions.json"), "w") as f:
    json.dump(topic_distributions, f)

print(f"\n💾 LDA model saved  → {MODELS_DIR}/lda_model.gensim")
print(f"💾 Topic dists      → {OUTPUT_DIR}/topic_distributions.json")
print(f"\n✅ LDA step complete! Coherence: {score:.4f}")
