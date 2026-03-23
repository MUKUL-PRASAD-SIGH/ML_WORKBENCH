"""
STEP 4 — Transformer Embeddings (DistilBERT)
=============================================
Generates dense semantic embeddings for each document using DistilBERT.
Uses mean-pooling over the last hidden state as the sentence vector.
Processes in mini-batches for speed and saves embeddings as numpy arrays.
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
BATCH_SIZE  = 16
MAX_LENGTH  = 256          # truncate long reviews
NUM_SAMPLES = 2000         # match step 1
OUTPUT_DIR  = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥  Device: {DEVICE}")

# ─────────────────────────────────────────────
# Load data & model
# ─────────────────────────────────────────────
print("📂 Loading raw data...")
with open(os.path.join(OUTPUT_DIR, "raw_data.json"), "r", encoding="utf-8") as f:
    data = json.load(f)
texts = data["texts"][:NUM_SAMPLES]

print(f"🤖 Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ─────────────────────────────────────────────
# Embedding function (batched)
# ─────────────────────────────────────────────
def mean_pool(last_hidden_state, attention_mask):
    """Masked mean-pooling of the last hidden state."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def get_embeddings_batch(batch_texts: list[str]) -> np.ndarray:
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    vecs = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return vecs.cpu().numpy()

# ─────────────────────────────────────────────
# Process all texts
# ─────────────────────────────────────────────
print(f"\n🔢 Generating embeddings for {NUM_SAMPLES} documents...")
all_embeddings = []

for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
    batch = texts[start : start + BATCH_SIZE]
    emb   = get_embeddings_batch(batch)
    all_embeddings.append(emb)

embeddings = np.vstack(all_embeddings)
print(f"\n✅ Embeddings shape: {embeddings.shape}  (samples × hidden_dim)")

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, "bert_embeddings.npy"), embeddings)
print(f"💾 Embeddings saved → {OUTPUT_DIR}/bert_embeddings.npy")
