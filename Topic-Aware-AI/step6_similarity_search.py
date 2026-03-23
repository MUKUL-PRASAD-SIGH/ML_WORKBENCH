"""
STEP 6 — Similarity & Search System
=====================================
Implements a mini semantic search engine over the IMDB corpus using
the hybrid (BERT + LDA) feature vectors.

Usage:
    python step6_similarity_search.py --query "a wonderful romantic comedy"
"""

import os
import json
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora
from gensim.models import LdaModel
from transformers import AutoTokenizer, AutoModel
import torch

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
OUTPUT_DIR = "data"
MODELS_DIR = "models"

# ─────────────────────────────────────────────
# Load pre-built assets
# ─────────────────────────────────────────────
def load_assets():
    print("📂 Loading corpus and models...")

    with open(os.path.join(OUTPUT_DIR, "raw_data.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus_texts   = data["texts"]
    corpus_labels  = data["labels"]
    hybrid_feats   = np.load(os.path.join(OUTPUT_DIR, "hybrid_features.npy"))

    dictionary = corpora.Dictionary.load(os.path.join(MODELS_DIR, "lda_dictionary.gensim"))
    lda        = LdaModel.load(os.path.join(MODELS_DIR, "lda_model.gensim"))

    tokenizer  = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model      = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    print(f"✅ Loaded {len(corpus_texts)} documents, hybrid dim = {hybrid_feats.shape[1]}")
    return corpus_texts, corpus_labels, hybrid_feats, dictionary, lda, tokenizer, model


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
CUSTOM_STOP = STOPWORDS.union({"film", "movie", "br", "one", "would", "could", "like"})

def get_bert_embedding(text, tokenizer, model, max_len=256):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    with torch.no_grad():
        out = model(**inputs)
    mask  = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
    vec   = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return vec.squeeze().numpy()


def get_lda_vector(text, dictionary, lda, num_topics=5):
    tokens = [t for t in simple_preprocess(text, deacc=True)
              if t not in CUSTOM_STOP and len(t) > 2]
    bow    = dictionary.doc2bow(tokens)
    dist   = lda.get_document_topics(bow, minimum_probability=0.0)
    vec    = np.zeros(num_topics)
    for tid, prob in dist:
        vec[tid] = prob
    return vec


def make_hybrid(text, dictionary, lda, tokenizer, model, num_topics=5):
    bert = get_bert_embedding(text, tokenizer, model)
    bert = bert / (np.linalg.norm(bert) + 1e-9)

    lda_v = get_lda_vector(text, dictionary, lda, num_topics)
    lda_v = lda_v / (np.linalg.norm(lda_v) + 1e-9)

    return np.concatenate([bert, lda_v])


# ─────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────
def search(query, corpus_texts, corpus_labels, hybrid_feats,
           dictionary, lda, tokenizer, model, top_k=5):
    print(f"\n🔍 Query: \"{query}\"")
    print("─" * 60)

    query_vec = make_hybrid(query, dictionary, lda, tokenizer, model)
    sims      = cosine_similarity([query_vec], hybrid_feats)[0]
    top_idx   = np.argsort(sims)[::-1][:top_k]

    label_map = {0: "Negative ❌", 1: "Positive ✅"}
    for rank, idx in enumerate(top_idx, 1):
        snippet = corpus_texts[idx][:200].replace("\n", " ")
        print(f"\n  [{rank}] Similarity: {sims[idx]:.4f}  |  {label_map[corpus_labels[idx]]}")
        print(f"       {snippet}...")

    return top_idx, sims[top_idx]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid semantic search over IMDB")
    parser.add_argument("--query", type=str,
                        default="a thrilling action movie with great visual effects",
                        help="Query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    texts, labels, hybrid_feats, dictionary, lda, tokenizer, model = load_assets()
    search(args.query, texts, labels, hybrid_feats, dictionary, lda, tokenizer, model, args.top_k)
