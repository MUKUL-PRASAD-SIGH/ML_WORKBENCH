# Topic-Aware Semantic AI System

> **Advanced Hybrid NLP** — Gensim LDA + DistilBERT Transformers

---

## 🎯 What This Project Does

Builds a **Hybrid NLP System** that combines two complementary views of text:

| Component | Tool | What It Captures |
|-----------|------|-----------------|
| **Semantic Embeddings** | DistilBERT (Transformers) | Deep contextual meaning |
| **Topic Distributions** | LDA (Gensim) | Interpretable topic structure |
| **Hybrid Vector** | Concatenation | Both meaning + topics |

---

## 🧱 Architecture

```
Raw Text (IMDB)
      │
      ├──► Preprocessing (Gensim) ──► BoW Corpus ──► LDA Model
      │                                                    │
      │                                              Topic Vector (5-dim)
      │                                                    │
      └──► Tokenizer ──► DistilBERT ──► Mean Pooling       │
                                              │             │
                                    BERT Vector (768-dim)   │
                                              │             │
                                              ▼             ▼
                                         Concatenate (L2 normalised)
                                              │
                                     Hybrid Vector (773-dim)
                                              │
                              ┌───────────────┴─────────────────┐
                              │                                   │
                        Cosine Similarity            (Future: Classifier)
                        Semantic Search
```

---

## 📂 Project Structure

```
Topic-Aware-AI/
├── step1_load_data.py          # Load 2000 IMDB reviews
├── step2_preprocess.py         # Tokenise + remove stop-words
├── step3_lda_topics.py         # Train LDA (5 topics)
├── step4_bert_embeddings.py    # Generate DistilBERT embeddings (batched)
├── step5_combine_features.py   # Fuse → 773-dim hybrid vector
├── step6_similarity_search.py  # Semantic search engine
├── run_pipeline.py             # Master runner (steps 1–5)
├── requirements.txt            # Dependencies
├── data/                       # Auto-created: raw + processed data
└── models/                     # Auto-created: LDA model + dictionary
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python run_pipeline.py
```

This runs Steps 1–5 end-to-end (~10–20 min depending on hardware).

### 3. Run Semantic Search

```bash
python step6_similarity_search.py --query "a thrilling action movie"
python step6_similarity_search.py --query "romantic comedy with happy ending" --top_k 3
```

---

## 📋 Step-by-Step Guide

### Step 1 — Load Dataset
```bash
python step1_load_data.py
```
Downloads IMDB from HuggingFace Hub, saves `data/raw_data.json`.

### Step 2 — Preprocess
```bash
python step2_preprocess.py
```
Applies Gensim `simple_preprocess` + stop-word removal → `data/processed_tokens.json`

### Step 3 — LDA Topics
```bash
python step3_lda_topics.py
```
Trains LDA (5 topics, 10 passes), prints coherence score → `models/lda_model.gensim`

### Step 4 — BERT Embeddings
```bash
python step4_bert_embeddings.py
```
Batch-embeds all 2000 reviews with DistilBERT → `data/bert_embeddings.npy`

### Step 5 — Combine
```bash
python step5_combine_features.py
```
L2-normalises and concatenates both → `data/hybrid_features.npy`

### Step 6 — Search
```bash
python step6_similarity_search.py --query "your query here"
```
Finds the top-k most similar reviews using cosine similarity on hybrid vectors.

---

## 🧠 Key Concepts

### Why Hybrid?

- **BERT alone** → great meaning, but treats every topic the same
- **LDA alone** → interpretable topics, but misses nuance
- **Combined** → richer representation that captures both aspects

### Hybrid Vector Formula

```
hybrid(doc) = L2_norm(BERT_embed(doc)) ⊕ L2_norm(LDA_dist(doc))
```

### LDA Topics (example output)

```
Topic 0: "good time great performance acting role"
Topic 1: "horror scary suspense kill blood"
Topic 2: "funny comedy laugh hilarious joke"
Topic 3: "story plot character drama life"
Topic 4: "action war battle fight adventure"
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | IMDB (HuggingFace Datasets) |
| Split used | Train set |
| Samples | 2,000 |
| Classes | Positive / Negative |
| Balance | 50 / 50 |

---

## 🔧 Configuration

| Parameter | Default | Location |
|-----------|---------|----------|
| Num LDA topics | 5 | `step3_lda_topics.py` |
| LDA passes | 10 | `step3_lda_topics.py` |
| BERT model | `distilbert-base-uncased` | `step4_bert_embeddings.py` |
| Batch size | 16 | `step4_bert_embeddings.py` |
| Max sequence length | 256 | `step4_bert_embeddings.py` |
| Num samples | 2,000 | `step1_load_data.py` |

---

## ⚡ Performance Tips

- **GPU**: If CUDA is available, Step 4 runs 5–10× faster automatically.
- **Reduce `NUM_SAMPLES`** in `step1_load_data.py` to 500 for a quick test run.
- **Increase `NUM_TOPICS`** to 10–15 for a richer topic model.

---

## 🗺️ What's Next

| Option | Feature |
|--------|---------|
| **A** | Train a classifier using hybrid features (logistic regression / SVM) |
| **B** | Visualise topics + embeddings (UMAP / PCA plots) |
| **C** | Build a full search engine with a Streamlit UI |

---

## 📚 References

- [Gensim LDA](https://radimrehurek.com/gensim/models/ldamodel.html)
- [HuggingFace DistilBERT](https://huggingface.co/distilbert-base-uncased)
- [IMDB Dataset](https://huggingface.co/datasets/imdb)
- [Latent Dirichlet Allocation (Blei et al., 2003)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
