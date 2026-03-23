"""
CODE DUMP — Topic-Aware AI System (All Steps)
===============================================
Complete reference of all source code in one file.
Auto-updated when new features are added.
Last updated: 2026-03-23
"""

# ============================================================
# STEP 1 — Load IMDB Dataset
# ============================================================
STEP1 = '''
from datasets import load_dataset
import json, os

dataset = load_dataset("imdb")
texts  = dataset["train"]["text"][:2000]
labels = dataset["train"]["label"][:2000]

os.makedirs("data", exist_ok=True)
with open("data/raw_data.json", "w", encoding="utf-8") as f:
    json.dump({"texts": list(texts), "labels": list(labels)}, f)
'''

# ============================================================
# STEP 2 — Preprocess
# ============================================================
STEP2 = '''
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import json

with open("data/raw_data.json") as f:
    data = json.load(f)

STOP = STOPWORDS.union({"film","movie","br","one","would","could","like"})

processed = [
    [t for t in simple_preprocess(text, deacc=True)
     if t not in STOP and len(t) > 2]
    for text in data["texts"]
]

with open("data/processed_tokens.json", "w") as f:
    json.dump(processed, f)
'''

# ============================================================
# STEP 3 — LDA
# ============================================================
STEP3 = '''
from gensim import corpora
from gensim.models import LdaModel
import json, numpy as np

with open("data/processed_tokens.json") as f:
    processed = json.load(f)

dictionary = corpora.Dictionary(processed)
dictionary.filter_extremes(no_below=5, no_above=0.7)
corpus     = [dictionary.doc2bow(doc) for doc in processed]

lda = LdaModel(corpus=corpus, id2word=dictionary,
               num_topics=5, passes=10, random_state=42,
               alpha="auto", eta="auto")

# per-doc topic distributions
dists = []
for bow in corpus:
    d   = lda.get_document_topics(bow, minimum_probability=0.0)
    vec = np.array([prob for _, prob in sorted(d, key=lambda x: x[0])])
    dists.append(vec.tolist())

dictionary.save("models/lda_dictionary.gensim")
lda.save("models/lda_model.gensim")
with open("data/topic_distributions.json", "w") as f:
    json.dump(dists, f)
'''

# ============================================================
# STEP 4 — BERT Embeddings
# ============================================================
STEP4 = '''
from transformers import AutoTokenizer, AutoModel
import torch, numpy as np, json

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model     = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

with open("data/raw_data.json") as f:
    texts = json.load(f)["texts"]

def embed_batch(batch):
    inp = tokenizer(batch, return_tensors="pt",
                    truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        out = model(**inp)
    mask = inp["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
    return ((out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(1e-9)).numpy()

embeddings = np.vstack([embed_batch(texts[i:i+16]) for i in range(0, len(texts), 16)])
np.save("data/bert_embeddings.npy", embeddings)
'''

# ============================================================
# STEP 5 — Combine
# ============================================================
STEP5 = '''
import numpy as np, json

bert = np.load("data/bert_embeddings.npy")
with open("data/topic_distributions.json") as f:
    lda = np.array(json.load(f))

def l2(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

hybrid = np.concatenate([l2(bert), l2(lda)], axis=1)  # (2000, 773)
np.save("data/hybrid_features.npy", hybrid)

with open("data/raw_data.json") as f:
    labels = np.array(json.load(f)["labels"])
np.save("data/labels.npy", labels)
'''

# ============================================================
# STEP 6 — Similarity Search
# ============================================================
STEP6 = '''
# python step6_similarity_search.py --query "your query here"
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

hybrid = np.load("data/hybrid_features.npy")
query_vec = make_hybrid(query_text)          # uses BERT + LDA helpers
sims      = cosine_similarity([query_vec], hybrid)[0]
top_k_idx = np.argsort(sims)[::-1][:5]
'''
