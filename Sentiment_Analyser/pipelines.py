"""
pipelines.py  ─  All inference + training logic for the NLP Dashboard
"""
from __future__ import annotations
import os, re, time, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── global handles ─────────────────────────────────────────────────────────
_model_nltk  = None
_vec_nltk    = None
_nlp_spacy   = None
_model_spacy = None
_vec_spacy   = None
_classifier  = None

_nltk_ready  = False
_spacy_ready = False
_trans_ready = False

# ═══════════════════════════════════════════════════════════════════════════
# LOADERS
# ═══════════════════════════════════════════════════════════════════════════
def load_nltk_pipeline() -> bool:
    global _model_nltk, _vec_nltk, _nltk_ready
    if _nltk_ready: return True
    mp = os.path.join(MODELS_DIR, "nltk_model.pkl")
    vp = os.path.join(MODELS_DIR, "nltk_vectorizer.pkl")
    if not (os.path.exists(mp) and os.path.exists(vp)): return False
    _model_nltk = joblib.load(mp)
    _vec_nltk   = joblib.load(vp)
    import nltk
    for r in ("punkt", "stopwords", "punkt_tab"):
        nltk.download(r, quiet=True)
    _nltk_ready = True
    return True

def load_spacy_pipeline() -> bool:
    global _nlp_spacy, _model_spacy, _vec_spacy, _spacy_ready
    if _spacy_ready: return True
    mp = os.path.join(MODELS_DIR, "spacy_model.pkl")
    vp = os.path.join(MODELS_DIR, "spacy_vectorizer.pkl")
    if not (os.path.exists(mp) and os.path.exists(vp)): return False
    import spacy
    _nlp_spacy   = spacy.load("en_core_web_sm", disable=["parser","ner"])
    _model_spacy = joblib.load(mp)
    _vec_spacy   = joblib.load(vp)
    _spacy_ready = True
    return True

def load_transformer_pipeline() -> bool:
    global _classifier, _trans_ready
    if _trans_ready: return True
    from transformers import pipeline
    _classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1, truncation=True, max_length=512,
    )
    _trans_ready = True
    return True

# ═══════════════════════════════════════════════════════════════════════════
# STEP-BY-STEP PREPROCESSORS (returns each intermediate stage)
# ═══════════════════════════════════════════════════════════════════════════
def nltk_steps(text: str) -> dict:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    sw = set(stopwords.words("english"))

    step1 = text
    step2 = text.lower()
    step3 = re.sub(r"[^a-z\s]", "", step2)
    step4 = word_tokenize(step3)
    removed_stops  = [t for t in step4 if t in sw or len(t) <= 1]
    step5 = [t for t in step4 if t not in sw and len(t) > 1]
    final = " ".join(step5)
    return {
        "raw":          step1,
        "lowercased":   step2,
        "punct_removed":step3,
        "tokenized":    step4,
        "stops_removed":step5,
        "removed_words":removed_stops,
        "final":        final,
    }

def spacy_steps(text: str) -> dict:
    doc    = _nlp_spacy(text.lower())
    tokens_raw      = [tok.text        for tok in doc]
    lemmas_all      = [(tok.text, tok.lemma_) for tok in doc]
    removed_stops   = [tok.text for tok in doc if tok.is_stop or tok.is_punct or len(tok.text)<=1]
    kept            = [tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_punct and len(tok.text)>1]
    return {
        "raw":          text,
        "lowercased":   text.lower(),
        "tokenized":    tokens_raw,
        "lemma_pairs":  lemmas_all,
        "removed_words":removed_stops,
        "stops_removed":kept,
        "final":        " ".join(kept),
    }

def bert_steps(text: str) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    enc    = tok(text, return_tensors="pt", truncation=True, max_length=512)
    ids    = enc["input_ids"][0].tolist()
    tokens = tok.convert_ids_to_tokens(ids)
    return {
        "raw":    text,
        "tokens": tokens,
        "ids":    ids,
        "n_tokens": len(tokens),
    }

# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE WITH EXPLANATIONS
# ═══════════════════════════════════════════════════════════════════════════
def predict_nltk(text: str) -> dict:
    t0       = time.perf_counter()
    steps    = nltk_steps(text)
    X        = _vec_nltk.transform([steps["final"]])
    pred     = _model_nltk.predict(X)[0]
    proba    = _model_nltk.predict_proba(X)[0]
    elapsed  = (time.perf_counter() - t0) * 1000

    # Top contributing TF-IDF features
    feat_names = _vec_nltk.get_feature_names_out()
    coefs      = _model_nltk.coef_[0]
    X_arr      = X.toarray()[0]
    contrib    = [(feat_names[i], float(X_arr[i] * coefs[i]))
                  for i in X_arr.nonzero()[0]]
    contrib.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "label":      "Positive" if pred == 1 else "Negative",
        "confidence": float(max(proba)),
        "proba_neg":  float(proba[0]),
        "proba_pos":  float(proba[1]),
        "time_ms":    round(elapsed, 2),
        "steps":      steps,
        "top_features": contrib[:10],
    }

def predict_spacy(text: str) -> dict:
    t0       = time.perf_counter()
    steps    = spacy_steps(text)
    X        = _vec_spacy.transform([steps["final"]])
    pred     = _model_spacy.predict(X)[0]
    proba    = _model_spacy.predict_proba(X)[0]
    elapsed  = (time.perf_counter() - t0) * 1000

    feat_names = _vec_spacy.get_feature_names_out()
    coefs      = _model_spacy.coef_[0]
    X_arr      = X.toarray()[0]
    contrib    = [(feat_names[i], float(X_arr[i] * coefs[i]))
                  for i in X_arr.nonzero()[0]]
    contrib.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "label":      "Positive" if pred == 1 else "Negative",
        "confidence": float(max(proba)),
        "proba_neg":  float(proba[0]),
        "proba_pos":  float(proba[1]),
        "time_ms":    round(elapsed, 2),
        "steps":      steps,
        "top_features": contrib[:10],
    }

def predict_transformer(text: str) -> dict:
    t0     = time.perf_counter()
    steps  = bert_steps(text)
    result = _classifier(text)[0]
    elapsed= (time.perf_counter() - t0) * 1000
    label  = result["label"].capitalize()
    score  = float(result["score"])
    return {
        "label":      label,
        "confidence": round(score, 4),
        "proba_neg":  round(1 - score if label == "Positive" else score, 4),
        "proba_pos":  round(score if label == "Positive" else 1 - score, 4),
        "time_ms":    round(elapsed, 2),
        "steps":      steps,
        "top_features": [],
    }

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING  (with step-by-step callback for live UI updates)
# ═══════════════════════════════════════════════════════════════════════════
def train_nltk_model(texts, labels, progress_cb=None):
    """
    Train the NLTK pipeline. progress_cb(step: str, pct: float) is called
    at each stage so Streamlit can update a progress bar live.
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, classification_report

    sw = set(stopwords.words("english"))

    def log(msg, pct):
        if progress_cb: progress_cb(msg, pct)

    log("⬇️  Downloading NLTK resources …", 0.05)
    for r in ("punkt", "stopwords", "punkt_tab"):
        nltk.download(r, quiet=True)

    log("🔡  Lowercasing & removing punctuation …", 0.15)
    cleaned = [re.sub(r"[^a-z\s]", "", t.lower()) for t in texts]

    log("✂️  Tokenising …", 0.25)
    tokenized = [word_tokenize(t) for t in cleaned]

    log("🚫  Removing stopwords …", 0.38)
    filtered = [" ".join(tok for tok in toks if tok not in sw and len(tok) > 1)
                for toks in tokenized]

    log("📐  Building TF-IDF matrix (20k features, 1–2 grams) …", 0.50)
    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))
    X   = vec.fit_transform(filtered)

    log(f"🏋️  Training Logistic Regression on {X.shape[0]:,} × {X.shape[1]:,} matrix …", 0.70)
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    model.fit(X, labels)

    log("📊  Evaluating on training data …", 0.88)
    preds  = model.predict(X)
    acc    = accuracy_score(labels, preds)
    report = classification_report(labels, preds,
                                   target_names=["Negative", "Positive"],
                                   output_dict=True)

    log("💾  Saving model …", 0.95)
    joblib.dump(model, os.path.join(MODELS_DIR, "nltk_model.pkl"))
    joblib.dump(vec,   os.path.join(MODELS_DIR, "nltk_vectorizer.pkl"))

    # reload into globals
    global _model_nltk, _vec_nltk, _nltk_ready
    _model_nltk  = model
    _vec_nltk    = vec
    _nltk_ready  = True

    log("✅  Done!", 1.0)
    return {"accuracy": acc, "report": report, "vocab_size": len(vec.vocabulary_)}


def train_spacy_model(texts, labels, progress_cb=None):
    import spacy
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, classification_report

    def log(msg, pct):
        if progress_cb: progress_cb(msg, pct)

    log("📦  Loading spaCy en_core_web_sm …", 0.05)
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # ── Batch lemmatisation with live progress updates ──────────────────────
    # nlp.pipe() processes texts in batches (much faster than one-by-one)
    # We fire a progress callback every UPDATE_EVERY docs so the bar moves.
    n_texts      = len(texts)
    UPDATE_EVERY = 200          # callback frequency
    processed    = []

    log(f"🔤  Lemmatising {n_texts:,} texts in batches (batch_size=256) …", 0.25)
    for i, doc in enumerate(
        nlp.pipe([t.lower() for t in texts], batch_size=256), start=1
    ):
        processed.append(" ".join(
            tok.lemma_ for tok in doc
            if not tok.is_stop and not tok.is_punct and len(tok.text) > 1
        ))
        # fire callback every UPDATE_EVERY docs → progress 0.25 → 0.48
        if i % UPDATE_EVERY == 0 or i == n_texts:
            pct = 0.25 + (i / n_texts) * 0.23
            log(f"🔤  Lemmatised {i:,} / {n_texts:,} texts …", pct)

    log("📐  Building TF-IDF matrix (20k features, 1–2 grams) …", 0.50)
    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))
    X   = vec.fit_transform(processed)

    log(f"🏋️  Training Logistic Regression on {X.shape[0]:,} × {X.shape[1]:,} matrix …", 0.70)
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    model.fit(X, labels)

    log("📊  Evaluating …", 0.88)
    preds  = model.predict(X)
    acc    = accuracy_score(labels, preds)
    report = classification_report(labels, preds,
                                   target_names=["Negative", "Positive"],
                                   output_dict=True)

    log("💾  Saving model …", 0.95)
    joblib.dump(model, os.path.join(MODELS_DIR, "spacy_model.pkl"))
    joblib.dump(vec,   os.path.join(MODELS_DIR, "spacy_vectorizer.pkl"))

    global _nlp_spacy, _model_spacy, _vec_spacy, _spacy_ready
    _nlp_spacy   = nlp
    _model_spacy = model
    _vec_spacy   = vec
    _spacy_ready = True

    log("✅  Done!", 1.0)
    return {"accuracy": acc, "report": report, "vocab_size": len(vec.vocabulary_)}


def load_default_dataset(n=5000):
    """Load SST-2 (or a subset) for quick training."""
    from datasets import load_dataset
    ds = load_dataset("sst2", split="train")
    texts  = ds["sentence"][:n]
    labels = ds["label"][:n]
    return list(texts), list(labels)


# ═══════════════════════════════════════════════════════════════════════════
# 4TH PIPELINE — FINE-TUNED DistilBERT
# ═══════════════════════════════════════════════════════════════════════════
FINETUNED_DIR = os.path.join(MODELS_DIR, "finetuned_distilbert")

_finetuned_pipe  = None
_finetuned_ready = False


def load_finetuned_pipeline() -> bool:
    global _finetuned_pipe, _finetuned_ready
    if _finetuned_ready:
        return True
    if not os.path.isdir(FINETUNED_DIR):
        return False
    from transformers import pipeline as hf_pipeline
    _finetuned_pipe  = hf_pipeline(
        "sentiment-analysis",
        model=FINETUNED_DIR,
        device=-1, truncation=True, max_length=512,
    )
    _finetuned_ready = True
    return True


def predict_finetuned(text: str) -> dict:
    """Inference with the locally fine-tuned DistilBERT."""
    t0     = time.perf_counter()
    result = _finetuned_pipe(text)[0]
    elapsed= (time.perf_counter() - t0) * 1000
    raw    = result["label"]          # "Negative" / "Positive" (from id2label)
    label  = raw.capitalize() if raw.lower() in ("negative","positive") else (
             "Negative" if raw in ("LABEL_0","LABEL_1") and result["score"] < 0.5 else "Positive"
    )
    score  = float(result["score"])
    # if label came from custom id2label the score IS for that label
    if raw.upper().startswith("LABEL_"):
        is_pos = (raw == "LABEL_1")
    else:
        is_pos = (label == "Positive")
    p_pos = score if is_pos else 1 - score
    p_neg = 1 - p_pos
    return {
        "label":      "Positive" if is_pos else "Negative",
        "confidence": round(score, 4),
        "proba_neg":  round(p_neg, 4),
        "proba_pos":  round(p_pos, 4),
        "time_ms":    round(elapsed, 2),
        "steps":      {},
        "top_features": [],
    }


def finetune_distilbert(
    texts, labels,
    epochs: int = 2,
    batch_size: int = 16,
    max_samples: int = 2000,
    progress_cb=None,
):
    """
    Fine-tune distilbert-base-uncased from scratch on the provided dataset.
    progress_cb(msg: str, pct: float, extra: dict) fires at every logging step.
    extra may contain {"losses": [{"step":int,"loss":float}, ...]}
    Returns: {"accuracy":float, "report":dict, "loss_history":list, "n_train":int, "n_eval":int}
    """
    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments, TrainerCallback,
    )
    from sklearn.metrics import accuracy_score, classification_report

    def log(msg, pct, extra=None):
        if progress_cb:
            progress_cb(msg, pct, extra or {})

    # Auto-fix missing or outdated 'accelerate' for the current python environment
    try:
        import accelerate
    except ImportError:
        log("🚀  Auto-installing missing 'accelerate' library (one-time setup) …", 0.01)
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate>=1.1.0", "transformers[torch]"])
        import accelerate # Ensure it's available after install


    # ── cap samples for CPU sanity ─────────────────────────────────────────
    if len(texts) > max_samples:
        log(f"⚠️  Capping at {max_samples:,} samples for CPU speed …", 0.02)
        texts  = list(texts[:max_samples])
        labels = list(labels[:max_samples])

    log(f"📦  Loading tokenizer: distilbert-base-uncased …", 0.05)
    MODEL_BASE = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_BASE)

    log("🔢  Tokenising dataset …", 0.12)
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True,
            padding="max_length", max_length=128
        )
    ds_full  = Dataset.from_dict({"text": texts, "label": labels})
    split    = ds_full.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"].map(tokenize_fn, batched=True)
    eval_ds  = split["test"].map(tokenize_fn, batched=True)

    log("🧠  Initialising DistilBERT classifier head (random weights, NOT SST-2 tuned) …", 0.20)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"},
        label2id={"Negative": 0, "Positive": 1},
    )

    # ── live callback ──────────────────────────────────────────────────────
    n_train      = len(train_ds)
    total_steps  = max(1, (n_train // batch_size) * epochs)
    loss_history = []

    class LiveCB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            loss     = logs.get("loss")
            eval_loss = logs.get("eval_loss")
            step     = state.global_step
            if loss is not None:
                loss_history.append({"step": step, "loss": float(loss)})
                pct = 0.25 + (step / total_steps) * 0.58
                log(
                    f"📉  Step {step}/{total_steps}  ·  Loss: {loss:.4f}",
                    min(pct, 0.83),
                    {"losses": loss_history},
                )
            if eval_loss is not None:
                log(
                    f"📊  Epoch eval loss: {eval_loss:.4f}",
                    0.86,
                    {"losses": loss_history},
                )

    logging_steps = max(1, total_steps // 20)

    log(
        f"🏋️  Starting fine-tuning: {epochs} epoch(s) · "
        f"batch={batch_size} · {n_train:,} train samples …",
        0.25,
    )
    args = TrainingArguments(
        output_dir=FINETUNED_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",  # Fix for newer transformers versions
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=logging_steps,
        use_cpu=True,            # use_cpu replaces no_cuda in newer transformers versions
        report_to="none",
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[LiveCB()],
    )
    trainer.train()

    log("💾  Saving fine-tuned model to disk …", 0.92)
    os.makedirs(FINETUNED_DIR, exist_ok=True)
    trainer.save_model(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)

    log("📊  Evaluating on held-out set …", 0.95)
    preds_out = trainer.predict(eval_ds)
    preds     = np.argmax(preds_out.predictions, axis=1)
    acc       = accuracy_score(eval_ds["label"], preds)
    report    = classification_report(
        eval_ds["label"], preds,
        target_names=["Negative", "Positive"],
        output_dict=True,
    )

    # ── reload into global handle ──────────────────────────────────────────
    from transformers import pipeline as hf_pipeline
    global _finetuned_pipe, _finetuned_ready
    _finetuned_pipe  = hf_pipeline(
        "sentiment-analysis", model=FINETUNED_DIR,
        device=-1, truncation=True, max_length=512,
    )
    _finetuned_ready = True

    log("✅  Fine-tuning complete!", 1.0, {"losses": loss_history})
    return {
        "accuracy":     acc,
        "report":       report,
        "loss_history": loss_history,
        "n_train":      n_train,
        "n_eval":       len(eval_ds),
    }

