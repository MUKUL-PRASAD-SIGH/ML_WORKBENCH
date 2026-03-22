# NLP Model Comparison Dashboard — Feature Documentation (Educational Edition)

> **Last Updated:** 2026-03-22  
> **Stack:** Python · NLTK · spaCy · HuggingFace Transformers · scikit-learn · Streamlit · Plotly

---

## 📋 Project Overview

An **educational, interactive web dashboard** that shows every single step of how three NLP
pipelines process a sentence — from raw text to final prediction — with full visual explanations.

---

## 🗂 File Structure

```
Sentiment_Analyser/
├── app.py                   # Streamlit dashboard — main entry point
├── pipelines.py             # Inference, training, and step-breakdown logic
├── requirements.txt         # Python dependencies
├── nlp_comparison.ipynb     # Research notebook
├── FEATURES.md              # ← This file
└── models/                  # Created at training time
    ├── nltk_model.pkl
    ├── nltk_vectorizer.pkl
    ├── spacy_model.pkl
    └── spacy_vectorizer.pkl
```

---

## ✅ Implemented Features

### 1. 🔬 Step-by-Step Analyser Tab
The **flagship feature**. For any input sentence, shows:

| Pipeline | Steps shown |
|----------|-------------|
| NLTK | Raw → Lowercase → Punct removal → Tokenise → Stopword removal → TF-IDF → LR → Prediction |
| spaCy | Raw → Lowercase → Tokenise → Lemmatise (with original→lemma pairs) → Remove stops → TF-IDF → LR → Prediction |
| DistilBERT | Raw → WordPiece tokens (coloured) → Token IDs → Encoder → CLS → Softmax → Prediction |

- **Token highlighting:** green (kept), red strikethrough (removed), purple (lemma differs), gold (special BERT tokens)
- **Probability gauges** — semicircular dial per model
- **Feature contribution bar charts** — shows which TF-IDF tokens pushed the NLTK/spaCy prediction (green = toward Positive, red = toward Negative)

### 2. 🏋️ Train Models Tab
- **Dataset source:** SST-2 auto-download (slider for 1k–67k samples) OR upload own CSV
- **Live progress callback** — updates Streamlit progress bar + log box at every training step
- **NLTK Train button** — runs full pipeline: download NLTK data → clean → tokenise → TF-IDF → LR → save `.pkl`
- **spaCy Train button** — same but with lemmatisation; includes spaCy model download if missing
- **DistilBERT load button** — downloads pre-trained checkpoint (~270 MB), no training needed
- **Post-training metrics:** accuracy, F1 (Positive), vocab size

### 3. 📂 Batch / CSV Tab
- Upload CSV with `text` column (+ optional `label` for accuracy scoring)
- Runs all loaded models row-by-row with a live progress bar
- Shows accuracy vs. ground truth per model if labels provided
- CSV download of all results

### 4. 📊 Compare Models Tab
- Populated automatically after running the Analyser
- Multi-panel Plotly chart: Confidence · Inference Time · P(Positive) vs P(Negative)
- Radar chart across all models
- Full summary DataFrame

### 5. 📖 How It Works Tab
- Expandable explainers per pipeline
- Trade-off comparison table (accuracy, speed, negation, size, interpretability)
- Resume-ready bullet point

---

## ⚙️ Setup & Run

```bash
# 1. Install
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Launch
streamlit run app.py
```

Open http://localhost:8501

> Models train **inside the dashboard** → no separate script needed.

---

## 📈 Expected Model Performance (SST-2)

| Model       | Accuracy | Speed (CPU) |
|-------------|----------|-------------|
| NLTK        | ~83%     | 2–8 ms      |
| spaCy       | ~84%     | 4–12 ms     |
| DistilBERT  | ~91%     | 80–200 ms   |

---

## 🔥 Roadmap

| Feature | Status |
|---------|--------|
| Attention heatmap visualisation for BERT | 🔲 Planned |
| GPU toggle | 🔲 Planned |
| Streamlit Cloud deployment | 🔲 Planned |
| LIME / SHAP explainability for BERT | 🔲 Planned |
| Session history accuracy trend chart | 🔲 Planned |
