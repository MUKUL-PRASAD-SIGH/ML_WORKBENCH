# NLP Model Comparison Dashboard

> **Live Demo:** [https://sentilyticz.streamlit.app/](https://sentilyticz.streamlit.app/)  
> **Last Updated:** 2026-03-22  
> **Architecture:** Streamlit (Python)  
> **Core Stack:** Python · NLTK · spaCy · HuggingFace (RoBERTa) · scikit-learn · Plotly

---

## 🏗️ Architecture Comparison: Streamlit vs Vercel

When building AI applications, choosing the right deployment environment is essential. Here is why this app uses Streamlit over Vercel:

### 🔵 Streamlit (Current Workspace)
- **Use Case:** Built for data apps / ML demos.
- **Workflow:** You write pure Python → the UI appears automatically.
- **Advantage:** Unbeatable for rapidly prototyping ML models and dashboards without needing any frontend context.

### ⚫ Vercel
- **Use Case:** Built for production-ready consumer web apps.
- **Workflow:** Uses frontend frameworks (Next.js, React) to build complex client-side architecture.
- **Advantage:** Offers infinite scale, highly-customizable user experiences, and edge caching for global audiences.

---

## 📋 Project Overview

An **educational, interactive web dashboard** that shows every single step of how three NLP pipelines process a sentence — from raw text to final prediction — with full visual explanations. It allows you to visualize tf-idf internals, neural logits, and seamlessly fine-tune a model locally via the GUI.

---

## 🗂 File Structure

```
Sentiment_Analyser/
├── app.py                   # Streamlit dashboard — main GUI entry point
├── pipelines.py             # Inference, tokenizer bridges, and callback logic
├── requirements.txt         # Python dependencies
├── nlp_comparison.ipynb     # Research notebook
└── models/                  # Created at training time (cache)
    ├── nltk_model.pkl
    ├── nltk_vectorizer.pkl
    ├── spacy_model.pkl
    ├── spacy_vectorizer.pkl
    └── finetuned_distilbert/
```

---

## ✅ Implemented Features

### 1. 🔬 Step-by-Step Analyser Tab
The **flagship feature**. For any input sentence, shows:

| Pipeline | Steps shown |
|----------|-------------|
| **NLTK** | Raw → Lowercase → Punct removal → Tokenise → Stopword removal → TF-IDF → LR → Prediction |
| **spaCy** | Raw → Lowercase → Tokenise → Lemmatise (with original→lemma pairs) → Remove stops → TF-IDF → LR → Prediction |
| **RoBERTa** | Raw → WordPiece tokens (coloured) → Token IDs → Encoder → CLS → Softmax → Prediction |

- **Token highlighting:** green (kept), red strikethrough (removed), purple (lemma differs), gold (special BERT tokens)
- **Probability gauges** — smooth circular dials plotting exact confidence spreads per model.
- **Feature contribution bar charts** — shows which TF-IDF tokens pushed the NLTK/spaCy prediction (green = toward Positive, red = toward Negative). Fully interactive Plotly backend.

### 2. 🏋️ Train Models Tab
- **Dataset source:** SST-2 auto-download (slider for 1k–67k samples) OR upload your own CSV data.
- **Live progress hooks** — updates Streamlit progress bars + live loss charting seamlessly at every training step.
- **NLTK Train button** — runs full pipeline: download NLTK data → clean → tokenise → TF-IDF → LR → save `.pkl`.
- **spaCy Train button** — same but with lemmatisation; automatically triggers background spaCy model downloads if missing.
- **Pre-trained RoBERTa Load button** — downloads the ~499 MB `cardiffnlp/twitter-roberta-base-sentiment` state.
- **DistilBERT Fine-tuning button** — lets the user execute `Trainer` gradient descent over multiple epochs directly from the frontend to customize global sentiment performance natively. 

### 3. 📂 Batch / CSV Tab
- Upload a standard CSV with a `text` column (and optional `label` for accuracy grading).
- Seamlessly runs all active internal models row-by-row with an animated live progress bar.
- Shows accuracy vs. ground truth gradients per model (if labels are provided).
- Triggers a 1-click CSV download of all generated inference results.

### 4. 📊 Compare Models Tab
- Evaluates your recent dashboard inputs and aggregates them fully autonomously.
- Multi-panel Plotly chart matrix: Confidence profiles, graphical Inference Time tracking, and P(Positive) mapping.
- Live-generated Radar Chart contrasting metric advantages across all loaded model types simultaneously.

### 5. 📖 How It Works Tab
- Streamlit vs Vercel design architecture breakdown.
- Expandable step-by-step logic explainers per pipeline type showing exactly how machine bias occurs.
- Active trade-off comparison matrix contrasting speed metrics, storage weights, and interpretability.

---

## ⚙️ Setup & Run

```bash
# 1. Install
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Launch
streamlit run app.py
```

Open `http://localhost:8501` to view the GUI.

> ⚠️ Note: Deep learning models (like RoBERTa) will train **inside the dashboard session**; there's no separate script needed to compile binaries.

---

## 📈 Expected Model Performance (SST-2 Baseline)

| Model       | Accuracy | Speed (CPU) |
|-------------|----------|-------------|
| NLTK        | ~83%     | 2–8 ms      |
| spaCy       | ~84%     | 4–12 ms     |
| RoBERTa     | ~93%     | 80–200 ms   |

---

## 🔥 Roadmap

| Feature | Status |
|---------|--------|
| Multi-class Label Mapping | ✅ Implemented |
| UUID UI Instance Stability | ✅ Implemented |
| Attention heatmap visualisation for Transformer | 🔲 Planned |
| GPU Execution Toggle | 🔲 Planned |
| LIME / SHAP explainability overlay | 🔲 Planned |
