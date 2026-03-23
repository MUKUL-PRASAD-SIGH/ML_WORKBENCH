"""
run_pipeline.py — Full Pipeline Runner
=======================================
Runs all 6 steps of the Topic-Aware AI System sequentially.
Each step's output is the input to the next.

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import time

STEPS = [
    ("Step 1 — Load IMDB Dataset",        "step1_load_data.py"),
    ("Step 2 — Preprocess (Gensim)",      "step2_preprocess.py"),
    ("Step 3 — Train LDA Topics",         "step3_lda_topics.py"),
    ("Step 4 — BERT Embeddings",          "step4_bert_embeddings.py"),
    ("Step 5 — Combine Hybrid Features",  "step5_combine_features.py"),
]

SEP = "=" * 65

print(SEP)
print("  🚀  TOPIC-AWARE AI SYSTEM — Full Pipeline")
print(SEP)

total_start = time.time()

for idx, (name, script) in enumerate(STEPS, 1):
    print(f"\n[{idx}/{len(STEPS)}] {name}")
    print("─" * 65)
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, script],
        check=False,
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n❌ Step {idx} FAILED (exit code {result.returncode}). Aborting pipeline.")
        sys.exit(1)

    print(f"\n⏱  Finished in {elapsed:.1f}s")

total = time.time() - total_start
print(f"\n{SEP}")
print(f"✅  All steps complete in {total:.1f}s")
print(f"   Run search engine:  python step6_similarity_search.py --query \"your text\"")
print(SEP)
