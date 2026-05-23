---
title: ResumeRadar
emoji: 🎯
colorFrom: teal
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🎯 ResumeRadar

**Scan your resume. Spot the gaps. Land the job.**

Upload or paste a job description + your resume → get a **semantic match score**, **prioritised missing keywords**, and **section-specific edit suggestions**.

## How it works

1. Both texts are encoded into dense vector embeddings using **`all-MiniLM-L6-v2`** (sentence-transformers).
2. **Cosine similarity** between the two embedding vectors produces the match score (0–100%).
3. **TF-IDF** (scikit-learn, unigrams + bigrams) extracts the most important keywords from the job description; any not found in the resume are flagged as missing.
4. Actionable edit suggestions are generated from the score tier and the missing-keyword list.

## Stack

| Layer | Library |
|---|---|
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Similarity | `sklearn.metrics.pairwise.cosine_similarity` |
| Keyword extraction | `sklearn.feature_extraction.text.TfidfVectorizer` |
| UI | `Streamlit` |
| Deployment | HuggingFace Spaces |

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```
