---
title: ResumeRadar
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🎯 ResumeRadar — AI-Powered Resume & Job Description Matcher

**Scan your resume. Spot the gaps. Land the job.**

> 🔗 **[Live Demo → huggingface.co/spaces/HansikaG/resume-radar](https://huggingface.co/spaces/HansikaG/resume-radar)**

---

## What it does

Recruiters and ATS systems filter resumes based on keyword and semantic alignment with the job description. ResumeRadar helps job seekers close that gap before they apply.

Paste or upload a **job description** and your **resume** (PDF, DOCX, or plain text) — the app instantly returns:

- **Semantic match score** — how closely your resume aligns with the role, using sentence embeddings (not just keyword counting)
- **Prioritised missing keywords** — the most important JD terms absent from your resume, ranked High / Medium / Low
- **Section-specific edit suggestions** — actionable tips mapped to Summary, Skills, and Experience sections

---

## Key Features

- Upload PDF or DOCX resumes directly — no copy-pasting needed
- Semantic understanding via sentence embeddings, not just string matching
- TF-IDF keyword extraction identifies what the job description actually emphasises
- Clean, recruiter-style output with colour-coded keyword chips and suggestion cards
- Deployed and accessible publicly — no setup required for end users

---

## Tech Stack

| What | How |
|---|---|
| Semantic embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Match scoring | Cosine similarity (`scikit-learn`) |
| Keyword extraction | TF-IDF with unigrams + bigrams (`scikit-learn`) |
| File parsing | `pdfplumber` (PDF) · `python-docx` (DOCX) |
| UI | `Streamlit` |
| Deployment | Docker · HuggingFace Spaces |

---

## Why I built this

Tailoring a resume for every job application is time-consuming and it's hard to know what to change. This project applies NLP techniques — sentence embeddings and cosine similarity — to make that feedback instant and objective. It's a practical demonstration of semantic search applied to a real, everyday problem.

---

## Run locally

```bash
git clone https://github.com/hansikagaidhani/resume-radar.git
cd resume-radar
pip install -r requirements.txt
streamlit run app.py
```
