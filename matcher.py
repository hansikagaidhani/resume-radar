from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ── Embeddings & cosine similarity ────────────────────────────────────────────

def compute_match_score(job_desc: str, resume: str) -> float:
    """Encode both texts with all-MiniLM-L6-v2 and return cosine similarity [0, 1]."""
    model = _get_model()
    embeddings = model.encode([job_desc, resume], convert_to_numpy=True)
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(np.clip(score, 0.0, 1.0))


# ── Keyword extraction ────────────────────────────────────────────────────────

def _tfidf_keywords(text: str, top_n: int = 50) -> list[tuple[str, float]]:
    """Return (keyword, tfidf_score) pairs ranked by importance for a single document."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=top_n,
        # Allow letters, digits, +, #, . so terms like C++, .NET, Node.js survive
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b",
    )
    try:
        matrix = vectorizer.fit_transform([text])
        names = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        return [(kw, sc) for kw, sc in ranked if sc > 0]
    except ValueError:
        return []


def _assign_priority(rank: int, total: int) -> str:
    """Map ranked position to high / medium / low priority tier."""
    if total == 0:
        return "medium"
    ratio = rank / total
    if ratio < 0.33:
        return "high"
    if ratio < 0.66:
        return "medium"
    return "low"


# ── Gap analysis ──────────────────────────────────────────────────────────────

def analyse_gaps(job_desc: str, resume: str) -> dict:
    """
    Full NLP analysis pipeline.

    Returns
    -------
    score      float  cosine similarity [0, 1]
    pct        int    [0, 100]
    verdict    str    Strong / Moderate / Weak Match
    missing    list[dict]  {"keyword": str, "priority": "high"|"medium"|"low"}
    present    list[str]   JD keywords already in the resume
    suggestions list[dict] {"section": str, "text": str}
    """
    score = compute_match_score(job_desc, resume)
    pct = int(score * 100)

    if pct >= 75:
        verdict = "Strong Match"
    elif pct >= 55:
        verdict = "Moderate Match"
    else:
        verdict = "Weak Match"

    # Keywords found in the job description, ranked by TF-IDF importance
    jd_keywords = _tfidf_keywords(job_desc, top_n=50)
    resume_lower = resume.lower()

    missing_pairs = [(kw, sc) for kw, sc in jd_keywords if kw.lower() not in resume_lower]
    present = [kw for kw, _ in jd_keywords if kw.lower() in resume_lower]

    missing = [
        {"keyword": kw, "priority": _assign_priority(i, len(missing_pairs))}
        for i, (kw, _) in enumerate(missing_pairs)
    ]

    suggestions = _build_suggestions(pct, missing)

    return {
        "score": score,
        "pct": pct,
        "verdict": verdict,
        "missing": missing,
        "present": present,
        "suggestions": suggestions,
    }


# ── Suggestion generation ─────────────────────────────────────────────────────

def _build_suggestions(pct: int, missing: list[dict]) -> list[dict]:
    tips: list[dict] = []

    high = [m["keyword"] for m in missing if m["priority"] == "high"]
    medium = [m["keyword"] for m in missing if m["priority"] == "medium"]

    # Overall score advice
    if pct >= 75:
        tips.append({
            "section": "General",
            "text": (
                "Excellent semantic match! Quantify your achievements "
                "(e.g., 'Reduced API latency by 40%', 'Led a team of 6 engineers') "
                "to stand out among equally-qualified candidates."
            ),
        })
    elif pct >= 55:
        tips.append({
            "section": "Experience",
            "text": (
                "Good match. Rewrite 2–3 experience bullets to directly mirror the "
                "language and priorities stated in the job description."
            ),
        })
    else:
        tips.append({
            "section": "Summary",
            "text": (
                "Weak semantic similarity. Add or rewrite a Professional Summary "
                "(3–4 sentences) that maps your background directly to the role's core requirements."
            ),
        })
        tips.append({
            "section": "Summary",
            "text": (
                "Mirror the job title and key phrases from the posting in your "
                "resume header or summary section — recruiters and ATS scanners look for this."
            ),
        })

    # High-priority missing keywords → Skills
    if high:
        kws = "  ·  ".join(f"`{kw}`" for kw in high[:8])
        tips.append({
            "section": "Skills",
            "text": (
                f"**Critical keywords missing** — add these to your Skills or "
                f"Core Competencies section:  {kws}."
            ),
        })

    # Medium-priority keywords → Experience bullets
    if medium:
        kws = "  ·  ".join(f"`{kw}`" for kw in medium[:6])
        tips.append({
            "section": "Experience",
            "text": (
                f"Weave these terms naturally into your experience bullets: {kws}."
            ),
        })

    # Structural advice when many keywords are absent
    if len(missing) > 15:
        tips.append({
            "section": "Skills",
            "text": (
                "You're missing a large portion of JD keywords. Consider adding a dedicated "
                "'Technical Skills' or 'Core Competencies' section — it lets you surface "
                "keywords without forcing them into every bullet point."
            ),
        })

    if pct < 65:
        tips.append({
            "section": "Experience",
            "text": (
                "Use active verbs (Led, Built, Designed, Automated, Delivered) and include "
                "measurable outcomes: numbers, percentages, team sizes, timelines."
            ),
        })

    tips.append({
        "section": "General",
        "text": (
            "After editing, re-run the matcher to confirm your score improves — "
            "and consider pasting your updated resume into an ATS simulator for a final check."
        ),
    })

    return tips
