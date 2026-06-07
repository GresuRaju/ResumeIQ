"""
nlp_engine.py — Core NLP logic
- Skill extraction
- TF-IDF cosine similarity + optional BERT/sentence-transformer similarity
- ATS score calculation
- Candidate scoring pipeline
- Resume gap analysis
- Multi-JD support helpers
"""

import re
import math
from collections import Counter

# ── Skill Vocabulary ──────────────────────────────────────────────────────────
SKILL_LIST = [
    # Programming
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "rust", "scala",
    "r programming", "matlab", "sql", "bash", "shell scripting", "kotlin", "swift", "php", "ruby",
    # ML / DL Frameworks
    "machine learning", "ml", "deep learning", "nlp", "natural language processing",
    "scikit-learn", "tensorflow", "pytorch", "keras", "xgboost", "lightgbm", "catboost",
    "huggingface", "transformers", "bert", "gpt", "llm", "generative ai", "computer vision",
    # NLP specific
    "tfidf", "tf-idf", "word2vec", "fasttext", "embeddings", "word embeddings",
    "text classification", "named entity recognition", "ner", "sentiment analysis",
    "tokenization", "lemmatization", "spacy", "nltk", "gensim",
    # Data
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "data analysis", "data science", "statistics", "feature engineering",
    # MLOps / Deployment
    "docker", "kubernetes", "mlflow", "airflow", "fastapi", "flask", "django",
    "rest api", "rest", "api", "microservices",
    # Cloud
    "aws", "gcp", "azure", "cloud", "s3", "ec2", "sagemaker", "vertex ai",
    # Tools
    "git", "github", "ci/cd", "linux", "jupyter",
    # Soft skills
    "communication", "teamwork", "problem-solving", "leadership", "attention to detail",
    "agile", "scrum",
    # Data Engineering
    "spark", "kafka", "airflow", "hadoop", "bigquery", "postgresql", "mongodb",
    "mysql", "redis", "elasticsearch",
]


def preprocess(text: str) -> list:
    """Lowercase, remove punctuation, tokenize."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def extract_skills(text: str) -> list:
    """Return list of skills found in text."""
    text_lower = text.lower()
    found = []
    for skill in SKILL_LIST:
        if " " in skill:
            if skill in text_lower:
                found.append(skill)
        else:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found.append(skill)
    return found


def tfidf_cosine_similarity(doc_a: str, doc_b: str) -> float:
    """
    Compute TF-IDF cosine similarity between two documents.
    Falls back to pure Python if scikit-learn not available.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vect = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vect.fit_transform([doc_a, doc_b])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(score)
    except ImportError:
        pass

    # Pure Python fallback
    tokens_a = preprocess(doc_a)
    tokens_b = preprocess(doc_b)
    vocab = list(set(tokens_a + tokens_b))

    def tf(tokens, word):
        count = tokens.count(word)
        return count / len(tokens) if tokens else 0

    def idf(word):
        n_docs = 2
        df = sum(1 for tokens in [tokens_a, tokens_b] if word in tokens)
        return math.log((n_docs + 1) / (df + 1)) + 1

    def tfidf_vec(tokens):
        return [tf(tokens, w) * idf(w) for w in vocab]

    vec_a = tfidf_vec(tokens_a)
    vec_b = tfidf_vec(tokens_b)

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Load BERT model once at startup (not per request) ─────────────────────────
# This avoids reloading the 80MB model on every resume — critical for batch runs.
_BERT_MODEL = None

def _get_bert_model():
    global _BERT_MODEL
    if _BERT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _BERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _BERT_MODEL = None
    return _BERT_MODEL


def bert_cosine_similarity(doc_a: str, doc_b: str) -> float:
    """
    Compute semantic similarity using sentence-transformers (BERT-based).
    Uses a module-level cached model — loaded once, reused for all resumes.
    Falls back to TF-IDF if sentence-transformers is not installed.
    """
    model = _get_bert_model()
    if model is None:
        return tfidf_cosine_similarity(doc_a, doc_b)
    try:
        from sentence_transformers import util
        # Truncate to 512 tokens — MiniLM limit
        emb_a = model.encode(doc_a[:1024], convert_to_tensor=True)
        emb_b = model.encode(doc_b[:1024], convert_to_tensor=True)
        score = float(util.cos_sim(emb_a, emb_b)[0][0])
        return max(0.0, min(1.0, score))
    except Exception:
        return tfidf_cosine_similarity(doc_a, doc_b)


def combined_similarity(doc_a: str, doc_b: str, use_bert: bool = True) -> dict:
    """
    Returns both TF-IDF and BERT similarities.
    BERT is ON by default — use_bert=False only for testing/fallback.
    Combined score: 35% TF-IDF + 65% BERT (semantic understanding dominates).
    """
    tfidf_sim = tfidf_cosine_similarity(doc_a, doc_b)
    if use_bert:
        bert_sim = bert_cosine_similarity(doc_a, doc_b)
        combined = 0.35 * tfidf_sim + 0.65 * bert_sim
    else:
        bert_sim  = None
        combined  = tfidf_sim
    return {"tfidf": tfidf_sim, "bert": bert_sim, "combined": combined}


def calc_ats_score(
    resume_text: str,
    jd_text: str,
    fresher_mode: bool = False,
    bert_sim: float = None,         # Pass in pre-computed BERT score to avoid double computation
) -> dict:
    """
    Calculate ATS score with BERT semantic similarity baked in.

    Score breakdown:
      - Rule-based checks  : 50 pts  (6 criteria, ~8.3 pts each)
      - Keyword match      : 25 pts  (how many JD skills appear in resume)
      - Semantic fit (BERT): 25 pts  (how well resume CONTENT matches JD)

    This mirrors how real ATS tools work — keyword presence alone is not enough,
    the semantic meaning of the resume must align with the job description.
    """
    jd_skills     = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    matched       = [s for s in resume_skills if s in jd_skills]
    keyword_ratio = len(matched) / len(jd_skills) if jd_skills else 0

    exp_match  = re.search(r'(\d+)\s*\+?\s*year', resume_text.lower())
    exp_years  = int(exp_match.group(1)) if exp_match else 0

    has_internship = bool(re.search(
        r'intern|internship|trainee|apprentice', resume_text, re.IGNORECASE
    ))
    has_projects = bool(re.search(
        r'project|built|developed|implemented|created|designed',
        resume_text, re.IGNORECASE
    ))
    word_count = len(resume_text.split())

    # ── Rule-based criteria ───────────────────────────────────────────────────
    if fresher_mode:
        experience_criterion = {
            "label": "Internship / Projects Present",
            "pass": has_internship or has_projects
        }
    else:
        experience_criterion = {
            "label": "Relevant Experience (≥2 yrs)",
            "pass": exp_years >= 2
        }

    criteria = [
        {"label": "Keyword Match ≥ 50%",       "pass": keyword_ratio >= 0.5},
        experience_criterion,
        {
            "label": "Contact Info Present",
            "pass": bool(
                re.search(r'\b[\w.-]+@[\w.-]+\.\w+\b', resume_text)
                or "linkedin" in resume_text.lower()
                or re.search(r'\+?\d[\d\s\-]{8,}\d', resume_text)
            )
        },
        {
            "label": "Education Section",
            "pass": bool(re.search(
                r'b\.?tech|bachelor|master|m\.?tech|phd|degree|engineering|university|college|b\.?e\b',
                resume_text, re.IGNORECASE
            ))
        },
        {
            "label": "Proper Sections",
            "pass": bool(re.search(
                r'experience|projects?|skills?|education|summary|objective|work history',
                resume_text, re.IGNORECASE
            ))
        },
        {"label": "Optimal Length (80–800 words)", "pass": 80 <= word_count <= 800},
    ]

    pass_count = sum(1 for c in criteria if c["pass"])

    # ── Three-component ATS scoring ───────────────────────────────────────────
    # 1. Rule-based checks  → max 50 pts
    rules_score   = (pass_count / len(criteria)) * 50

    # 2. Keyword match      → max 25 pts
    keyword_score = keyword_ratio * 25

    # 3. Semantic fit       → max 25 pts (BERT if available, else TF-IDF)
    if bert_sim is None:
        # Compute fresh — only happens if caller didn't pass it in
        bert_sim = bert_cosine_similarity(resume_text, jd_text)
    semantic_score = bert_sim * 25

    raw_ats = rules_score + keyword_score + semantic_score

    # ── Fresher mode slight rebalance ─────────────────────────────────────────
    # Give more weight to keyword + semantic, less to experience-heavy rules
    if fresher_mode:
        raw_ats = (pass_count / len(criteria)) * 40 + keyword_ratio * 30 + bert_sim * 30

    ats_score = min(int(raw_ats), 97)

    # ── Score breakdown (for UI display) ─────────────────────────────────────
    breakdown = {
        "rules_pts":    round(rules_score, 1),
        "keyword_pts":  round(keyword_score, 1),
        "semantic_pts": round(semantic_score, 1),
        "total":        ats_score,
    }

    return {
        "ats_score":      ats_score,
        "criteria":       criteria,
        "matched_skills": matched,
        "keyword_ratio":  keyword_ratio,
        "exp_years":      exp_years,
        "has_internship": has_internship,
        "has_projects":   has_projects,
        "bert_sim_used":  round(bert_sim, 4),
        "breakdown":      breakdown,        # New — for UI score breakdown card
    }


def generate_gap_analysis(resume_text: str, jd_text: str) -> dict:
    """
    Generate a gap analysis report for a candidate.
    Returns missing skills, suggestions, and a feedback summary.
    """
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    matched = [s for s in resume_skills if s in jd_skills]
    missing = [s for s in jd_skills if s not in resume_skills]

    exp_match = re.search(r'(\d+)\s*\+?\s*year', resume_text.lower())
    exp_years = int(exp_match.group(1)) if exp_match else 0

    has_contact = bool(
        re.search(r'\b[\w.-]+@[\w.-]+\.\w+\b', resume_text)
        or "linkedin" in resume_text.lower()
    )
    has_education = bool(re.search(
        r'b\.?tech|bachelor|master|m\.?tech|phd|degree|university|college',
        resume_text, re.IGNORECASE
    ))
    has_summary = bool(re.search(
        r'summary|objective|profile|about me', resume_text, re.IGNORECASE
    ))

    suggestions = []
    if missing:
        suggestions.append(f"Add these missing skills to your resume: {', '.join(missing[:6])}")
    if not has_contact:
        suggestions.append("Add your email, phone, or LinkedIn profile URL")
    if not has_education:
        suggestions.append("Include an Education section with your degree details")
    if not has_summary:
        suggestions.append("Add a Professional Summary or Objective at the top")
    if exp_years == 0:
        suggestions.append("Highlight any internships, freelance work, or side projects")

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": round(len(matched) / len(jd_skills) * 100, 1) if jd_skills else 0,
        "exp_years": exp_years,
        "suggestions": suggestions,
        "has_contact": has_contact,
        "has_education": has_education,
        "has_summary": has_summary,
    }


def score_candidate(
    resume_text: str,
    jd_text: str,
    fresher_mode: bool = False,
    use_bert: bool = True,          # ON by default now
) -> dict:
    """
    Full candidate scoring pipeline.
    BERT is computed once and shared between similarity scoring and ATS scoring.
    """
    # ── Step 1: Compute similarities (BERT runs here, once) ──────────────────
    sim_data = combined_similarity(resume_text, jd_text, use_bert=use_bert)
    sim      = sim_data["combined"]

    # ── Step 2: ATS score — pass in pre-computed bert_sim to avoid recompute ─
    bert_val = sim_data["bert"] if sim_data["bert"] is not None else 0.0
    ats_data = calc_ats_score(
        resume_text, jd_text,
        fresher_mode=fresher_mode,
        bert_sim=bert_val,
    )

    resume_skills = extract_skills(resume_text)
    matched       = ats_data["matched_skills"]
    keyword_ratio = ats_data["keyword_ratio"]
    exp_years     = ats_data["exp_years"]

    # ── Step 3: Match score formula ───────────────────────────────────────────
    if fresher_mode:
        raw          = 0.60 * keyword_ratio + 0.40 * sim
        match_score  = int(raw * 100)
        if ats_data["has_internship"]:
            match_score += 8
        elif ats_data["has_projects"]:
            match_score += 5
        match_score = max(match_score, int(keyword_ratio * 55))
        match_score = min(match_score, 95)
    else:
        exp_score   = min(exp_years / 4.0, 1.0)
        raw         = 0.40 * keyword_ratio + 0.35 * sim + 0.25 * exp_score
        match_score = min(int(raw * 100 + 10), 99)

    # ── Step 4: Gap analysis ─────────────────────────────────────────────────
    gap = generate_gap_analysis(resume_text, jd_text)

    return {
        "match_score":    match_score,
        "ats_score":      ats_data["ats_score"],
        "ats_breakdown":  ats_data["breakdown"],   # New — rules/keyword/semantic pts
        "criteria":       ats_data["criteria"],
        "matched_skills": matched,
        "all_skills":     resume_skills,
        "sim":            round(sim_data["tfidf"] * 100, 1),
        "bert_sim":       round(sim_data["bert"]  * 100, 1) if sim_data["bert"] is not None else None,
        "skill_score":    round(keyword_ratio * 100, 1),
        "exp_years":      exp_years,
        "fresher_mode":   fresher_mode,
        "has_internship": ats_data["has_internship"],
        "has_projects":   ats_data["has_projects"],
        "gap_analysis":   gap,
    }


def score_against_multiple_jds(
    resume_text: str,
    jd_dict: dict,
    fresher_mode: bool = False,
    use_bert: bool = True,          # ON by default
) -> dict:
    """
    Score a single resume against multiple job descriptions.
    jd_dict: {"Job Title": "JD text", ...}
    Returns dict of scores per JD.
    """
    results = {}
    for jd_title, jd_text in jd_dict.items():
        results[jd_title] = score_candidate(
            resume_text, jd_text,
            fresher_mode=fresher_mode,
            use_bert=use_bert,
        )
    return results
