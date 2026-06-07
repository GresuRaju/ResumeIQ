# ResumeIQ — Smart Resume Screening System

> Upload resumes. Paste a job description. Get instant ATS scores, ML-ranked candidates, skill gap analysis, and automated shortlist + rejection emails.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square)
![BERT](https://img.shields.io/badge/BERT-sentence--transformers-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## What It Does

ResumeIQ is a full-stack resume screening tool that automates the shortlisting process for recruiters. It combines rule-based ATS scoring with BERT semantic similarity and TF-IDF matching to rank candidates fairly and transparently.

**No black boxes. Every score is explainable.**

---

## Features

### Core Screening
- **Match Score Ranking** — 0–99 weighted score per candidate, sorted automatically
- **ATS Score** — 3-component scoring: rule checks (50pts) + keyword match (25pts) + BERT semantic fit (25pts)
- **Skill Gap Analysis** — matched vs missing skills shown as chips per candidate
- **Fresher Mode** — reweights formula for 0–1 yr candidates, bonus for internships (+8) and projects (+5)
- **BERT Semantic Similarity** — always on by default, understands synonyms and context
- **TF-IDF Cosine Similarity** — keyword-level document similarity
- **Candidate Comparison** — side-by-side scores across all metrics
- **Shortlist Panel** — custom ATS + Match Score thresholds via sliders

### Input Sources
- **Manual Upload** — drag and drop PDF/TXT resumes
- **Google Drive** — paste a folder URL, all resumes fetched automatically
- **Gmail Inbox** — scan inbox by date range (7 days to 1 year or custom), extracts PDF/TXT attachments and plain-text resume emails

### Email Automation
- **AI-Crafted Emails** — HR writes a rough note, Gemini 2.5 Flash crafts a professional email
- **Shortlist Emails** — personalised per candidate, sent via Gmail SMTP
- **Rejection Emails** — respectful rejection with candidate's specific skill gaps included
- **3D Envelope Animation** — visual send animation, green for shortlist, red for rejection
- **One-Click Send** — sends to all eligible candidates in one click

### Export
- **CSV Download** — full results including TF-IDF, BERT, ATS breakdown, verdict
- **SQLite Database** — shortlisted and rejected candidates stored automatically

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| NLP Similarity | sentence-transformers (all-MiniLM-L6-v2) |
| Keyword Similarity | TF-IDF + Cosine Similarity (scikit-learn) |
| AI Email Crafting | Gemini 2.5 Flash |
| Resume Parsing | pdfplumber / pdfminer |
| Email Sending | Gmail SMTP |
| Email Fetching | Gmail IMAP |
| Drive Fetching | Google Drive API v3 |
| Database | SQLite |
| Frontend | Vanilla HTML/CSS/JS + Three.js |

---

## Project Structure

```
ResumeIQ/
├── main.py                  # FastAPI backend — all endpoints
├── utils/
│   ├── nlp_engine.py        # BERT + TF-IDF scoring, ATS logic, gap analysis
│   └── parser.py            # PDF and TXT text extraction
├── workspace.html           # Main screening dashboard
├── email.html               # Email center (shortlist + rejection)
├── resumeiq_final.html      # Landing page
├── resumeiq.db              # SQLite database (auto-created)
├── .env                     # API keys (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/resumeiq.git
cd resumeiq
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create `.env` file
```
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

Get your Gemini API key → https://aistudio.google.com/app/apikey  
Get your Google API key → https://console.cloud.google.com (enable Drive API)

### 5. Add to top of `main.py`
```python
from dotenv import load_dotenv
load_dotenv()
```

### 6. Run the backend
```bash
python main.py
```

Backend runs at `http://localhost:8000`

### 7. Open frontend
Open `workspace.html` in your browser directly, or serve with any static server.

---

## Requirements

```
fastapi
uvicorn
python-multipart
scikit-learn
sentence-transformers
pdfplumber
google-generativeai
python-dotenv
httpx
plotly 
```

---

## Scoring Formula

### Experienced Mode
```
Match Score = 0.40 × Skill Overlap
            + 0.35 × Combined Similarity (TF-IDF + BERT)
            + 0.25 × Experience Score (capped at 4 years)
```

### Fresher Mode
```
Match Score = 0.60 × Skill Overlap
            + 0.40 × Combined Similarity
            + 8pts  if internship detected
            + 5pts  if projects detected
```

### ATS Score (0–97)
```
ATS Score = Rules/Checks      (max 50pts — 6 criteria)
          + Keyword Match      (max 25pts)
          + BERT Semantic Fit  (max 25pts)
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/screen` | Screen uploaded resumes |
| POST | `/api/screen-fetched` | Screen Drive/Gmail fetched resumes |
| POST | `/api/craft-email` | AI-craft shortlist email |
| POST | `/api/craft-rejection` | AI-craft rejection email |
| POST | `/api/send-emails` | Send emails via Gmail SMTP |
| POST | `/api/fetch-drive` | Fetch resumes from Google Drive folder |
| POST | `/api/fetch-gmail` | Fetch resumes from Gmail inbox by date range |
| GET  | `/api/shortlisted` | Get shortlisted candidates from DB |
| GET  | `/api/rejected` | Get rejected candidates from DB |

---

## Gmail Setup (for email sending + inbox fetching)

1. Create or use a Gmail account dedicated to hiring
2. Go to **Google Account → Security → 2-Step Verification → ON**
3. Then **Security → App Passwords**
4. Generate an App Password for "ResumeIQ"
5. Use that 16-digit password in the Email Center

The same App Password works for both sending emails and scanning the inbox.

---

## Deployment

**Quick summary:**
- Backend → [Render.com](https://render.com) (free tier)
- Frontend → [Netlify](https://netlify.com) (free tier)
- Set `GEMINI_API_KEY` in Render dashboard environment variables
- Total cost: **$0**

---

## Author

**Gresu Raju**  
B.Tech AIML · 2026  
Vignan's Lara Institute of Technology, Guntur, Andhra Pradesh


---

## License

MIT License — free to use, modify, and distribute.
"# ResumeIQ" 
