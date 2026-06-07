import os
import re
import json
import smtplib
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.parser import extract_text_from_pdf, extract_text_from_txt
from utils.nlp_engine import (
    extract_skills, calc_ats_score,
    score_candidate, generate_gap_analysis
)

app = FastAPI(title="ResumeIQ — Batch Screening Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SQLite DB setup ───────────────────────────────────────────────────────────
DB_PATH = "resumeiq.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS shortlisted (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT,
            email       TEXT,
            filename    TEXT,
            match_score INTEGER,
            ats_score   INTEGER,
            exp_years   REAL,
            screened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS rejected (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT,
            email          TEXT,
            filename       TEXT,
            match_score    INTEGER,
            ats_score      INTEGER,
            missing_skills TEXT,
            screened_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Email extractor ───────────────────────────────────────────────────────────
def extract_email_from_text(text: str) -> str:
    match = re.search(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b', text)
    return match.group(0).lower() if match else ""

# ── Bool parser ───────────────────────────────────────────────────────────────
def parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — Screen resumes
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/screen")
async def screen_candidates(
    selected_model: str = Form("SVM (RBF Kernel)"),
    fresher_mode:   str = Form("false"),
    min_match:      int = Form(60),
    min_ats:        int = Form(50),
    job_description: str = Form(...),
    resumes: List[UploadFile] = File(...)
):
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description is empty.")
    if not resumes:
        raise HTTPException(status_code=400, detail="No resume files uploaded.")

    fresher      = parse_bool(fresher_mode)
    bert         = True
    jd_skills    = extract_skills(job_description)
    batch_results = []

    for resume_file in resumes:
        filename = resume_file.filename or ""
        if not filename:
            continue
        try:
            raw_bytes    = await resume_file.read()
            display_name = (
                filename
                .replace(".pdf", "").replace(".txt", "")
                .replace("_", " ").strip()
            )

            if filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(raw_bytes)
            else:
                text = extract_text_from_txt(raw_bytes)

            if not text.strip():
                text = filename

            scores = score_candidate(
                text, job_description,
                fresher_mode=fresher,
                use_bert=bert
            )

            match_score = int(scores.get("match_score", 0))
            ats_score   = int(scores.get("ats_score", 0))
            skill_score = float(scores.get("skill_score", 0.0))
            sim         = float(scores.get("sim", 0.0))
            bert_sim    = scores.get("bert_sim")
            exp_years   = float(scores.get("exp_years", 0.0))
            matched     = list(scores.get("matched_skills", []))
            criteria    = scores.get("criteria", [])

            gap_data = scores.get("gap_analysis")
            if gap_data and isinstance(gap_data, dict):
                missing_skills = list(gap_data.get("missing_skills", []))
                suggestions    = list(gap_data.get("suggestions", []))
            else:
                gap_data       = generate_gap_analysis(text, job_description)
                missing_skills = list(gap_data.get("missing_skills", []))
                suggestions    = list(gap_data.get("suggestions", []))

            if selected_model == "Logistic Regression":
                match_score = max(0, min(100, match_score - 2))

            # Extract candidate email from resume text
            candidate_email = extract_email_from_text(text)

            is_shortlisted = match_score >= min_match and ats_score >= min_ats

            # Save shortlisted candidates to DB
            if is_shortlisted and candidate_email:
                conn = sqlite3.connect(DB_PATH)
                c    = conn.cursor()
                c.execute("DELETE FROM shortlisted WHERE email = ?", (candidate_email,))
                c.execute("""
                    INSERT INTO shortlisted (name, email, filename, match_score, ats_score, exp_years)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (display_name, candidate_email, filename, match_score, ats_score, exp_years))
                conn.commit()
                conn.close()

            # Save rejected candidates to DB
            if not is_shortlisted and candidate_email:
                conn = sqlite3.connect(DB_PATH)
                c    = conn.cursor()
                c.execute("DELETE FROM rejected WHERE email = ?", (candidate_email,))
                c.execute("""
                    INSERT INTO rejected (name, email, filename, match_score, ats_score, missing_skills)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (display_name, candidate_email, filename, match_score, ats_score,
                      json.dumps(missing_skills)))
                conn.commit()
                conn.close()

            batch_results.append({
                "name":           display_name,
                "filename":       filename,
                "email":          candidate_email,
                "match_score":    match_score,
                "ats_score":      ats_score,
                "ats_breakdown":  scores.get("ats_breakdown", {}),
                "skill_score":    skill_score,
                "sim":            round(sim, 4),
                "bert_sim":       round(float(bert_sim), 4) if bert_sim is not None else None,
                "matched_skills": matched,
                "missing_skills": missing_skills,
                "criteria":       criteria,
                "exp_years":      exp_years,
                "suggestions":    suggestions,
                "is_shortlisted": is_shortlisted,
            })

        except Exception as e:
            print(f"[SKIP] {filename}: {e}")
            continue

    if not batch_results:
        raise HTTPException(status_code=400, detail="Could not extract text from any uploaded files.")

    batch_results.sort(key=lambda x: x["match_score"], reverse=True)

    return {
        "success":        True,
        "total_screened": len(batch_results),
        "selected_model": selected_model,
        "jd_skills":      jd_skills,
        "results":        batch_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — Craft email using Gemini 2.5 Flash
# ══════════════════════════════════════════════════════════════════════════════
class CraftEmailRequest(BaseModel):
    raw_note: str          # HR's rough note
    candidate_name: str    # To personalise "Dear Priya,"

@app.post("/api/craft-email")
async def craft_email(req: CraftEmailRequest):
    """
    Takes HR's rough note and crafts a professional shortlist email.
    Uses Gemini 2.5 Flash — high token limit, pro tier.
    API key loaded from .env — never exposed to frontend.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model  = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""You are an HR email writer.
Craft a short, professional shortlist notification email for a candidate named {req.candidate_name}.

HR's note: "{req.raw_note}"

Rules:
- Start with: Dear {req.candidate_name},
- Keep it under 120 words
- Sound warm and professional  
- End with: Best regards,\\nHiring Team
- Do NOT include a subject line
- Return just the email body, nothing else"""

        response = model.generate_content(prompt)
        body     = response.text.strip()
        return {"success": True, "email_body": body}

    except Exception as e:
        print(f"[Gemini fallback] {e}")
        body = f"""Dear {req.candidate_name},

We are pleased to inform you that your profile has been shortlisted after our initial screening.

{req.raw_note}

We look forward to connecting with you soon. Please feel free to reach out if you have any questions.

Best regards,
Hiring Team"""
        return {"success": True, "email_body": body}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — Send emails to shortlisted candidates
# ══════════════════════════════════════════════════════════════════════════════
class SendEmailsRequest(BaseModel):
    sender_email:    str   # company Gmail
    sender_password: str   # Gmail App Password
    subject:         str
    candidates: list       # [{name, email, email_body}]

@app.post("/api/send-emails")
async def send_emails(req: SendEmailsRequest):
    """
    Sends individual personalised emails to each shortlisted candidate.
    Uses Gmail SMTP — completely free.
    """
    if not req.candidates:
        raise HTTPException(status_code=400, detail="No candidates to email.")

    sent    = []
    failed  = []

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(req.sender_email, req.sender_password)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gmail login failed: {str(e)}")

    for candidate in req.candidates:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = req.subject
            msg["From"]    = req.sender_email
            msg["To"]      = candidate["email"]

            # Plain text part
            msg.attach(MIMEText(candidate["email_body"], "plain"))

            server.sendmail(req.sender_email, candidate["email"], msg.as_string())
            sent.append(candidate["name"])
        except Exception as e:
            failed.append({"name": candidate["name"], "reason": str(e)})

    server.quit()

    return {
        "success": True,
        "sent":    sent,
        "failed":  failed,
        "total":   len(sent),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT — Craft rejection email using Gemini
# ══════════════════════════════════════════════════════════════════════════════
class CraftRejectionRequest(BaseModel):
    raw_note:       str
    candidate_name: str
    missing_skills: str   # comma-separated string of missing skills

@app.post("/api/craft-rejection")
async def craft_rejection(req: CraftRejectionRequest):
    gap_line = f"Key areas to develop: {req.missing_skills}." if req.missing_skills else ""
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model  = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""You are a professional HR email writer.
Write a short, respectful rejection email for a candidate named {req.candidate_name}.

{f'Their skill gaps: {req.missing_skills}' if req.missing_skills else ''}
{f'HR note: {req.raw_note}' if req.raw_note else ''}

Rules:
- Start with: Dear {req.candidate_name},
- Be respectful, warm, and encouraging — not cold
- Mention the skill gaps briefly so they know what to improve (use [SKILL_GAPS] as placeholder)
- Keep it under 130 words
- End with: Best regards,\\nHiring Team
- Return only the email body, no subject line"""

        response = model.generate_content(prompt)
        body     = response.text.strip()
        return {"success": True, "email_body": body}
    except Exception as e:
        print(f"[Gemini rejection fallback] {e}")
        body = f"""Dear {req.candidate_name},

Thank you for taking the time to apply and for your interest in this role.

After careful review, we will not be moving forward with your application at this time. {gap_line}

{req.raw_note or "We encourage you to keep building your skills and welcome you to apply again in the future."}

Best regards,
Hiring Team"""
        return {"success": True, "email_body": body}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT — Get rejected candidates from DB
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/rejected")
async def get_rejected():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT name, email, filename, match_score, ats_score, missing_skills, screened_at FROM rejected ORDER BY match_score DESC")
    rows = c.fetchall()
    conn.close()
    return {
        "success": True,
        "count":   len(rows),
        "candidates": [
            {
                "name":           r[0], "email": r[1], "filename": r[2],
                "match_score":    r[3], "ats_score": r[4],
                "missing_skills": json.loads(r[5]) if r[5] else [],
                "screened_at":    r[6],
            }
            for r in rows
        ]
    }
@app.get("/api/shortlisted")
async def get_shortlisted():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT name, email, filename, match_score, ats_score, exp_years, screened_at FROM shortlisted ORDER BY match_score DESC")
    rows = c.fetchall()
    conn.close()
    return {
        "success": True,
        "count":   len(rows),
        "candidates": [
            {
                "name": r[0], "email": r[1], "filename": r[2],
                "match_score": r[3], "ats_score": r[4],
                "exp_years": r[5], "screened_at": r[6]
            }
            for r in rows
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT — Fetch resumes from Google Drive folder
# ══════════════════════════════════════════════════════════════════════════════
class DriveFetchRequest(BaseModel):
    folder_url: str

@app.post("/api/fetch-drive")
async def fetch_drive(req: DriveFetchRequest):
    """
    Fetches PDF/TXT files from a public Google Drive folder.
    Extracts folder ID from URL and uses Drive API (no auth needed for public folders).
    Returns list of {filename, content_b64} dicts.
    """
    import re, base64, httpx

    # Extract folder ID from URL
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', req.folder_url)
    if not match:
        # Maybe they pasted just the ID
        folder_id = req.folder_url.strip()
    else:
        folder_id = match.group(1)

    try:
        # List files in public folder via Drive API v3
        api_key  = os.getenv("GOOGLE_API_KEY", "")
        list_url = f"https://www.googleapis.com/drive/v3/files"
        params   = {
            "q":       f"'{folder_id}' in parents and (mimeType='application/pdf' or mimeType='text/plain')",
            "fields":  "files(id,name,mimeType)",
            "key":     api_key,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(list_url, params=params)
        if r.status_code != 200:
            raise Exception(f"Drive API error: {r.text}")

        drive_files = r.json().get("files", [])
        if not drive_files:
            return {"success": False, "error": "No PDF or TXT files found in that folder."}

        # Download each file
        results = []
        async with httpx.AsyncClient(timeout=30) as client:
            for f in drive_files[:30]:   # cap at 30
                dl_url  = f"https://www.googleapis.com/drive/v3/files/{f['id']}?alt=media&key={api_key}"
                dl_resp = await client.get(dl_url)
                if dl_resp.status_code == 200:
                    results.append({
                        "filename":    f["name"],
                        "content_b64": base64.b64encode(dl_resp.content).decode(),
                        "mime_type":   f["mimeType"],
                    })

        return {"success": True, "files": results, "count": len(results)}

    except Exception as e:
        return {"success": False, "error": str(e), "files": []}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT — Fetch resumes from Gmail inbox attachments
# ══════════════════════════════════════════════════════════════════════════════
class GmailFetchRequest(BaseModel):
    email:          str
    app_password:   str
    subject_filter: str = ""
    date_range:     str = "7"   # days — "7", "30", "90", or "SINCE DD-Mon-YYYY"

@app.post("/api/fetch-gmail")
async def fetch_gmail(req: GmailFetchRequest):
    """
    Connects to Gmail via IMAP.
    Scans by DATE RANGE — not a fixed count — so 10,000 emails work fine.
    Smarter detection: subject keywords + attachment filename + inline body text.
    """
    import imaplib
    import email as email_lib
    import base64
    from datetime import datetime, timedelta

    # ── Build IMAP date string ────────────────────────────────────────────────
    def to_imap_date(d: datetime) -> str:
        return d.strftime("%d-%b-%Y")   # e.g. "01-Jan-2025"

    since_str = None
    before_str = None   # IMAP BEFORE filter for custom end date

    if "|" in req.date_range:
        # Custom range: "2025-01-01|2025-06-01"
        parts = req.date_range.split("|")
        try:
            since_str  = to_imap_date(datetime.strptime(parts[0].strip(), "%Y-%m-%d"))
            before_str = to_imap_date(datetime.strptime(parts[1].strip(), "%Y-%m-%d") + timedelta(days=1))
        except ValueError:
            return {"success": False, "error": "Invalid custom date format.", "files": []}
    else:
        # Preset: number of days
        try:
            days = int(req.date_range)
            since_str = to_imap_date(datetime.now() - timedelta(days=days))
        except ValueError:
            since_str = req.date_range.replace("SINCE ", "").strip()

    # ── Keywords that indicate a job application email ────────────────────────
    APPLICATION_SUBJECT_KEYWORDS = [
        "application", "applying", "apply", "resume", "cv",
        "job application", "position", "opportunity", "candidate",
        "cover letter", "interest in", "opening",
    ]

    # ── Filename patterns that indicate a resume attachment ───────────────────
    RESUME_FILENAME_KEYWORDS = [
        "resume", "cv", "curriculum", "vitae", "application",
        "portfolio", "profile",
    ]
    RESUME_EXTENSIONS = [".pdf", ".txt", ".doc", ".docx"]

    results    = []
    seen_names = set()   # deduplicate by filename

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(req.email, req.app_password)
        mail.select("inbox")

        # ── Build search query ────────────────────────────────────────────────
        # Always filter by date — this is the core fix for scale
        if before_str:
            search_criteria = f'(SINCE {since_str} BEFORE {before_str})'
        else:
            search_criteria = f'SINCE {since_str}'

        # Add subject filter if provided
        if req.subject_filter.strip():
            search_criteria = f'({search_criteria} SUBJECT "{req.subject_filter}")'

        _, data = mail.search(None, search_criteria)
        mail_ids = data[0].split()

        if not mail_ids or mail_ids == [b'']:
            mail.logout()
            range_info = f"{since_str} to {before_str}" if before_str else f"last {req.date_range} days"
            return {
                "success": False,
                "error":   f"No emails found in range: {range_info}.",
                "files":   [],
                "scanned": 0,
            }

        # Process newest first
        mail_ids = list(reversed(mail_ids))
        scanned  = 0

        for mid in mail_ids:
            try:
                _, msg_data = mail.fetch(mid, "(RFC822)")
                msg         = email_lib.message_from_bytes(msg_data[0][1])
                scanned    += 1

                subject  = str(msg.get("Subject", "")).lower()
                sender   = str(msg.get("From",    ""))
                is_application_email = any(kw in subject for kw in APPLICATION_SUBJECT_KEYWORDS)

                found_attachment = False

                # ── Scan all parts of the email ───────────────────────────────
                for part in msg.walk():
                    ctype    = part.get_content_type()
                    cdispo   = str(part.get("Content-Disposition", ""))
                    filename = part.get_filename()

                    # ── Case 1: Named attachment ──────────────────────────────
                    if filename:
                        fname_lower = filename.lower()
                        ext_ok      = any(fname_lower.endswith(e) for e in RESUME_EXTENSIONS)
                        name_ok     = any(kw in fname_lower for kw in RESUME_FILENAME_KEYWORDS)

                        # Accept if: (resume keyword in name OR it's a PDF/TXT)
                        # AND (it's an attachment OR the email looks like an application)
                        is_resume_file = ext_ok and (name_ok or is_application_email or "attachment" in cdispo)

                        if is_resume_file and filename not in seen_names:
                            payload = part.get_payload(decode=True)
                            if payload and len(payload) > 500:   # skip tiny/corrupt files
                                seen_names.add(filename)
                                # Extract sender name for display
                                sender_name = sender.split("<")[0].strip().strip('"') or filename
                                results.append({
                                    "filename":    filename,
                                    "content_b64": base64.b64encode(payload).decode(),
                                    "mime_type":   ctype,
                                    "sender":      sender_name,
                                    "subject":     msg.get("Subject",""),
                                })
                                found_attachment = True

                    # ── Case 2: Plain text body — candidate pasted resume ─────
                    # Only if no attachment found and email looks like an application
                    elif (
                        not found_attachment
                        and is_application_email
                        and ctype == "text/plain"
                        and "attachment" not in cdispo
                    ):
                        body_text = part.get_payload(decode=True)
                        if body_text:
                            try:
                                body_str = body_text.decode("utf-8", errors="ignore")
                            except Exception:
                                body_str = ""

                            # Only treat as resume if body is substantial (>200 words)
                            word_count = len(body_str.split())
                            if word_count > 200:
                                sender_name = sender.split("<")[0].strip().strip('"') or "Candidate"
                                safe_name   = re.sub(r'[^a-zA-Z0-9_\- ]', '', sender_name)[:30]
                                fname       = f"{safe_name}_resume.txt"
                                if fname not in seen_names:
                                    seen_names.add(fname)
                                    results.append({
                                        "filename":    fname,
                                        "content_b64": base64.b64encode(body_str.encode()).decode(),
                                        "mime_type":   "text/plain",
                                        "sender":      sender_name,
                                        "subject":     msg.get("Subject",""),
                                    })

            except Exception as e:
                print(f"[GMAIL SKIP] email {mid}: {e}")
                continue

        mail.logout()

        if not results:
            return {
                "success": False,
                "error":   f"No resume attachments found in {scanned} emails over the last {req.date_range} days.",
                "files":   [],
                "scanned": scanned,
            }

        return {
            "success": True,
            "files":   results,
            "count":   len(results),
            "scanned": scanned,
            "message": f"Found {len(results)} resume(s) from {scanned} emails scanned.",
        }

    except imaplib.IMAP4.error as e:
        return {"success": False, "error": f"Gmail login failed: {str(e)}", "files": []}
    except Exception as e:
        return {"success": False, "error": str(e), "files": []}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT — Screen pre-fetched resumes (Drive / Gmail base64 files)
# ══════════════════════════════════════════════════════════════════════════════
class FetchedFile(BaseModel):
    filename:    str
    content_b64: str
    mime_type:   str = "application/pdf"

class ScreenFetchedRequest(BaseModel):
    job_description: str
    selected_model:  str  = "SVM (RBF Kernel)"
    fresher_mode:    bool = False
    min_match:       int  = 60
    min_ats:         int  = 50
    files:           List[FetchedFile]

@app.post("/api/screen-fetched")
async def screen_fetched(req: ScreenFetchedRequest):
    """
    Same as /api/screen but accepts pre-fetched base64 files
    instead of multipart uploads. Used for Drive and Gmail inputs.
    """
    import base64

    if not req.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description is empty.")
    if not req.files:
        raise HTTPException(status_code=400, detail="No files provided.")

    jd_skills    = extract_skills(req.job_description)
    batch_results = []

    for f in req.files:
        try:
            raw_bytes    = base64.b64decode(f.content_b64)
            display_name = (
                f.filename
                .replace(".pdf","").replace(".txt","")
                .replace("_"," ").strip()
            )

            if f.filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(raw_bytes)
            else:
                text = extract_text_from_txt(raw_bytes)

            if not text.strip():
                text = f.filename

            scores = score_candidate(
                text, req.job_description,
                fresher_mode=req.fresher_mode,
                use_bert=True
            )

            match_score = int(scores.get("match_score", 0))
            ats_score   = int(scores.get("ats_score", 0))

            if req.selected_model == "Logistic Regression":
                match_score = max(0, min(100, match_score - 2))

            gap_data       = scores.get("gap_analysis", {})
            missing_skills = list(gap_data.get("missing_skills", []))
            suggestions    = list(gap_data.get("suggestions", []))
            candidate_email = extract_email_from_text(text)
            is_shortlisted  = match_score >= req.min_match and ats_score >= req.min_ats

            # Save to DB same as manual upload
            if candidate_email:
                conn = sqlite3.connect(DB_PATH)
                c    = conn.cursor()
                if is_shortlisted:
                    c.execute("DELETE FROM shortlisted WHERE email=?", (candidate_email,))
                    c.execute("INSERT INTO shortlisted (name,email,filename,match_score,ats_score,exp_years) VALUES (?,?,?,?,?,?)",
                              (display_name, candidate_email, f.filename, match_score, ats_score, scores.get("exp_years",0)))
                else:
                    c.execute("DELETE FROM rejected WHERE email=?", (candidate_email,))
                    c.execute("INSERT INTO rejected (name,email,filename,match_score,ats_score,missing_skills) VALUES (?,?,?,?,?,?)",
                              (display_name, candidate_email, f.filename, match_score, ats_score, json.dumps(missing_skills)))
                conn.commit()
                conn.close()

            batch_results.append({
                "name":           display_name,
                "filename":       f.filename,
                "email":          candidate_email,
                "match_score":    match_score,
                "ats_score":      ats_score,
                "ats_breakdown":  scores.get("ats_breakdown", {}),
                "skill_score":    float(scores.get("skill_score", 0)),
                "sim":            round(float(scores.get("sim", 0)), 4),
                "bert_sim":       scores.get("bert_sim"),
                "matched_skills": list(scores.get("matched_skills", [])),
                "missing_skills": missing_skills,
                "criteria":       scores.get("criteria", []),
                "exp_years":      float(scores.get("exp_years", 0)),
                "suggestions":    suggestions,
                "is_shortlisted": is_shortlisted,
            })
        except Exception as e:
            print(f"[SKIP] {f.filename}: {e}")
            continue

    if not batch_results:
        raise HTTPException(status_code=400, detail="Could not extract text from any files.")

    batch_results.sort(key=lambda x: x["match_score"], reverse=True)
    return {
        "success":        True,
        "total_screened": len(batch_results),
        "selected_model": req.selected_model,
        "jd_skills":      jd_skills,
        "results":        batch_results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
