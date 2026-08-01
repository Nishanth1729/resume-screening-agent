# Resume Screening Agent

**My agent takes** a job description and a folder of resumes (TXT/DOCX/PDF), **and produces** an ordered, scored shortlist (JSON + CSV) with a written reason for every candidate.

A runnable, transparent NLP agent that ranks a batch of resumes against a job description. Built for reviewer reproducibility: the included 10 synthetic resumes and job description run locally, end to end, with **no API key**.

## Features

- Parses PDF, DOCX, and TXT resumes
- Extracts skills, experience, and education
- Computes hybrid relevance scores using TF-IDF and optional semantic similarity
- Generates explainable rankings with detailed reasoning
- Exports ranked results to CSV and JSON
- Processes batches of resumes in a single run

## What it does

`Job description + resume folder → ordered JSON/CSV shortlist with scores and reasons.`

The agent reads documents, extracts a configurable skills catalogue, detects experience and education signals, calculates TF-IDF or optional embedding cosine similarity against the JD, and writes ranked results plus an audit log. The included TXT demo uses only the Python standard library; the two listed dependencies only enable DOCX and PDF input.

## Setup instructions

Requires Python 3.10+.

**1. Install**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## 2. Configure API Keys
None needed. This agent is fully deterministic (TF-IDF + regex-based extraction) and runs offline — see [Approach & model choice](#approach--model-choice) for why.

**3. Run end to end**
```bash
python src/resume_agent.py --jd data/job_description.txt --resumes data/resumes --output outputs/ranked_candidates
```
This produces `outputs/ranked_candidates.json`, `.csv`, and `.audit.json`, plus a ranked shortlist printed to the console.

**4. Run the regression tests**
```bash
python -m unittest discover -s tests -v
```

## Agent-specific deliverables

As required for the Resume Screening Agent:

| Deliverable | Location |
|---|---|
| Job Description (JD) | [`data/job_description.txt`](data/job_description.txt) |
| Folder of sample resumes | [`data/resumes/`](data/resumes/) — 10 synthetic candidates |
| Ranked output (CSV/JSON) | [`samples/ranked_candidates.csv`](samples/ranked_candidates.csv), [`samples/ranked_candidates.json`](samples/ranked_candidates.json) — reproducible demo output |
| Note explaining the scoring method | [Scoring method](#scoring-method) below |

## Sample output preview

Top 3 of 10 ranked candidates from the included demo run (full results in `samples/`):

| Rank | Name | Overall | Matched skills | Reasoning (excerpt) |
|---|---|---:|---|---|
| 1 | Aisha Khan | 58.8 | 11/12 JD skills | 3 yrs experience, Master's, LLM/RAG/FastAPI/Docker matched |
| 2 | Elena Petrov | 54.9 | 11/12 JD skills | 6 yrs experience, PhD, missing FastAPI only |
| 3 | Charu Iyer | 52.3 | 9/12 JD skills | 2 yrs experience, Bachelor's, missing LLM/RAG + data analysis |

## Scoring method

The final score is a weighted, auditable blend:

| Component | Weight | Calculation |
|---|---:|---|
| NLP relevance | 45% | TF-IDF by default; optional semantic embedding or hybrid similarity |
| Required skills | 35% | Share of JD catalogue skills found in the resume |
| Experience | 15% | Detected years, capped at the JD requirement |
| Education | 5% | Degree signal detected in the resume |

The JSON output contains every component score, detected years, matched/missing skills, and the reason string shown to a reviewer — nothing is hidden inside a single opaque number.

## Approach & model choice

We deliberately **do not use an LLM to calculate scores**. Deterministic NLP (TF-IDF cosine similarity) plus rule-based extraction (regex for skills/years/education) makes every ranking reproducible, auditable, and free of hallucinated qualifications — the same resume and JD always produce the same score, and every score can be traced back to matched text.

This trades off the semantic flexibility an LLM or embedding model would offer (e.g. recognizing "led a team of 5 engineers" as a seniority signal without the word "years") for full transparency. An LLM could be layered on top later purely to rewrite the `reasoning` field in more natural language, without ever being allowed to change the underlying score — keeping the ranking itself auditable while improving the write-up.

### Optional semantic embeddings

For semantic comparison, install the optional local dependency (no API key is needed):

```bash
pip install -r requirements-semantic.txt
python src/resume_agent.py --jd data/job_description.txt --resumes data/resumes --output outputs/semantic_ranking --similarity semantic
```

`--similarity hybrid` averages TF-IDF and embedding similarity. Semantic and hybrid output retain `tfidf_similarity_score`, making the baseline directly comparable. The default stays offline TF-IDF; semantic mode downloads the named Sentence Transformers model once when it is first used. Pass a role-specific JSON skill list with `--skills data/ai_research_skills.json` instead of relying on the global catalogue.

This generates `semantic_ranking.json`, `semantic_ranking.csv`, and `semantic_ranking.audit.json`.

## Design trade-offs, limitations, and what we'd improve with more time

- TF-IDF is inexpensive, explainable, and works offline, but it doesn't understand semantic equivalents as well as embeddings would (e.g. "led model deployment" vs. "MLOps").
- Skills are matched against a small explicit catalogue (`SKILLS` in `resume_agent.py`); a production deployment would expand this per role.
- Experience and education use conservative regex signals (`year`/`years`/`yr`/`yrs`) and can still miss unusual résumé formatting or non-English degree titles.
- PDF extraction depends on the PDF having a text layer; scanned PDFs would need OCR (e.g. Tesseract) — not implemented here.
- This is a decision-support tool, not an automated hiring decision-maker. A human should review the evidence, apply consistent policy, and audit for bias before acting on any ranking.
- Every run writes an audit log containing input SHA-256 fingerprints, scoring weights, the skills catalogue, and the selected method/model. This supports reproducibility and human review, but does not replace monitoring selection outcomes for disparate impact.
- **With more time**, we'd add: OCR fallback for scanned PDFs; date-range experience parsing; and the optional LLM-narrative layer described above.

## Future Improvements

- Dynamic skill extraction from the job description
- Experience extraction from employment date ranges
- OCR support for scanned PDFs
- Section-aware resume parsing
- Ranking evaluation with NDCG and Precision@K
- Configurable weighting profiles for different job roles

## Project Structure

```text
data/
  job_description.txt
  resumes/
outputs/                    created when you run the agent
samples/                    checked-in example outputs
src/
  resume_agent.py
tests/
  test_agent.py
requirements.txt
requirements-semantic.txt
README.md
```
