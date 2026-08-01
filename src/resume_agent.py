#!/usr/bin/env python3
"""Rank resumes against a job description using transparent NLP scoring."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SKILLS = {
    "python", "sql", "machine learning", "deep learning", "nlp", "pandas",
    "scikit-learn", "tensorflow", "pytorch", "aws", "azure", "gcp", "docker",
    "kubernetes", "fastapi", "flask", "git", "tableau", "power bi", "excel",
    "statistics", "data analysis", "data visualization", "spark", "airflow",
    "llm", "rag", "langchain", "postgresql", "mysql", "javascript", "java",
}


@dataclass
class Candidate:
    filename: str
    name: str
    similarity_score: float
    skills_score: float
    experience_score: float
    education_score: float
    overall_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    years_experience: float | None
    education: str
    reasoning: str


def read_document(path: Path) -> str:
    """Read TXT, DOCX or PDF resumes. Optional parser packages give clear errors."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(path).paragraphs)
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    raise ValueError(f"Unsupported document type: {path.name}")


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def extract_skills(text: str) -> set[str]:
    text = normalized(text)
    # Accept the common plural spelling "LLMs" while keeping all other
    # catalogue entries exact, so unrelated words cannot create a skill hit.
    def found(skill: str) -> bool:
        expression = r"llms?" if skill == "llm" else re.escape(skill)
        return bool(re.search(r"(?<!\w)" + expression + r"(?!\w)", text))
    return {skill for skill in SKILLS if found(skill)}


def extract_years(text: str) -> float | None:
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:year|years|yr|yrs)\s+(?:of\s+)?experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\+?\s*(?:year|years|yr|yrs)",
    ]
    values = []
    for pattern in patterns:
        values.extend(float(v) for v in re.findall(pattern, normalized(text)))
    return max(values) if values else None


def extract_required_years(jd: str) -> float | None:
    return extract_years(jd)


def extract_education(text: str) -> str:
    value = normalized(text)
    if re.search(r"\b(ph\.?d|doctorate)\b", value):
        return "PhD"
    if re.search(r"\b(master|m\.?s\.?|mtech|m\.tech)\b", value):
        return "Master's"
    if re.search(r"\b(bachelor|b\.?s\.?|btech|b\.tech)\b", value):
        return "Bachelor's"
    if re.search(r"\b(diploma|associate)\b", value):
        return "Diploma/Associate"
    return "Not detected"


def candidate_name(text: str, fallback: str) -> str:
    for line in text.splitlines()[:5]:
        line = line.strip()
        if 2 <= len(line) <= 70 and re.fullmatch(r"[A-Za-z .'-]+", line):
            return line.title()
    return fallback.replace("_", " ").title()


def tokens(text: str) -> list[str]:
    """Return unigram and adjacent-bigram tokens for a compact TF-IDF model."""
    words = re.findall(r"[a-z][a-z+#.-]*", normalized(text))
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def tfidf_similarity(query: str, documents: list[str]) -> list[float]:
    """Dependency-free TF-IDF cosine similarity; deterministic and inspectable."""
    document_tokens = [tokens(query), *(tokens(doc) for doc in documents)]
    document_frequency = Counter(token for doc in document_tokens for token in set(doc))
    count = len(document_tokens)

    def vector(items: list[str]) -> dict[str, float]:
        frequencies = Counter(items)
        return {token: (1 + math.log(freq)) * math.log((1 + count) / (1 + document_frequency[token])) + 1 for token, freq in frequencies.items()}

    query_vector = vector(document_tokens[0])
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
    output = []
    for doc in document_tokens[1:]:
        candidate = vector(doc)
        dot = sum(value * candidate.get(token, 0.0) for token, value in query_vector.items())
        candidate_norm = math.sqrt(sum(value * value for value in candidate.values()))
        output.append(dot / (query_norm * candidate_norm) if candidate_norm else 0.0)
    return output


def make_reasoning(matched: Iterable[str], missing: Iterable[str], years: float | None, required: float | None) -> str:
    parts = []
    if matched:
        parts.append("Matched skills: " + ", ".join(sorted(matched)))
    else:
        parts.append("No catalogue skills matched")
    if missing:
        parts.append("Missing JD skills: " + ", ".join(sorted(missing)))
    if years is not None:
        unit = "year" if years == 1 else "years"
        parts.append(f"Detected experience: {years:g} {unit}")
    if required is not None and (years is None or years < required):
        parts.append(f"below the JD's {required:g}-year signal")
    return "; ".join(parts) + "."


def rank_resumes(jd: str, paths: list[Path]) -> list[Candidate]:
    texts = [read_document(p) for p in paths]
    # Word and two-word features balance direct skill hits with contextual relevance.
    similarities = tfidf_similarity(jd, texts)
    jd_skills = extract_skills(jd)
    required_years = extract_required_years(jd)
    results = []
    for path, text, similarity in zip(paths, texts, similarities):
        skills = extract_skills(text)
        matched, missing = skills & jd_skills, jd_skills - skills
        skills_score = 100 * len(matched) / max(len(jd_skills), 1)
        years = extract_years(text)
        experience_score = 50.0 if required_years is None else (0.0 if years is None else min(100.0, years / required_years * 100))
        education = extract_education(text)
        education_score = 100.0 if education in {"Bachelor's", "Master's", "PhD"} else 40.0
        # Weights are intentionally visible: relevance 45%, required skills 35%, experience 15%, education 5%.
        overall = 0.45 * similarity * 100 + 0.35 * skills_score + 0.15 * experience_score + 0.05 * education_score
        results.append(Candidate(path.name, candidate_name(text, path.stem), round(similarity * 100, 1), round(skills_score, 1), round(experience_score, 1), round(education_score, 1), round(overall, 1), sorted(matched), sorted(missing), years, education, make_reasoning(matched, missing, years, required_years)))
    return sorted(results, key=lambda item: item.overall_score, reverse=True)


def save_results(results: list[Candidate], output_base: Path) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output_base.with_suffix(".json"), output_base.with_suffix(".csv")
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rank", "name", "filename", "overall_score", "similarity_score", "skills_score", "experience_score", "education_score", "matched_skills", "missing_skills", "years_experience", "education", "reasoning"])
        writer.writeheader()
        for rank, result in enumerate(results, 1):
            row = asdict(result)
            row.update(rank=rank, matched_skills=", ".join(result.matched_skills), missing_skills=", ".join(result.missing_skills))
            writer.writerow(row)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank TXT, DOCX, and PDF resumes against a job description.")
    parser.add_argument("--jd", required=True, type=Path, help="Job description file (.txt)")
    parser.add_argument("--resumes", required=True, type=Path, help="Directory containing resumes")
    parser.add_argument("--output", default="outputs/ranked_candidates", type=Path, help="Output path without extension")
    args = parser.parse_args()
    paths = sorted(p for p in args.resumes.iterdir() if p.suffix.lower() in {".txt", ".pdf", ".docx"})
    if not paths:
        raise SystemExit("No supported resumes found (.txt, .pdf, .docx).")
    results = rank_resumes(read_document(args.jd), paths)
    json_path, csv_path = save_results(results, args.output)
    print(f"Ranked {len(results)} candidates. Saved {json_path} and {csv_path}.")
    for rank, result in enumerate(results, 1):
        print(f"{rank:>2}. {result.name:<22} {result.overall_score:>5.1f}  {result.reasoning}")


if __name__ == "__main__":
    main()
