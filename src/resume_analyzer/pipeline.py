"""Step 6a: the whole pipeline in one call, plus UI-ready table helpers.

    result = analyze(resume_bytes, "resume.pdf", jd_text)

Everything the demo shows comes from `AnalysisResult`. This module has no
Streamlit dependency, so it is unit-tested; app.py only draws what it returns.

Stages (matches the architecture in the project proposal):
    1 Resume input            extraction.extract_text_from_bytes
    2 Document understanding  sections.parse_resume
    3 Candidate profile       profile.build_profile
    4 Job understanding       jd.parse_job_description
    5 Matching (baseline)     matching.TfidfMatcher      <- embeddings come later
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .extraction import extract_text_from_bytes
from .jd import parse_job_description
from .matching import Matcher, TfidfMatcher, baseline_similarity
from .profile import _STRENGTH_RANK, build_profile
from .schema import (CandidateProfile, ExtractionResult, JobDescription, RequirementMatch,
                     ResumeSentence)
from .sections import parse_resume

ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_RESUMES = ROOT / "data" / "sample_resumes"
SAMPLE_JDS = ROOT / "data" / "sample_jds"

# Stage status shown in the demo. Update as the project grows.
STAGES = [
    ("1", "Resume input", "done"),
    ("2", "Document understanding", "done"),
    ("3", "Candidate profile", "done"),
    ("4", "Job understanding", "done"),
    ("5", "Semantic matching", "baseline"),     # TF-IDF now, embeddings next
    ("6", "Gap analysis", "planned"),
    ("7", "Resume quality engine", "planned"),
    ("8", "Explainable report", "planned"),
]
STAGE_LABELS = {"done": "Done", "baseline": "Baseline done (embeddings next)",
                "planned": "Planned"}


@dataclass
class AnalysisResult:
    extraction: ExtractionResult
    sentences: List[ResumeSentence]
    profile: CandidateProfile
    jd: JobDescription
    matches: List[RequirementMatch]
    matcher_name: str
    baseline_score: float
    warnings: List[str] = field(default_factory=list)


def analyze(resume_bytes: bytes, filename: str, jd_text: str,
            matcher: Optional[Matcher] = None) -> AnalysisResult:
    """Run stages 1-5 on an uploaded resume and a pasted job description."""
    matcher = matcher or TfidfMatcher()
    extraction = extract_text_from_bytes(resume_bytes, filename)
    warnings = list(extraction.warnings)
    if not extraction.ok:
        warnings.append("No text could be extracted from the resume.")
    sentences = parse_resume(extraction.text)
    profile = build_profile(sentences)
    jd = parse_job_description(jd_text)
    if not jd.requirements:
        warnings.append("No requirements were found in the job description. "
                        "Check that it has Responsibilities / Requirements sections.")
    matches = matcher.match(profile, jd, top_k=3) if jd.requirements else []
    return AnalysisResult(extraction, sentences, profile, jd, matches, matcher.name,
                          baseline_similarity(matches), warnings)


# --------------------------------------------------------------------------
# Sample files
# --------------------------------------------------------------------------
def list_sample_resumes() -> List[str]:
    return sorted(p.name for p in SAMPLE_RESUMES.glob("*") if p.suffix.lower() in (".pdf", ".docx"))


def list_sample_jds() -> List[str]:
    return sorted(p.name for p in SAMPLE_JDS.glob("*.txt"))


def read_sample_resume(name: str) -> bytes:
    return (SAMPLE_RESUMES / Path(name).name).read_bytes()      # .name blocks path tricks


def read_sample_jd(name: str) -> str:
    return (SAMPLE_JDS / Path(name).name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Table rows for the UI (plain dicts, so no pandas needed here)
# --------------------------------------------------------------------------
def section_rows(sentences: List[ResumeSentence]) -> List[Dict]:
    return [{"#": s.index, "Section": s.section, "Type": s.kind, "Text": s.text,
             "Belongs to": s.parent or ""} for s in sentences]


def skill_rows(profile: CandidateProfile) -> List[Dict]:
    return [{"Skill": r.skill, "Category": r.category, "Evidence": r.strength,
             "Mentions": len(r.mentions),
             "Where": ", ".join(sorted({m.section for m in r.mentions}))}
            for r in profile.skills]


def requirement_rows(jd: JobDescription) -> List[Dict]:
    rows = []
    for r in jd.requirements:
        alts = " | ".join("/".join(g) for g in r.skill_groups if len(g) > 1)
        rows.append({"#": r.index, "Priority": r.priority, "Type": r.kind, "Requirement": r.text,
                     "Skills": ", ".join(r.skills), "Any-of": alts,
                     "Min years": r.min_years if r.min_years is not None else "",
                     "Degree": r.degree_level or ""})
    return rows


def strength_counts(profile: CandidateProfile) -> Dict[str, int]:
    counts = {"demonstrated": 0, "supported": 0, "claimed": 0}
    for r in profile.skills:
        counts[r.strength] += 1
    return counts


def jd_skill_coverage(profile: CandidateProfile, jd: JobDescription) -> Dict[str, List[str]]:
    """Which JD skill requirements the resume mentions by name, and how strongly.

    Respects "any-of" groups: "Flask, Django or FastAPI" is satisfied by Flask alone.
    Name-level check only (no semantics); the real gap analysis comes with the
    embedding stage.
    """
    have = {r.skill: r.strength for r in profile.skills}
    groups: Dict[tuple, str] = {}                 # skill group -> priority (must wins)
    for req in jd.requirements:
        for g in req.skill_groups:
            key = tuple(g)
            if groups.get(key) != "must":
                groups[key] = req.priority
    # A skill required on its own already covers any optional group it belongs to.
    out = {"must_found": [], "must_missing": [], "nice_found": [], "nice_missing": []}
    for key, prio in groups.items():
        found = [sk for sk in key if sk in have]
        if found:
            best = max(found, key=lambda sk: _STRENGTH_RANK[have[sk]])
            text = f"{best} ({have[best]})" if len(key) == 1 else \
                f"{' / '.join(key)}: has {best} ({have[best]})"
        else:
            text = " / ".join(key) if len(key) == 1 else f"any of {' / '.join(key)}"
        out[f"{prio}_{'found' if found else 'missing'}"].append(text)
    return out


def format_duration(months: int) -> str:
    """Approximate duration for display: 22 -> '1y 10m', 3 -> '3m', 0 -> 'none'."""
    if not months:
        return "none"
    y, m = divmod(months, 12)
    return " ".join(p for p in (f"{y}y" if y else "", f"{m}m" if m else "") if p)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:80]
