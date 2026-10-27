"""Step 4: job description text -> structured requirements.

    classify_lines(text)         label every line (title / heading / about /
                                 responsibility / requirement / nice)
    parse_job_description(text)  -> JobDescription with one `Requirement` per
                                 bullet/sentence, each carrying priority, skills,
                                 skill alternatives, years and degree level

Like Step 2 this is rule-based on purpose: transparent, fast, and a baseline
that a learned model can be compared against.

CLI:
    python -m resume_analyzer.jd path/to/jd.txt
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .extraction import clean_text
from .profile import _LEVEL_PATTERNS, _LEVEL_RANK
from .schema import JobDescription, Requirement
from .skills import SkillHit, extract_skills

# --------------------------------------------------------------------------
# Headings
# --------------------------------------------------------------------------
_HEADING_EXACT = {
    # responsibility
    "responsibilities": "responsibility", "key responsibilities": "responsibility",
    "duties": "responsibility", "what youll do": "responsibility",
    "what you will do": "responsibility", "your role": "responsibility",
    "role and responsibilities": "responsibility", "day to day": "responsibility",
    # requirement
    "requirements": "requirement", "qualifications": "requirement",
    "required skills": "requirement", "required qualifications": "requirement",
    "minimum qualifications": "requirement", "basic qualifications": "requirement",
    "skills": "requirement", "technical skills": "requirement", "key skills": "requirement",
    "must have": "requirement", "must haves": "requirement",
    "what were looking for": "requirement", "what youll bring": "requirement",
    "who you are": "requirement", "what you need": "requirement",
    "skills required": "requirement", "eligibility": "requirement", "your profile": "requirement",
    # nice to have
    "nice to have": "nice", "good to have": "nice", "preferred": "nice",
    "preferred qualifications": "nice", "preferred skills": "nice", "bonus": "nice",
    "bonus points": "nice", "plus": "nice", "desirable": "nice", "desired skills": "nice",
    "additional skills": "nice",
    # not matched against the resume
    "about the role": "about", "about us": "about", "about the company": "about",
    "overview": "about", "job summary": "about", "summary": "about", "benefits": "about",
    "perks": "about", "what we offer": "about", "why join us": "about",
    "how to apply": "about", "location": "about", "compensation": "about", "salary": "about",
}


def _norm_heading(line: str) -> str:
    line = line.lower().replace("\u2019", "").replace("'", "")
    line = re.sub(r"[^a-z0-9\s/-]", " ", line)
    return re.sub(r"\s+", " ", line).strip(" -:")


def detect_jd_heading(line: str) -> Optional[Tuple[str, bool]]:
    """Return (kind, skills_context) if `line` is a JD section heading."""
    raw = line.strip()
    if not raw or raw.startswith("- ") or raw.endswith((".", ",", ";")):
        return None
    norm = _norm_heading(raw)
    if not norm or len(norm.split()) > 6:
        return None
    kind = _HEADING_EXACT.get(norm)
    if kind is None:                                   # suffix rules
        first = norm.split()[0]
        if norm.endswith("responsibilities"):
            kind = "responsibility"
        elif norm.endswith(("requirements", "qualifications", "skills")):
            kind = "nice" if first in ("preferred", "desired", "additional", "bonus") \
                else "requirement"
    if kind is None:
        return None
    return kind, norm.endswith("skills") or norm in ("must have", "must haves")


# --------------------------------------------------------------------------
# Line classification
# --------------------------------------------------------------------------
_NICE_CUE = re.compile(
    r"\b(preferred|nice to have|bonus|(?:is |are )?a plus|good to have|desirable|"
    r"an advantage|an asset)\b", re.I)
_RESP_CUE = re.compile(
    r"^(?:you will|you'll|responsible for|your role)|\byou will\b|"
    r"^(?:develop|design|build|write|maintain|collaborate|deploy|create|implement|lead|manage|"
    r"test|analy[sz]e|present|monitor|integrate|train|clean|review|debug)\b", re.I)
_REQ_CUE = re.compile(
    r"\b(should|must|required|requirements?|experience (?:with|in)|knowledge of|proficien\w+|"
    r"familiar\w*|degree|diploma|years?|ability to|understanding of|skilled)\b", re.I)


def _cue_kind(text: str) -> str:
    """Fallback for lines that sit before any heading (or in heading-less JDs)."""
    t = re.sub(r"^[-\s]+", "", text)
    if _NICE_CUE.search(t):
        return "nice"
    if _RESP_CUE.search(t):
        return "responsibility"
    if _REQ_CUE.search(t):
        return "requirement"
    return "about"


@dataclass
class JDLine:
    text: str
    kind: Optional[str]          # title|about|responsibility|requirement|nice|None(blank)
    is_heading: bool = False
    skills_context: bool = False


_INLINE_HEADING = re.compile(r"^([A-Za-z' /&-]{3,40}):\s+(\S.*)$")


def _split_inline_headings(text: str) -> str:
    """'Requirements: Python, SQL' -> two lines (heading, then content)."""
    out = []
    for line in text.split("\n"):
        m = _INLINE_HEADING.match(line.strip())
        if m and detect_jd_heading(m.group(1)):
            out += [m.group(1) + ":", m.group(2)]
        else:
            out.append(line)
    return "\n".join(out)


def classify_lines(text: str) -> List[JDLine]:
    out: List[JDLine] = []
    section: Optional[str] = None
    skills_ctx = False
    seen_title = False
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            out.append(JDLine("", None))
            continue
        head = detect_jd_heading(line)
        if head:
            section, skills_ctx = head
            out.append(JDLine(line, section, True, skills_ctx))
            seen_title = True
            continue
        if not seen_title:
            seen_title = True
            if len(line) <= 90 and not line.endswith(".") and not line.startswith("- "):
                out.append(JDLine(line, "title"))
                continue
        kind = section if section is not None else _cue_kind(line)
        out.append(JDLine(line, kind, False, skills_ctx))
    return out


# --------------------------------------------------------------------------
# Requirement details: years, degree, skill alternatives
# --------------------------------------------------------------------------
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:-\s*\d+(?:\.\d+)?|to\s*\d+(?:\.\d+)?)?\s*\+?\s*years?\b", re.I)
_NUMBERED = re.compile(r"^\d+[.)]\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def parse_min_years(text: str) -> Optional[float]:
    m = _YEARS_RE.search(text)
    return float(m.group(1)) if m else None


def parse_degree_level(text: str) -> Optional[str]:
    """Lowest degree level the text accepts ('B.Tech, B.E. or M.Tech' -> bachelor)."""
    if not re.search(r"degree|diploma|b\.?\s?tech|b\.?e\b|bca|bsc|b\.?sc|m\.?\s?tech|mca|msc|"
                     r"m\.?sc|mba|bachelor|master|ph\.?d|pursuing", text, re.I):
        return None
    found = [lvl for lvl, rx in _LEVEL_PATTERNS if rx.search(text) and lvl != "secondary"]
    if not found:
        return "bachelor" if re.search(r"\bdegree\b", text, re.I) else None
    return min(found, key=_LEVEL_RANK.get)


_ALT_GAP = re.compile(r"^\s*(?:,\s*or|/|\bor\b|and/or)\s*$", re.I)
_AND_GAP = re.compile(r"^\s*(?:,\s*and|,|&|\band\b)\s*$", re.I)


def build_skill_groups(text: str, hits: List[SkillHit]) -> List[List[str]]:
    """Group skills joined by 'or' or '/' as alternatives (any one will do).

    "Python (Flask, Django or FastAPI)" -> [["Python"], ["Flask","Django","FastAPI"]]
    "Git, Docker and CI/CD"             -> [["Git"], ["Docker"], ["CI/CD"]]
    """
    if not hits:
        return []
    hits = sorted(hits, key=lambda h: h.start)
    runs: List[List[SkillHit]] = [[hits[0]]]
    alt = [False]
    for prev, cur in zip(hits, hits[1:]):
        gap = text[prev.end:cur.start]
        is_alt, is_and = bool(_ALT_GAP.match(gap)), bool(_AND_GAP.match(gap))
        if is_alt or is_and:
            runs[-1].append(cur)
            alt[-1] = alt[-1] or is_alt
        else:
            runs.append([cur])
            alt.append(False)
    groups: List[List[str]] = []
    for run, is_alt in zip(runs, alt):
        if is_alt:
            groups.append([h.skill for h in run])
        else:
            groups += [[h.skill] for h in run]
    return groups


# --------------------------------------------------------------------------
# Main parser
# --------------------------------------------------------------------------
def parse_job_description(text: str) -> JobDescription:
    text = _split_inline_headings(clean_text(text))
    jd = JobDescription()
    items: List[dict] = []          # {text, kind, skills_ctx}

    for ln in classify_lines(text):
        if not ln.text or ln.is_heading:
            continue
        if ln.kind == "title":
            jd.title = ln.text
            continue
        if ln.kind == "about":
            jd.about.append(re.sub(r"^-\s+", "", ln.text))
            continue

        is_bullet = ln.text.startswith("- ") or bool(_NUMBERED.match(ln.text))
        content = _NUMBERED.sub("", re.sub(r"^-\s+", "", ln.text)).strip()
        prev = items[-1] if items else None
        # A plain line continues the bullet above when it starts in lower case, or when
        # that bullet has not ended yet (no closing punctuation): wrapped lines often
        # begin with a capitalised tech name ("...knowledge of Python and\nDjango").
        unfinished_bullet = bool(prev and prev["bullet"]
                                 and not prev["text"].rstrip().endswith((".", "!", "?", ":", ";")))
        if (not is_bullet and prev and prev["kind"] == ln.kind
                and (content[:1].islower() or unfinished_bullet)):
            prev["text"] += " " + content                    # wrapped line
            continue
        pieces = [content] if is_bullet else _SENTENCE_SPLIT.split(content)
        for piece in pieces:
            items.append({"text": piece.strip(), "kind": ln.kind,
                          "skills_ctx": ln.skills_context, "bullet": is_bullet})

    must, nice = [], []
    for it in items:
        txt = it["text"]
        is_nice = it["kind"] == "nice" or bool(_NICE_CUE.search(txt))
        hits = extract_skills(txt, section="skills" if it["skills_ctx"] else "requirements")
        req = Requirement(
            index=len(jd.requirements), text=txt,
            kind="responsibility" if it["kind"] == "responsibility" else "qualification",
            priority="nice" if is_nice else "must",
            skills=[h.skill for h in hits],
            skill_groups=build_skill_groups(txt, hits),
            min_years=parse_min_years(txt),
            degree_level=parse_degree_level(txt),
        )
        jd.requirements.append(req)
        for sk in req.skills:
            bucket = nice if is_nice else must
            if sk not in bucket:
                bucket.append(sk)

    jd.must_skills = must
    jd.nice_skills = [s for s in nice if s not in must]       # must wins
    mandatory = [r for r in jd.requirements if r.priority == "must"]
    years = [r.min_years for r in mandatory if r.min_years is not None]
    levels = [r.degree_level for r in mandatory if r.degree_level]
    jd.min_years = max(years) if years else None
    jd.degree_level = min(levels, key=_LEVEL_RANK.get) if levels else None
    return jd


def jd_from_file(path: str) -> JobDescription:
    from pathlib import Path
    return parse_job_description(Path(path).read_text(encoding="utf-8"))


def _main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m resume_analyzer.jd <job_description.txt>")
        return 2
    jd = jd_from_file(argv[1])
    print(f"Title: {jd.title}")
    print(f"Degree: {jd.degree_level}   Min years: {jd.min_years}")
    print(f"MUST skills: {', '.join(jd.must_skills)}")
    print(f"NICE skills: {', '.join(jd.nice_skills)}\n")
    for r in jd.requirements:
        extra = []
        if r.min_years is not None:
            extra.append(f"{r.min_years:g}+ yrs")
        if r.degree_level:
            extra.append(f"degree:{r.degree_level}")
        alts = [g for g in r.skill_groups if len(g) > 1]
        if alts:
            extra.append("any-of " + " | ".join("/".join(g) for g in alts))
        print(f"{r.index:>2} [{r.priority:<4} {r.kind[:5]}] {r.text}")
        if r.skills or extra:
            print(f"       skills={r.skills}  {'; '.join(extra)}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
