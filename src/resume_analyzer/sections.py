"""Step 2: clean text -> section-tagged evidence sentences.

Two stages:
  1. `detect_sections(text)`: find heading lines, tag every line with a section.
  2. `parse_resume(text)`: merge wrapped lines, split into bullets / sentences,
     and attach each bullet to the role or project title it belongs to.

Heading detection is rule-based on purpose: it is fast, explainable, and gives a
baseline that a learned classifier can be compared against later.

CLI:
    python -m resume_analyzer.sections path/to/resume.pdf
"""
from __future__ import annotations

import re
import sys
from typing import List, Optional, Tuple

from .schema import ResumeSentence

# --------------------------------------------------------------------------
# Heading vocabulary
# --------------------------------------------------------------------------
# Whole-line aliases (multi-word headings that keyword matching would miss).
ALIASES = {
    "work history": "experience", "employment history": "experience",
    "professional background": "experience",
    "academic background": "education",
    "about me": "summary", "career objective": "summary",
    "tech stack": "skills",
}

# Single keywords. A line is a heading if it contains one of these AND looks
# like a heading (see `detect_heading`).
KEYWORDS = {
    "summary": {"summary", "objective", "profile"},
    "education": {"education", "qualifications", "qualification", "academics"},
    "experience": {"experience", "internships", "internship", "employment"},
    "projects": {"projects", "project"},
    "skills": {"skills", "technologies", "competencies", "expertise", "skillset"},
    "certifications": {"certifications", "certification", "certificates", "courses",
                       "licenses"},
    "achievements": {"achievements", "achievement", "awards", "honours", "honors",
                     "accomplishments"},
    "other": {"interests", "hobbies", "references", "declaration"},
}
_KEYWORD_TO_SECTION = {w: sec for sec, ws in KEYWORDS.items() for w in ws}

# A heading never contains these (e.g. "Experience with Python").
_STOPWORDS = {"with", "in", "using", "for", "on", "at", "to", "the", "a", "an",
              "by", "from", "gained"}
_BULLET_PREFIX = "- "


def _normalise(line: str) -> str:
    line = line.lower().strip().rstrip(":").strip()
    line = re.sub(r"[^a-z0-9&\s-]", " ", line)
    return re.sub(r"\s+", " ", line).strip()


def detect_heading(line: str) -> Optional[str]:
    """Return the canonical section name if `line` is a section heading."""
    raw = line.strip()
    if not raw or raw.startswith(_BULLET_PREFIX):
        return None
    if any(ch in raw for ch in "|@") or any(ch.isdigit() for ch in raw):
        return None
    if raw.endswith((".", ",", ";")):
        return None

    norm = _normalise(raw)
    tokens = [t for t in norm.split() if t not in ("&", "and")]
    if not tokens or len(norm.split()) > 4:
        return None
    if any(t in _STOPWORDS for t in tokens):
        return None

    if norm in ALIASES:
        return ALIASES[norm]

    matched = [(_KEYWORD_TO_SECTION[t], i) for i, t in enumerate(tokens)
               if t in _KEYWORD_TO_SECTION]
    if not matched:
        return None

    looks_like_heading = (
        raw.isupper()                       # SKILLS & TOOLS
        or raw.endswith(":")                # Technical Skills:
        or tokens[-1] in _KEYWORD_TO_SECTION  # Personal Projects, Work Experience
    )
    if not looks_like_heading:              # e.g. the project title "Education Loan Tracker"
        return None
    return matched[-1][0] if tokens[-1] in _KEYWORD_TO_SECTION else matched[0][0]


def detect_sections(text: str) -> List[Tuple[str, str, bool]]:
    """Tag each line. Returns [(line, section, is_heading)].

    Lines before the first heading are the resume "header" (name, contacts).
    Blank lines inherit the current section.
    """
    out: List[Tuple[str, str, bool]] = []
    current = "header"
    for line in text.split("\n"):
        sec = detect_heading(line)
        if sec:
            current = sec
            out.append((line, sec, True))
        else:
            out.append((line, current, False))
    return out


# --------------------------------------------------------------------------
# Sentence / bullet building
# --------------------------------------------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# Prose sections get split into sentences. Education / skills / certifications /
# header lines are entries, never prose: "B.E. Computer Engineering, ..." is ONE line.
_PROSE_SECTIONS = {"summary", "experience", "projects", "achievements", "other"}
_TITLES = {"dr.", "mr.", "ms.", "mrs.", "prof.", "vs.", "e.g.", "i.e.", "etc."}


def split_sentences(text: str) -> List[str]:
    """Split on '. ' + capital, but never right after an abbreviation (B.E., M.Sc., Dr.)."""
    parts = _SENTENCE_SPLIT.split(text)
    out: List[str] = []
    for part in parts:
        if out:
            last = out[-1].split()[-1].lower() if out[-1].split() else ""
            is_abbrev = last in _TITLES or bool(re.search(r"[a-z]\.[a-z]", last))
            if is_abbrev:
                out[-1] += " " + part
                continue
        out.append(part)
    return out
# Sections where free-text lines wrap. In education / skills / certifications
# every line is normally its own entry, so we never merge there.
_MERGE_SECTIONS = {"summary", "experience", "projects", "achievements", "other"}


def _is_continuation(cur: str) -> bool:
    """A line that starts in lower case is the wrapped second half of the line above.

    (An earlier version also merged any long line without a full stop, but
    education entries are long and have no full stop, so it glued separate
    entries together. Lower-case start is the safer signal.)
    """
    return cur[:1].islower()


def parse_resume(text: str) -> List[ResumeSentence]:
    """Turn cleaned resume text into section-tagged evidence units."""
    units: List[dict] = []                # working list, finalised at the end
    for line, section, is_heading in detect_sections(text):
        line = line.strip()
        if not line or is_heading:
            continue

        is_bullet = line.startswith(_BULLET_PREFIX)
        content = line[len(_BULLET_PREFIX):].strip() if is_bullet else line

        prev = units[-1] if units else None
        if (not is_bullet and prev is not None and prev["section"] == section
                and section in _MERGE_SECTIONS and prev["kind"] != "title"
                and _is_continuation(content)):
            prev["text"] += " " + content          # merge wrapped line
            continue

        kind = "bullet" if is_bullet else "text"
        if (kind == "text" and section in ("experience", "projects")
                and len(content) < 110 and not content.endswith(".")):
            kind = "title"                          # role / project heading line
        units.append({"text": content, "section": section, "kind": kind})

    # Split paragraph-style text (e.g. summaries) into sentences.
    sentences: List[ResumeSentence] = []
    parent: Optional[str] = None
    last_section = None
    for u in units:
        if u["section"] != last_section:
            parent, last_section = None, u["section"]
        if u["kind"] == "title":
            parent = u["text"]
        pieces = (split_sentences(u["text"])
                  if u["kind"] == "text" and u["section"] in _PROSE_SECTIONS
                  else [u["text"]])
        for piece in pieces:
            sentences.append(ResumeSentence(
                index=len(sentences), text=piece.strip(), section=u["section"],
                kind=u["kind"],
                parent=parent if u["kind"] == "bullet" else None,
            ))
    return sentences


def _main(argv: List[str]) -> int:
    from .extraction import extract_text
    if len(argv) != 2:
        print("usage: python -m resume_analyzer.sections <resume.pdf|docx|txt>")
        return 2
    res = extract_text(argv[1])
    for s in parse_resume(res.text):
        tag = f"{s.section}/{s.kind}"
        parent = f"  [in: {s.parent}]" if s.parent else ""
        print(f"{s.index:>3} {tag:<20} {s.text}{parent}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
