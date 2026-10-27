"""Step 3b: tagged sentences -> structured CandidateProfile.

The key idea is *evidence*: every skill in the profile keeps the sentences
that mention it, and is graded by where it appears:

    demonstrated  used inside an experience / project entry
    supported     appears in education, certifications or achievements
    claimed       only listed in a Skills line or the summary

A skill that is only "claimed" is weaker evidence than one that shows up in a
bullet describing what was built; later steps use this for gap analysis.

CLI:
    python -m resume_analyzer.profile path/to/resume.pdf
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from typing import Dict, List, Optional, Tuple

from .schema import (CandidateProfile, EducationEntry, ExperienceEntry, ProjectEntry,
                     ResumeSentence, SkillMention, SkillRecord)
from .skills import extract_skills

_STRENGTH_RANK = {"claimed": 0, "supported": 1, "demonstrated": 2}
_SECTION_STRENGTH = {
    "experience": "demonstrated", "projects": "demonstrated",
    "education": "supported", "certifications": "supported", "achievements": "supported",
    "skills": "claimed", "summary": "claimed", "other": "claimed",
}

# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_RANGE_RE = re.compile(
    rf"(?:({_MON})\s+)?((?:19|20)\d{{2}})\s*(?:[-–—]|\bto\b)\s*"
    rf"(?:({_MON})\s+)?((?:19|20)\d{{2}}|present|current|now)", re.I)
_SINGLE_YEAR_RE = re.compile(rf"\((?:({_MON})\s+)?((?:19|20)\d{{2}})\)", re.I)
_PAREN_DATES_RE = re.compile(r"\s*\([^)]*(?:19|20)\d{2}[^)]*\)")


def _month_num(token: Optional[str]) -> Optional[int]:
    return _MONTHS.get(token[:3].lower()) if token else None


def parse_duration(line: str, today: Optional[dt.date] = None) -> Tuple[Optional[int], bool]:
    """Return (months, approximate) for the date range in a role line.

    Month+year on both ends -> exact (inclusive). Year-only ranges are
    approximate (end_year - start_year, at least 6 months). A single year is
    taken as ~6 months and flagged approximate.
    """
    today = today or dt.date.today()
    m = _RANGE_RE.search(line)
    if m:
        sm, sy, em, ey = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        present = ey.lower() in ("present", "current", "now")
        end_year = today.year if present else int(ey)
        end_month = today.month if present else _month_num(em)
        start_month = _month_num(sm)
        if start_month and end_month:
            months = (end_year - sy) * 12 + (end_month - start_month) + 1
            return max(months, 1), False
        return max((end_year - sy) * 12, 6), True
    if _SINGLE_YEAR_RE.search(line):
        return 6, True
    return None, False


# --------------------------------------------------------------------------
# Education
# --------------------------------------------------------------------------
# Degree codes are case-sensitive on purpose: "BE" / "ME" must not match the words "be"/"me".
_LEVEL_PATTERNS = [
    ("phd", re.compile(r"\b(?:Ph\.?\s?D|Doctorate)\b", re.I)),
    ("master", re.compile(r"\b(?:M\.?\s?Tech|M\.?\s?Sc|MCA|MBA|M\.?E|M\.?Com)\.?(?![A-Za-z])"
                         r"|\b[Mm]aster(?:'?s)?\b")),
    ("bachelor", re.compile(r"\b(?:B\.?\s?Tech|B\.?E|B\.?\s?Sc|BCA|BBA|B\.?\s?Com)\.?(?![A-Za-z])"
                           r"|\b[Bb]achelor(?:'?s)?\b")),
    ("diploma", re.compile(r"\b(?:Diploma|Polytechnic)\b", re.I)),
    ("secondary", re.compile(r"\b(?:Senior Secondary|Higher Secondary|Class\s*(?:XII|XI|X|12|10)|"
                             r"12th|10th|HSC|SSC|CBSE|ICSE)\b", re.I)),
]
_LEVEL_RANK = {"secondary": 0, "diploma": 1, "bachelor": 2, "master": 3, "phd": 4}
_CGPA_RES = [re.compile(r"CGPA[:\s]*([0-9]+(?:\.[0-9]+)?)", re.I),
             re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*CGPA", re.I)]


def _education_level(text: str) -> Optional[str]:
    for level, rx in _LEVEL_PATTERNS:      # ordered highest first
        if rx.search(text):
            return level
    return None


def _cgpa(text: str) -> Optional[float]:
    for rx in _CGPA_RES:
        m = rx.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


# --------------------------------------------------------------------------
# Role line -> (role, org)
# --------------------------------------------------------------------------
def _split_role(title: str) -> Tuple[str, str]:
    clean = _PAREN_DATES_RE.sub("", title).strip()
    for sep in (",", " at ", "|"):
        if sep in clean:
            a, b = clean.split(sep, 1)
            return a.strip(), b.strip()
    return clean, ""


# --------------------------------------------------------------------------
# Main builder
# --------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Header lines that are contact details: never searched for skills (emails, URLs, phones).
_CONTACT_RE = re.compile(r"@|https?://|github\.com|linkedin\.com", re.I)
_PHONE_RE = re.compile(r"\+?\d[\d\s()-]{7,}\d")


def is_contact_line(text: str) -> bool:
    """Email / URL / phone number. A phone needs >= 10 digits, so a date range such as
    "(2022 - 2025)" (8 digits) is not mistaken for one."""
    if _CONTACT_RE.search(text):
        return True
    return any(sum(c.isdigit() for c in m.group(0)) >= 10 for m in _PHONE_RE.finditer(text))
_LINK_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:github\.com|linkedin\.com)/[\w./-]+", re.I)


def build_profile(sentences: List[ResumeSentence],
                  today: Optional[dt.date] = None) -> CandidateProfile:
    prof = CandidateProfile(sentences=sentences)

    # ---- header: name, email, links
    header = [s.text for s in sentences if s.section == "header"]
    if header:
        prof.name = header[0]
    for line in header:
        if not prof.email and (m := _EMAIL_RE.search(line)):
            prof.email = m.group(0)
        prof.links += [l for l in _LINK_RE.findall(line) if l not in prof.links]

    prof.summary = " ".join(s.text for s in sentences if s.section == "summary")

    # ---- skills with evidence (header lines are never searched: URLs, emails)
    found: Dict[str, SkillRecord] = {}
    sent_skills: Dict[int, List[str]] = {}
    name_idx = next((x.index for x in sentences if x.section == "header"), None)
    for s in sentences:
        # Skip only the name line and contact lines. (Skipping the whole header
        # broke resumes without headings, where Step 2 labels every line "header".)
        if s.section == "header" and (s.index == name_idx or is_contact_line(s.text)):
            continue
        hits = extract_skills(s.text, section=s.section)
        sent_skills[s.index] = [h.skill for h in hits]
        strength = _SECTION_STRENGTH.get(s.section, "claimed")
        for h in hits:
            rec = found.get(h.skill)
            if rec is None:
                rec = found[h.skill] = SkillRecord(h.skill, h.category, strength)
            if _STRENGTH_RANK[strength] > _STRENGTH_RANK[rec.strength]:
                rec.strength = strength
            rec.mentions.append(SkillMention(s.index, s.section, s.text, s.parent))
    prof.skills = sorted(found.values(),
                         key=lambda r: (-_STRENGTH_RANK[r.strength], -len(r.mentions), r.skill))

    # ---- education / experience / projects
    cur_exp: Optional[ExperienceEntry] = None
    cur_proj: Optional[ProjectEntry] = None
    for s in sentences:
        if s.section == "education":
            level, gpa = _education_level(s.text), _cgpa(s.text)
            if level is None and gpa is not None and prof.education \
                    and prof.education[-1].cgpa is None:
                prof.education[-1].cgpa = gpa          # "CGPA: 8.4/10" line under a degree
                prof.education[-1].text += " | " + s.text
            else:
                prof.education.append(EducationEntry(s.text, level, gpa))
        elif s.section == "experience":
            if s.kind == "title" or cur_exp is None:
                role, org = _split_role(s.text)
                months, approx = parse_duration(s.text, today)
                cur_exp = ExperienceEntry(
                    title=s.text, role=role, org=org, months=months, months_approx=approx,
                    is_internship="intern" in s.text.lower())
                prof.experience.append(cur_exp)
                if s.kind == "title":
                    continue
            cur_exp.bullets.append(s.text)
            for sk in sent_skills.get(s.index, []):
                if sk not in cur_exp.skills:
                    cur_exp.skills.append(sk)
        elif s.section == "projects":
            if s.kind == "title" or cur_proj is None:
                cur_proj = ProjectEntry(title=s.text)
                prof.projects.append(cur_proj)
                if s.kind == "title":
                    continue
            cur_proj.bullets.append(s.text)
            for sk in sent_skills.get(s.index, []):
                if sk not in cur_proj.skills:
                    cur_proj.skills.append(sk)

    levels = [e.level for e in prof.education if e.level]
    prof.highest_degree = max(levels, key=_LEVEL_RANK.get) if levels else None
    prof.total_experience_months = sum(e.months or 0 for e in prof.experience)
    return prof


def profile_from_file(path: str, today: Optional[dt.date] = None) -> CandidateProfile:
    from .extraction import extract_text
    from .sections import parse_resume
    return build_profile(parse_resume(extract_text(path).text), today)


def _main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m resume_analyzer.profile <resume.pdf|docx|txt>")
        return 2
    p = profile_from_file(argv[1])
    print(f"Name: {p.name}   Email: {p.email}   Links: {', '.join(p.links) or '-'}")
    print(f"Highest degree: {p.highest_degree}   "
          f"Experience: ~{p.total_experience_months} months "
          f"({len(p.experience)} entries)   Projects: {len(p.projects)}")
    for e in p.education:
        print(f"  EDU  [{e.level}] {e.text}" + (f"  (CGPA {e.cgpa})" if e.cgpa else ""))
    for e in p.experience:
        tag = f"~{e.months}mo{'*' if e.months_approx else ''}"
        print(f"  EXP  {e.role} @ {e.org}  {tag}  skills: {', '.join(e.skills)}")
    for pr in p.projects:
        print(f"  PROJ {pr.title}  skills: {', '.join(pr.skills)}")
    print("\nSKILLS (strength, #mentions):")
    for r in p.skills:
        print(f"  {r.skill:<22} {r.strength:<13} {len(r.mentions)}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
