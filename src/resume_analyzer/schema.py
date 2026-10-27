"""Shared data contract between the two halves of the pipeline.

Person A (extraction, sections, profile) PRODUCES `ResumeSentence` lists.
Person B (JD splitting, matching, dashboard) CONSUMES them, and produces
`Requirement` lists from job descriptions.

Keep this file small and stable so both people can work in parallel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Canonical section names used across the whole project.
SECTIONS = ("header", "summary", "education", "experience", "projects",
            "skills", "certifications", "achievements", "other", "unknown")


@dataclass
class ExtractionResult:
    """Output of Step 1: raw text plus diagnostics."""
    text: str
    source_path: str
    source_type: str              # "pdf" | "docx" | "txt"
    n_pages: int = 1
    n_chars: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.n_chars > 0


@dataclass
class ResumeSentence:
    """One unit of resume evidence (a bullet or sentence) tagged by section."""
    index: int
    text: str
    section: str = "unknown"      # one of SECTIONS (filled in Step 2)
    kind: str = "text"            # "bullet" | "title" | "text"
    parent: Optional[str] = None  # the role/project title a bullet belongs to


@dataclass
class Requirement:
    """One job requirement / responsibility (Step 4). This is the unit that
    Step 5 matches against resume evidence."""
    index: int
    text: str
    kind: Optional[str] = None            # "responsibility" | "qualification"
    priority: str = "must"                # "must" | "nice"
    skills: List[str] = field(default_factory=list)
    # Skills joined by "or" / "/" are alternatives: any ONE satisfies the group.
    # Single-item groups are individually required. e.g. [["Python"], ["Flask","Django"]]
    skill_groups: List[List[str]] = field(default_factory=list)
    min_years: Optional[float] = None     # "3+ years" -> 3
    degree_level: Optional[str] = None    # lowest degree level accepted


@dataclass
class JobDescription:
    title: str = ""
    requirements: List[Requirement] = field(default_factory=list)
    about: List[str] = field(default_factory=list)       # company blurb, benefits (not matched)
    must_skills: List[str] = field(default_factory=list)
    nice_skills: List[str] = field(default_factory=list)
    min_years: Optional[float] = None
    degree_level: Optional[str] = None


# --------------------------------------------------------------------------
# Step 3: candidate profile (the structured view of a resume)
# --------------------------------------------------------------------------
@dataclass
class SkillMention:
    """One place in the resume where a skill appears: this is the 'evidence'."""
    sentence_index: int
    section: str
    text: str
    parent: Optional[str] = None


@dataclass
class SkillRecord:
    skill: str
    category: str
    # "demonstrated": used in an experience/project entry
    # "supported":    appears in education / certifications / achievements
    # "claimed":      only listed in Skills / Summary
    strength: str
    mentions: List[SkillMention] = field(default_factory=list)


@dataclass
class EducationEntry:
    text: str
    level: Optional[str] = None       # secondary | diploma | bachelor | master | phd
    cgpa: Optional[float] = None


@dataclass
class ExperienceEntry:
    title: str                        # the raw role line
    role: str = ""
    org: str = ""
    months: Optional[int] = None      # approximate duration
    months_approx: bool = False       # True when only years were given
    is_internship: bool = False
    bullets: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)


@dataclass
class ProjectEntry:
    title: str
    bullets: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)


@dataclass
class CandidateProfile:
    name: str = ""
    email: str = ""
    links: List[str] = field(default_factory=list)
    summary: str = ""
    education: List[EducationEntry] = field(default_factory=list)
    experience: List[ExperienceEntry] = field(default_factory=list)
    projects: List[ProjectEntry] = field(default_factory=list)
    skills: List[SkillRecord] = field(default_factory=list)
    highest_degree: Optional[str] = None
    total_experience_months: int = 0
    sentences: List[ResumeSentence] = field(default_factory=list)

    def skill_names(self) -> List[str]:
        return [s.skill for s in self.skills]

    def get_skill(self, name: str) -> Optional[SkillRecord]:
        return next((s for s in self.skills if s.skill.lower() == name.lower()), None)


# --------------------------------------------------------------------------
# Step 5: matching requirements to resume evidence
# --------------------------------------------------------------------------
@dataclass
class Evidence:
    """A resume sentence offered as evidence for one requirement."""
    sentence_index: int
    text: str
    section: str
    parent: Optional[str]
    score: float                  # similarity in [0, 1]; meaning depends on the matcher


@dataclass
class RequirementMatch:
    requirement: Requirement
    evidence: List[Evidence] = field(default_factory=list)   # best first

    @property
    def best_score(self) -> float:
        return self.evidence[0].score if self.evidence else 0.0
