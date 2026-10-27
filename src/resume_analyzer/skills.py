"""Step 3a: find skills in text using a curated taxonomy.

Why not just search for words?  Because
  * "Java" must not match inside "JavaScript",  "SQL" not inside "MySQL";
  * "C", "R" and "Go" are also ordinary letters/words;
  * "GitHub Actions" should not also count as "GitHub".
The extractor handles all three (word boundaries, ambiguity rules, and
longest-match-wins span suppression).

Taxonomy format: see the `_format` note in skills_taxonomy.json.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

TAXONOMY_PATH = Path(__file__).with_name("skills_taxonomy.json")

# Characters that make a match "part of a bigger token" if adjacent.
_LEFT = r"(?<![A-Za-z0-9+#.])"
_RIGHT = r"(?![A-Za-z0-9+#])"
_CTX_BEFORE = re.compile(r"\b(?:using|in|with|via|on)\s+$", re.I)


@dataclass(frozen=True)
class _Alias:
    skill: str
    category: str
    regex: Pattern[str]
    needs_context: bool


@dataclass(frozen=True)
class SkillHit:
    skill: str
    category: str
    start: int
    end: int


def _parse_token(token: str) -> Tuple[str, bool, bool]:
    case_sensitive = needs_context = False
    while token and token[0] in "!?":
        case_sensitive |= token[0] == "!"
        needs_context |= token[0] == "?"
        token = token[1:]
    return token.strip(), case_sensitive, needs_context


class SkillTaxonomy:
    def __init__(self, path: Path = TAXONOMY_PATH):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.category_of: Dict[str, str] = {}
        self.aliases: List[_Alias] = []
        for category, entries in raw.items():
            if category.startswith("_"):
                continue
            for entry in entries:
                tokens = entry.split("|")
                canonical = _parse_token(tokens[0])[0]
                self.category_of[canonical] = category
                for tok in tokens:
                    name, cs, ctx = _parse_token(tok)
                    if not name:
                        continue
                    flags = 0 if cs else re.IGNORECASE
                    rx = re.compile(_LEFT + re.escape(name) + _RIGHT, flags)
                    self.aliases.append(_Alias(canonical, category, rx, ctx))

    @property
    def skills(self) -> List[str]:
        return sorted(self.category_of)


@lru_cache(maxsize=1)
def default_taxonomy() -> SkillTaxonomy:
    return SkillTaxonomy()


def extract_skills(text: str, section: Optional[str] = None,
                   taxonomy: Optional[SkillTaxonomy] = None) -> List[SkillHit]:
    """Return the skills mentioned in `text` (one hit per skill per text).

    `section` matters only for ambiguous names such as C / R / Go: they count
    in a Skills section, or when written as "using R", "in C", etc.
    """
    tax = taxonomy or default_taxonomy()
    matches: List[SkillHit] = []
    for al in tax.aliases:
        for m in al.regex.finditer(text):
            if al.needs_context and section != "skills" \
                    and not _CTX_BEFORE.search(text[:m.start()]):
                continue
            matches.append(SkillHit(al.skill, al.category, m.start(), m.end()))

    # Longest match wins: drop a hit that sits inside a longer hit of another skill
    # (e.g. "GitHub" inside "GitHub Actions", "React" inside "React Native").
    kept: List[SkillHit] = []
    for h in matches:
        swallowed = any(
            o.skill != h.skill and o.start <= h.start and h.end <= o.end
            and (o.end - o.start) > (h.end - h.start)
            for o in matches
        )
        if not swallowed:
            kept.append(h)

    seen, out = set(), []
    for h in sorted(kept, key=lambda x: x.start):
        if h.skill not in seen:
            seen.add(h.skill)
            out.append(h)
    return out
