"""Step 5: match each job requirement to the resume sentences that answer it.

This is the TF-IDF BASELINE. Every matcher has the same interface:

    matcher.match(profile, jd, top_k=3) -> List[RequirementMatch]

so the embedding matcher added later (Step 6+) can be dropped in and compared
on exactly the same evaluation set.

How the baseline works
  1. Evidence pool = every resume sentence except the name and contact lines.
  2. Tokenise (keeping c++, node.js, ci/cd, scikit-learn), lower-case, drop
     stop-words, and strip a plural "s" so "APIs" matches "API".
  3. Fit TF-IDF over the resume sentences + requirements together.
  4. Rank evidence for each requirement by cosine similarity.

Its built-in weakness is the point of the project: it only sees shared WORDS.
"RESTful endpoints" does not match "REST APIs", and "B.Tech" does not match
"Bachelor's degree".

CLI:
    python -m resume_analyzer.matching resume.pdf job_description.txt
"""
from __future__ import annotations

import re
import sys
from typing import List

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .profile import is_contact_line
from .schema import CandidateProfile, Evidence, JobDescription, RequirementMatch, ResumeSentence

# --------------------------------------------------------------------------
# Text preparation
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9]+(?:[.+#/\-][a-z0-9]+)*(?:[+#]+)?")

# Words that appear in nearly every job posting and say nothing about the skill.
_JD_BOILERPLATE = {
    "experience", "knowledge", "strong", "proficiency", "familiarity", "ability", "good",
    "skills", "skill", "understanding", "exposure", "solid", "basic", "excellent", "years",
    "year", "etc", "work", "working", "using", "hands-on", "including", "related", "field",
    "plus", "bonus", "preferred", "must", "required", "currently", "least",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | _JD_BOILERPLATE


def _stem(tok: str) -> str:
    """Tiny plural stripper: apis->api, tests->test, databases->database."""
    # Plain words only: never touch technical tokens like node.js, ci/cd, c++.
    if tok.isalpha() and len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> List[str]:
    toks = _TOKEN.findall(text.lower())
    return [_stem(t) for t in toks if t not in STOP_WORDS and not t.isdigit()]


def evidence_pool(profile: CandidateProfile) -> List[ResumeSentence]:
    """Resume sentences that can serve as evidence (not the name / contact lines)."""
    first = next((s.index for s in profile.sentences if s.section == "header"), None)
    return [s for s in profile.sentences
            if not (s.section == "header" and (s.index == first or is_contact_line(s.text)))]


# --------------------------------------------------------------------------
# Matchers
# --------------------------------------------------------------------------
class Matcher:
    name = "base"

    def match(self, profile: CandidateProfile, jd: JobDescription,
              top_k: int = 3) -> List[RequirementMatch]:
        pool = evidence_pool(profile)
        reqs = jd.requirements
        if not pool or not reqs:
            return [RequirementMatch(r) for r in reqs]
        sims = self.similarity([s.text for s in pool], [r.text for r in reqs])
        results = []
        for r, row in zip(reqs, sims):
            order = np.argsort(-row, kind="stable")
            ev = [Evidence(pool[i].index, pool[i].text, pool[i].section, pool[i].parent,
                           float(row[i])) for i in order[:top_k] if row[i] > 0]
            results.append(RequirementMatch(r, ev))
        return results

    def similarity(self, evidence_texts: List[str], requirement_texts: List[str]) -> np.ndarray:
        """Return a (n_requirements x n_evidence) similarity matrix."""
        raise NotImplementedError


class TfidfMatcher(Matcher):
    name = "tfidf"

    def __init__(self, ngram_range=(1, 1)):
        self.ngram_range = ngram_range

    def similarity(self, evidence_texts, requirement_texts):
        vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                              ngram_range=self.ngram_range, sublinear_tf=True)
        try:
            vec.fit(evidence_texts + requirement_texts)
            E = vec.transform(evidence_texts)
            R = vec.transform(requirement_texts)
        except ValueError:                       # empty vocabulary
            return np.zeros((len(requirement_texts), len(evidence_texts)))
        return (R @ E.T).toarray()               # rows are L2-normalised: dot = cosine


def baseline_similarity(matches: List[RequirementMatch]) -> float:
    """Priority-weighted mean of each requirement's best score, as a 0-100 number.

    NOT the final fit score. It is the naive "one number" the proposal argues
    against, kept only so the baseline has something to report.
    """
    if not matches:
        return 0.0
    w = [1.0 if m.requirement.priority == "must" else 0.5 for m in matches]
    return 100.0 * sum(wi * m.best_score for wi, m in zip(w, matches)) / sum(w)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _main(argv: List[str]) -> int:
    from .jd import jd_from_file
    from .profile import profile_from_file
    if len(argv) != 3:
        print("usage: python -m resume_analyzer.matching <resume> <job_description.txt>")
        return 2
    prof, jd = profile_from_file(argv[1]), jd_from_file(argv[2])
    matches = TfidfMatcher().match(prof, jd)
    print(f"{prof.name}  vs  {jd.title}   [{TfidfMatcher.name}]")
    print(f"Baseline similarity: {baseline_similarity(matches):.1f}  (naive; not the final score)\n")
    for m in matches:
        r = m.requirement
        print(f"[{r.priority:<4}] {r.text}")
        if not m.evidence:
            print("        (no overlapping evidence)")
        for e in m.evidence:
            ctx = f"  [{e.parent}]" if e.parent else ""
            print(f"      {e.score:.2f}  ({e.section}) {e.text[:90]}{ctx}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
