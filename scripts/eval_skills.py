"""Evaluate Step 3 (skill extraction) against hand-labeled gold skills.

Compares two systems on the same resumes:
  * KEYWORD BASELINE  - the "basic project" approach: case-insensitive whole-word
                        search for each skill's canonical name. No aliases, no
                        ambiguity rules, no longest-match logic.
  * OURS              - taxonomy + aliases + word boundaries + ambiguity rules
                        + longest-match suppression (resume_analyzer.skills).

Metrics: micro-averaged precision / recall / F1 over all resumes, per-resume
scores, and every false positive / false negative.

Run from the project root:  python scripts/eval_skills.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from resume_analyzer.extraction import clean_text                 # noqa: E402
from resume_analyzer.sections import parse_resume                 # noqa: E402
from resume_analyzer.skills import default_taxonomy               # noqa: E402
from resume_analyzer.profile import build_profile                 # noqa: E402
from eval_sections import load_labeled, TAG                       # noqa: E402


def load_resume_text(rel_path: str) -> str:
    path = ROOT / "data" / rel_path
    if rel_path.startswith("labeled/"):
        lines, _ = load_labeled(path)
        return clean_text("\n".join(lines))
    return clean_text(path.read_text(encoding="utf-8"))


def ours(text: str):
    profile = build_profile(parse_resume(text))
    return set(profile.skill_names())


def keyword_baseline(text: str):
    found = set()
    for skill in default_taxonomy().skills:
        if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", text, re.I):
            found.add(skill)
    return found


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(system):
    gold = json.loads((ROOT / "data" / "gold_skills.json").read_text())
    TP = FP = FN = 0
    rows, mistakes = [], []
    for rel, g in gold.items():
        if rel.startswith("_"):
            continue
        g = set(g)
        pred = system(load_resume_text(rel))
        tp, fp, fn = len(g & pred), len(pred - g), len(g - pred)
        TP += tp; FP += fp; FN += fn
        rows.append((rel, *prf(tp, fp, fn)))
        mistakes.append((rel, sorted(pred - g), sorted(g - pred)))
    p, r, f = prf(TP, FP, FN)
    return dict(p=p, r=r, f1=f, tp=TP, fp=FP, fn=FN, rows=rows, mistakes=mistakes)


def main():
    base, mine = evaluate(keyword_baseline), evaluate(ours)
    print("SKILL EXTRACTION: micro-averaged over", len(mine["rows"]), "labeled resumes")
    print("=" * 68)
    print(f"{'system':<22}{'precision':>10}{'recall':>10}{'F1':>8}   TP/FP/FN")
    for name, r in (("keyword baseline", base), ("ours (taxonomy+rules)", mine)):
        print(f"{name:<22}{r['p']:>10.3f}{r['r']:>10.3f}{r['f1']:>8.3f}   "
              f"{r['tp']}/{r['fp']}/{r['fn']}")
    print("\nPer resume F1 (baseline -> ours):")
    for (rel, *_b), (_, *_m) in zip(base["rows"], mine["rows"]):
        print(f"  {rel:<36}{_b[2]:.2f} -> {_m[2]:.2f}")
    print("\nOur errors:")
    for rel, fps, fns in mine["mistakes"]:
        if fps or fns:
            print(f"  {rel}\n     false positives: {fps or '-'}\n     missed:          {fns or '-'}")


if __name__ == "__main__":
    main()
