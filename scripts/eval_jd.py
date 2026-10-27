"""Evaluate Step 4 (job description parsing) on the hand-labeled JDs.

Three measurements
  1. Line labelling: is each line a title / heading / about / responsibility /
     requirement / nice-to-have?  (accuracy + heading precision/recall/F1)
  2. Skill extraction from the JD: micro P/R/F1 against gold, next to a plain
     keyword baseline (which also cannot tell must-have from nice-to-have).
  3. Priority: of the skills found correctly, how many were put in the right
     bucket (must vs nice)?

Run from the project root:  python scripts/eval_jd.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from resume_analyzer.extraction import clean_text           # noqa: E402
from resume_analyzer.jd import classify_lines, parse_job_description, _split_inline_headings  # noqa: E402
from eval_sections import load_labeled                        # noqa: E402
from eval_skills import keyword_baseline, prf                 # noqa: E402

LABELED = ROOT / "data" / "labeled_jds"


def eval_lines():
    correct = total = tp = fp = fn = 0
    errors, per_file = [], []
    for path in sorted(LABELED.glob("*.txt")):
        lines, gold = load_labeled(path)
        cleaned = clean_text("\n".join(lines)).split("\n")
        assert len(cleaned) == len(lines), path.name
        pred = [l for l in classify_lines("\n".join(cleaned))]
        fc = 0
        for text, (g_kind, g_head), p in zip(cleaned, gold, pred):
            ok = (g_kind == p.kind)
            tp += g_head and p.is_heading
            fp += (not g_head) and p.is_heading
            fn += g_head and not p.is_heading
            fc += ok
            if not ok or g_head != p.is_heading:
                errors.append((path.name, text, g_kind, p.kind))
        correct += fc
        total += len(lines)
        per_file.append((path.name, fc, len(lines)))
    p, r, f = prf(tp, fp, fn)
    return dict(acc=correct / total, correct=correct, total=total, hp=p, hr=r, hf=f,
                errors=errors, per_file=per_file)


def eval_skills_and_priority():
    gold = json.loads((ROOT / "data" / "gold_jd_skills.json").read_text())
    TP = FP = FN = 0
    bTP = bFP = bFN = 0
    prio_ok = prio_total = 0
    mistakes = []
    for name, g in gold.items():
        if name.startswith("_"):
            continue
        text = (LABELED / name).read_text(encoding="utf-8")
        import re
        text = "\n".join(re.sub(r"^\{(?:H:)?[a-z]+\}\s?", "", l) for l in text.splitlines())
        jd = parse_job_description(text)
        g_must, g_nice = set(g["must"]), set(g["nice"])
        p_must, p_nice = set(jd.must_skills), set(jd.nice_skills)
        g_all, p_all = g_must | g_nice, p_must | p_nice

        TP += len(g_all & p_all); FP += len(p_all - g_all); FN += len(g_all - p_all)
        base = keyword_baseline(clean_text(text))
        bTP += len(g_all & base); bFP += len(base - g_all); bFN += len(g_all - base)

        for sk in g_all & p_all:
            prio_total += 1
            prio_ok += ((sk in g_must) == (sk in p_must))
        mistakes.append((name, sorted(p_all - g_all), sorted(g_all - p_all),
                         sorted(sk for sk in g_all & p_all if (sk in g_must) != (sk in p_must))))
    return dict(ours=prf(TP, FP, FN), base=prf(bTP, bFP, bFN), counts=(TP, FP, FN),
                bcounts=(bTP, bFP, bFN), prio=prio_ok / prio_total, prio_ok=prio_ok,
                prio_total=prio_total, mistakes=mistakes)


def main():
    L, S = eval_lines(), eval_skills_and_priority()
    print("JOB DESCRIPTION PARSING: 6 hand-labeled JDs")
    print("=" * 66)
    print("1) Line labelling")
    print(f"   accuracy {L['acc']:.3f} ({L['correct']}/{L['total']} lines)   "
          f"heading P={L['hp']:.3f} R={L['hr']:.3f} F1={L['hf']:.3f}")
    for name, c, t in L["per_file"]:
        print(f"     {name:<24}{c}/{t}")
    for name, text, g, p in L["errors"]:
        print(f"     ERR {name}: '{text[:55]}' gold={g} pred={p}")
    print("\n2) Skill extraction (micro, must+nice combined)")
    print(f"   {'system':<22}{'P':>7}{'R':>7}{'F1':>7}   TP/FP/FN")
    for name, key, ck in (("keyword baseline", "base", "bcounts"), ("ours", "ours", "counts")):
        p, r, f = S[key]
        print(f"   {name:<22}{p:>7.3f}{r:>7.3f}{f:>7.3f}   {'/'.join(map(str, S[ck]))}")
    print(f"\n3) Must-vs-nice priority accuracy: {S['prio']:.3f} "
          f"({S['prio_ok']}/{S['prio_total']} correctly found skills)")
    print("   (the keyword baseline has no notion of priority)")
    print("\nSkill errors per JD:")
    for name, fps, fns, wrong in S["mistakes"]:
        if fps or fns or wrong:
            print(f"   {name}\n      false positives: {fps or '-'}\n"
                  f"      missed:          {fns or '-'}\n      wrong priority:  {wrong or '-'}")


if __name__ == "__main__":
    main()
