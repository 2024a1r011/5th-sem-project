"""Run every evaluation and print one results table (also used by the demo app).

    python scripts/eval_all.py

Each row compares a baseline with our method on the same hand-labeled data.
Numbers come from small, team-labeled sets: indicative, not final.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def summary():
    """Return a list of dict rows: Component, Metric, Baseline, Ours, Test set."""
    import eval_jd
    import eval_matching
    import eval_sections
    import eval_skills
    from resume_analyzer.matching import TfidfMatcher

    rows = []

    def add(component, metric, baseline, ours, data):
        rows.append({"Component": component, "Metric": metric, "Baseline": baseline,
                     "Ours": ours, "Test set": data})

    f = lambda x: f"{x:.3f}"                                           # noqa: E731

    s = eval_sections.evaluate()
    add("Section detection", "Heading F1", "-", f(s["f1"]), "6 labeled resumes")
    add("Section detection", "Line accuracy", "-", f(s["accuracy"]), "6 labeled resumes")

    ours, base = eval_skills.evaluate(eval_skills.ours), eval_skills.evaluate(eval_skills.keyword_baseline)
    add("Resume skill extraction", "F1", f(base["f1"]), f(ours["f1"]), "9 labeled resumes")
    add("Resume skill extraction", "Precision", f(base["p"]), f(ours["p"]), "9 labeled resumes")
    add("Resume skill extraction", "Recall", f(base["r"]), f(ours["r"]), "9 labeled resumes")

    L, S = eval_jd.eval_lines(), eval_jd.eval_skills_and_priority()
    add("Job description parsing", "Line label accuracy", "-", f(L["acc"]), "6 labeled JDs")
    add("Job description parsing", "Skill F1", f(S["base"][2]), f(S["ours"][2]), "6 labeled JDs")
    add("Job description parsing", "Must-vs-nice accuracy", "n/a", f(S["prio"]), "6 labeled JDs")

    m = eval_matching.evaluate(TfidfMatcher())
    for label, key in (("Hit@1", "hit1"), ("Hit@3", "hit3"), ("MRR", "mrr"),
                       ("Evidence-vs-none AUROC", "auroc")):
        add("Requirement matching", label, f(m[key]), "embeddings: next stage",
            f"{m['n_answerable'] + m['n_unanswerable']} labeled requirements")
    return rows


def main():
    rows = summary()
    cols = ["Component", "Metric", "Baseline", "Ours", "Test set"]
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(line)
    print("-" * len(line))
    last = None
    for r in rows:
        comp = r["Component"] if r["Component"] != last else ""
        last = r["Component"]
        vals = [comp] + [str(r[c]) for c in cols[1:]]
        print("  ".join(v.ljust(w) for v, w in zip(vals, widths)))
    print("\nNote: matching 'Baseline' column is TF-IDF; the embedding matcher is the next stage.")


if __name__ == "__main__":
    main()
