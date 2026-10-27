"""Evaluate Step 5 (requirement -> resume evidence matching).

Gold: data/gold_matches.json. For each requirement, which resume sentences are
relevant evidence (or none at all).

Metrics, computed for ANY matcher with the standard interface
    matcher.match(profile, jd, top_k) -> List[RequirementMatch]
so the embedding matcher added later is scored on the identical set.

  Answerable requirements (resume has evidence):
      Hit@1   top-ranked sentence is relevant
      Hit@3   a relevant sentence is in the top 3
      MRR     mean reciprocal rank of the first relevant sentence
  Separation (does the score tell "has evidence" from "no evidence"?):
      AUROC of the best score over answerable vs unanswerable requirements
      (0.5 = score is no better than chance, 1.0 = perfectly separates)

A relevant sentence only counts as retrieved if its score is > 0.

Run from the project root:  python scripts/eval_matching.py
"""
import json
import re
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from resume_analyzer.jd import parse_job_description          # noqa: E402
from resume_analyzer.matching import TfidfMatcher, evidence_pool   # noqa: E402
from resume_analyzer.profile import build_profile             # noqa: E402
from resume_analyzer.sections import parse_resume             # noqa: E402
from eval_skills import load_resume_text                      # noqa: E402

LABELED_JDS = ROOT / "data" / "labeled_jds"
_TAG = re.compile(r"^\{(?:H:)?[a-z]+\}\s?")


def load_jd_text(name: str) -> str:
    raw = (LABELED_JDS / name).read_text(encoding="utf-8")
    return "\n".join(_TAG.sub("", l) for l in raw.splitlines())


def load_gold():
    """Load and VALIDATE the gold file: every requirement labeled exactly once,
    every relevant substring present in the resume. Catches typos early."""
    gold = json.loads((ROOT / "data" / "gold_matches.json").read_text())
    cases = []
    for pair in gold["pairs"]:
        profile = build_profile(parse_resume(load_resume_text(pair["resume"])))
        jd = parse_job_description(load_jd_text(pair["jd"]))
        pool_texts = [s.text for s in evidence_pool(profile)]
        labeled = {}
        for item in pair["items"]:
            hits = [r for r in jd.requirements if r.text.startswith(item["req"])]
            assert len(hits) == 1, f"{pair['jd']}: prefix '{item['req']}' matched {len(hits)} requirements"
            for sub in item["relevant"]:
                assert any(sub in t for t in pool_texts), \
                    f"{pair['resume']}: '{sub}' not found in any resume sentence"
            labeled[hits[0].index] = item["relevant"]
        missing = [r.text for r in jd.requirements if r.index not in labeled]
        assert not missing, f"{pair['jd']}: unlabeled requirements {missing}"
        cases.append(dict(resume=pair["resume"], jd=pair["jd"], profile=profile, job=jd,
                          labels=labeled))
    return cases


def evaluate(matcher, cases=None, top_k=3):
    cases = cases or load_gold()
    hit1 = hit3 = n_ans = 0
    rr_sum = 0.0
    ans_scores, un_scores = [], []
    per_pair, failures = [], []
    for c in cases:
        matches = matcher.match(c["profile"], c["job"], top_k=max(top_k, 10))
        p_hit1 = p_n = 0
        for m in matches:
            rel_subs = c["labels"][m.requirement.index]
            if not rel_subs:
                un_scores.append(m.best_score)
                continue
            ans_scores.append(m.best_score)
            n_ans += 1; p_n += 1
            rank = next((i + 1 for i, e in enumerate(m.evidence)
                         if any(sub in e.text for sub in rel_subs)), None)
            hit1 += rank == 1; hit3 += (rank is not None and rank <= 3)
            p_hit1 += rank == 1
            rr_sum += 1.0 / rank if rank else 0.0
            if rank != 1:
                top = m.evidence[0].text[:70] if m.evidence else "(nothing retrieved)"
                failures.append((c["jd"], m.requirement.text[:62], rank, top, rel_subs[0][:45]))
        per_pair.append((f"{Path(c['resume']).stem[:16]} x {c['jd'][:14]}", p_hit1, p_n))
    auc = (roc_auc_score([1] * len(ans_scores) + [0] * len(un_scores), ans_scores + un_scores)
           if ans_scores and un_scores else float("nan"))
    return dict(n_answerable=n_ans, n_unanswerable=len(un_scores), hit1=hit1 / n_ans,
                hit3=hit3 / n_ans, mrr=rr_sum / n_ans, auroc=auc,
                mean_ans=sum(ans_scores) / len(ans_scores),
                mean_un=sum(un_scores) / len(un_scores) if un_scores else 0.0,
                per_pair=per_pair, failures=failures)


def report(name, r):
    print(f"MATCHING EVALUATION: {name}")
    print("=" * 70)
    print(f"{r['n_answerable']} answerable requirements, {r['n_unanswerable']} with no evidence "
          f"in the resume\n")
    print(f"  Hit@1 {r['hit1']:.3f}    Hit@3 {r['hit3']:.3f}    MRR {r['mrr']:.3f}")
    print(f"  Separation AUROC {r['auroc']:.3f}   "
          f"(mean best score: answerable {r['mean_ans']:.3f} vs no-evidence {r['mean_un']:.3f})")
    print("\n  Hit@1 per resume x JD pair:")
    for label, h, n in r["per_pair"]:
        print(f"    {label:<34}{h}/{n}")
    print(f"\n  Misses ({len(r['failures'])}): requirement -> what it retrieved instead")
    for jd, req, rank, top, gold in r["failures"]:
        got = f"rank {rank}" if rank else "not retrieved"
        print(f"    [{jd[:3]}] {req}\n          {got}; top1: {top}\n          needed: ...{gold}...")


if __name__ == "__main__":
    report("TF-IDF baseline", evaluate(TfidfMatcher()))
