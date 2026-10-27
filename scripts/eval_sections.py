"""Evaluate Step 2 (section detection) on the hand-labeled resumes.

Each file in data/labeled/ has one tag per non-blank line:
    {H:experience} Work History      <- a heading line
    {experience}   Built REST APIs   <- a content line in that section

Metrics
  * Heading detection: precision / recall / F1 (is this line a heading?)
  * Section accuracy:  fraction of ALL non-blank lines given the right section
  * Per-resume breakdown, and every mistake listed (so we can fix or explain it)

Run from the project root:  python scripts/eval_sections.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from resume_analyzer.extraction import clean_text          # noqa: E402
from resume_analyzer.sections import detect_sections        # noqa: E402

TAG = re.compile(r"^\{(H:)?([a-z]+)\}\s?(.*)$")


def load_labeled(path):
    lines, gold = [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        m = TAG.match(raw)
        if not m:
            raise ValueError(f"{path.name}: untagged line: {raw!r}")
        is_head, section, text = bool(m.group(1)), m.group(2), m.group(3)
        lines.append(text)
        gold.append((section, is_head))
    return lines, gold


def evaluate():
    tp = fp = fn = 0
    correct = total = 0
    per_file, errors = [], []

    for path in sorted((ROOT / "data" / "labeled").glob("*.txt")):
        lines, gold = load_labeled(path)
        cleaned = clean_text("\n".join(lines)).split("\n")
        assert len(cleaned) == len(lines), f"{path.name}: cleaning changed line count"
        pred = [(s, h) for _, s, h in detect_sections("\n".join(cleaned))]

        f_correct = 0
        for line, (g_sec, g_head), (p_sec, p_head) in zip(cleaned, gold, pred):
            tp += g_head and p_head
            fp += (not g_head) and p_head
            fn += g_head and (not p_head)
            ok = (g_sec == p_sec)
            f_correct += ok
            if not ok or g_head != p_head:
                errors.append((path.name, line, g_sec, p_sec, g_head, p_head))
        correct += f_correct
        total += len(lines)
        per_file.append((path.name, f_correct, len(lines)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1, accuracy=correct / total,
                correct=correct, total=total, per_file=per_file, errors=errors,
                tp=tp, fp=fp, fn=fn)


def main():
    r = evaluate()
    print("SECTION DETECTION: rule-based baseline")
    print("=" * 60)
    print(f"Heading detection   P={r['precision']:.3f}  R={r['recall']:.3f}  "
          f"F1={r['f1']:.3f}   (TP={r['tp']} FP={r['fp']} FN={r['fn']})")
    print(f"Line-level section accuracy: {r['accuracy']:.3f}  "
          f"({r['correct']}/{r['total']} lines)")
    print("\nPer resume:")
    for name, c, t in r["per_file"]:
        print(f"  {name:<28} {c}/{t}  ({c / t:.0%})")
    print(f"\nErrors ({len(r['errors'])}):")
    for name, line, g, p, gh, ph in r["errors"]:
        print(f"  {name}: '{line[:50]}'  gold={g}{'(H)' if gh else ''} "
              f"pred={p}{'(H)' if ph else ''}")


if __name__ == "__main__":
    main()
