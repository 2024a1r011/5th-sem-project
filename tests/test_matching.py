import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from resume_analyzer.jd import parse_job_description, jd_from_file                 # noqa: E402
from resume_analyzer.matching import (TfidfMatcher, baseline_similarity, evidence_pool,  # noqa: E402
                                      tokenize)
from resume_analyzer.profile import build_profile, is_contact_line, profile_from_file  # noqa: E402
from resume_analyzer.schema import CandidateProfile, JobDescription, Requirement, RequirementMatch  # noqa: E402
from resume_analyzer.sections import parse_resume, split_sentences                  # noqa: E402

SAMPLES = ROOT / "data" / "sample_resumes"
JDS = ROOT / "data" / "sample_jds"


class TestRegressions(unittest.TestCase):
    """Bugs found while building Step 5. Keep them fixed."""

    def test_degree_line_not_split_at_abbreviation(self):
        sents = parse_resume("Education\nB.E. Computer Engineering, Sample College, 2025")
        self.assertEqual([s.text for s in sents],
                         ["B.E. Computer Engineering, Sample College, 2025"])

    def test_abbreviation_guard_in_prose(self):
        self.assertEqual(
            split_sentences("Final-year B.E. student. Built APIs. Keen to learn."),
            ["Final-year B.E. student.", "Built APIs.", "Keen to learn."])

    def test_date_range_is_not_a_phone_number(self):
        self.assertFalse(is_contact_line("BCA, Sample College (2022 - 2025)"))
        self.assertFalse(is_contact_line("Intern, X (Jun 2025 - Aug 2025)"))
        self.assertTrue(is_contact_line("+91 98765 00001"))
        self.assertTrue(is_contact_line("a@b.com | Ludhiana"))
        self.assertTrue(is_contact_line("github.com/someone"))

    def test_no_heading_resume_keeps_its_evidence(self):
        text = ("Aman Gill\naman@example.com\nBCA, Sample College (2022 - 2025)\n"
                "Web Developer Intern, PixelWorks (2024)\nHTML, CSS, JavaScript, Git")
        pool = [s.text for s in evidence_pool(build_profile(parse_resume(text)))]
        self.assertIn("BCA, Sample College (2022 - 2025)", pool)
        self.assertNotIn("Aman Gill", pool)
        self.assertNotIn("aman@example.com", pool)


class TestTokenize(unittest.TestCase):
    def test_keeps_technical_tokens(self):
        toks = tokenize("Wrote C++ and Node.js code with scikit-learn and CI/CD")
        for t in ("c++", "node.js", "scikit-learn", "ci/cd"):
            self.assertIn(t, toks)

    def test_plural_and_stopwords(self):
        self.assertEqual(tokenize("REST APIs"), tokenize("REST API"))
        self.assertEqual(tokenize("the experience with strong knowledge"), [])
        self.assertNotIn("2025", tokenize("In 2025"))


class TestMatcher(unittest.TestCase):
    def setUp(self):
        self.prof = profile_from_file(str(SAMPLES / "resume_ananya.pdf"))
        self.jd = jd_from_file(str(JDS / "jd_backend_engineer.txt"))
        self.matches = TfidfMatcher().match(self.prof, self.jd)

    def test_one_result_per_requirement(self):
        self.assertEqual(len(self.matches), len(self.jd.requirements))
        self.assertTrue(all(isinstance(m, RequirementMatch) for m in self.matches))

    def test_rest_requirement_finds_rest_bullet(self):
        m = self.matches[0]
        self.assertIn("REST APIs in Flask", m.evidence[0].text)
        self.assertEqual(m.evidence[0].section, "experience")
        self.assertIn("TechNova", m.evidence[0].parent)

    def test_sorted_bounded_and_limited(self):
        for m in self.matches:
            scores = [e.score for e in m.evidence]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertTrue(all(0 < sc <= 1.0 for sc in scores))
            self.assertLessEqual(len(scores), 3)
        self.assertLessEqual(len(TfidfMatcher().match(self.prof, self.jd, top_k=1)[0].evidence), 1)

    def test_no_overlap_means_no_evidence(self):
        jd = parse_job_description("Requirements\n- Quantum chromodynamics")
        self.assertEqual(TfidfMatcher().match(self.prof, jd)[0].evidence, [])
        self.assertEqual(TfidfMatcher().match(self.prof, jd)[0].best_score, 0.0)

    def test_evidence_never_includes_name_or_contact(self):
        for m in self.matches:
            for e in m.evidence:
                self.assertNotEqual(e.text, "Ananya Sharma")
                self.assertNotIn("@", e.text)

    def test_empty_inputs_do_not_crash(self):
        self.assertEqual(TfidfMatcher().match(CandidateProfile(), self.jd)[0].evidence, [])
        self.assertEqual(TfidfMatcher().match(self.prof, JobDescription()), [])

    def test_interface_for_future_matchers(self):
        self.assertEqual(TfidfMatcher.name, "tfidf")
        self.assertTrue(callable(TfidfMatcher().match))


class TestBaselineSimilarity(unittest.TestCase):
    def test_nice_requirements_count_half(self):
        def m(priority, score):
            r = Requirement(0, "x", priority=priority)
            from resume_analyzer.schema import Evidence
            return RequirementMatch(r, [Evidence(0, "t", "experience", None, score)])
        # must 1.0 -> weight 1; nice 0.0 -> weight 0.5  => 1/(1.5) = 66.67
        self.assertAlmostEqual(baseline_similarity([m("must", 1.0), m("nice", 0.0)]), 66.667, places=2)
        self.assertEqual(baseline_similarity([]), 0.0)


class TestEvaluationFloor(unittest.TestCase):
    def test_gold_file_is_valid_and_baseline_has_signal(self):
        from eval_matching import evaluate, load_gold
        cases = load_gold()                         # also validates every label
        self.assertEqual(sum(len(c["labels"]) for c in cases), 72)
        r = evaluate(TfidfMatcher(), cases)
        self.assertGreaterEqual(r["hit3"], 0.60)
        self.assertGreaterEqual(r["mrr"], 0.55)
        self.assertGreaterEqual(r["auroc"], 0.75)


if __name__ == "__main__":
    unittest.main()
