import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from resume_analyzer.extraction import extract_text      # noqa: E402
from resume_analyzer.sections import detect_heading, parse_resume  # noqa: E402


class TestHeadingDetection(unittest.TestCase):
    def test_true_headings(self):
        cases = {
            "Education": "education", "EXPERIENCE": "experience",
            "Work Experience": "experience", "Technical Skills:": "skills",
            "SKILLS & TOOLS": "skills", "Courses & Certifications": "certifications",
            "Work History": "experience", "Personal Projects": "projects",
            "AWARDS & HONOURS": "achievements", "About Me": "summary",
            "Qualifications": "education", "Interests": "other",
        }
        for line, expected in cases.items():
            self.assertEqual(detect_heading(line), expected, line)

    def test_not_headings(self):
        for line in [
            "Education Loan Tracker",                       # project title
            "Experience with Python and Django, and a strong interest in data.",
            "- Built a web app",                            # bullet
            "Skills gained: debugging",                     # has stopword 'gained'
            "Languages: Python, Java",                      # skills content
            "ananya@example.com | Ludhiana",                # contact line
            "Backend Developer Intern, TechNova (2025)",    # role title
            "",
        ]:
            self.assertIsNone(detect_heading(line), line)


class TestParseResume(unittest.TestCase):
    def setUp(self):
        self.s = parse_resume(extract_text(
            ROOT / "data" / "sample_resumes" / "resume_ananya.pdf").text)

    def test_sections_present(self):
        secs = {x.section for x in self.s}
        for sec in ("header", "summary", "education", "experience",
                    "projects", "skills", "certifications"):
            self.assertIn(sec, secs)

    def test_headings_not_emitted_as_sentences(self):
        texts = [x.text.lower() for x in self.s]
        for h in ("education", "experience", "projects", "skills"):
            self.assertNotIn(h, texts)

    def test_education_lines_stay_separate(self):
        edu = [x.text for x in self.s if x.section == "education"]
        self.assertEqual(len(edu), 3)

    def test_bullets_know_their_parent(self):
        b = next(x for x in self.s if "Redis caching" in x.text)
        self.assertEqual(b.kind, "bullet")
        self.assertIn("TechNova", b.parent)
        p = next(x for x in self.s if "RESTful endpoints" in x.text)
        self.assertEqual(p.parent, "Student Management System")

    def test_indices_are_sequential(self):
        self.assertEqual([x.index for x in self.s], list(range(len(self.s))))

    def test_wrapped_bullet_is_merged(self):
        text = ("Experience\nIoT Intern, SmartHome (2025)\n"
                "- Led a team of four students to build an inventory dashboard that reduced manual\n"
                "stock counting time for the college store.\n"
                "- Programmed ESP32 boards.")
        bullets = [x for x in parse_resume(text) if x.kind == "bullet"]
        self.assertEqual(len(bullets), 2)
        self.assertTrue(bullets[0].text.endswith("college store."))

    def test_summary_split_into_sentences(self):
        text = "Summary\nBuilt APIs in Flask. Keen to learn about cloud systems."
        sents = [x.text for x in parse_resume(text) if x.section == "summary"]
        self.assertEqual(len(sents), 2)


class TestEvaluation(unittest.TestCase):
    def test_baseline_quality_floor(self):
        """Regression guard: fail if a change makes section detection worse."""
        from eval_sections import evaluate
        r = evaluate()
        self.assertGreaterEqual(r["f1"], 0.95)
        self.assertGreaterEqual(r["accuracy"], 0.90)


if __name__ == "__main__":
    unittest.main()
