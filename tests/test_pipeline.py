import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from resume_analyzer import pipeline as pl     # noqa: E402

JD = pl.read_sample_jd("jd_backend_engineer.txt")
ANANYA = pl.read_sample_resume("resume_ananya.pdf")


class TestAnalyze(unittest.TestCase):
    def setUp(self):
        self.r = pl.analyze(ANANYA, "resume_ananya.pdf", JD)

    def test_end_to_end(self):
        self.assertEqual(self.r.profile.name, "Ananya Sharma")
        self.assertEqual(len(self.r.jd.requirements), 12)
        self.assertEqual(len(self.r.matches), 12)
        self.assertGreater(self.r.baseline_score, 0)
        self.assertEqual(self.r.matcher_name, "tfidf")
        self.assertEqual(self.r.warnings, [])

    def test_docx_and_txt_uploads(self):
        docx = pl.analyze(pl.read_sample_resume("resume_ananya.docx"), "a.docx", JD)
        self.assertEqual(docx.profile.name, "Ananya Sharma")
        txt = (ROOT / "data" / "source_text" / "resume_ananya.txt").read_bytes()
        self.assertEqual(pl.analyze(txt, "a.txt", JD).profile.name, "Ananya Sharma")

    def test_unreadable_pdf_gives_warning_not_crash(self):
        r = pl.analyze(b"%PDF-1.4 not really a pdf", "broken.pdf", JD)
        self.assertTrue(any("No text" in w or "scanned" in w for w in r.warnings))

    def test_unsupported_type_raises(self):
        with self.assertRaises(ValueError):
            pl.analyze(b"MZ", "virus.exe", JD)

    def test_jd_without_requirements_warns(self):
        r = pl.analyze(ANANYA, "a.pdf", "Hello")
        self.assertEqual(r.matches, [])
        self.assertTrue(any("No requirements" in w for w in r.warnings))


class TestSamples(unittest.TestCase):
    def test_listing(self):
        self.assertIn("resume_ananya.pdf", pl.list_sample_resumes())
        self.assertIn("jd_backend_engineer.txt", pl.list_sample_jds())
        self.assertNotIn("resume_ananya.txt", pl.list_sample_resumes())

    def test_path_traversal_is_blocked(self):
        with self.assertRaises(FileNotFoundError):
            pl.read_sample_resume("../../requirements.txt")
        with self.assertRaises(FileNotFoundError):
            pl.read_sample_jd("../../README.md")

    def test_safe_filename(self):
        self.assertEqual(pl.safe_filename("my resume (1).pdf"), "my resume _1_.pdf")
        self.assertNotIn("/", pl.safe_filename("../../etc/passwd"))


class TestRows(unittest.TestCase):
    def setUp(self):
        self.r = pl.analyze(ANANYA, "resume_ananya.pdf", JD)

    def test_row_shapes(self):
        self.assertEqual(len(pl.section_rows(self.r.sentences)), len(self.r.sentences))
        self.assertEqual(len(pl.skill_rows(self.r.profile)), len(self.r.profile.skills))
        req = pl.requirement_rows(self.r.jd)
        self.assertEqual(len(req), 12)
        self.assertIn("Flask/Django/FastAPI", req[6]["Any-of"])

    def test_strength_counts_add_up(self):
        c = pl.strength_counts(self.r.profile)
        self.assertEqual(sum(c.values()), len(self.r.profile.skills))
        self.assertGreater(c["demonstrated"], 0)

    def test_coverage_understands_alternatives(self):
        cov = pl.jd_skill_coverage(self.r.profile, self.r.jd)
        self.assertEqual(set(cov["must_missing"]), {"Integration Testing", "CI/CD"})
        self.assertIn("Flask / Django / FastAPI: has Flask (demonstrated)", cov["must_found"])
        self.assertIn("PostgreSQL / MySQL: has PostgreSQL (claimed)", cov["must_found"])
        self.assertIn("AWS / GCP / Azure: has AWS (supported)", cov["nice_found"])
        self.assertEqual(cov["nice_missing"], [])

    def test_coverage_reports_a_fully_missing_group(self):
        weak = pl.analyze(pl.read_sample_resume("resume_simran.pdf"), "s.pdf", JD)
        cov = pl.jd_skill_coverage(weak.profile, weak.jd)
        self.assertIn("any of Flask / Django / FastAPI", cov["must_missing"])
        self.assertIn("any of AWS / GCP / Azure", cov["nice_missing"])

    def test_format_duration(self):
        self.assertEqual(pl.format_duration(22), "1y 10m")
        self.assertEqual(pl.format_duration(12), "1y")
        self.assertEqual(pl.format_duration(3), "3m")
        self.assertEqual(pl.format_duration(0), "none")

    def test_stage_table(self):
        self.assertEqual(len(pl.STAGES), 8)
        self.assertTrue(all(s[2] in pl.STAGE_LABELS for s in pl.STAGES))


class TestEvalAll(unittest.TestCase):
    def test_summary_rows(self):
        import eval_all
        rows = eval_all.summary()
        self.assertGreaterEqual(len(rows), 12)
        self.assertEqual({"Component", "Metric", "Baseline", "Ours", "Test set"}, set(rows[0]))
        comps = {r["Component"] for r in rows}
        self.assertEqual(len(comps), 4)


if __name__ == "__main__":
    unittest.main()
