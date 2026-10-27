import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from resume_analyzer.extraction import clean_text, extract_text, extract_text_from_bytes  # noqa: E402

SAMPLES = ROOT / "data" / "sample_resumes"


class TestCleanText(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(clean_text(""), "")

    def test_bullets_normalised(self):
        out = clean_text("• one\n▪ two\n● three\n– four")
        self.assertEqual(out.split("\n"), ["- one", "- two", "- three", "- four"])

    def test_cid_bullets(self):
        out = clean_text("(cid:127) Built REST APIs\nplain (cid:12) text")
        self.assertEqual(out.split("\n"), ["- Built REST APIs", "plain text"])

    def test_ligatures_and_nbsp(self):
        self.assertEqual(clean_text("ﬁne\u00a0work"), "fine work")

    def test_dehyphenation(self):
        self.assertEqual(clean_text("develop-\nment"), "development")

    def test_page_numbers_removed(self):
        self.assertEqual(clean_text("Skills\nPage 2\nPython\n3 of 4"), "Skills\nPython")

    def test_blank_lines_collapsed(self):
        self.assertEqual(clean_text("a\n\n\n\n\nb"), "a\n\nb")


class TestExtractFiles(unittest.TestCase):
    def check(self, name, must_have):
        res = extract_text(SAMPLES / name)
        self.assertTrue(res.ok, f"{name}: no text extracted")
        self.assertNotIn("(cid:", res.text)
        for needle in must_have:
            self.assertIn(needle, res.text, f"{name}: missing '{needle}'")
        return res

    def test_pdf_strong_resume(self):
        res = self.check("resume_ananya.pdf",
                         ["Ananya Sharma", "Flask", "REST APIs", "Redis caching"])
        self.assertEqual(res.source_type, "pdf")
        self.assertEqual(res.n_pages, 1)
        self.assertIn("- Built REST APIs", res.text)

    def test_pdf_weak_resume(self):
        self.check("resume_simran.pdf", ["Simran Kaur", "HTML", "Java"])

    def test_docx(self):
        res = self.check("resume_ananya.docx", ["Ananya Sharma", "Flask"])
        self.assertEqual(res.source_type, "docx")

    def test_docx_inside_table(self):
        res = self.check("resume_rohan_table.docx", ["Rohan Mehta", "Power BI"])
        self.assertTrue(any("table" in w.lower() for w in res.warnings))

    def test_bytes_helper(self):
        data = (SAMPLES / "resume_rohan.pdf").read_bytes()
        res = extract_text_from_bytes(data, "upload.pdf")
        self.assertIn("Rohan Mehta", res.text)
        self.assertEqual(res.source_path, "upload.pdf")

    def test_unsupported_and_missing(self):
        with self.assertRaises(FileNotFoundError):
            extract_text(SAMPLES / "nope.pdf")
        with self.assertRaises(ValueError):
            extract_text(ROOT / "scripts" / "generate_samples.py")


if __name__ == "__main__":
    unittest.main()
