import datetime as dt
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from resume_analyzer.profile import (profile_from_file, parse_duration,   # noqa: E402
                                     _education_level, _split_role, build_profile)
from resume_analyzer.sections import parse_resume                          # noqa: E402

TODAY = dt.date(2026, 10, 27)
SAMPLES = ROOT / "data" / "sample_resumes"


class TestDuration(unittest.TestCase):
    def test_month_year_range_is_inclusive(self):
        self.assertEqual(parse_duration("Intern (Jun 2025 - Aug 2025)", TODAY), (3, False))

    def test_year_only_range_is_approximate(self):
        self.assertEqual(parse_duration("Developer (2022 - 2023)", TODAY), (12, True))

    def test_single_year(self):
        self.assertEqual(parse_duration("Intern, X (2024)", TODAY), (6, True))

    def test_present(self):
        self.assertEqual(parse_duration("Dev (Jan 2025 - Present)", TODAY), (22, False))

    def test_no_dates(self):
        self.assertEqual(parse_duration("Developer, Acme", TODAY), (None, False))


class TestEducationAndRoles(unittest.TestCase):
    def test_levels(self):
        cases = {
            "B.E. in Information Technology": "bachelor", "B.Tech in CSE": "bachelor",
            "BCA, Sample College": "bachelor", "M.Sc. Statistics": "master",
            "MCA, Sample University": "master", "Class XII, Sample School, 93%": "secondary",
            "Senior Secondary (CBSE)": "secondary", "Be careful to be on time": None,
            "Mastering Python": None,
        }
        for text, level in cases.items():
            self.assertEqual(_education_level(text), level, text)

    def test_split_role(self):
        self.assertEqual(
            _split_role("Backend Developer Intern, TechNova Solutions (Jun 2025 - Aug 2025)"),
            ("Backend Developer Intern", "TechNova Solutions"))


class TestProfile(unittest.TestCase):
    def setUp(self):
        self.p = profile_from_file(str(SAMPLES / "resume_ananya.pdf"), TODAY)

    def test_basics(self):
        self.assertEqual(self.p.name, "Ananya Sharma")
        self.assertEqual(self.p.email, "ananya.sharma@example.com")
        self.assertEqual(self.p.highest_degree, "bachelor")
        self.assertEqual(self.p.education[0].cgpa, 8.4)
        self.assertEqual(len(self.p.education), 2)       # CGPA line merged into degree

    def test_experience_and_projects(self):
        e = self.p.experience[0]
        self.assertTrue(e.is_internship)
        self.assertEqual(e.months, 3)
        self.assertEqual(len(e.bullets), 4)
        self.assertEqual([x.title for x in self.p.projects],
                         ["Student Management System", "Sentiment Analysis of Product Reviews"])
        self.assertIn("Flask", self.p.projects[0].skills)

    def test_evidence_strength(self):
        self.assertEqual(self.p.get_skill("Flask").strength, "demonstrated")
        self.assertEqual(self.p.get_skill("Docker").strength, "claimed")   # only in Skills line
        self.assertEqual(self.p.get_skill("AWS").strength, "supported")    # certification only

    def test_every_skill_has_evidence_sentences(self):
        for rec in self.p.skills:
            self.assertGreaterEqual(len(rec.mentions), 1)
        redis = self.p.get_skill("Redis")
        self.assertTrue(any("Redis caching" in m.text for m in redis.mentions))
        self.assertTrue(any(m.parent and "TechNova" in m.parent for m in redis.mentions))

    def test_urls_in_header_are_not_skills(self):
        self.assertIsNone(self.p.get_skill("GitHub"))   # header has github.com/...

    def test_resume_without_headings_still_gets_skills(self):
        text = "Aman Gill\naman@example.com\nBCA, Sample College (2022 - 2025)\nHTML, CSS, JavaScript, Git"
        prof = build_profile(parse_resume(text), TODAY)
        self.assertEqual(set(prof.skill_names()), {"HTML", "CSS", "JavaScript", "Git"})

    def test_weak_resume(self):
        w = profile_from_file(str(SAMPLES / "resume_simran.pdf"), TODAY)
        self.assertEqual(w.total_experience_months, 0)
        self.assertEqual(set(w.skill_names()), {"HTML", "CSS", "Java", "MySQL", "C++"})


if __name__ == "__main__":
    unittest.main()
