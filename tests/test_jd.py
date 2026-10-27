import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from resume_analyzer.jd import (detect_jd_heading, parse_job_description, parse_min_years,   # noqa: E402
                                parse_degree_level, build_skill_groups, jd_from_file,
                                _split_inline_headings)
from resume_analyzer.skills import extract_skills                                             # noqa: E402

SAMPLES = ROOT / "data" / "sample_jds"


class TestHeadings(unittest.TestCase):
    def test_headings(self):
        cases = {
            "Responsibilities": "responsibility", "What you'll do": "responsibility",
            "Key Responsibilities": "responsibility", "Requirements:": "requirement",
            "What we're looking for": "requirement", "Required Skills": "requirement",
            "Qualifications": "requirement", "Nice to have": "nice",
            "Preferred Qualifications": "nice", "Bonus points": "nice",
            "About the role": "about", "Benefits": "about", "What we offer": "about",
        }
        for line, kind in cases.items():
            self.assertEqual(detect_jd_heading(line)[0], kind, line)

    def test_not_headings(self):
        for line in ["- Experience with Python.", "We are looking for a Backend Engineer.",
                     "Strong proficiency in Python (Flask, Django or FastAPI).", ""]:
            self.assertIsNone(detect_jd_heading(line), line)

    def test_skills_context_flag(self):
        self.assertTrue(detect_jd_heading("Required Skills")[1])
        self.assertFalse(detect_jd_heading("Requirements")[1])

    def test_inline_heading_split(self):
        out = _split_inline_headings("Requirements: Python, SQL")
        self.assertEqual(out.split("\n"), ["Requirements:", "Python, SQL"])


class TestDetails(unittest.TestCase):
    def test_years(self):
        self.assertEqual(parse_min_years("3+ years of experience"), 3)
        self.assertEqual(parse_min_years("1-2 years of experience"), 1)
        self.assertEqual(parse_min_years("At least 2 years"), 2)
        self.assertIsNone(parse_min_years("A 6-month internship"))

    def test_degree_takes_lowest_accepted(self):
        self.assertEqual(parse_degree_level("Bachelor's degree in CS"), "bachelor")
        self.assertEqual(parse_degree_level("B.Tech, B.E. or M.Tech in CS"), "bachelor")
        self.assertEqual(parse_degree_level("MCA or M.Tech"), "master")
        self.assertEqual(parse_degree_level("A degree in IT"), "bachelor")
        self.assertEqual(parse_degree_level("Pursuing a PhD"), "phd")
        self.assertIsNone(parse_degree_level("Experience with Python"))

    def test_alternatives_grouping(self):
        t = "Strong proficiency in Python (Flask, Django or FastAPI)."
        self.assertEqual(build_skill_groups(t, extract_skills(t)),
                         [["Python"], ["Flask", "Django", "FastAPI"]])

    def test_and_lists_are_individually_required(self):
        t = "Familiarity with Git, Docker and CI/CD pipelines."
        self.assertEqual(build_skill_groups(t, extract_skills(t)),
                         [["Git"], ["Docker"], ["CI/CD"]])

    def test_such_as_alternatives(self):
        t = "Experience with SQL databases such as PostgreSQL or MySQL."
        self.assertEqual(build_skill_groups(t, extract_skills(t)),
                         [["SQL"], ["PostgreSQL", "MySQL"]])


class TestParsing(unittest.TestCase):
    def setUp(self):
        self.jd = jd_from_file(str(SAMPLES / "jd_backend_engineer.txt"))

    def test_structure(self):
        self.assertEqual(self.jd.title, "Backend Engineer (Junior)")
        self.assertEqual(len(self.jd.requirements), 12)
        self.assertEqual(len(self.jd.about), 1)
        self.assertEqual(self.jd.degree_level, "bachelor")
        kinds = [r.kind for r in self.jd.requirements]
        self.assertEqual(kinds.count("responsibility"), 5)

    def test_priorities_and_skills(self):
        self.assertIn("Docker", self.jd.must_skills)
        self.assertNotIn("Docker", self.jd.nice_skills)
        self.assertEqual(set(self.jd.nice_skills), {"Caching", "Redis", "AWS", "GCP", "Azure"})
        self.assertEqual({r.priority for r in self.jd.requirements[-2:]}, {"nice"})

    def test_indices_sequential(self):
        self.assertEqual([r.index for r in self.jd.requirements], list(range(12)))

    def test_years_and_degree_fullstack(self):
        jd = jd_from_file(str(SAMPLES / "jd4_fullstack.txt"))
        self.assertEqual(jd.min_years, 3)
        self.assertEqual(jd.degree_level, "bachelor")      # B.Tech/B.E./MCA -> lowest
        self.assertEqual(jd.nice_skills, ["AWS", "Azure", "GraphQL"])

    def test_nice_cue_overrides_section(self):
        jd = parse_job_description("Requirements\n- Python\n- Knowledge of GraphQL is a plus.")
        self.assertEqual([r.priority for r in jd.requirements], ["must", "nice"])

    def test_numbered_items(self):
        jd = parse_job_description("Responsibilities\n1. Build APIs\n2) Write tests")
        self.assertEqual([r.text for r in jd.requirements], ["Build APIs", "Write tests"])

    def test_wrapped_bullet_merges(self):
        jd = parse_job_description(
            "Requirements\n- Strong knowledge of Python and\nDjango frameworks.\n- Git")
        self.assertEqual(len(jd.requirements), 2)
        self.assertIn("Django frameworks", jd.requirements[0].text)

    def test_paragraph_split_into_sentences(self):
        jd = parse_job_description("Responsibilities\nYou will build APIs. You will write tests.")
        self.assertEqual(len(jd.requirements), 2)

    def test_skills_heading_enables_single_letter_languages(self):
        jd = parse_job_description("Required Skills\nC, R, Go")
        self.assertEqual(jd.must_skills, ["C", "R", "Go"])
        jd2 = parse_job_description("Requirements\n- Knowledge of the Go-to-market process")
        self.assertNotIn("Go", jd2.must_skills)

    def test_headingless_jd_uses_cues(self):
        jd = jd_from_file(str(SAMPLES / "jd6_noheadings.txt"))
        self.assertEqual(len(jd.about), 1)
        self.assertEqual([r.kind for r in jd.requirements],
                         ["responsibility", "responsibility", "qualification",
                          "qualification", "qualification"])
        self.assertEqual([r.priority for r in jd.requirements][-2:], ["nice", "nice"])
        self.assertIsNone(jd.degree_level)       # the degree is only "preferred"

    def test_benefits_are_not_requirements(self):
        jd = parse_job_description("Requirements\n- Python\nBenefits\n- Health insurance")
        self.assertEqual(len(jd.requirements), 1)
        self.assertEqual(jd.about, ["Health insurance"])


class TestEvaluationFloor(unittest.TestCase):
    def test_quality_floor(self):
        from eval_jd import eval_lines, eval_skills_and_priority
        L, S = eval_lines(), eval_skills_and_priority()
        self.assertGreaterEqual(L["acc"], 0.90)
        self.assertGreaterEqual(S["ours"][2], 0.90)
        self.assertGreaterEqual(S["ours"][2], S["base"][2])
        self.assertGreaterEqual(S["prio"], 0.90)


if __name__ == "__main__":
    unittest.main()
