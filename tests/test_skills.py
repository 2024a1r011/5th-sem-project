import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from resume_analyzer.skills import extract_skills, default_taxonomy   # noqa: E402


def names(text, section=None):
    return {h.skill for h in extract_skills(text, section)}


class TestBoundaries(unittest.TestCase):
    def test_java_not_in_javascript(self):
        self.assertEqual(names("Built a UI in JavaScript"), {"JavaScript"})
        self.assertEqual(names("Java and JavaScript"), {"Java", "JavaScript"})

    def test_sql_not_inside_other_databases(self):
        self.assertEqual(names("MySQL and PostgreSQL"), {"MySQL", "PostgreSQL"})
        self.assertEqual(names("SQLite with SQLAlchemy"), {"SQLite", "SQLAlchemy"})
        self.assertEqual(names("Wrote SQL queries"), {"SQL"})

    def test_cpp_and_csharp_are_not_c(self):
        self.assertEqual(names("C++, C#", "skills"), {"C++", "C#"})

    def test_longest_match_wins(self):
        self.assertEqual(names("CI pipeline using GitHub Actions"), {"CI/CD", "GitHub Actions"})
        self.assertEqual(names("An app in React Native"), {"React Native"})


class TestAmbiguousNames(unittest.TestCase):
    def test_c_and_r_in_skills_section(self):
        self.assertEqual(names("Languages: C, R, Python", "skills"), {"C", "R", "Python"})

    def test_c_and_r_need_context_elsewhere(self):
        self.assertIn("R", names("Analysed data using R and Python", "experience"))
        self.assertIn("C", names("Wrote firmware in C for the board", "experience"))
        self.assertNotIn("C", names("Class C students", "experience"))
        self.assertNotIn("R", names("Joined the R&D team", "experience"))

    def test_excel_is_case_sensitive(self):
        self.assertIn("Excel", names("Advanced Excel dashboards"))
        self.assertNotIn("Excel", names("I excel at teamwork"))


class TestAliases(unittest.TestCase):
    def test_rest_variants(self):
        for t in ("Built REST APIs", "RESTful endpoints", "Django REST services"):
            self.assertIn("REST API", names(t), t)

    def test_node_and_testing_aliases(self):
        self.assertIn("Node.js", names("Services in Node.js."))
        self.assertIn("Unit Testing", names("Wrote unit tests with pytest"))

    def test_one_hit_per_skill(self):
        self.assertEqual(len(extract_skills("Python, python3 and Python")), 1)


class TestTaxonomy(unittest.TestCase):
    def test_loads(self):
        tax = default_taxonomy()
        self.assertGreater(len(tax.skills), 100)
        self.assertEqual(tax.category_of["Docker"], "devops_cloud")


class TestEvaluationFloor(unittest.TestCase):
    def test_quality_floor_and_beats_baseline(self):
        from eval_skills import evaluate, ours, keyword_baseline
        mine, base = evaluate(ours), evaluate(keyword_baseline)
        self.assertGreaterEqual(mine["f1"], 0.95)
        self.assertGreater(mine["f1"], base["f1"])


if __name__ == "__main__":
    unittest.main()
