import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from resume_agent import extract_skills, extract_years, load_skills, rank_resumes, save_audit_log, save_results  # noqa: E402


class ResumeAgentTests(unittest.TestCase):
    def test_parses_singular_year_and_plural_llms(self):
        self.assertEqual(1.0, extract_years("1 year of experience building models"))
        self.assertEqual(2.0, extract_years("Experience: 2 yrs"))
        self.assertIn("llm", extract_skills("Implemented LLMs for support."))

    def test_loads_role_specific_skill_catalogue(self):
        root = Path(__file__).parents[1]
        skills = load_skills(root / "data" / "ai_research_skills.json")
        self.assertIn("rag", skills)
        self.assertNotIn("javascript", skills)

    def test_ranks_all_sample_resumes_and_writes_both_formats(self):
        root = Path(__file__).parents[1]
        jd = (root / "data" / "job_description.txt").read_text(encoding="utf-8")
        paths = sorted((root / "data" / "resumes").glob("*.txt"))
        results = rank_resumes(jd, paths)
        self.assertEqual(10, len(results))
        self.assertEqual("Aisha Khan", results[0].name)
        farhan = next(result for result in results if result.name == "Farhan Ali")
        self.assertEqual(1.0, farhan.years_experience)
        self.assertEqual(50.0, farhan.experience_score)
        self.assertTrue(all(a.overall_score >= b.overall_score for a, b in zip(results, results[1:])))
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, csv_path = save_results(results, Path(temp_dir) / "ranking")
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            audit_path = save_audit_log(Path(temp_dir) / "ranking", root / "data" / "job_description.txt", paths, "tfidf", "all-MiniLM-L6-v2", {"python"})
            self.assertIn('"human_review_required": true', audit_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
