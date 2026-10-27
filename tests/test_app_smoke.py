"""Smoke test for app.py using a stand-in for Streamlit.

What this DOES catch: NameErrors, wrong attribute names, bad data handed to
widgets, crashes on empty or odd inputs, and the stop-on-missing-input flow.
What it does NOT catch: a wrong Streamlit argument name or a layout problem.
Those only show up in a real `streamlit run app.py`.
"""
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from resume_analyzer import pipeline as pl     # noqa: E402


class StopApp(BaseException):          # real st.stop() also unwinds the script
    pass


def fake_streamlit(resume_mode="Sample resume", jd_mode="Sample JD", upload=None,
                   pasted="", button=False, sample_resume=None, sample_jd=None):
    st = MagicMock(name="streamlit")
    st.session_state = {}
    st.cache_data = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)
    st.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]
    st.columns.side_effect = lambda n, *a, **k: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    st.stop.side_effect = StopApp

    def selectbox(label, options, *a, **k):
        if label == "Choose a sample" and sample_resume:
            return sample_resume
        if label == "Choose a sample JD" and sample_jd:
            return sample_jd
        return options[0] if options else None

    def radio(label, options, *a, **k):
        return resume_mode if label == "Resume source" else jd_mode

    st.selectbox.side_effect = selectbox
    st.radio.side_effect = radio
    st.file_uploader.return_value = upload
    st.text_area.return_value = pasted
    st.button.return_value = button
    st.sidebar = st                    # sidebar calls hit the same recorder
    return st


def run_app(**kw):
    st = fake_streamlit(**kw)
    with patch.dict(sys.modules, {"streamlit": st}):
        try:
            runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
        except StopApp:
            pass
    return st


class TestApp(unittest.TestCase):
    def test_happy_path_with_samples(self):
        st = run_app()
        st.error.assert_not_called()
        st.title.assert_called_once()
        # sections, skills, requirements, roadmap (the evaluation table needs a click)
        self.assertEqual(st.dataframe.call_count, 4)
        for call in st.dataframe.call_args_list:         # every table is a list of dicts
            rows = call.args[0]
            self.assertIsInstance(rows, list)
            if rows:
                self.assertIsInstance(rows[0], dict)
        self.assertTrue(st.progress.called)              # evidence bars drawn
        for call in st.progress.call_args_list:
            self.assertTrue(0.0 <= call.args[0] <= 1.0)

    def test_every_sample_pair_runs(self):
        for resume in pl.list_sample_resumes():
            for jd in ("jd_backend_engineer.txt", "jd6_noheadings.txt", "jd5_embedded.txt"):
                st = run_app(sample_resume=resume, sample_jd=jd)
                st.error.assert_not_called()

    def test_no_resume_stops_politely(self):
        st = run_app(resume_mode="Upload a file", upload=None)
        st.info.assert_called()
        st.error.assert_not_called()
        self.assertEqual(st.tabs.call_count, 0)

    def test_empty_pasted_jd_stops(self):
        st = run_app(jd_mode="Paste text", pasted="   ")
        st.info.assert_called()
        self.assertEqual(st.tabs.call_count, 0)

    def test_uploaded_resume_and_pasted_jd(self):
        up = SimpleNamespace(getvalue=lambda: pl.read_sample_resume("resume_rohan.pdf"),
                             name="Rohan's resume (final).pdf")
        st = run_app(resume_mode="Upload a file", upload=up, jd_mode="Paste text",
                     pasted=pl.read_sample_jd("jd_data_analyst.txt"))
        st.error.assert_not_called()
        self.assertTrue(st.metric.called)

    def test_bad_upload_shows_error_not_traceback(self):
        up = SimpleNamespace(getvalue=lambda: b"MZ", name="virus.exe")
        st = run_app(resume_mode="Upload a file", upload=up)
        st.error.assert_called_once()

    def test_jd_without_requirements_warns(self):
        st = run_app(jd_mode="Paste text", pasted="Just some text without sections")
        st.warning.assert_called()
        st.error.assert_not_called()

    def test_evaluation_button(self):
        st = run_app(button=True)
        st.error.assert_not_called()
        self.assertIn("eval_rows", st.session_state)
        self.assertGreaterEqual(len(st.session_state["eval_rows"]), 12)
        self.assertEqual(st.dataframe.call_count, 5)      # the evaluation table appears


if __name__ == "__main__":
    unittest.main()
