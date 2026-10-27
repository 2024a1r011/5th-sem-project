"""Streamlit demo for the Resume Quality & Job Fit Analyzer (mid-project, stages 1-5).

Run from the project root:
    streamlit run app.py

This file only draws things. All the logic lives in src/resume_analyzer/pipeline.py,
which is unit-tested without Streamlit.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from resume_analyzer import pipeline as pl                      # noqa: E402

st.set_page_config(page_title="Resume Quality & Job Fit Analyzer", layout="wide")


# --------------------------------------------------------------------------
# Cached work
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_analysis(resume_bytes: bytes, filename: str, jd_text: str):
    return pl.analyze(resume_bytes, filename, jd_text)


@st.cache_data(show_spinner=False)
def run_evaluations():
    import eval_all
    return eval_all.summary()


# --------------------------------------------------------------------------
# Header + sidebar inputs
# --------------------------------------------------------------------------
st.title("Resume Quality & Job Fit Analyzer")
st.caption("Prototype at the mid-project review: stages 1 to 5 of 8 are working. "
           "Scores here are this project's own analysis, not an ATS score.")

st.sidebar.header("1. Resume")
resume_mode = st.sidebar.radio("Resume source", ["Sample resume", "Upload a file"])
resume_bytes, resume_name = None, ""
if resume_mode == "Sample resume":
    samples = pl.list_sample_resumes()
    resume_name = st.sidebar.selectbox("Choose a sample", samples)
    if resume_name:
        resume_bytes = pl.read_sample_resume(resume_name)
else:
    up = st.sidebar.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
    if up is not None:
        resume_bytes, resume_name = up.getvalue(), pl.safe_filename(up.name)

st.sidebar.header("2. Job description")
jd_mode = st.sidebar.radio("Job description source", ["Sample JD", "Paste text"])
jd_text = ""
if jd_mode == "Sample JD":
    jd_choice = st.sidebar.selectbox("Choose a sample JD", pl.list_sample_jds())
    if jd_choice:
        jd_text = pl.read_sample_jd(jd_choice)
        with st.sidebar.expander("Preview"):
            st.text(jd_text)
else:
    jd_text = st.sidebar.text_area("Paste the job description", value="", height=300)

if not resume_bytes or not jd_text.strip():
    st.info("Pick or upload a resume and choose or paste a job description in the sidebar.")
    st.stop()

try:
    with st.spinner("Analyzing..."):
        result = run_analysis(resume_bytes, resume_name, jd_text)
except Exception as exc:                                        # show, don't crash
    st.error(f"Could not analyze this input: {exc}")
    st.stop()

for w in result.warnings:
    st.warning(w)

prof, jd = result.profile, result.jd
tabs = st.tabs(["Overview", "Resume understanding", "Skills & evidence",
                "Job requirements", "Evidence matches", "Evaluation", "Roadmap"])

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tabs[0]:
    st.subheader(f"{prof.name or 'Candidate'}  vs  {jd.title or 'Job'}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Highest degree", prof.highest_degree or "not found")
    c2.metric("Experience (approx.)", pl.format_duration(prof.total_experience_months))
    c3.metric("Skills found", len(prof.skills))
    c4.metric("Job requirements", len(jd.requirements))

    counts = pl.strength_counts(prof)
    st.markdown("**How well is each skill backed up?**")
    s1, s2, s3 = st.columns(3)
    s1.metric("Demonstrated", counts["demonstrated"],
              help="Used in an experience or project entry")
    s2.metric("Supported", counts["supported"],
              help="Appears in education, certifications or achievements")
    s3.metric("Only claimed", counts["claimed"],
              help="Listed in a Skills line or summary, with no example of use")

    st.markdown("**Job skills found on the resume**")
    cov = pl.jd_skill_coverage(prof, jd)
    a, b = st.columns(2)
    with a:
        st.markdown("Must-have")
        for item in cov["must_found"]:
            st.write("✅ " + item)
        for item in cov["must_missing"]:
            st.write("❌ " + item)
    with b:
        st.markdown("Nice-to-have")
        for item in cov["nice_found"]:
            st.write("✅ " + item)
        for item in cov["nice_missing"]:
            st.write("❌ " + item)
    st.caption("Name-level check that understands 'A or B' alternatives. It does not yet "
               "recognise related evidence (e.g. unit tests for 'integration testing'); "
               "that is what the embedding stage adds.")

    st.metric("Baseline similarity (TF-IDF)", f"{result.baseline_score:.0f} / 100",
              help="A naive word-overlap number, shown only as the baseline. The final "
                   "report will explain its scores instead of giving one unexplained number.")

# --------------------------------------------------------------------------
# Resume understanding
# --------------------------------------------------------------------------
with tabs[1]:
    ex = result.extraction
    st.write(f"**{ex.source_type.upper()}**, {ex.n_pages} page(s), {ex.n_chars} characters extracted.")
    with st.expander("Extracted text"):
        st.text(ex.text)
    st.markdown("Every line is tagged with its section; bullets are linked to the role or "
                "project they belong to.")
    sections = sorted({s.section for s in result.sentences})
    chosen = st.selectbox("Filter by section", ["all"] + sections)
    rows = pl.section_rows(result.sentences)
    if chosen != "all":
        rows = [r for r in rows if r["Section"] == chosen]
    st.dataframe(rows, hide_index=True)

# --------------------------------------------------------------------------
# Skills & evidence
# --------------------------------------------------------------------------
with tabs[2]:
    st.markdown("Each skill is graded by where it appears, and keeps the sentences that prove it.")
    st.dataframe(pl.skill_rows(prof), hide_index=True)
    if prof.skills:
        pick = st.selectbox("Show the evidence for a skill", [r.skill for r in prof.skills])
        rec = prof.get_skill(pick)
        if rec:
            st.write(f"**{rec.skill}**: {rec.strength}")
            for m in rec.mentions:
                where = f"{m.section}" + (f", in '{m.parent}'" if m.parent else "")
                st.write(f"- ({where}) {m.text}")

# --------------------------------------------------------------------------
# Job requirements
# --------------------------------------------------------------------------
with tabs[3]:
    d1, d2, d3 = st.columns(3)
    d1.metric("Degree required", jd.degree_level or "not specified")
    d2.metric("Min. years", f"{jd.min_years:g}" if jd.min_years is not None else "not specified")
    d3.metric("Must / nice skills", f"{len(jd.must_skills)} / {len(jd.nice_skills)}")
    st.dataframe(pl.requirement_rows(jd), hide_index=True)
    if jd.about:
        with st.expander("Company / benefits text (not matched)"):
            for line in jd.about:
                st.write("- " + line)

# --------------------------------------------------------------------------
# Evidence matches
# --------------------------------------------------------------------------
with tabs[4]:
    st.markdown(f"For each requirement, the closest resume sentences by **{result.matcher_name.upper()}** "
                "similarity. The bar shows overlap in words, not a verdict.")
    view = st.selectbox("Show", ["all requirements", "must-have only", "nice-to-have only",
                                 "no evidence found"])
    for m in result.matches:
        r = m.requirement
        if view == "must-have only" and r.priority != "must":
            continue
        if view == "nice-to-have only" and r.priority != "nice":
            continue
        if view == "no evidence found" and m.evidence:
            continue
        tag = "MUST" if r.priority == "must" else "nice"
        with st.expander(f"[{tag}] {r.text}", expanded=bool(m.evidence) and r.index < 3):
            if not m.evidence:
                st.warning("No overlapping words with any resume sentence. TF-IDF only sees "
                           "shared words, so related evidence phrased differently is missed.")
            for e in m.evidence:
                st.progress(min(max(e.score, 0.0), 1.0))
                st.write(f"**{e.score:.2f}**  {e.text}")
                st.caption(f"{e.section}" + (f"  |  {e.parent}" if e.parent else ""))

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
with tabs[5]:
    st.markdown("Every component is measured against hand-labeled data and compared with a "
                "simple baseline. The test sets are small and team-labeled, so read these as "
                "indicative, not final.")
    if st.button("Run all evaluations"):
        with st.spinner("Evaluating..."):
            st.session_state["eval_rows"] = run_evaluations()
    if "eval_rows" in st.session_state:
        st.dataframe(st.session_state["eval_rows"], hide_index=True)
        st.caption("Requirement matching: the TF-IDF baseline is scored; the embedding matcher "
                   "will be scored on the same labeled requirements.")

# --------------------------------------------------------------------------
# Roadmap
# --------------------------------------------------------------------------
with tabs[6]:
    st.markdown("Pipeline: Resume, Document understanding, Quality analysis, Candidate profile, "
                "Job description, Semantic matching, Gap analysis, Explainable report.")
    st.dataframe([{"Stage": n, "Component": name, "Status": pl.STAGE_LABELS[status]}
                  for n, name, status in pl.STAGES], hide_index=True)
    st.markdown("**Next:** sentence-embedding matcher compared against TF-IDF; "
                "strong / partial / missing classification; resume quality scoring; "
                "explainable report with recommendations.")
