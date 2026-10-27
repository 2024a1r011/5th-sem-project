# AI-Powered Resume Quality & Job Fit Analyzer

Document-intelligence pipeline: resume -> sections -> candidate profile -> semantic
matching against a job description -> gap analysis -> explainable report.

## Setup
    pip install -r requirements.txt            # Steps 1-6 (includes scikit-learn and streamlit)
    python scripts/generate_samples.py         # optional: the sample PDFs/DOCX are already included
    python -m unittest discover -s tests       # or: pytest

If `pip install streamlit` fails on a very new Python (3.14), create a virtual environment
with Python 3.12 or 3.13 and install there.

## Run the demo (Step 6)
    streamlit run app.py                       # opens in the browser
    python scripts/eval_all.py                 # all results in one table

## Try Step 1 (text extraction)
    cd src
    python -m resume_analyzer.extraction ../data/sample_resumes/resume_ananya.pdf

## Try Step 2 (sections + evidence sentences)
    cd src
    python -m resume_analyzer.sections ../data/sample_resumes/resume_ananya.pdf
    cd .. && python scripts/eval_sections.py      # accuracy on hand-labeled resumes

## Try Step 3 (candidate profile + skill evidence)
    cd src
    python -m resume_analyzer.profile ../data/sample_resumes/resume_ananya.pdf
    cd .. && python scripts/eval_skills.py        # skill extraction vs keyword baseline

## Try Step 4 (job description -> requirements)
    cd src
    python -m resume_analyzer.jd ../data/sample_jds/jd_backend_engineer.txt
    cd .. && python scripts/eval_jd.py            # line labels, skills, must-vs-nice priority

## Try Step 5 (requirement -> evidence matching, TF-IDF baseline)
    cd src
    python -m resume_analyzer.matching ../data/sample_resumes/resume_ananya.pdf ../data/sample_jds/jd_backend_engineer.txt
    cd .. && python scripts/eval_matching.py      # Hit@1 / Hit@3 / MRR / AUROC vs hand labels

## Layout
    src/resume_analyzer/schema.py       shared data format (agree on this first)
    src/resume_analyzer/extraction.py   Step 1: PDF/DOCX/TXT -> clean text
    src/resume_analyzer/sections.py     Step 2: section detection + bullets/sentences
    src/resume_analyzer/skills.py       Step 3: taxonomy-based skill extraction
    src/resume_analyzer/skills_taxonomy.json   curated skills (edit this to add skills)
    src/resume_analyzer/profile.py      Step 3: CandidateProfile with graded skill evidence
    data/gold_skills.json               hand-labeled skills for evaluation
    src/resume_analyzer/jd.py           Step 4: JD -> Requirement list (priority, skills, years, degree)
    data/labeled_jds/ + gold_jd_skills.json   hand-labeled JDs for evaluation
    scripts/eval_jd.py                  JD parsing metrics
    src/resume_analyzer/matching.py     Step 5: TF-IDF matcher (shared Matcher interface)
    src/resume_analyzer/pipeline.py     Step 6: analyze() runs stages 1-5; UI table helpers
    app.py                              Step 6: Streamlit demo (thin; logic is in pipeline.py)
    scripts/eval_all.py                 one results table for every component
    docs/MIDREVIEW_NOTES.md             status, results, demo script, likely questions
    data/gold_matches.json              hand-labeled relevant evidence (8 pairs, 72 requirements)
    scripts/eval_matching.py            retrieval metrics; any future matcher plugs in here
    scripts/eval_skills.py              skill extraction P/R/F1 vs keyword baseline
    data/labeled/                       hand-labeled resumes (gold set for evaluation)
    scripts/eval_sections.py            precision/recall/F1 + accuracy report
    data/source_text/                   synthetic resumes (strong / medium / weak)
    data/sample_resumes/                generated PDF/DOCX versions
    data/sample_jds/                    sample job descriptions
    tests/                              unit tests

## Status
- [x] Step 1  Extraction
- [x] Step 2  Section detection (heading F1 1.00, line accuracy 95.8% on 6 labeled resumes)
- [x] Step 3  Candidate profile (skill F1 0.995 vs 0.942 keyword baseline on 9 labeled resumes)
- [x] Step 4  JD -> requirements (line labels 100%, skill F1 0.964 vs 0.949 baseline, priority 100% on 6 labeled JDs)
- [x] Step 5  TF-IDF baseline matching (Hit@1 0.68, Hit@3 0.76, MRR 0.72, AUROC 0.84 on 72 labeled requirements)
- [x] Step 6  Streamlit demo (mid-project milestone reached)

## Known limitations (be upfront about these in the review)
- Gold sets are small and written by the team; real resumes/JDs will be messier.
- Section/heading detection is rule-based; resumes with no headings fall back to "header".
- Skill matching is exact-phrase + aliases: "Review code" is not matched to "Code Review",
  "predictive models" not to "Predictive Modeling". The embedding matcher (second half) targets this.
- "X or Y" alternatives collapse to a single required skill if one option is not in the taxonomy.
- Year-only date ranges give approximate experience durations.
- TF-IDF matches shared words only: it cannot link "B.Tech" to "Bachelor's degree" or
  "query indexing" to "optimise database queries", and a certificate title can outrank a work bullet.
