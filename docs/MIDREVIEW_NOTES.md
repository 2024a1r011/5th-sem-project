# Mid-project review notes (about 50% complete)

## Status against the proposal's 8-stage pipeline
| # | Stage | Status |
|---|-------|--------|
| 1 | Resume input (PDF/DOCX/TXT) | Done |
| 2 | Document understanding (sections, bullets, parents) | Done |
| 3 | Candidate profile (skills graded by evidence) | Done |
| 4 | Job understanding (requirements, priority, any-of skills) | Done |
| 5 | Semantic matching | TF-IDF baseline done, embeddings next |
| 6 | Gap analysis (strong / partial / missing) | Planned |
| 7 | Resume quality engine | Planned |
| 8 | Explainable report + recommendations | Planned |

## Results so far (python scripts/eval_all.py)
| Component | Metric | Baseline | Ours |
|---|---|---|---|
| Section detection | Heading F1 / line accuracy | - | 1.000 / 0.958 |
| Resume skill extraction | F1 | 0.942 (keyword) | 0.995 |
| JD parsing | Skill F1 / must-vs-nice accuracy | 0.949 / n.a. | 0.964 / 1.000 |
| Requirement matching | Hit@1 / Hit@3 / MRR / AUROC | 0.680 / 0.760 / 0.717 / 0.837 (TF-IDF) | embeddings next |

All test sets are small and labeled by the team (6-9 resumes, 6 JDs, 72 requirements).
Say so. Expect lower numbers on real, messy documents.

## 5-minute demo flow (streamlit run app.py)
1. Overview tab, Ananya vs backend JD: skills graded demonstrated / supported / claimed;
   the any-of handling ("Flask / Django / FastAPI: has Flask"); only CI/CD and Integration
   Testing are genuinely missing.
2. Resume understanding: sections and bullets linked to "Backend Developer Intern, TechNova".
3. Evidence matches: "REST APIs" requirement finds "Built REST APIs in Flask..." (works).
4. Same tab, filter "no evidence found" and the "Collaborate with frontend engineers and
   product managers" requirement: it matched "Sentiment Analysis of Product Reviews" on the
   shared word "product". Also: degree requirements never match "B.Tech". This is the
   limitation the embedding stage is designed to fix.
5. Swap to the weak resume (resume_simran.pdf): few skills, no experience, very low overlap.
6. Evaluation tab: click Run all evaluations.

## Likely questions
- "Is this novel?" No, and the proposal says so. The contribution is whole-document
  understanding, evidence-graded skills, explainability, and measured evaluation.
- "Is the score an ATS score?" No. It is this project's own analysis. The single
  TF-IDF number is only a baseline; the final report will explain its scores.
- "Why TF-IDF first?" It is the baseline the embedding method must beat on the same labeled set.
- "How reliable are the numbers?" Small team-labeled sets; shown as indicative.
- "What did evaluation teach you?" Four real bugs found by measuring: header skipped on
  resumes without headings, sentence splitter cutting "B.E.", a date range read as a
  phone number, and a plural-stripper breaking "node.js".

## Known limitations
- Rule-based section detection: resumes with no headings fall back to "header".
- Skill matching is exact phrases + aliases ("Review code" != "Code Review").
- "X or Y" alternatives collapse if one option is outside the skills taxonomy.
- TF-IDF cannot match meaning across different words.
- Year-only dates give approximate experience durations.
