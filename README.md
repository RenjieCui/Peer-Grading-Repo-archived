# PeerGrading Starter

This repo ingests Microsoft Forms exports + a students database + a weekly presentations setup file,
computes peer-grading scores, and produces per-presenter summaries and email-ready drafts.

## Folder expectations (SharePoint/OneDrive)
We assume you already keep live files in something like:

- `PeerGrading/Input/students_db/students.xlsx`
- `PeerGrading/Input/form_exports/forms_responses.xlsx`
- `PeerGrading/Input/week_setup/2026-W09/presentations.csv`

For local testing, download copies into `data/example/` or point env vars to your synced OneDrive paths.

## Quick start
1. Create a venv and install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` -> `.env` and set file paths.

3. Update `config/forms_mapping.yaml` so its column names match your Forms export headers.

4. Run:
   ```bash
   python scripts/run_pipeline.py
   ```

Outputs go to `outputs/<WEEK_ID>/`.

## Scoring
- 6 rubric questions (3 + 3): weight 0.1 each
- general grade question: weight 0.4
Final per-response score is the weighted sum (scale 1..5).
Final per-presenter score is the mean across unique raters (deduped by latest submission before deadline).
