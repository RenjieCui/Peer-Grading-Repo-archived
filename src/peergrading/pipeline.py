from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd

from .io import read_yaml, load_students, load_week_setup, load_forms_export
from .compute import build_rubric, compute_weighted_score, parse_submitted_at, dedupe_latest_before_deadline
from .utils import compute_deadline_for_week, normalize_week_id

def resolve_presenter(df: pd.DataFrame, week_setup: pd.DataFrame, students: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    choice = df["presenter_choice"].astype(str).str.strip()

    # try match week_setup by presenter_display or presenter_email
    ws = week_setup.copy()
    ws["presenter_display_norm"] = ws["presenter_display"].astype(str).str.strip()
    ws["presenter_email_norm"] = ws["presenter_email"].astype(str).str.strip().str.lower()

    # map by exact display
    display_map = {d: e for d, e in zip(ws["presenter_display_norm"], ws["presenter_email_norm"]) if d}
    email_map = {e: e for e in ws["presenter_email_norm"] if e}

    def _resolve(x: str) -> str:
        xl = x.lower()
        if xl in email_map:
            return email_map[xl]
        if x in display_map:
            return display_map[x]
        # fallback: try students by name/email
        if "@" in xl:
            return xl
        hit = students[students["name"].str.lower() == xl]
        if len(hit) == 1:
            return hit.iloc[0]["email"]
        return ""

    df["presenter_email_resolved"] = choice.apply(_resolve)
    return df

def build_outputs(cleaned: pd.DataFrame, week_setup: pd.DataFrame, students: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Presenter summary
    pres = (
        cleaned.dropna(subset=["weighted_score"])
        .groupby(["week_id", "presenter_email_resolved"], as_index=False)
        .agg(
            n_raters=("rater_email", "nunique"),
            mean_score=("weighted_score", "mean"),
            std_score=("weighted_score", "std"),
        )
    )
    # Attach presentation_id/topic if available
    ws = week_setup[["presenter_email", "presentation_id", "topic"]].copy()
    ws["presenter_email"] = ws["presenter_email"].str.lower()
    pres = pres.merge(ws, how="left", left_on="presenter_email_resolved", right_on="presenter_email").drop(columns=["presenter_email"])

    # Rater activity
    rater = (
        cleaned.dropna(subset=["weighted_score"])
        .groupby(["week_id", "rater_email"], as_index=False)
        .agg(
            presentations_graded=("presenter_email_resolved", "nunique"),
            last_submitted_at=("submitted_at", "max"),
        )
    )
    # Missing raters per presentation (relative to full roster)
    roster = set(students["email"].str.lower().tolist())
    missing_rows = []
    for _, row in ws.iterrows():
        presenter_email = str(row["presenter_email"]).lower()
        graded_by = set(cleaned.loc[cleaned["presenter_email_resolved"] == presenter_email, "rater_email"].str.lower().tolist())
        missing = sorted(roster - graded_by - {presenter_email})  # usually presenter shouldn't grade themselves
        missing_rows.append({
            "week_id": cleaned["week_id"].iloc[0] if len(cleaned) else "",
            "presentation_id": row.get("presentation_id", ""),
            "presenter_email": presenter_email,
            "missing_raters_count": len(missing),
            "missing_raters": ";".join(missing),
        })
    missing_df = pd.DataFrame(missing_rows)

    return {
        "presenter_summary": pres,
        "rater_activity": rater,
        "missing_raters_by_presentation": missing_df,
    }

def generate_email_drafts(presenter_summary: pd.DataFrame, cleaned: pd.DataFrame, students: pd.DataFrame) -> dict[str, str]:
    # returns mapping filename -> markdown body
    name_by_email = {e: n for e, n in zip(students["email"].str.lower(), students["name"])}
    drafts = {}
    for _, row in presenter_summary.iterrows():
        presenter_email = str(row["presenter_email_resolved"]).lower()
        presenter_name = name_by_email.get(presenter_email, presenter_email)
        pres_id = row.get("presentation_id", "")
        topic = row.get("topic", "")
        mean_score = row.get("mean_score", None)
        n_raters = int(row.get("n_raters", 0) or 0)

        # collect comments (if present)
        comments = []
        if "comment" in cleaned.columns:
            sub = cleaned[(cleaned["presenter_email_resolved"].str.lower() == presenter_email) & cleaned["comment"].notna()]
            comments = [str(c).strip() for c in sub["comment"].tolist() if str(c).strip()]

        lines = []
        lines.append(f"Hi {presenter_name},")
        lines.append("")
        lines.append(f"Here are your peer-grading results for **{pres_id}** {f'({topic})' if topic else ''}:")
        lines.append("")
        if mean_score is not None and pd.notna(mean_score):
            lines.append(f"- **Average score:** {float(mean_score):.2f} / 5.00")
        lines.append(f"- **Number of raters:** {n_raters}")
        lines.append("")
        lines.append("### Breakdown (per rater)")
        sub = cleaned[(cleaned["presenter_email_resolved"].str.lower() == presenter_email)].copy()
        sub = sub.dropna(subset=["weighted_score"]).sort_values("weighted_score", ascending=False)
        if len(sub):
            for _, r in sub.iterrows():
                rater = str(r.get("rater_email", "")).lower()
                rname = name_by_email.get(rater, rater)
                lines.append(f"- {rname}: {float(r['weighted_score']):.2f}")
        else:
            lines.append("- (No valid submissions found before the deadline.)")
        lines.append("")
        if comments:
            lines.append("### Comments")
            for c in comments[:50]:
                lines.append(f"- {c}")
            lines.append("")
        lines.append("Best,")
        lines.append("Course Team")

        fname = f"{pres_id or presenter_email.replace('@','_at_')}.md"
        drafts[fname] = "\n".join(lines)
    return drafts

def run_pipeline(
    students_db_path: str,
    forms_export_path: str,
    week_setup_path: str,
    out_dir: str,
    week_id: str | None = None,
    local_tz: str = "Europe/Zurich",
) -> Path:
    cfg_mapping = read_yaml(Path(__file__).resolve().parents[2] / "config" / "forms_mapping.yaml")
    cfg_rubric = read_yaml(Path(__file__).resolve().parents[2] / "config" / "rubric.yaml")
    rubric = build_rubric(cfg_rubric)

    # Determine week_id
    if week_id and str(week_id).strip():
        week_id_norm = normalize_week_id(str(week_id))
    else:
        now = datetime.now(ZoneInfo(local_tz))
        week_id_norm = f"{now.isocalendar().year:04d}-W{now.isocalendar().week:02d}"

    deadline = compute_deadline_for_week(week_id_norm, local_tz)

    students = load_students(students_db_path)
    week_setup = load_week_setup(week_setup_path, week_id_norm)
    forms_raw = load_forms_export(forms_export_path)

    # Rename columns based on mapping
    meta = cfg_mapping.get("metadata_columns", {})
    rub_cols = cfg_mapping.get("rubric_columns", {})
    free_cols = cfg_mapping.get("free_text_columns", {})

    needed = {**meta, **rub_cols, **free_cols}
    # Validate mapped columns exist (ignore empty mappings)
    missing = []
    for internal, col in needed.items():
        if not col:
            continue
        if col not in forms_raw.columns:
            missing.append((internal, col))
    if missing:
        msg = "\n".join([f"- {k}: '{c}'" for k, c in missing])
        raise ValueError(f"Your Forms export is missing mapped columns (check config/forms_mapping.yaml):\n{msg}")

    df = pd.DataFrame()
    # copy over metadata
    for internal, col in meta.items():
        df[internal] = forms_raw[col] if col else pd.NA
    for internal, col in rub_cols.items():
        df[internal] = forms_raw[col] if col else pd.NA
    for internal, col in free_cols.items():
        if col:
            df[internal] = forms_raw[col]
        else:
            df[internal] = pd.NA

    df["week_id"] = week_id_norm
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()

    # normalize emails
    if "rater_email" in df.columns:
        df["rater_email"] = df["rater_email"].astype(str).str.strip().str.lower().replace({"nan": ""})
    if "rater_name" in df.columns:
        df["rater_name"] = df["rater_name"].astype(str).str.strip()

    # submitted time
    df["submitted_at"] = parse_submitted_at(df["submitted_at"], local_tz)

    # resolve presenter
    df = resolve_presenter(df, week_setup=week_setup, students=students)

    # compute weighted score
    df["weighted_score"] = compute_weighted_score(df, rubric=rubric)

    # filter & dedupe
    df_dedup = dedupe_latest_before_deadline(df, deadline=deadline)

    # outputs
    out_path = Path(out_dir) / week_id_norm
    out_path.mkdir(parents=True, exist_ok=True)

    cleaned_path = out_path / "cleaned_responses.csv"
    df_dedup.to_csv(cleaned_path, index=False)

    outputs = build_outputs(df_dedup, week_setup=week_setup, students=students)
    outputs["presenter_summary"].to_csv(out_path / "presenter_summary.csv", index=False)
    outputs["rater_activity"].to_csv(out_path / "rater_activity.csv", index=False)
    outputs["missing_raters_by_presentation"].to_csv(out_path / "missing_raters_by_presentation.csv", index=False)

    # email drafts
    drafts = generate_email_drafts(outputs["presenter_summary"], df_dedup, students=students)
    emails_dir = out_path / "emails"
    emails_dir.mkdir(exist_ok=True)
    for fname, body in drafts.items():
        (emails_dir / fname).write_text(body, encoding="utf-8")

    # run meta
    meta_path = out_path / "run_meta.txt"
    meta_path.write_text(
        f"week_id={week_id_norm}\n" +
        f"deadline_local={deadline.isoformat()}\n" +
        f"forms_export={forms_export_path}\n" +
        f"students_db={students_db_path}\n" +
        f"week_setup={week_setup_path}\n",
        encoding="utf-8"
    )

    return out_path
