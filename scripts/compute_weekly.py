#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


LIKERT_MAP = {
    "bad 1": 1,
    "poor 2": 2,
    "fair 3": 3,
    "good 4": 4,
    "excellent 5": 5,
}


def likert_to_int(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    # Common patterns: "Bad 1", "Excellent 5", or just "1".."5"
    low = s.lower()
    if low in LIKERT_MAP:
        return LIKERT_MAP[low]
    # extract last digit 1..5
    m = re.search(r"([1-5])\s*$", s)
    if m:
        return int(m.group(1))
    # already numeric?
    try:
        v = int(float(s))
        if 1 <= v <= 5:
            return v
    except Exception:
        pass
    return None


@dataclass
class WeekWindow:
    week_id: str
    start_local: pd.Timestamp
    end_local: pd.Timestamp
    deadline_local: pd.Timestamp
    tz: str


def _to_local(ts_utc: pd.Timestamp, tz: str) -> pd.Timestamp:
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    return ts_utc.tz_convert(tz)


def load_week_windows(path: Path) -> List[WeekWindow]:
    dfw = pd.read_csv(path)
    required = {"week_id", "start_local", "end_local", "deadline_local", "tz"}
    missing = required - set(dfw.columns)
    if missing:
        raise ValueError(f"week_windows.csv missing columns: {sorted(missing)}")

    windows: List[WeekWindow] = []
    for _, row in dfw.iterrows():
        tz = str(row["tz"]).strip()
        start = pd.to_datetime(row["start_local"])
        end = pd.to_datetime(row["end_local"])
        deadline = pd.to_datetime(row["deadline_local"])

        # Localize as local time (naive -> tz-aware)
        start = start.tz_localize(tz)
        end = end.tz_localize(tz)
        deadline = deadline.tz_localize(tz)

        windows.append(
            WeekWindow(
                week_id=str(row["week_id"]).strip(),
                start_local=start,
                end_local=end,
                deadline_local=deadline,
                tz=tz,
            )
        )
    return windows


def assign_week(local_dt: pd.Timestamp, windows: List[WeekWindow]) -> Optional[str]:
    for w in windows:
        # Week membership by [start, end)
        if w.start_local <= local_dt < w.end_local:
            return w.week_id
    return None


def sanitize_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-zA-Z0-9 _.-]", "", s)
    s = s.replace(" ", "_")
    return s[:120] if s else "unknown"


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # try case-insensitive match
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def main():
    ap = argparse.ArgumentParser(description="Compute weekly peer-grading summaries from Forms exports.")
    ap.add_argument(
        "--responses",
        default="PeerGrading/Input/form_exports/forms_responses.xlsx",
        help="Path to forms_responses.xlsx copied into the repo.",
    )
    ap.add_argument(
        "--week-windows",
        default="PeerGrading/Input/week_setup/week_windows.csv",
        help="Path to week_windows.csv",
    )
    ap.add_argument(
        "--out-dir",
        default="PeerGrading/Output",
        help="Output directory",
    )
    ap.add_argument(
        "--week",
        default="ALL",
        help="Which week_id to compute (e.g., W01). Use ALL to compute all weeks present in week_windows.",
    )
    ap.add_argument(
        "--tz",
        default="Europe/Zurich",
        help="Timezone for week assignment and deadlines (should match week_windows.csv).",
    )
    args = ap.parse_args()

    responses_path = Path(args.responses)
    week_path = Path(args.week_windows)
    out_dir = Path(args.out_dir)

    if not responses_path.exists():
        raise FileNotFoundError(
            f"Responses file not found: {responses_path}\n"
            f"B planında: OneDrive'daki forms_responses.xlsx'i indirip bu path'e kopyalamalısın."
        )
    if not week_path.exists():
        raise FileNotFoundError(f"Week windows file not found: {week_path}")

    windows = load_week_windows(week_path)
    # Optional: ensure windows tz aligns with args.tz (not mandatory)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mails").mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(responses_path)

    # Normalize column names (strip)
    df.columns = [str(c).strip() for c in df.columns]

    # Identify timestamp columns
    col_received = find_col(df, ["ReceivedAtUTC", "ReceivedAtUtc", "receivedatutc"])
    col_submitted = find_col(df, ["SubmittedAt", "SubmissionTime", "CompletionTime", "submittedat"])

    # Parse timestamps
    ts = None
    if col_received:
        ts = pd.to_datetime(df[col_received], utc=True, errors="coerce")
    if ts is None or ts.isna().all():
        if col_submitted:
            # SubmittedAt may be local or UTC; we attempt parse and assume UTC if tz-naive
            ts2 = pd.to_datetime(df[col_submitted], errors="coerce")
            # If naive, treat as UTC
            ts2 = ts2.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
            ts = ts2
        else:
            raise ValueError("No usable timestamp column found (ReceivedAtUTC or SubmittedAt).")

    df["_ts_utc"] = ts
    df = df[~df["_ts_utc"].isna()].copy()

    # Convert to local
    df["_ts_local"] = df["_ts_utc"].dt.tz_convert(args.tz)

    # Assign week_id
    df["week_id"] = df["_ts_local"].apply(lambda x: assign_week(x, windows))

    # Drop rows that are outside week windows
    df = df[~df["week_id"].isna()].copy()

    # Apply deadline filter per week
    deadline_map: Dict[str, pd.Timestamp] = {w.week_id: w.deadline_local for w in windows}
    df["_deadline_local"] = df["week_id"].map(deadline_map)
    df = df[df["_ts_local"] < df["_deadline_local"]].copy()

    # Identify core columns
    col_presenter = find_col(df, ["PresenterChoice", "Presenter", "Who is the presenter"])
    col_rater_email = find_col(df, ["RaterEmail", "ResponderEmail", "Responder's email", "ResponderEmailAddress"])
    col_rater_name = find_col(df, ["RaterName", "ResponderName", "Responder's name"])
    col_comment = find_col(df, ["Comment", "Comments", "comment"])

    if not col_presenter:
        raise ValueError("PresenterChoice column not found. Check your Excel table headers.")

    # Identify question columns (allow minor variants)
    q_cols_candidates = {
        "Q2_1": ["Q2_1", "Q2.1", "Q2 1"],
        "Q2_2": ["Q2_2", "Q2.2", "Q2 2"],
        "Q2_3": ["Q2_3", "Q2.3", "Q2 3"],
        "Q3_1": ["Q3_1", "Q3.1", "Q3 1"],
        "Q3_2": ["Q3_2", "Q3.2", "Q3 2"],
        "Q3_3": ["Q3_3", "Q3.3", "Q3 3"],
        "Q4": ["Q4", "Q_4", "Q 4", "General grade", "GeneralGrade"],
    }

    found_qcols: Dict[str, str] = {}
    for key, cands in q_cols_candidates.items():
        col = find_col(df, cands)
        if col:
            found_qcols[key] = col

    missing_q = [k for k in ["Q2_1", "Q2_2", "Q2_3", "Q3_1", "Q3_2", "Q3_3", "Q4"] if k not in found_qcols]
    if missing_q:
        raise ValueError(f"Missing required question columns in Excel: {missing_q}. Found: {list(df.columns)}")

    # Convert Likert to numeric
    for k, col in found_qcols.items():
        df[k] = df[col].apply(likert_to_int)

    # Drop rows with any missing rating
    df = df.dropna(subset=["Q2_1", "Q2_2", "Q2_3", "Q3_1", "Q3_2", "Q3_3", "Q4"]).copy()

    # Compute weighted score
    df["score"] = (
        0.1 * (df["Q2_1"] + df["Q2_2"] + df["Q2_3"] + df["Q3_1"] + df["Q3_2"] + df["Q3_3"])
        + 0.4 * df["Q4"]
    )

    # Choose which weeks to compute
    target_week = args.week.strip()
    if target_week != "ALL":
        df = df[df["week_id"] == target_week].copy()

    if df.empty:
        print("No rows to process after filtering. (Check week range / deadlines / timestamps).")
        return

    # Aggregations per (week_id, presenter)
    group_cols = ["week_id", col_presenter]
    agg = df.groupby(group_cols).agg(
        n_raters=("score", "count"),
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        mean_Q2_1=("Q2_1", "mean"),
        mean_Q2_2=("Q2_2", "mean"),
        mean_Q2_3=("Q2_3", "mean"),
        mean_Q3_1=("Q3_1", "mean"),
        mean_Q3_2=("Q3_2", "mean"),
        mean_Q3_3=("Q3_3", "mean"),
        mean_Q4=("Q4", "mean"),
    ).reset_index().rename(columns={col_presenter: "PresenterChoice"})

    # Save per-week outputs
    for week_id, dfw in agg.groupby("week_id"):
        week_out = out_dir / f"weekly_summary_{week_id}.csv"
        dfw.sort_values(["mean_score", "n_raters"], ascending=[False, False]).to_csv(week_out, index=False)

        # Peer log (who graded whom) - keep minimal but useful
        log_cols = ["week_id", "_ts_local", "score", "PresenterChoice"]
        if col_rater_email:
            log_cols.insert(3, col_rater_email)
        if col_rater_name:
            log_cols.insert(3, col_rater_name)
        if col_comment:
            log_cols.append(col_comment)

        dflog = df.rename(columns={col_presenter: "PresenterChoice"}).copy()
        dflog["_ts_local"] = dflog["_ts_local"].astype(str)
        dflog = dflog[log_cols].copy()
        dflog.to_csv(out_dir / f"peer_log_{week_id}.csv", index=False)

        # Mail drafts (no sending)
        mails_dir = out_dir / "mails" / week_id
        mails_dir.mkdir(parents=True, exist_ok=True)

        df_week_raw = df[df["week_id"] == week_id].copy()
        df_week_raw = df_week_raw.rename(columns={col_presenter: "PresenterChoice"})

        for presenter, dfp in df_week_raw.groupby("PresenterChoice"):
            presenter_safe = sanitize_filename(str(presenter))
            mean_score = float(dfp["score"].mean())
            n = int(dfp["score"].count())

            comments = []
            if col_comment and col_comment in dfp.columns:
                for c in dfp[col_comment].fillna("").astype(str).tolist():
                    c = c.strip()
                    if c and c.lower() not in {"nan", "none"}:
                        comments.append(c)

            lines = []
            lines.append(f"Subject: Peer feedback summary ({week_id})")
            lines.append("")
            lines.append(f"Hi {presenter},")
            lines.append("")
            lines.append(f"Here is your peer-assessment summary for {week_id}:")
            lines.append(f"- Number of reviewers: {n}")
            lines.append(f"- Final score (weighted): {mean_score:.2f} / 5.00")
            lines.append("")
            lines.append("Per-question averages (1–5):")
            lines.append(f"- Q2_1: {dfp['Q2_1'].mean():.2f}")
            lines.append(f"- Q2_2: {dfp['Q2_2'].mean():.2f}")
            lines.append(f"- Q2_3: {dfp['Q2_3'].mean():.2f}")
            lines.append(f"- Q3_1: {dfp['Q3_1'].mean():.2f}")
            lines.append(f"- Q3_2: {dfp['Q3_2'].mean():.2f}")
            lines.append(f"- Q3_3: {dfp['Q3_3'].mean():.2f}")
            lines.append(f"- Q4  : {dfp['Q4'].mean():.2f}")
            lines.append("")
            if comments:
                lines.append("Comments:")
                for i, c in enumerate(comments, 1):
                    lines.append(f"{i}. {c}")
                lines.append("")
            else:
                lines.append("Comments: (no written comments submitted)")
                lines.append("")

            lines.append("Best regards,")
            lines.append("Course team")

            (mails_dir / f"{presenter_safe}.txt").write_text("\n".join(lines), encoding="utf-8")

    # Master summary
    agg.to_csv(out_dir / "weekly_summary_ALL.csv", index=False)
    print(f"Done. Outputs written under: {out_dir}")


if __name__ == "__main__":
    main()
