#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

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
    low = s.lower()
    if low in LIKERT_MAP:
        return LIKERT_MAP[low]
    m = re.search(r"([1-5])\s*$", s)
    if m:
        return int(m.group(1))
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
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_week_windows(path: Path) -> List[WeekWindow]:
    dfw = pd.read_csv(path)
    required = {"week_id", "start_local", "end_local", "deadline_local", "tz"}
    missing = required - set(dfw.columns)
    if missing:
        raise ValueError(f"week_windows.csv missing columns: {sorted(missing)}")

    windows: List[WeekWindow] = []
    for _, row in dfw.iterrows():
        tz = str(row["tz"]).strip()
        start = pd.to_datetime(row["start_local"]).tz_localize(tz)
        end = pd.to_datetime(row["end_local"]).tz_localize(tz)
        deadline = pd.to_datetime(row["deadline_local"]).tz_localize(tz)
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
        if w.start_local <= local_dt < w.end_local:
            return w.week_id
    return None


def load_students(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    # xlsx/csv destekle
    if path.suffix.lower() in [".xlsx", ".xls"]:
        s = pd.read_excel(path)
    else:
        s = pd.read_csv(path)

    s.columns = [str(c).strip() for c in s.columns]

    # Senin dosyaya göre kolon isimleri
    email_col = find_col(s, ["E-mail", "Email", "E-mail address", "Mail"])
    first_col = find_col(s, ["First name", "Firstname", "First Name"])
    last_col  = find_col(s, ["Last name", "Lastname", "Last Name", "Surname", "Family name"])
    name_col  = find_col(s, ["StudentName", "Name", "Full name", "Full Name"])

    if not email_col:
        raise ValueError("Student list must contain an email column (e.g., 'E-mail').")

    s["StudentEmail"] = s[email_col].astype(str).str.strip().str.lower()

    if name_col:
        s["StudentName"] = s[name_col].astype(str).fillna("").str.strip()
    else:
        fn = s[first_col].astype(str).fillna("").str.strip() if first_col else ""
        ln = s[last_col].astype(str).fillna("").str.strip() if last_col else ""
        if isinstance(fn, str):  # güvenlik (nadiren olur)
            fn = ""
        if isinstance(ln, str):
            ln = ""
        s["StudentName"] = (fn + " " + ln).str.strip()

    s = s[["StudentEmail", "StudentName"]].dropna(subset=["StudentEmail"]).drop_duplicates()
    s = s[s["StudentEmail"] != ""]
    return s



def main():
    ap = argparse.ArgumentParser(description="Compute weekly peer-grading summaries from Forms exports.")
    ap.add_argument("--responses", default="PeerGrading/Input/form_exports/forms_responses.xlsx")
    ap.add_argument("--week-windows", default="PeerGrading/Input/week_setup/week_windows.csv")
    ap.add_argument("--students", default="PeerGrading/Input/students_db/StudentListDB.xlsx")
    ap.add_argument("--out-dir", default="PeerGrading/Output")
    ap.add_argument("--week", default="ALL")
    ap.add_argument("--tz", default="Europe/Zurich")
    args = ap.parse_args()

    responses_path = Path(args.responses)
    week_path = Path(args.week_windows)
    students_path = Path(args.students)
    out_dir = Path(args.out_dir)

    if not responses_path.exists():
        raise FileNotFoundError(f"Responses file not found: {responses_path}")
    if not week_path.exists():
        raise FileNotFoundError(f"Week windows file not found: {week_path}")

    windows = load_week_windows(week_path)
    deadline_map: Dict[str, pd.Timestamp] = {w.week_id: w.deadline_local for w in windows}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mails").mkdir(parents=True, exist_ok=True)
    master_dir = out_dir / "master"
    master_dir.mkdir(parents=True, exist_ok=True)

    # Read responses
    df = pd.read_excel(responses_path)
    df.columns = [str(c).strip() for c in df.columns]

    col_response_id = find_col(df, ["ResponseId", "ResponseID", "responseid"])
    col_received = find_col(df, ["ReceivedAtUTC", "ReceivedAtUtc"])
    col_submitted = find_col(df, ["SubmittedAt"])

    # Timestamp parse (prefer ReceivedAtUTC)
    ts = None
    if col_received:
        ts = pd.to_datetime(df[col_received], utc=True, errors="coerce")
    if ts is None or ts.isna().all():
        if col_submitted:
            ts2 = pd.to_datetime(df[col_submitted], errors="coerce")
            ts2 = ts2.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
            ts = ts2
        else:
            raise ValueError("No usable timestamp column found (ReceivedAtUTC or SubmittedAt).")

    df["_ts_utc"] = ts
    df = df[~df["_ts_utc"].isna()].copy()
    df["_ts_local"] = df["_ts_utc"].dt.tz_convert(args.tz)

    # Week assignment
    df["week_id"] = df["_ts_local"].apply(lambda x: assign_week(x, windows))
    df = df[~df["week_id"].isna()].copy()

    # Deadline filter
    df["_deadline_local"] = df["week_id"].map(deadline_map)
    df = df[df["_ts_local"] < df["_deadline_local"]].copy()

    col_presenter = find_col(df, ["PresenterChoice"])
    col_rater_email = find_col(df, ["RaterEmail"])
    col_rater_name = find_col(df, ["RaterName"])
    col_comment = find_col(df, ["Comment"])

    if not col_presenter:
        raise ValueError("PresenterChoice column not found.")

    # Questions
    q_cols_candidates = {
        "Q2_1": ["Q2_1"],
        "Q2_2": ["Q2_2"],
        "Q2_3": ["Q2_3"],
        "Q3_1": ["Q3_1"],
        "Q3_2": ["Q3_2"],
        "Q3_3": ["Q3_3"],
        "Q4": ["Q4", "Q_4"],
    }
    found_qcols: Dict[str, str] = {}
    for key, cands in q_cols_candidates.items():
        col = find_col(df, cands)
        if col:
            found_qcols[key] = col
    missing_q = [k for k in ["Q2_1","Q2_2","Q2_3","Q3_1","Q3_2","Q3_3","Q4"] if k not in found_qcols]
    if missing_q:
        raise ValueError(f"Missing required question columns: {missing_q}")

    for k, col in found_qcols.items():
        df[k] = df[col].apply(likert_to_int)

    df = df.dropna(subset=["Q2_1","Q2_2","Q2_3","Q3_1","Q3_2","Q3_3","Q4"]).copy()

    # Normalize rater email
    if col_rater_email:
        df[col_rater_email] = df[col_rater_email].astype(str).str.strip().str.lower()

    # Weighted score
    df["score"] = (
        0.1 * (df["Q2_1"] + df["Q2_2"] + df["Q2_3"] + df["Q3_1"] + df["Q3_2"] + df["Q3_3"])
        + 0.4 * df["Q4"]
    )

    # Filter by week argument
    target_week = args.week.strip()
    if target_week != "ALL":
        df = df[df["week_id"] == target_week].copy()

    if df.empty:
        print("No rows to process after filtering. (Check week range / deadlines / timestamps).")
        return

    # Deduplicate: same rater grading same presenter in same week -> keep latest
    if col_rater_email:
        df = df.sort_values("_ts_utc")
        df = df.drop_duplicates(subset=["week_id", col_presenter, col_rater_email], keep="last").copy()

    # Weekly summary per (week, presenter)
    agg = df.groupby(["week_id", col_presenter]).agg(
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

    # Save per-week outputs + mail drafts + peer logs
    for week_id, dfw in agg.groupby("week_id"):
        dfw = dfw.sort_values(["mean_score", "n_raters"], ascending=[False, False])
        dfw.to_csv(out_dir / f"weekly_summary_{week_id}.csv", index=False)

        # Peer log (deduped)
        dflog = df[df["week_id"] == week_id].rename(columns={col_presenter: "PresenterChoice"}).copy()
        dflog["_ts_local"] = dflog["_ts_local"].astype(str)

        keep_cols = ["week_id", "_ts_local", "PresenterChoice", "score"]
        if col_response_id and col_response_id in dflog.columns:
            keep_cols.insert(1, col_response_id)
        if col_rater_email:
            keep_cols.insert(2, col_rater_email)
        if col_rater_name:
            keep_cols.insert(2, col_rater_name)
        if col_comment:
            keep_cols.append(col_comment)

        dflog = dflog[keep_cols].copy()
        dflog.to_csv(out_dir / f"peer_log_{week_id}.csv", index=False)

        # Mail drafts
        mails_dir = out_dir / "mails" / week_id
        mails_dir.mkdir(parents=True, exist_ok=True)

        for presenter, dfp in df[df["week_id"] == week_id].rename(columns={col_presenter: "PresenterChoice"}).groupby("PresenterChoice"):
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

    # Master edges (week-by-week birikir)
    edges_cols = {
        "week_id": df["week_id"],
        "PresenterChoice": df[col_presenter].astype(str),
        "score": df["score"],
        "ts_utc": df["_ts_utc"].astype(str),
        "ts_local": df["_ts_local"].astype(str),
    }
    if col_response_id:
        edges_cols["ResponseId"] = df[col_response_id].astype(str)
    if col_rater_email:
        edges_cols["RaterEmail"] = df[col_rater_email].astype(str).str.lower()
    if col_rater_name:
        edges_cols["RaterName"] = df[col_rater_name].astype(str)
    if col_comment:
        edges_cols["Comment"] = df[col_comment].fillna("").astype(str)

    new_edges = pd.DataFrame(edges_cols)

    edges_path = master_dir / "grade_edges.csv"
    if edges_path.exists():
        old = pd.read_csv(edges_path)
        all_edges = pd.concat([old, new_edges], ignore_index=True)
    else:
        all_edges = new_edges

    # Dedup master by ResponseId if exists, else by (week,rater,presenter,ts_utc)
    if "ResponseId" in all_edges.columns:
        all_edges = all_edges.drop_duplicates(subset=["ResponseId"], keep="last")
    else:
        key = ["week_id", "RaterEmail", "PresenterChoice", "ts_utc"]
        key = [k for k in key if k in all_edges.columns]
        all_edges = all_edges.drop_duplicates(subset=key, keep="last")

    all_edges.to_csv(edges_path, index=False)

    # Matrices across all weeks
    if "RaterEmail" in all_edges.columns:
        mat_count = all_edges.pivot_table(
            index="RaterEmail", columns="PresenterChoice", values="score", aggfunc="count", fill_value=0
        )
        mat_mean = all_edges.pivot_table(
            index="RaterEmail", columns="PresenterChoice", values="score", aggfunc="mean"
        )
        mat_count.to_csv(master_dir / "grade_matrix_count.csv")
        mat_mean.to_csv(master_dir / "grade_matrix_mean.csv")

    # Attendance per week
    students = load_students(students_path)
    if students is not None and "RaterEmail" in all_edges.columns:
        for week_id in sorted(all_edges["week_id"].dropna().unique().tolist()):
            ew = all_edges[all_edges["week_id"] == week_id].copy()
            per = ew.groupby("RaterEmail").agg(
                n_submissions=("score", "count"),
                last_submission=("ts_local", "max"),
            ).reset_index().rename(columns={"RaterEmail": "StudentEmail"})

            att = students.merge(per, on="StudentEmail", how="left")
            att["n_submissions"] = att["n_submissions"].fillna(0).astype(int)
            att["submitted"] = (att["n_submissions"] > 0).astype(int)
            att["last_submission"] = att["last_submission"].fillna("")
            att = att.sort_values(["submitted", "n_submissions"], ascending=[True, False])

            att.to_csv(out_dir / f"attendance_{week_id}.csv", index=False)

        # Master attendance snapshot (latest)
        # (istersen burada ayrıca toplam submit sayısı vs da ekleyebiliriz)
    else:
        print("students.csv missing or RaterEmail missing -> attendance outputs skipped.")

    # Master summary
    agg.to_csv(out_dir / "weekly_summary_ALL.csv", index=False)

    print(f"Done. Outputs written under: {out_dir}")


if __name__ == "__main__":
    main()
