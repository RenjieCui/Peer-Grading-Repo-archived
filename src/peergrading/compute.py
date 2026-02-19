from __future__ import annotations

import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass

from .utils import parse_likert_value

@dataclass
class Rubric:
    weights: dict[str, float]
    label_to_value: dict[str, int]
    scale_min: int = 1
    scale_max: int = 5

def build_rubric(rubric_cfg: dict) -> Rubric:
    scale = rubric_cfg.get("scale", {})
    return Rubric(
        weights=rubric_cfg["weights"],
        label_to_value=scale.get("label_to_value", {}),
        scale_min=int(scale.get("min", 1)),
        scale_max=int(scale.get("max", 5)),
    )

def compute_weighted_score(df: pd.DataFrame, rubric: Rubric) -> pd.Series:
    # Parse each rubric column -> numeric
    score = pd.Series(0.0, index=df.index)
    weight_sum = 0.0
    for key, w in rubric.weights.items():
        if key not in df.columns:
            raise ValueError(f"Missing rubric field '{key}' in cleaned dataframe.")
        vals = df[key].apply(lambda x: parse_likert_value(x, rubric.label_to_value))
        df[key + "_num"] = vals
        score = score + vals.astype(float) * float(w)
        weight_sum += float(w)
    # If any required item missing -> NaN score
    required_num_cols = [k + "_num" for k in rubric.weights.keys()]
    any_missing = df[required_num_cols].isna().any(axis=1)
    score = score.where(~any_missing, other=pd.NA)
    # normalize if weights not summing to 1 (just in case)
    if abs(weight_sum - 1.0) > 1e-9:
        score = score / weight_sum
    return score

def parse_submitted_at(series: pd.Series, tz_name: str) -> pd.Series:
    tz = ZoneInfo(tz_name)
    dt = pd.to_datetime(series, errors="coerce")
    # If tz-naive, localize; if tz-aware, convert
    def _fix(x):
        if pd.isna(x):
            return pd.NaT
        if getattr(x, "tzinfo", None) is None:
            return x.replace(tzinfo=tz)
        return x.astimezone(tz)
    return dt.apply(_fix)

def dedupe_latest_before_deadline(df: pd.DataFrame, deadline: datetime) -> pd.DataFrame:
    # Keep latest submission per (rater_email, presenter_resolved, week_id) before deadline
    df = df.copy()
    df["late_flag"] = df["submitted_at"].apply(lambda x: (pd.notna(x) and x > deadline))
    df = df[(df["submitted_at"].isna()) | (df["submitted_at"] <= deadline)].copy()
    df = df.sort_values(["week_id", "presenter_email_resolved", "rater_email", "submitted_at"], na_position="last")
    df["dup_rank"] = df.groupby(["week_id", "presenter_email_resolved", "rater_email"]).cumcount()
    # Keep last row per group (largest rank)
    idx = df.groupby(["week_id", "presenter_email_resolved", "rater_email"])["dup_rank"].idxmax()
    return df.loc[idx].drop(columns=["dup_rank"])
