from __future__ import annotations

import pandas as pd
from pathlib import Path
import yaml

def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_students(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # normalize
    df.columns = [c.strip() for c in df.columns]
    required = {"student_id", "name", "email"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"students_db missing columns: {sorted(missing)}. Expected at least {sorted(required)}")
    df["email"] = df["email"].astype(str).str.strip().str.lower()
    df["name"] = df["name"].astype(str).str.strip()
    df["student_id"] = df["student_id"].astype(str).str.strip()
    return df

def load_week_setup(path: str | Path, week_id: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"week_setup file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = {"week_id", "presentation_id", "presenter_email"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"presentations.csv missing columns: {sorted(missing)}. Expected at least {sorted(required)}")
    df = df[df["week_id"].astype(str).str.strip() == week_id].copy()
    df["presenter_email"] = df["presenter_email"].astype(str).str.strip().str.lower()
    if "presenter_display" in df.columns:
        df["presenter_display"] = df["presenter_display"].astype(str).str.strip()
    else:
        df["presenter_display"] = ""
    if "presenter_student_id" in df.columns:
        df["presenter_student_id"] = df["presenter_student_id"].astype(str).str.strip()
    else:
        df["presenter_student_id"] = ""
    if "topic" not in df.columns:
        df["topic"] = ""
    return df

def load_forms_export(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Forms export not found: {path}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df
