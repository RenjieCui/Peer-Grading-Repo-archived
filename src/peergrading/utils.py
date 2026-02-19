from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

_DIGIT_RE = re.compile(r"([1-5])")

def parse_likert_value(x, label_to_value: dict[str, int] | None = None) -> Optional[int]:
    """Parse a Forms Likert cell into an int in [1..5]."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        # allow 1..5
        xi = int(x)
        if 1 <= xi <= 5:
            return xi
        return None
    s = str(x).strip()
    if not s:
        return None
    if label_to_value and s in label_to_value:
        return int(label_to_value[s])
    m = _DIGIT_RE.search(s)
    if m:
        return int(m.group(1))
    return None

def compute_deadline_for_week(week_id: str, tz_name: str, hour: int = 12, minute: int = 0) -> datetime:
    """Compute Saturday 12:00 local time for ISO week (YYYY-Www)."""
    # week_id like 2026-W09
    year_str, week_str = week_id.split("-W")
    year, week = int(year_str), int(week_str)
    # ISO weekday: Monday=1 ... Saturday=6
    dt = datetime.fromisocalendar(year, week, 6).replace(hour=hour, minute=minute, second=0, microsecond=0)
    return dt.replace(tzinfo=ZoneInfo(tz_name))

def normalize_week_id(week_id: str) -> str:
    # Accept variants like "2026W9" etc. Keep strict output.
    week_id = week_id.strip()
    if "-W" in week_id:
        y, w = week_id.split("-W")
        return f"{int(y):04d}-W{int(w):02d}"
    if "W" in week_id:
        y, w = week_id.split("W")
        return f"{int(y):04d}-W{int(w):02d}"
    raise ValueError(f"Invalid week_id format: {week_id}")
