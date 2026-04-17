# src/transcripts/transcript_mapping.py
"""
Apple transcript -> fiscal period mapping.

Supports three approaches (in priority order):
1) Parse explicit "Qx YYYY" from transcript text if present.
2) Parse call date from transcript text and map it to Apple fiscal quarter.
3) (Fallback) Parse call date from filename like: 2016-Apr-26-AAPL.txt

Apple fiscal calendar (practical mapping by call month):
- Late Jan/Feb calls usually correspond to Q1
- Late Apr/May calls usually correspond to Q2
- Late Jul/Aug calls usually correspond to Q3
- Late Oct/Nov calls usually correspond to Q4 (and FY)

This is "good enough" for a demo and can be refined later using SEC period_end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple


MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}


@dataclass(frozen=True)
class TranscriptMeta:
    fiscal_year: int
    fp: str                      # Q1/Q2/Q3/Q4
    call_date: Optional[str] = None
    confidence: str = "medium"   # high if explicit quarter found; medium if date-based; low if unknown


def parse_quarter_from_text(text: str) -> Optional[Tuple[int, str]]:
    """
    Looks for patterns like:
      "Q2 2016" or "Q3 FY2017" or "FQ4 2016"
    Returns (fiscal_year, fp)
    """
    patterns = [
        r"\bQ([1-4])\s*(?:FY)?\s*(20\d{2})\b",
        r"\bFQ([1-4])\s*(20\d{2})\b",
        r"\bQ([1-4])\s*FY\s*(20\d{2})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            q = m.group(1)
            yr = int(m.group(2))
            return (yr, f"Q{q}")
    return None


def parse_call_date_from_text(text: str) -> Optional[str]:
    """
    Tries to parse a call date from transcript header, like:
      "APRIL 26, 2016"
      "OCTOBER 25, 2016"
      "Jul 26, 2016"
    Returns ISO date string YYYY-MM-DD if found.
    """
    # Full month name: APRIL 26, 2016
    m = re.search(
        r"\b(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(\d{1,2}),\s*(20\d{2})\b",
        text,
        re.IGNORECASE,
    )
    if m:
        month_name = m.group(1)[:3].upper()
        day = int(m.group(2))
        year = int(m.group(3))
        dt = datetime(year, MONTHS[month_name], day)
        return dt.strftime("%Y-%m-%d")

    # Abbrev month: Jul 26, 2016
    m2 = re.search(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),\s*(20\d{2})\b",
        text,
        re.IGNORECASE,
    )
    if m2:
        month = m2.group(1)[:3].upper()
        day = int(m2.group(2))
        year = int(m2.group(3))
        dt = datetime(year, MONTHS[month], day)
        return dt.strftime("%Y-%m-%d")

    return None


def parse_call_date_from_filename(filename: str) -> Optional[str]:
    """
    Fallback parse for filenames like:
      2016-Apr-26-AAPL.txt
      2017-Nov-02-AAPL.txt

    Returns ISO date string YYYY-MM-DD if found.
    """
    # strip directories + extension
    base = filename.split("/")[-1]
    if base.lower().endswith(".txt"):
        base = base[:-4]

    parts = base.split("-")
    if len(parts) < 3:
        return None

    year_str = parts[0]
    mon_str = parts[1][:3].upper()
    day_str = parts[2]

    if not year_str.isdigit() or not day_str.isdigit():
        return None
    if mon_str not in MONTHS:
        return None

    try:
        dt = datetime(int(year_str), MONTHS[mon_str], int(day_str))
    except ValueError:
        return None

    return dt.strftime("%Y-%m-%d")


def map_call_date_to_apple_fiscal(call_date_iso: str) -> Tuple[int, str]:
    """Legacy: Apple (sep FY). Use map_call_date_to_fiscal for configurable calendar."""
    return map_call_date_to_fiscal(call_date_iso, "sep")


def map_call_date_to_fiscal(call_date_iso: str, fiscal_year_end: str = "sep") -> Tuple[int, str]:
    """
    Map call date -> (fiscal_year, fp) by fiscal calendar.

    fiscal_year_end: "sep" (Apple) or "dec" (Google, Microsoft)
    """
    dt = datetime.strptime(call_date_iso, "%Y-%m-%d")
    m = dt.month
    y = dt.year

    if fiscal_year_end == "dec":
        # Dec FY: Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec
        # Jan/Feb call reports Q4 prior year; Apr/May reports Q1; etc.
        if m in (1, 2):
            return (y - 1, "Q4")
        if m in (4, 5):
            return (y, "Q1")
        if m in (7, 8):
            return (y, "Q2")
        if m in (10, 11):
            return (y, "Q3")
        if m == 3:
            return (y, "Q1")
        if m == 6:
            return (y, "Q2")
        if m == 9:
            return (y, "Q3")
        if m == 12:
            return (y, "Q4")
        return (y, "Q4")
    else:
        # Sep FY (Apple): Q1=Oct-Dec, Q2=Jan-Mar, Q3=Apr-Jun, Q4=Jul-Sep
        if m in (1, 2):
            return (y, "Q1")
        if m in (4, 5):
            return (y, "Q2")
        if m in (7, 8):
            return (y, "Q3")
        if m in (10, 11):
            return (y, "Q4")
        if m == 3:
            return (y, "Q1")
        if m == 6:
            return (y, "Q2")
        if m == 9:
            return (y, "Q4")
        if m == 12:
            return (y, "Q1")
        return (y, "Q4")


def detect_transcript_meta(
    text: str, filename: Optional[str] = None, fiscal_year_end: str = "sep"
) -> TranscriptMeta:
    """
    Returns fiscal_year + fp for an Apple transcript.

    Priority:
      1) explicit quarter in text -> confidence=high
      2) call date in text -> confidence=medium
      3) call date in filename -> confidence=medium
      4) unknown -> confidence=low
    """
    q = parse_quarter_from_text(text)
    if q:
        yr, fp = q
        call_date = parse_call_date_from_text(text) or (parse_call_date_from_filename(filename) if filename else None)
        return TranscriptMeta(fiscal_year=yr, fp=fp, call_date=call_date, confidence="high")

    call_date = parse_call_date_from_text(text)
    if call_date:
        yr, fp = map_call_date_to_fiscal(call_date, fiscal_year_end)
        return TranscriptMeta(fiscal_year=yr, fp=fp, call_date=call_date, confidence="medium")

    if filename:
        call_date2 = parse_call_date_from_filename(filename)
        if call_date2:
            yr, fp = map_call_date_to_fiscal(call_date2, fiscal_year_end)
            return TranscriptMeta(fiscal_year=yr, fp=fp, call_date=call_date2, confidence="medium")

    return TranscriptMeta(fiscal_year=-1, fp="UNKNOWN", call_date=None, confidence="low")
