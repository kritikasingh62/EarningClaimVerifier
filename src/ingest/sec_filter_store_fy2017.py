"""
Filter cached SEC companyfacts JSON locally and store only the needed facts in SQLite.

Goal (FY2017 demo):
- Keep FY2016 + FY2017 quarter facts for a small whitelist of metrics.
- Avoid mixing YTD vs quarterly values.
- Derive Q4 for additive metrics using: FY (10-K) - 9M (Q3 10-Q).
- Support segment cherry-pick for Apple via Services revenue when available.

Inputs:
  data/raw/sec/companyfacts/{TICKER}.json (fetched by sec_fetch_companyfacts.py)

Outputs:
  SQLite table: financial_facts_filtered
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Config / Whitelist
# -----------------------------

# Metric -> candidate XBRL tags (first existing is used)
METRICS = {
    "revenue_total": [
        "us-gaap:SalesRevenueNet",
        "us-gaap:Revenues",
    ],
    "revenue_services": [
        "us-gaap:SalesRevenueServicesNet",
        "us-gaap:SalesRevenueServicesGross",
    ],
    "net_income": ["us-gaap:NetIncomeLoss"],
    "gross_profit": ["us-gaap:GrossProfit"],
    "cost_of_revenue": ["us-gaap:CostOfRevenue"],
    "eps_diluted": ["us-gaap:EarningsPerShareDiluted"],
}


ALLOWED_FORMS_Q = {"10-Q", "10-Q/A"}          # quarter filings
ALLOWED_FORMS_FY = {"10-K", "10-K/A"}         # annual filings

TARGET_YEARS = {2015, 2016, 2017}  # include 2015 for YoY growth verification

RAW_COMPANYFACTS_DIR = Path("data/raw/sec/companyfacts")
DEFAULT_SQLITE_PATH = Path("db/kip.sqlite")


# -----------------------------
# Data model
# -----------------------------

@dataclass
class FactRow:
    metric: str
    source_tag: str
    ticker: str
    fiscal_year: int
    fp: str                   # Q1/Q2/Q3/FY/Q4_DERIVED
    form: str                 # 10-Q / 10-K / etc
    filed: str
    period_start: Optional[str]
    period_end: str
    value: float
    unit: str
    is_derived: int = 0
    derived_note: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def days_between(start: str, end: str) -> int:
    return (parse_date(end) - parse_date(start)).days

def is_quarter_duration(start: Optional[str], end: str) -> bool:
    if not start or not end:
        return False
    d = days_between(start, end)
    return 75 <= d <= 120

def is_nine_month_duration(start: Optional[str], end: str) -> bool:
    if not start or not end:
        return False
    d = days_between(start, end)
    return 240 <= d <= 320

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def split_tag(tag: str) -> Tuple[str, str]:
    taxonomy, concept = tag.split(":", 1)
    return taxonomy, concept

def get_fact_items(companyfacts: Dict[str, Any], tag: str) -> List[Dict[str, Any]]:
    """
    Returns a flat list of observation dicts across all units.
    """
    taxonomy, concept = split_tag(tag)
    node = companyfacts.get("facts", {}).get(taxonomy, {}).get(concept)
    if not node:
        return []
    units = node.get("units", {})
    items: List[Dict[str, Any]] = []
    for unit_name, arr in units.items():
        for it in arr:
            it2 = dict(it)
            it2["_unit"] = unit_name
            items.append(it2)
    return items

def pick_tag_that_yields_quarters(
    companyfacts: dict,
    ticker: str,
    metric: str,
    candidates: list[str],
) -> str | None:
    """
    FIX:
    Instead of choosing the first tag that merely 'exists' in the JSON,
    choose the first tag that actually produces usable quarterly (10-Q) rows.

    Why:
    Some tags exist in companyfacts but only as yearly totals or weird frames.
    That caused revenue_total to never get inserted into SQLite.
    """
    for tag in candidates:
        try:
            rows = extract_q1_q3_quarters(companyfacts, ticker, metric, tag)
            if rows:
                return tag
        except Exception:
            # If a tag has unexpected structure, ignore and try next
            continue
    return None


def dedupe_keep_latest_by_end(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For duplicate period_end values, keep the one with the latest 'filed'.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for it in items:
        end = it.get("end")
        filed = it.get("filed") or ""
        if not end:
            continue
        if end not in best:
            best[end] = it
        else:
            if filed and (filed > (best[end].get("filed") or "")):
                best[end] = it
    return list(best.values())


# -----------------------------
# Extraction logic
# -----------------------------

DURATION_METRICS = {"revenue_total", "revenue_services", "net_income", "cost_of_revenue", "gross_profit"}
POINT_METRICS = {"eps_diluted"}  # EPS behaves more like a point/rate; do not derive Q4.

def extract_q1_q3_quarters(
    companyfacts: Dict[str, Any],
    ticker: str,
    metric: str,
    tag: str
) -> List[FactRow]:
    """
    Extract quarter-only Q1-Q3 facts from 10-Q filings for FY2016 & FY2017.
    Avoid YTD mixing by requiring ~quarter duration for duration metrics.
    """
    items = get_fact_items(companyfacts, tag)

    # Only 10-Q forms, FY in {2016, 2017}, and fp in {Q1, Q2, Q3}
    items = [
        it for it in items
        if (it.get("form") in ALLOWED_FORMS_Q)
        and (it.get("fy") in TARGET_YEARS)
        and (it.get("fp") in {"Q1", "Q2", "Q3"})
        and (it.get("end") is not None)
        and (it.get("val") is not None)
    ]

    if metric in DURATION_METRICS:
        items = [
            it for it in items
            if is_quarter_duration(it.get("start"), it.get("end"))
        ]
    # SEC context can include prior-year comparatives (fy=2016,fp=Q2 but end=2015-03-28).
    # Keep only rows where period_end matches the fiscal quarter.
    # Q2/Q3 end in Mar/Jun of fy; Q1 ends Dec of fy-1.
    def period_end_matches_quarter(it) -> bool:
        end = it.get("end") or ""
        fp = it.get("fp", "")
        fy = int(it.get("fy", 0))
        if not end or len(end) < 4:
            return True
        end_year = int(end[:4])
        if fp == "Q1":
            return end_year == fy - 1 or end_year == fy  # Dec of prior year
        if fp in {"Q2", "Q3"}:
            return end_year == fy  # Mar, Jun of fy
        return True
    items = [it for it in items if period_end_matches_quarter(it)]
    # EPS: skip duration filter

    items = dedupe_keep_latest_by_end(items)

    out: List[FactRow] = []
    for it in items:
        out.append(FactRow(
            metric=metric,
            source_tag=tag,
            ticker=ticker,
            fiscal_year=int(it["fy"]),
            fp=str(it["fp"]),
            form=str(it["form"]),
            filed=str(it.get("filed") or ""),
            period_start=it.get("start"),
            period_end=str(it["end"]),
            value=float(it["val"]),
            unit=str(it.get("_unit") or ""),
            is_derived=0,
            derived_note=None,
        ))
    return out


def extract_fy_totals(
    companyfacts: Dict[str, Any],
    ticker: str,
    metric: str,
    tag: str
) -> List[Dict[str, Any]]:
    """
    Extract FY totals for duration metrics from 10-K for FY2016 & FY2017.
    Returned as raw items for derivation logic.
    """
    items = get_fact_items(companyfacts, tag)
    items = [
        it for it in items
        if (it.get("form") in ALLOWED_FORMS_FY)
        and (it.get("fy") in TARGET_YEARS)
        and (it.get("fp") == "FY")
        and (it.get("end") is not None)
        and (it.get("val") is not None)
    ]
    return dedupe_keep_latest_by_end(items)


def extract_9m_totals_from_q3(
    companyfacts: Dict[str, Any],
    ticker: str,
    metric: str,
    tag: str
) -> List[Dict[str, Any]]:
    """
    Extract 9-month YTD totals from Q3 10-Q for duration metrics.
    Used only to derive Q4 as: FY - 9M
    """
    items = get_fact_items(companyfacts, tag)
    items = [
        it for it in items
        if (it.get("form") in ALLOWED_FORMS_Q)
        and (it.get("fy") in TARGET_YEARS)
        and (it.get("fp") == "Q3")
        and (it.get("end") is not None)
        and (it.get("val") is not None)
    ]
    # specifically 9-month duration
    items = [
        it for it in items
        if is_nine_month_duration(it.get("start"), it.get("end"))
    ]
    return dedupe_keep_latest_by_end(items)


def derive_q4_for_duration_metric(
    ticker: str,
    metric: str,
    tag: str,
    fy_items: List[Dict[str, Any]],
    nine_m_items: List[Dict[str, Any]],
) -> List[FactRow]:
    """
    Derive Q4 rows for FY2016 and FY2017:
      Q4 = FY_total - NineMonth_total
    Stores as fp="Q4_DERIVED"
    """
    fy_by_year = {int(it["fy"]): it for it in fy_items}
    nm_by_year = {int(it["fy"]): it for it in nine_m_items}

    out: List[FactRow] = []
    for year in sorted(TARGET_YEARS):
        if year not in fy_by_year or year not in nm_by_year:
            continue
        fy = fy_by_year[year]
        nm = nm_by_year[year]

        q4_val = float(fy["val"]) - float(nm["val"])
        q4_end = str(fy["end"])
        q4_start = str(nm["end"])  # end of Q3 is boundary for derived Q4

        note = f"Derived Q4 = FY({fy.get('accn','')}) - 9M({nm.get('accn','')})"

        out.append(FactRow(
            metric=metric,
            source_tag=tag,
            ticker=ticker,
            fiscal_year=year,
            fp="Q4_DERIVED",
            form=str(fy.get("form", "10-K")),
            filed=str(fy.get("filed") or ""),
            period_start=q4_start,
            period_end=q4_end,
            value=q4_val,
            unit=str(fy.get("_unit") or nm.get("_unit") or ""),
            is_derived=1,
            derived_note=note,
        ))
    return out


def build_filtered_rows_for_ticker(companyfacts: dict, ticker: str) -> list:
    rows = []

    for metric, candidates in METRICS.items():
        tag = pick_tag_that_yields_quarters(companyfacts, ticker, metric, candidates)
        if not tag:
            continue

        # 1) Extract Q1-Q3 quarterly points from 10-Q
        q_rows = extract_q1_q3_quarters(companyfacts, ticker, metric, tag)
        rows.extend(q_rows)

        # 2) If this is a duration metric (revenue, net income, gross profit etc),
        # derive Q4 = FY - 9M
        if metric in DURATION_METRICS:
            fy_items = extract_fy_totals(companyfacts, ticker, metric, tag)
            nm_items = extract_9m_totals_from_q3(companyfacts, ticker, metric, tag)
            rows.extend(derive_q4_for_duration_metric(ticker, metric, tag, fy_items, nm_items))

    return rows

def derive_and_insert_gross_margin(conn, ticker: str) -> int:
    """
    Insert derived gross_margin (%) per (fiscal_year, fp) for a ticker.

    gross_margin = (gross_profit / revenue_total) * 100

    Returns: number of rows inserted.
    """
    ticker = ticker.upper()

    # Pull matching quarter points for revenue_total and gross_profit
    sql = """
    SELECT
        r.ticker,
        r.fiscal_year,
        r.fp,
        r.period_end,
        r.form,
        r.filed,
        r.value AS revenue_total,
        g.value AS gross_profit
    FROM financial_facts_filtered r
    JOIN financial_facts_filtered g
      ON r.ticker = g.ticker
     AND r.fiscal_year = g.fiscal_year
     AND r.fp = g.fp
    WHERE r.ticker = ?
      AND r.metric = 'revenue_total'
      AND g.metric = 'gross_profit'
      AND r.unit = 'USD'
      AND g.unit = 'USD'
      AND r.value IS NOT NULL
      AND g.value IS NOT NULL
      AND r.value != 0
    """
    rows = conn.execute(sql, (ticker,)).fetchall()

    inserted = 0

    for row in rows:
        fiscal_year = int(row["fiscal_year"])
        fp = row["fp"]
        period_end = row["period_end"]
        form = row["form"]
        filed = row["filed"]

        revenue_total = float(row["revenue_total"])
        gross_profit = float(row["gross_profit"])

        gross_margin = (gross_profit / revenue_total) * 100.0

        # Upsert behavior: delete existing derived gross_margin row for same quarter
        conn.execute(
            """
            DELETE FROM financial_facts_filtered
            WHERE ticker = ?
              AND metric = 'gross_margin'
              AND fiscal_year = ?
              AND fp = ?
            """,
            (ticker, fiscal_year, fp),
        )

        # Insert new derived row
        conn.execute(
            """
            INSERT INTO financial_facts_filtered
            (ticker, metric, fiscal_year, fp, period_end, value, unit, source_tag, form, filed, is_derived, derived_note)
            VALUES
            (?, 'gross_margin', ?, ?, ?, ?, 'percent', 'derived:gross_profit/revenue_total', ?, ?, 1,
             'gross_margin = gross_profit / revenue_total * 100')
            """,
            (ticker, fiscal_year, fp, period_end, gross_margin, form, filed),
        )

        inserted += 1

    conn.commit()
    return inserted


# -----------------------------
# SQLite storage
# -----------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS financial_facts_filtered (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker TEXT NOT NULL,
  metric TEXT NOT NULL,
  source_tag TEXT NOT NULL,
  fiscal_year INTEGER NOT NULL,
  fp TEXT NOT NULL,
  form TEXT,
  filed TEXT,
  period_start TEXT,
  period_end TEXT NOT NULL,
  value REAL NOT NULL,
  unit TEXT,
  is_derived INTEGER DEFAULT 0,
  derived_note TEXT,
  UNIQUE(ticker, metric, fiscal_year, fp, period_end, source_tag)
);

CREATE INDEX IF NOT EXISTS idx_facts_ticker_metric_year_fp
ON financial_facts_filtered(ticker, metric, fiscal_year, fp);
"""

def connect_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()

def upsert_rows(conn: sqlite3.Connection, rows: List[FactRow]) -> None:
    sql = """
    INSERT OR REPLACE INTO financial_facts_filtered
    (ticker, metric, source_tag, fiscal_year, fp, form, filed, period_start, period_end, value, unit, is_derived, derived_note)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    conn.executemany(sql, [
        (r.ticker, r.metric, r.source_tag, r.fiscal_year, r.fp, r.form, r.filed,
         r.period_start, r.period_end, r.value, r.unit, r.is_derived, r.derived_note)
        for r in rows
    ])
    conn.commit()


# -----------------------------
# CLI
# -----------------------------

def load_tickers_from_config(config_path: str) -> List[str]:
    obj = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return [str(t).upper().strip() for t in obj.get("tickers", [])]

def main():
    parser = argparse.ArgumentParser(description="Filter cached SEC companyfacts into SQLite (FY2016+FY2017).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", help="Ticker symbol (e.g., AAPL)")
    group.add_argument("--config", help="Path to config/tickers.json")

    parser.add_argument("--raw-dir", default=str(RAW_COMPANYFACTS_DIR), help="Directory containing cached companyfacts JSONs.")
    parser.add_argument("--sqlite", default=str(DEFAULT_SQLITE_PATH), help="SQLite DB path.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    sqlite_path = Path(args.sqlite)

    conn = connect_sqlite(sqlite_path)
    init_db(conn)

    tickers = [args.ticker.upper()] if args.ticker else load_tickers_from_config(args.config)

    for t in tickers:
        path = raw_dir / f"{t}.json"
        if not path.exists():
            print(f"[{t}] Missing cached companyfacts at {path}. Fetch it first.")
            continue

        companyfacts = load_json(path)
        filtered = build_filtered_rows_for_ticker(companyfacts, t)
        upsert_rows(conn, filtered)
        n = derive_and_insert_gross_margin(conn, t)

        print(f"[{t}] stored {len(filtered)} base rows")
        print(f"[{t}] derived {n} gross_margin rows")

    conn.close()

if __name__ == "__main__":
    main()
