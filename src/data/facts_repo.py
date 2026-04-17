# src/data/facts_repo.py
"""
Facts query layer: SQLite -> Python

Why this exists:
- Keeps SQL in one place
- Makes validator code clean: get_value(), compute_yoy(), compute_qoq()
- Central place to enforce "which fp to use" (Q4_DERIVED etc.)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class FactPoint:
    ticker: str
    metric: str
    fiscal_year: int
    fp: str
    period_end: str
    value: float
    unit: str
    source_tag: str
    form: Optional[str] = None
    filed: Optional[str] = None
    is_derived: int = 0
    derived_note: Optional[str] = None


class FactsRepo:
    def __init__(self, sqlite_path: str = "db/kip.sqlite") -> None:
        self.sqlite_path = sqlite_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -----------------------------
    # Core fetch
    # -----------------------------
    def get_point(
        self,
        ticker: str,
        metric: str,
        fiscal_year: int,
        fp: str,
    ) -> Optional[FactPoint]:
        """
        Fetch a single fact row for (ticker, metric, fiscal_year, fp).

        Uses the fiscal_year column directly (SEC XBRL fy) for correct matching.
        For companies like Apple, Q2 FY2016 has period_end in Dec 2015, so matching
        on period_end year would incorrectly fail; fiscal_year is the authoritative FY.
        """
        sql = """
        SELECT *
        FROM financial_facts_filtered
        WHERE ticker = ?
            AND metric = ?
            AND fp = ?
            AND fiscal_year = ?
        ORDER BY period_end DESC, COALESCE(filed,'') DESC
        LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (ticker.upper(), metric, fp, fiscal_year)).fetchone()
            if not row:
                return None
            return FactPoint(
                ticker=row["ticker"],
                metric=row["metric"],
                fiscal_year=int(row["fiscal_year"]) if row["fiscal_year"] is not None else fiscal_year,
                fp=row["fp"],
                period_end=row["period_end"],
                value=float(row["value"]),
                unit=row["unit"] or "",
                source_tag=row["source_tag"] or "",
                form=row["form"],
                filed=row["filed"],
                is_derived=int(row["is_derived"] or 0),
                derived_note=row["derived_note"],
        )


    def get_value(self, ticker: str, metric: str, fiscal_year: int, fp: str) -> Optional[float]:
        pt = self.get_point(ticker, metric, fiscal_year, fp)
        return pt.value if pt else None

    # -----------------------------
    # Quarter helpers
    # -----------------------------
    @staticmethod
    def prev_fp(fp: str) -> Optional[Tuple[int, str]]:
        """
        Returns a (year_delta, previous_fp) mapping for QoQ comparison.
        Assumes Q4 is stored as 'Q4_DERIVED'.
        """
        fp = fp.upper()
        if fp == "Q1":
            return (-1, "Q4_DERIVED")
        if fp == "Q2":
            return (0, "Q1")
        if fp == "Q3":
            return (0, "Q2")
        if fp in {"Q4", "Q4_DERIVED"}:
            return (0, "Q3")
        return None

    # -----------------------------
    # Computations
    # -----------------------------
    def compute_yoy(self, ticker: str, metric: str, fiscal_year: int, fp: str) -> Optional[float]:
        """
        YoY% = (current - prior_year_same_q) / prior_year_same_q
        Returns a decimal (e.g., 0.15 = +15%).
        """
        cur = self.get_value(ticker, metric, fiscal_year, fp)
        prev = self.get_value(ticker, metric, fiscal_year - 1, fp)
        if cur is None or prev is None or prev == 0:
            return None
        return (cur - prev) / prev

    def compute_qoq(self, ticker: str, metric: str, fiscal_year: int, fp: str) -> Optional[float]:
        """
        QoQ% = (current - previous_quarter) / previous_quarter
        Returns a decimal.
        """
        cur = self.get_value(ticker, metric, fiscal_year, fp)
        prev_info = self.prev_fp(fp)
        if cur is None or prev_info is None:
            return None

        year_delta, prev_fp = prev_info
        prev = self.get_value(ticker, metric, fiscal_year + year_delta, prev_fp)

        if prev is None or prev == 0:
            return None
        return (cur - prev) / prev

    def compute_gross_margin(self, ticker: str, fiscal_year: int, fp: str) -> Optional[float]:
        """
        gross_margin = (revenue - cost_of_revenue) / revenue
        If cost_of_revenue is missing, tries gross_profit / revenue (if stored).
        Returns a decimal.
        """
        rev = self.get_value(ticker, "revenue_total", fiscal_year, fp)
        if rev is None or rev == 0:
            return None

        cor = self.get_value(ticker, "cost_of_revenue", fiscal_year, fp)
        if cor is not None:
            return (rev - cor) / rev

        gp = self.get_value(ticker, "gross_profit", fiscal_year, fp)
        if gp is not None:
            return gp / rev

        return None
