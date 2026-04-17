"""
Filter SEC companyfacts JSON for Apple FY2017 (4 quarters) and compute result scores.

Reads: data/raw/sec/companyfacts/AAPL.json
Outputs: JSON with quarterly values and aggregate scores for:
  - eps_diluted
  - gross_margin (derived: gross_profit / revenue_total)
  - gross_profit
  - net_income
  - revenue_total
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------------
# Config
# -----------------------------

METRICS = {
    "revenue_total": ["us-gaap:SalesRevenueNet", "us-gaap:Revenues"],
    "net_income": ["us-gaap:NetIncomeLoss"],
    "gross_profit": ["us-gaap:GrossProfit"],
    "eps_diluted": ["us-gaap:EarningsPerShareDiluted"],
}

ALLOWED_FORMS_Q = {"10-Q", "10-Q/A"}
ALLOWED_FORMS_FY = {"10-K", "10-K/A"}
TARGET_YEAR = 2017

DURATION_METRICS = {"revenue_total", "net_income", "gross_profit"}
POINT_METRICS = {"eps_diluted"}

RAW_DIR = Path("data/raw/sec/companyfacts")
DEFAULT_OUTPUT = Path("data/raw/sec/fy2017_metric_scores.json")


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def days_between(start: str, end: str) -> int:
    return (parse_date(end) - parse_date(start)).days


def is_quarter_duration(start: Optional[str], end: str) -> bool:
    if not start or not end:
        return False
    return 75 <= days_between(start, end) <= 120


def is_nine_month_duration(start: Optional[str], end: str) -> bool:
    if not start or not end:
        return False
    return 240 <= days_between(start, end) <= 320


def get_fact_items(companyfacts: Dict[str, Any], tag: str) -> List[Dict[str, Any]]:
    taxonomy, concept = tag.split(":", 1)
    node = companyfacts.get("facts", {}).get(taxonomy, {}).get(concept)
    if not node:
        return []
    items: List[Dict[str, Any]] = []
    for unit_name, arr in node.get("units", {}).items():
        for it in arr:
            it2 = dict(it)
            it2["_unit"] = unit_name
            items.append(it2)
    return items


def dedupe_keep_latest(items: List[Dict], key_fn) -> List[Dict]:
    best: Dict[str, Dict] = {}
    for it in items:
        k = key_fn(it)
        if not k:
            continue
        if k not in best or (it.get("filed") or "") > (best[k].get("filed") or ""):
            best[k] = it
    return list(best.values())


def period_end_matches_quarter(it: Dict) -> bool:
    end = it.get("end") or ""
    fp = it.get("fp", "")
    fy = int(it.get("fy", 0))
    if not end or len(end) < 4:
        return True
    end_year = int(end[:4])
    if fp == "Q1":
        return end_year == fy - 1 or end_year == fy
    if fp in {"Q2", "Q3"}:
        return end_year == fy
    return True


def extract_quarterly_2017(
    companyfacts: Dict[str, Any],
    metric: str,
    tag: str,
) -> List[Dict[str, Any]]:
    """Extract Q1-Q3 for 2017 with period_end filter."""
    items = get_fact_items(companyfacts, tag)
    items = [
        it for it in items
        if it.get("form") in ALLOWED_FORMS_Q
        and it.get("fy") == TARGET_YEAR
        and it.get("fp") in {"Q1", "Q2", "Q3"}
        and it.get("end")
        and it.get("val") is not None
    ]
    if metric in DURATION_METRICS:
        items = [it for it in items if is_quarter_duration(it.get("start"), it.get("end"))]
    items = [it for it in items if period_end_matches_quarter(it)]
    items = dedupe_keep_latest(items, lambda x: x.get("end"))
    return items


def extract_fy_2017(companyfacts: Dict[str, Any], tag: str) -> Optional[Dict]:
    items = get_fact_items(companyfacts, tag)
    items = [
        it for it in items
        if it.get("form") in ALLOWED_FORMS_FY
        and it.get("fy") == TARGET_YEAR
        and it.get("fp") == "FY"
        and it.get("end")
        and it.get("val") is not None
    ]
    items = dedupe_keep_latest(items, lambda x: x.get("end"))
    return items[0] if items else None


def extract_9m_2017(
    companyfacts: Dict[str, Any],
    tag: str,
) -> Optional[Dict]:
    items = get_fact_items(companyfacts, tag)
    items = [
        it for it in items
        if it.get("form") in ALLOWED_FORMS_Q
        and it.get("fy") == TARGET_YEAR
        and it.get("fp") == "Q3"
        and it.get("end")
        and it.get("val") is not None
    ]
    items = [it for it in items if is_nine_month_duration(it.get("start"), it.get("end"))]
    items = dedupe_keep_latest(items, lambda x: x.get("end"))
    return items[0] if items else None


def find_tag(companyfacts: Dict, metric: str) -> Optional[str]:
    for tag in METRICS.get(metric, []):
        items = extract_quarterly_2017(companyfacts, metric, tag)
        if items:
            return tag
    return None


def compute_fy2017_scores(
    companyfacts_path: Path = RAW_DIR / "AAPL.json",
) -> Dict[str, Any]:
    """Filter for 4 quarters of 2017 and compute metric scores."""
    data = json.loads(companyfacts_path.read_text(encoding="utf-8"))
    result: Dict[str, Any] = {
        "ticker": "AAPL",
        "fiscal_year": TARGET_YEAR,
        "quarters": ["Q1", "Q2", "Q3", "Q4"],
        "metrics": {},
    }

    # Base metrics
    by_quarter: Dict[str, Dict[str, float]] = {q: {} for q in ["Q1", "Q2", "Q3", "Q4"]}

    for metric in ["revenue_total", "net_income", "gross_profit", "eps_diluted"]:
        tag = find_tag(data, metric)
        if not tag:
            result["metrics"][metric] = {"quarterly": {}, "score": None, "data_points": 0}
            continue

        q_rows = extract_quarterly_2017(data, metric, tag)
        values: List[float] = []
        for it in q_rows:
            fp = it.get("fp")
            val = float(it["val"])
            by_quarter[fp][metric] = val
            values.append(val)

        # Q4: derive for duration metrics
        if metric in DURATION_METRICS:
            fy = extract_fy_2017(data, tag)
            nm = extract_9m_2017(data, tag)
            if fy and nm:
                q4_val = float(fy["val"]) - float(nm["val"])
                by_quarter["Q4"][metric] = q4_val
                values.append(q4_val)

        quarterly = {q: by_quarter[q].get(metric) for q in ["Q1", "Q2", "Q3", "Q4"]}
        if metric == "eps_diluted":
            score = sum(values) / len(values) if values else None
        else:
            score = sum(values) if values else None

        result["metrics"][metric] = {
            "quarterly": quarterly,
            "score": round(score, 2) if score is not None else None,
            "data_points": len(values),
        }

    # gross_margin (derived)
    rev = by_quarter
    gm_quarterly: Dict[str, Optional[float]] = {}
    gm_values: List[float] = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        r = rev[q].get("revenue_total")
        g = rev[q].get("gross_profit")
        if r is not None and g is not None and r != 0:
            gm = (g / r) * 100.0
            gm_quarterly[q] = round(gm, 2)
            gm_values.append(gm)
        else:
            gm_quarterly[q] = None
    result["metrics"]["gross_margin"] = {
        "quarterly": gm_quarterly,
        "score": round(sum(gm_values) / len(gm_values), 2) if gm_values else None,
        "data_points": len(gm_values),
    }

    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Compute FY2017 metric scores from SEC AAPL.json")
    parser.add_argument("--input", default=str(RAW_DIR / "AAPL.json"), help="Path to AAPL companyfacts JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to output JSON")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Companyfacts not found: {path}")

    scores = compute_fy2017_scores(path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
