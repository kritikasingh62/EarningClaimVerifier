"""verify_claims_from_json.py

Verify transcript claims against SEC XBRL facts stored in a merged JSON.

Outputs verdicts:
- VERIFIED: claim matches SEC within tolerance
- INACCURATE: claim differs beyond tolerance
- UNVERIFIABLE: cannot map/compute/compare

Key features:
- Robust SEC point selection when multiple facts share the same period end
  (prefers quarter-duration for quarterly claims, YTD/annual only when claim language indicates).
- Margin support (gross / operating / net) via derived computation.
- Expanded derived metrics (supports subtraction with a leading '-').
- Approximation-language detection ("about", "roughly", "over", etc.) widens tolerance.
- Period matching fallbacks (exact end → same month → closest end within window → guarded annual fallback).
- Stronger claim normalization (B/M/K, $ commas, percent, per-share cues).

Designed for Apple FY2017 examples but generally applicable with a quarter→end mapping.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Config (edit if you change company/year)
# -----------------------------

def _quarter_dates(fiscal_year_end: str, fiscal_year: int) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (QUARTER_TO_END_DATE, PRIOR_YEAR_END_DATE) for given fiscal calendar."""
    if fiscal_year_end == "dec":
        cur = {
            "Q1": f"{fiscal_year}-03-31",
            "Q2": f"{fiscal_year}-06-30",
            "Q3": f"{fiscal_year}-09-30",
            "Q4": f"{fiscal_year}-12-31",
        }
        prior = {
            "Q1": f"{fiscal_year - 1}-03-31",
            "Q2": f"{fiscal_year - 1}-06-30",
            "Q3": f"{fiscal_year - 1}-09-30",
            "Q4": f"{fiscal_year - 1}-12-31",
        }
    else:
        cur = {
            "Q1": f"{fiscal_year - 1}-12-31",
            "Q2": f"{fiscal_year}-04-01",
            "Q3": f"{fiscal_year}-07-01",
            "Q4": f"{fiscal_year}-09-30",
        }
        prior = {
            "Q1": f"{fiscal_year - 2}-12-26",
            "Q2": f"{fiscal_year - 1}-04-01",
            "Q3": f"{fiscal_year - 1}-07-01",
            "Q4": f"{fiscal_year - 1}-09-24",
        }
    return cur, prior


# Quarter -> period end date (Apple FY2017, default)
QUARTER_TO_END_DATE: Dict[str, str] = {
    "Q1": "2016-12-31",
    "Q2": "2017-04-01",
    "Q3": "2017-07-01",
    "Q4": "2017-09-30",
}

# Prior year same fiscal quarter end (for YoY growth)
PRIOR_YEAR_END_DATE: Dict[str, str] = {
    "Q1": "2015-12-26",  # might be missing depending on how you filtered SEC data
    "Q2": "2016-04-01",
    "Q3": "2016-07-01",
    "Q4": "2016-09-24",
}

# Tolerances (company-agnostic; allow for rounding, FX, constant-currency differences)
TOL_ABS_PCT = 0.05       # 5% relative tolerance for absolute values
TOL_GROWTH_PP = 4.0      # 4 percentage points for growth
TOL_EPS = 0.15           # $0.15 for EPS (share class, rounding)
TOL_PCT_MARGIN = 1.5     # 1.5 pp for margin %

# Metric equivalence: when primary gives material delta (|delta_pct| > 25%), try alternates
# Format: primary_metric -> list of alternates (same economic concept, different SEC tags)
# Revenue: consolidated vs segment vs goods - try all, pick best match
METRIC_ALIASES: Dict[str, List[str]] = {
    "us-gaap:SalesRevenueNet": [
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:SalesRevenueGoodsNet",
    ],
    "us-gaap:Revenues": [
        "us-gaap:SalesRevenueNet",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:SalesRevenueGoodsNet",
    ],
    "us-gaap:SalesRevenueServicesNet": [
        "us-gaap:SalesRevenueNet",
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueGoodsNet",
    ],
    "us-gaap:SalesRevenueGoodsNet": ["us-gaap:SalesRevenueNet", "us-gaap:Revenues"],
    "us-gaap:CostOfRevenue": ["us-gaap:CostOfGoodsAndServicesSold"],
    "us-gaap:CostOfGoodsAndServicesSold": ["us-gaap:CostOfRevenue"],
}

DEFAULT_CLAIMS_PATH = Path("data/claims_out/transcript_claims_by_sec_metric_2017_new.json")
# Default SEC sources. Prefer per-filing snapshots (10-Q/10-K) that include 2016/2017 comparatives.
# These help compute true quarterly flows (YTD differencing) and Q4 (FY - Q3YTD).
DEFAULT_SEC_JSON_PATHS: List[Path] = [
    Path("data/filtered/2017-Jan-31-AAPL_10K_10Q_ENDYEAR_2016_2017.json"),
    Path("data/filtered/2017-May-02-AAPL_10K_10Q_ENDYEAR_2016_2017.json"),
    Path("data/filtered/2017-Aug-01-AAPL_10K_10Q_ENDYEAR_2016_2017.json"),
    Path("data/filtered/2017-Nov-02-AAPL_10K_10Q_ENDYEAR_2016_2017.json"),
]
# Fallback: merged SEC JSON (older approach)
DEFAULT_SEC_JSON_PATH = Path("data/filtered/AAPL_10K_10Q_merged_filtered_endyear.json")
DEFAULT_OUTPUT_PATH = Path("data/claims_out/claim_verification_results.json")


# -----------------------------
# Derived metrics
# -----------------------------

# Derived metric name -> list of component metrics.
# A component may be prefixed with '-' to subtract.
DERIVED_METRICS: Dict[str, List[str]] = {
    "derived:CashPlusMarketableSecurities": [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:AvailableForSaleSecuritiesCurrent",
        "us-gaap:AvailableForSaleSecuritiesNoncurrent",
    ],
    "derived:FreeCashFlow": [
        "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        "-us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    ],
    "derived:TotalDebt": [
        "us-gaap:LongTermDebtCurrent",
        "us-gaap:LongTermDebtNoncurrent",
        "us-gaap:DebtCurrent",
    ],
}


# -----------------------------
# Regex helpers
# -----------------------------

_REPURCHASE_PARTIAL = re.compile(r"\b\d+(?:\.\d+)?\s*(?:m|million|mm)\s*(?:shares?|AAPL|co\.)", re.I)
_REPURCHASE_PROGRAM = re.compile(
    r"authorization|capital return program|completed \$\d+\.?\d*b\s+of\s+\$\d+|\$\d+\.?\d*b\s+in\s+share\s+repurchase",
    re.I,
)

_APPROX_WORDS = re.compile(r"\b(about|approximately|roughly|over|nearly|around|close to|just over|just under)\b", re.I)
_ANNUAL_CUES = re.compile(r"\b(full\s+year|fiscal\s+year|annual|year\s+ended|for\s+the\s+year)\b", re.I)
_YTD_CUES = re.compile(r"\b(ytd|year\s*to\s*date|first\s+(?:six|9|nine)\s+months|six\s+months|nine\s+months)\b", re.I)

_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")
_SCALE_B = re.compile(r"\b(b|bn|billion)\b", re.I)
_SCALE_M = re.compile(r"\b(m|mm|million)\b", re.I)
_SCALE_K = re.compile(r"\b(k|thousand)\b", re.I)
_PCT_IN_TEXT = re.compile(r"%|\bpercent\b|\bpts\b|\bpp\b", re.I)
_PER_SHARE_IN_TEXT = re.compile(r"\bper\s+share\b|\b/\s*share\b", re.I)


# -----------------------------
# Data model
# -----------------------------

@dataclass
class VerificationResult:
    claim: Dict[str, Any]
    quarter: str
    verdict: str  # VERIFIED | INACCURATE | UNVERIFIABLE
    sec_value: Optional[float] = None
    sec_unit: Optional[str] = None
    sec_period_end: Optional[str] = None
    delta: Optional[float] = None
    delta_pct: Optional[float] = None
    explanation: str = ""


# -----------------------------
# SEC access helpers
# -----------------------------

def load_sec_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sec_jsons(paths: List[Path]) -> Dict[str, Any]:
    """Load and merge multiple SEC JSON files into a single sec_data dict.

    Expected structure per file:
        {"facts": {taxonomy: {concept: {"units": {unit: [points...]}}}}}
    """
    merged: Dict[str, Any] = {"facts": {}}

    def _ensure(d: Dict[str, Any], *keys: str) -> Dict[str, Any]:
        cur = d
        for k in keys:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        return cur

    for p in paths:
        if not p:
            continue
        if not p.exists():
            continue
        data = load_sec_json(p)
        facts = data.get("facts", {})
        for tax, concepts in facts.items():
            for concept, node in concepts.items():
                units = node.get("units", {})
                for unit, pts in units.items():
                    tgt = _ensure(merged["facts"], tax, concept, "units")
                    tgt.setdefault(unit, [])
                    if isinstance(pts, list):
                        tgt[unit].extend(pts)

    # Deduplicate points per (taxonomy, concept, unit) by common identifying fields
    for tax, concepts in merged["facts"].items():
        for concept, node in concepts.items():
            units = node.get("units", {})
            for unit, pts in list(units.items()):
                seen = set()
                deduped = []
                for p in pts:
                    if not isinstance(p, dict):
                        continue
                    key = (
                        p.get("start"), p.get("end"), p.get("val"),
                        p.get("fp"), p.get("fy"), p.get("form"), p.get("frame")
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(p)
                units[unit] = deduped

    return merged

def _parse_date(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def _days_between(start: Optional[str], end: Optional[str]) -> Optional[int]:
    if not start or not end:
        return None
    ds = _parse_date(start)
    de = _parse_date(end)
    if not ds or not de:
        return None
    return abs((de - ds).days)


def _split_metric(sec_metric: str) -> Optional[Tuple[str, str]]:
    parts = sec_metric.split(":", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _collect_points(
    sec_data: Dict[str, Any],
    sec_metric: str,
    unit_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return a flat list of points for a metric across units.

    Each returned dict includes: val, unit, start, end, fp, form, filed, frame.
    """
    sp = _split_metric(sec_metric)
    if not sp:
        return []
    taxonomy, concept = sp

    fact_node = sec_data.get("facts", {}).get(taxonomy, {}).get(concept)
    if not fact_node:
        return []

    units = fact_node.get("units", {})
    unit_names = list(units.keys())
    if unit_hint and unit_hint in unit_names:
        unit_names = [unit_hint] + [u for u in unit_names if u != unit_hint]

    out: List[Dict[str, Any]] = []
    for u in unit_names:
        for p in units.get(u, []):
            if p.get("val") is None or p.get("end") is None:
                continue
            out.append({
                "val": float(p["val"]),
                "unit": u,
                "start": p.get("start"),
                "end": p.get("end"),
                "fp": p.get("fp"),
                "form": p.get("form"),
                "filed": p.get("filed"),
                "frame": p.get("frame"),
            })
    return out


def _wants_annual_or_ytd(quarter: str, raw_sentence: str) -> bool:
    if quarter == "FY":
        return True
    if raw_sentence and (_ANNUAL_CUES.search(raw_sentence) or _YTD_CUES.search(raw_sentence)):
        return True
    return False


def _approx_factor(raw_sentence: str) -> float:
    return 2.0 if raw_sentence and _APPROX_WORDS.search(raw_sentence) else 1.0


def _is_flow_metric(sec_metric: str) -> bool:
    """Heuristic: income statement/cash flow metrics tend to have start+end.

    We use the presence of any point with a 'start' as signal.
    """
    # Quick heuristic list for common flow metrics.
    flow_keywords = (
        "Revenues",
        "NetIncome",
        "OperatingExpenses",
        "NetCash",
        "Payments",
        "CostOfGoods",
        "GrossProfit",
        "OperatingIncome",
        "IncomeBefore",
        "EarningsPerShare",
    )
    return any(k in sec_metric for k in flow_keywords)


def _rank_point(
    p: Dict[str, Any],
    target_end: str,
    quarter: str,
    raw_sentence: str,
    prefer_short_duration: bool,
) -> Tuple[float, int, int, float]:
    """Return a sortable rank (higher is better).
    Key insight: consolidated figures are usually largest; segments are smaller.
    """
    score = 0.0

    # End-date match strength
    if p.get("end") == target_end:
        score += 1000
    elif p.get("end", "").startswith(target_end[:7]):
        score += 800

    # Prefer 10-Q for quarters when available
    if quarter in {"Q1", "Q2", "Q3", "Q4"} and p.get("form") == "10-Q":
        score += 60
    if quarter == "FY" and p.get("form") == "10-K":
        score += 60

    # Prefer fp matching quarter if present
    if quarter in {"Q1", "Q2", "Q3", "Q4", "FY"} and p.get("fp") == quarter:
        score += 40

    # Consolidated over segment: for flow metrics, larger |val| is usually consolidated.
    # Add magnitude as significant score so we pick 6.6B over 1B when both match.
    val = p.get("val")
    mag = abs(float(val)) if val is not None else 0.0
    if prefer_short_duration and mag >= 1:
        mag_score = min(150, math.log10(mag) * 50)
        score += mag_score

    # Duration preference: true quarter (70-120 days) preferred over YTD/instant
    dur = _days_between(p.get("start"), p.get("end"))
    if dur is None:
        score += 15
        dur_key = 10**9
    else:
        dur_key = dur
        if prefer_short_duration:
            if 70 <= dur <= 120:
                score += 25
            score += max(0, 20 - min(dur, 365) // 15)
        else:
            score += min(20, min(dur, 365) // 15)

    filed = p.get("filed") or ""
    filed_key = int(filed.replace("-", "")) if filed and filed[0].isdigit() else 0

    return (score, -dur_key, filed_key, mag)


def get_sec_value_best(
    sec_data: Dict[str, Any],
    sec_metric: str,
    target_end: str,
    quarter: str,
    raw_sentence: str,
    unit_hint: Optional[str] = None,
    quarter_to_end: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[float, str, str]]:
    """Best-effort SEC lookup for a metric at a target period end.

    Handles multiple facts sharing the same 'end' by selecting the most appropriate one.
    """
    if not target_end:
        return None

    points = _collect_points(sec_data, sec_metric, unit_hint=unit_hint)
    if not points:
        return None

    wants_annual = _wants_annual_or_ytd(quarter, raw_sentence)
    flow = _is_flow_metric(sec_metric)

    # For quarterly absolute flow claims, prefer short duration (true quarter).
    prefer_short_duration = flow and (quarter in {"Q1", "Q2", "Q3", "Q4"}) and not wants_annual

    # If the SEC facts at quarter end are only YTD/annual (common in filtered/merged XBRL),
    # derive true quarter values for flow metrics via differencing:
    #   Q2 = YTD(Q2) - YTD(Q1)
    #   Q3 = YTD(Q3) - YTD(Q2)
    #   Q4 = FY - YTD(Q3)
        # If the SEC facts at quarter end are only YTD/annual (common in filtered/merged XBRL),
    # derive true quarter values for flow metrics via differencing:
    #   Q2 = YTD(Q2) - YTD(Q1)
    #   Q3 = YTD(Q3) - YTD(Q2)
    #   Q4 = FY - YTD(Q3)
    if flow and quarter in {"Q2", "Q3", "Q4"} and not wants_annual:
        # Detect whether we already have a true-quarter duration point at this end date.
        def _is_true_quarter(p: Dict[str, Any]) -> bool:
            if p.get("end") != target_end:
                return False
            dur = _days_between(p.get("start"), p.get("end"))
            # ~13 weeks; allow some slack for fiscal calendars.
            return dur is not None and 70 <= dur <= 120

        has_true_q = any(_is_true_quarter(p) for p in points)
        if not has_true_q:
            qt = quarter_to_end or QUARTER_TO_END_DATE
            q1_end = qt.get("Q1")
            q2_end = qt.get("Q2")
            q3_end = qt.get("Q3")
            q4_end = qt.get("Q4")

            def _ytd(end: Optional[str], q: str) -> Optional[Tuple[float, str, str]]:
                if not end:
                    return None
                return get_sec_value_best(sec_data, sec_metric, end, q, "YTD", unit_hint=unit_hint, quarter_to_end=quarter_to_end)

            if quarter == "Q2":
                v2 = _ytd(q2_end, "Q2")
                v1 = _ytd(q1_end, "Q1")
                if v2 and v1 and v2[1] == v1[1]:
                    return (float(v2[0] - v1[0]), str(v2[1]), str(q2_end))
            elif quarter == "Q3":
                v3 = _ytd(q3_end, "Q3")
                v2 = _ytd(q2_end, "Q2")
                if v3 and v2 and v3[1] == v2[1]:
                    return (float(v3[0] - v2[0]), str(v3[1]), str(q3_end))
            elif quarter == "Q4":
                fy = get_sec_value_best(sec_data, sec_metric, q4_end or "", "FY", "full year", unit_hint=unit_hint) if q4_end else None
                v3 = _ytd(q3_end, "Q3")
                if fy and v3 and fy[1] == v3[1]:
                    return (float(fy[0] - v3[0]), str(fy[1]), str(q4_end))

    # ---- ALWAYS DO FALLBACK SEARCH (for flow + non-flow) ----

    # 1) exact end or same-month candidates
    same_month = [
        p for p in points
        if p.get("end") == target_end or (p.get("end", "").startswith(target_end[:7]))
    ]
    if same_month:
        same_month.sort(
            key=lambda p: _rank_point(p, target_end, quarter, raw_sentence, prefer_short_duration),
            reverse=True,
        )
        best = same_month[0]
        return (float(best["val"]), str(best["unit"]), str(best["end"]))

    # 2) closest end within window (period matching fallback)
    target_dt = _parse_date(target_end)
    if not target_dt:
        return None

    best_p: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[int, int]] = None  # (days_diff, -rank_score)

    for p in points:
        end = p.get("end")
        if not end:
            continue
        end_dt = _parse_date(end)
        if not end_dt:
            continue
        dd = abs((end_dt - target_dt).days)
        if dd <= 45:
            rank = _rank_point(p, end, quarter, raw_sentence, prefer_short_duration)
            score = rank[0]
            key = (dd, -score)
            if best_key is None or key < best_key:
                best_key = key
                best_p = p

    if best_p is not None:
        return (float(best_p["val"]), str(best_p["unit"]), str(best_p["end"]))

    return None



def get_sec_value_with_fallbacks(
    sec_data: Dict[str, Any],
    sec_metric: str,
    quarter: str,
    target_end: str,
    raw_sentence: str,
    quarter_to_end: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[float, str, str]]:
    """Layered SEC lookup with guarded annual fallback."""
    qt = quarter_to_end or QUARTER_TO_END_DATE
    r = get_sec_value_best(sec_data, sec_metric, target_end, quarter, raw_sentence, quarter_to_end=quarter_to_end)
    if r:
        return r

    alt_end = qt.get(quarter)
    if alt_end and alt_end != target_end:
        r = get_sec_value_best(sec_data, sec_metric, alt_end, quarter, raw_sentence, quarter_to_end=quarter_to_end)
        if r:
            return r

    if raw_sentence and _ANNUAL_CUES.search(raw_sentence):
        fy_end = qt.get("Q4")
        if fy_end:
            r = get_sec_value_best(sec_data, sec_metric, fy_end, "FY", raw_sentence, quarter_to_end=quarter_to_end)
            if r:
                return r

    return None


def get_derived_sec_value(
    sec_data: Dict[str, Any],
    derived_metric: str,
    quarter: str,
    period_end: str,
    raw_sentence: str,
    quarter_to_end: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[float, str, str]]:
    comps = DERIVED_METRICS.get(derived_metric)
    if not comps:
        return None

    total = 0.0
    used_unit: Optional[str] = None
    used_end: str = period_end

    for comp in comps:
        sign = 1.0
        metric = comp
        if isinstance(comp, str) and comp.startswith("-"):
            sign = -1.0
            metric = comp[1:]

        r = get_sec_value_with_fallbacks(sec_data, metric, quarter, period_end, raw_sentence, quarter_to_end=quarter_to_end)
        if r is None:
            return None
        val, unit, end = r
        used_unit = used_unit or unit
        used_end = end or used_end
        total += sign * float(val)

    return (total, used_unit or "USD", used_end)


# -----------------------------
# Normalization
# -----------------------------

def _parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        m = _NUM_RE.search(s.replace("$", ""))
        if not m:
            return None
        try:
            return float(m.group(0).replace(",", ""))
        except Exception:
            return None
    return None


def normalize_claim_value(claim: Dict[str, Any]) -> Optional[float]:
    """Normalize claim numeric value to match SEC comparison scale.

    Convention used here:
    - For USD amounts: compare in *billions*.
    - For percent: keep percent.
    - For per-share: keep USD/share.
    """
    raw = (claim.get("raw_sentence") or "")
    unit = (claim.get("unit") or "").strip().lower()

    v = _parse_number(claim.get("value"))
    if v is None:
        return None

    # Infer unit if missing
    if not unit:
        if _PER_SHARE_IN_TEXT.search(raw):
            unit = "usd/share"
        elif _PCT_IN_TEXT.search(raw):
            unit = "percent"
        elif "$" in raw:
            unit = "usd"

    # Per-share
    if "share" in unit:
        return float(v)

    # Percent
    if unit == "percent":
        return float(v)

    # USD: normalize to billions (SEC comparison scale)
    if unit == "usd":
        # Scale words: value may be raw ($) or already scaled
        if _SCALE_B.search(raw):
            # "24.8 billion" -> value may be 24.8 or 24800000000
            return float(v) / 1e9 if v >= 1e9 else float(v)
        if _SCALE_M.search(raw):
            # millions -> billions
            return float(v) / 1e9 if v >= 1e6 else float(v) / 1000.0
        if _SCALE_K.search(raw):
            return float(v) / 1e9 if v >= 1e3 else float(v) / 1e6

        # Magnitude-based: raw dollars -> billions
        if v >= 1e9:
            return float(v) / 1e9
        if v >= 1e6:
            return float(v) / 1e9
        return float(v)

    return float(v)


def normalize_sec_value(value: float, unit: str) -> float:
    """Normalize SEC value to the same scale as claims.

    - USD amounts: SEC is in dollars -> convert to billions.
    - Per-share & percent: leave as-is.
    """
    u = (unit or "").strip().upper()

    # Percent
    if "PERCENT" in u or u == "PURE":
        return float(value)

    # Per-share
    if "USD/SHARES" in u or "SHARES" in u and "USD" in u:
        return float(value)

    # USD dollars -> billions
    if u == "USD":
        return float(value) / 1e9

    # If weird unit, return raw
    return float(value)


# -----------------------------
# Computations: growth & margin
# -----------------------------

def compute_growth(
    sec_data: Dict[str, Any],
    sec_metric: str,
    quarter: str,
    frame: str,
    raw_sentence: str,
    quarter_to_end: Optional[Dict[str, str]] = None,
    prior_year_end: Optional[Dict[str, str]] = None,
) -> Optional[float]:
    """Compute YoY or QoQ growth % from SEC data."""
    q_ends = ["Q1", "Q2", "Q3", "Q4"]
    if quarter not in q_ends:
        return None

    qt = quarter_to_end or QUARTER_TO_END_DATE
    py = prior_year_end or PRIOR_YEAR_END_DATE
    idx = q_ends.index(quarter)
    cur_end = qt.get(quarter)
    if not cur_end:
        return None

    if frame == "YoY":
        prev_end = py.get(quarter)
    elif frame == "QoQ":
        if idx == 0:
            return None
        prev_end = qt.get(q_ends[idx - 1])
    else:
        return None

    if not prev_end:
        return None

    cur = get_sec_value_best(sec_data, sec_metric, cur_end, quarter, raw_sentence, quarter_to_end=quarter_to_end)
    prev = get_sec_value_best(sec_data, sec_metric, prev_end, (quarter if frame == "YoY" else q_ends[idx - 1]), raw_sentence, quarter_to_end=quarter_to_end)
    if not cur or not prev:
        return None

    cur_val = normalize_sec_value(cur[0], cur[1])
    prev_val = normalize_sec_value(prev[0], prev[1])

    if prev_val == 0:
        return None

    return ((cur_val - prev_val) / abs(prev_val)) * 100.0


def compute_margin(
    sec_data: Dict[str, Any],
    margin_type: str,
    quarter: str,
    period_end: str,
    raw_sentence: str,
    quarter_to_end: Optional[Dict[str, str]] = None,
) -> Optional[float]:
    """Compute a margin percentage for the period end.

    margin_type: 'gross' | 'operating' | 'net'
    """
    rev = get_sec_value_with_fallbacks(sec_data, "us-gaap:Revenues", quarter, period_end, raw_sentence, quarter_to_end=quarter_to_end)
    if not rev:
        rev = get_sec_value_with_fallbacks(sec_data, "us-gaap:SalesRevenueNet", quarter, period_end, raw_sentence, quarter_to_end=quarter_to_end)
    if not rev:
        return None
    rev_b = normalize_sec_value(rev[0], rev[1])
    if rev_b == 0:
        return None

    if margin_type == "gross":
        gp = get_sec_value_with_fallbacks(sec_data, "us-gaap:GrossProfit", quarter, period_end, raw_sentence, quarter_to_end=quarter_to_end)
        if gp:
            gp_b = normalize_sec_value(gp[0], gp[1])
            return (gp_b / rev_b) * 100.0

        cogs = get_sec_value_with_fallbacks(sec_data, "us-gaap:CostOfGoodsAndServicesSold", quarter, period_end, raw_sentence, quarter_to_end=quarter_to_end)
        if not cogs:
            return None
        cogs_b = normalize_sec_value(cogs[0], cogs[1])
        return ((rev_b - cogs_b) / rev_b) * 100.0

    if margin_type == "operating":
        op = get_sec_value_with_fallbacks(sec_data, "us-gaap:OperatingIncomeLoss", quarter, period_end, raw_sentence, quarter_to_end=quarter_to_end)
        if not op:
            return None
        op_b = normalize_sec_value(op[0], op[1])
        return (op_b / rev_b) * 100.0

    if margin_type == "net":
        ni = get_sec_value_with_fallbacks(sec_data, "us-gaap:NetIncomeLoss", quarter, period_end, raw_sentence, quarter_to_end=quarter_to_end)
        if not ni:
            return None
        ni_b = normalize_sec_value(ni[0], ni[1])
        return (ni_b / rev_b) * 100.0

    return None


def _infer_margin_type(sec_metric: str, raw_sentence: str) -> str:
    s = (raw_sentence or "").lower()
    if "operating margin" in s or "op margin" in s:
        return "operating"
    if "net margin" in s:
        return "net"
    if "gross margin" in s or re.search(r"\bgm\b", s):
        return "gross"

    # fallback: based on metric
    if "OperatingIncome" in sec_metric:
        return "operating"
    if "NetIncome" in sec_metric:
        return "net"
    return "gross"


# -----------------------------
# Main verification
# -----------------------------

def verify_one_claim(
    claim: Dict[str, Any],
    quarter: str,
    sec_data: Dict[str, Any],
    quarter_to_end: Optional[Dict[str, str]] = None,
    prior_year_end: Optional[Dict[str, str]] = None,
) -> VerificationResult:
    sec_metric = claim.get("sec_metric", "") or ""
    kind = claim.get("kind", "absolute") or "absolute"
    raw_sentence = claim.get("raw_sentence") or ""

    if not sec_metric or claim.get("value") is None:
        return VerificationResult(claim=claim, quarter=quarter, verdict="UNVERIFIABLE", explanation="Missing sec_metric or value.")

    qt = quarter_to_end or QUARTER_TO_END_DATE
    period_end = qt.get(quarter)
    if not period_end:
        return VerificationResult(claim=claim, quarter=quarter, verdict="UNVERIFIABLE", explanation=f"Unknown quarter {quarter}.")

    # Repurchase scope
    raw_l = raw_sentence.lower()
    if "PaymentsForRepurchaseOfCommonStock" in sec_metric or "StockRepurchasedAndRetired" in sec_metric:
        if _REPURCHASE_PROGRAM.search(raw_l):
            return VerificationResult(claim=claim, quarter=quarter, verdict="UNVERIFIABLE", explanation="Claim is program-level or cumulative; cannot verify against quarterly total.")
        if _REPURCHASE_PARTIAL.search(raw_l):
            return VerificationResult(claim=claim, quarter=quarter, verdict="UNVERIFIABLE", explanation="Claim appears to be partial repurchase (specific tranche); SEC reports total quarterly.")

    approx_mult = _approx_factor(raw_sentence)

    # Growth claims
    if kind == "growth":
        frame = claim.get("frame") or ""
        if frame not in {"YoY", "QoQ"}:
            return VerificationResult(claim=claim, quarter=quarter, verdict="UNVERIFIABLE", explanation="Growth claim missing frame (YoY/QoQ).")

        actual_pct = compute_growth(sec_data, sec_metric, quarter, frame, raw_sentence, quarter_to_end=quarter_to_end, prior_year_end=prior_year_end)
        if actual_pct is None:
            return VerificationResult(claim=claim, quarter=quarter, verdict="UNVERIFIABLE", explanation="Cannot compute growth from SEC data.")

        claim_pct = normalize_claim_value({**claim, "unit": "percent"})
        if claim_pct is None:
            return VerificationResult(claim=claim, quarter=quarter, verdict="UNVERIFIABLE", explanation="Could not normalize growth claim value.")

        tol_pp = TOL_GROWTH_PP * approx_mult
        delta_pp = float(claim_pct) - float(actual_pct)
        ok = abs(delta_pp) <= tol_pp

        return VerificationResult(
            claim=claim,
            quarter=quarter,
            verdict="VERIFIED" if ok else "INACCURATE",
            sec_value=float(actual_pct),
            sec_unit="percent",
            sec_period_end=period_end,
            delta=delta_pp,
            delta_pct=(delta_pp / actual_pct * 100) if actual_pct else None,
            explanation=f"SEC {frame} growth: {actual_pct:.2f}%.",
        )

    # Margin claims
    if kind == "margin":
        mtype = _infer_margin_type(sec_metric, raw_sentence)
        actual = compute_margin(sec_data, mtype, quarter, period_end, raw_sentence, quarter_to_end=quarter_to_end)
        if actual is None:
            return VerificationResult(claim=claim, quarter=quarter, verdict="UNVERIFIABLE", explanation="Cannot compute margin from SEC data (missing numerator or revenue).")

        claim_pct = normalize_claim_value({**claim, "unit": "percent"})
        if claim_pct is None:
            return VerificationResult(claim=claim, quarter=quarter, verdict="UNVERIFIABLE", explanation="Could not normalize margin claim value.")

        tol_pp = TOL_PCT_MARGIN * approx_mult
        delta_pp = float(claim_pct) - float(actual)
        ok = abs(delta_pp) <= tol_pp

        return VerificationResult(
            claim=claim,
            quarter=quarter,
            verdict="VERIFIED" if ok else "INACCURATE",
            sec_value=float(actual),
            sec_unit="percent",
            sec_period_end=period_end,
            delta=delta_pp,
            delta_pct=(delta_pp / actual * 100) if actual else None,
            explanation=f"SEC {mtype} margin: {actual:.2f}%.",
        )

    # Absolute claims
    # Special mapping: cash plus marketable
    effective_metric = sec_metric
    if (
        ("cash plus marketable" in raw_l or "cash and marketable" in raw_l)
        and "CashAndCashEquivalents" in sec_metric
    ):
        effective_metric = "derived:CashPlusMarketableSecurities"

    if effective_metric.startswith("derived:"):
        sec_tuple = get_derived_sec_value(sec_data, effective_metric, quarter, period_end, raw_sentence, quarter_to_end=quarter_to_end)
    else:
        sec_tuple = get_sec_value_with_fallbacks(sec_data, effective_metric, quarter, period_end, raw_sentence, quarter_to_end=quarter_to_end)

    if not sec_tuple:
        return VerificationResult(
            claim=claim,
            quarter=quarter,
            verdict="UNVERIFIABLE",
            explanation=f"No SEC data for {effective_metric} at (or near) period end {period_end}.",
        )

    sec_val, sec_unit, sec_end = sec_tuple

    claim_val_norm = normalize_claim_value(claim)
    if claim_val_norm is None:
        return VerificationResult(claim=claim, quarter=quarter, verdict="UNVERIFIABLE", explanation="Could not normalize claim value.")

    sec_val_norm = normalize_sec_value(sec_val, sec_unit)

    # EPS comparison
    unit = (claim.get("unit") or "").lower()
    if "earningspershare" in sec_metric.lower() or (unit and "share" in unit):
        tol = TOL_EPS * approx_mult
        delta = float(claim_val_norm) - float(sec_val_norm)
        ok = abs(delta) <= tol
        return VerificationResult(
            claim=claim,
            quarter=quarter,
            verdict="VERIFIED" if ok else "INACCURATE",
            sec_value=float(sec_val_norm),
            sec_unit=sec_unit,
            sec_period_end=sec_end,
            delta=delta,
            delta_pct=(delta / sec_val_norm * 100) if sec_val_norm else None,
            explanation=f"SEC EPS: ${sec_val_norm:.2f}.",
        )

    # General absolute comparison
    tol_pct = TOL_ABS_PCT * approx_mult
    if sec_val_norm == 0:
        ok = abs(claim_val_norm) < 1e-9
    else:
        ok = (abs(float(claim_val_norm) - float(sec_val_norm)) / abs(float(sec_val_norm))) <= tol_pct

    delta = float(claim_val_norm) - float(sec_val_norm)
    delta_pct = (delta / sec_val_norm * 100) if sec_val_norm else None
    vr = VerificationResult(
        claim=claim,
        quarter=quarter,
        verdict="VERIFIED" if ok else "INACCURATE",
        sec_value=float(sec_val_norm),
        sec_unit=sec_unit,
        sec_period_end=sec_end,
        delta=delta,
        delta_pct=delta_pct,
        explanation=f"SEC value: {sec_val_norm:.2f} {sec_unit}.",
    )

    # If INACCURATE, try alternate metrics when delta is material (>25%); pick best match
    if not ok and delta_pct is not None and abs(delta_pct) > 25.0 and not effective_metric.startswith("derived:"):
        best_vr: Optional[VerificationResult] = None
        best_abs_delta_pct = abs(delta_pct)

        for alt_metric in METRIC_ALIASES.get(effective_metric, []):
            if alt_metric == effective_metric:
                continue
            alt_tuple = get_sec_value_with_fallbacks(
                sec_data, alt_metric, quarter, period_end, raw_sentence, quarter_to_end=quarter_to_end
            )
            if not alt_tuple:
                continue
            alt_val, alt_unit, alt_end = alt_tuple
            alt_norm = normalize_sec_value(alt_val, alt_unit)
            if alt_norm == 0 and abs(claim_val_norm) < 1e-9:
                return VerificationResult(
                    claim=claim, quarter=quarter, verdict="VERIFIED",
                    sec_value=0.0, sec_unit=alt_unit, sec_period_end=alt_end,
                    delta=float(claim_val_norm), delta_pct=None,
                    explanation=f"SEC value (alt {alt_metric.split(':')[-1]}): 0.0 {alt_unit}.",
                )
            if alt_norm == 0:
                continue
            alt_delta = float(claim_val_norm) - float(alt_norm)
            alt_delta_pct = (alt_delta / abs(alt_norm)) * 100
            alt_ok = (abs(alt_delta) / abs(alt_norm)) <= tol_pct
            # Keep best match: prefer one that passes; among those, smallest |delta_pct|
            if alt_ok and abs(alt_delta_pct) < best_abs_delta_pct:
                best_abs_delta_pct = abs(alt_delta_pct)
                best_vr = VerificationResult(
                    claim=claim, quarter=quarter, verdict="VERIFIED",
                    sec_value=float(alt_norm), sec_unit=alt_unit, sec_period_end=alt_end,
                    delta=alt_delta, delta_pct=alt_delta_pct,
                    explanation=f"SEC value (alt {alt_metric.split(':')[-1]}): {alt_norm:.2f} {alt_unit}.",
                )
        if best_vr is not None:
            return best_vr

    return vr


def _result_dict(c: Dict[str, Any], vr: VerificationResult) -> Dict[str, Any]:
    return {
        "quarter": vr.quarter,
        "verdict": vr.verdict,
        "sec_metric": c.get("sec_metric"),
        "kind": c.get("kind"),
        "claim_value": c.get("value"),
        "claim_unit": c.get("unit"),
        "sec_value": vr.sec_value,
        "sec_unit": vr.sec_unit,
        "delta": vr.delta,
        "delta_pct": vr.delta_pct,
        "explanation": vr.explanation,
        "raw_sentence": (c.get("raw_sentence") or "")[:200],
    }


def verify_claims(
    claims_path: Path = DEFAULT_CLAIMS_PATH,
    sec_json_paths: Optional[List[Path]] = None,
    sec_json_path: Path = DEFAULT_SEC_JSON_PATH,
    output_path: Optional[Path] = None,
    fiscal_year_end: Optional[str] = None,
    fiscal_year: Optional[int] = None,
) -> Dict[str, Any]:
    with open(claims_path, "r", encoding="utf-8") as f:
        claims_data = json.load(f)

    fy = fiscal_year or claims_data.get("fiscal_year")
    qt, py = None, None
    if fiscal_year_end and fy is not None:
        qt, py = _quarter_dates(fiscal_year_end, int(fy))

    # Load SEC data.
    # Prefer per-filing snapshots when provided (these typically include both 2016 & 2017 comparatives).
    paths = sec_json_paths if sec_json_paths else DEFAULT_SEC_JSON_PATHS
    if paths:
        sec_data = load_sec_jsons(paths)
        # If nothing loaded (paths missing), fall back to the merged file.
        if not sec_data.get("facts"):
            sec_data = load_sec_json(sec_json_path)
    else:
        sec_data = load_sec_json(sec_json_path)

    results: List[Dict[str, Any]] = []
    by_verdict: Dict[str, int] = {"VERIFIED": 0, "INACCURATE": 0, "UNVERIFIABLE": 0}

    quarters_obj = claims_data.get("quarters", {})
    if not quarters_obj and claims_data.get("all_claims"):
        for c in claims_data["all_claims"]:
            q = c.get("_quarter", "UNKNOWN")
            vr = verify_one_claim(c, q, sec_data, quarter_to_end=qt, prior_year_end=py)
            by_verdict[vr.verdict] = by_verdict.get(vr.verdict, 0) + 1
            results.append(_result_dict(c, vr))
    else:
        for qk, qv in quarters_obj.items():
            q_fy = qv.get("fiscal_year") or fy
            q_qt, q_py = qt, py
            if fiscal_year_end and q_fy is not None:
                q_qt, q_py = _quarter_dates(fiscal_year_end, int(q_fy))
            for c in qv.get("claims", []):
                vr = verify_one_claim(c, qk, sec_data, quarter_to_end=q_qt, prior_year_end=q_py)
                by_verdict[vr.verdict] = by_verdict.get(vr.verdict, 0) + 1
                results.append(_result_dict(c, vr))

    out: Dict[str, Any] = {
        "ticker": claims_data.get("ticker"),
        "fiscal_year": claims_data.get("fiscal_year"),
        "claims_source": str(Path(claims_path).name),
        "sec_source": str(Path(sec_json_path).name),
        "summary": {"total": len(results), "by_verdict": by_verdict},
        "results": results,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved verification results to {output_path}")

    return out



def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify transcript claims against SEC JSON")
    parser.add_argument("--claims", default=str(DEFAULT_CLAIMS_PATH), help="Path to transcript claims JSON")
    parser.add_argument("--sec-json", default=str(DEFAULT_SEC_JSON_PATH), help="Path to merged/filtered SEC JSON")
    parser.add_argument(
        "--sec-jsons",
        nargs="*",
        default=[str(p) for p in DEFAULT_SEC_JSON_PATHS],
        help="Optional list of SEC JSON files (per-filing 10-Q/10-K snapshots). If provided, these are merged and preferred.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output path for verification results")
    parser.add_argument("--fiscal-year-end", choices=["sep", "dec"], default=None, help="Fiscal year end: sep (Apple) or dec (Google, Microsoft)")
    parser.add_argument("--fiscal-year", type=int, default=None, help="Fiscal year; uses claims file if omitted")
    args = parser.parse_args()

    out = verify_claims(
        claims_path=Path(args.claims),
        sec_json_paths=[Path(p) for p in (args.sec_jsons or [])],
        sec_json_path=Path(args.sec_json),
        output_path=Path(args.output),
        fiscal_year_end=args.fiscal_year_end,
        fiscal_year=args.fiscal_year,
    )

    s = out["summary"]
    print(f"\nVerification complete: {s['total']} claims")
    print(f"  VERIFIED:     {s['by_verdict'].get('VERIFIED', 0)}")
    print(f"  INACCURATE:   {s['by_verdict'].get('INACCURATE', 0)}")
    print(f"  UNVERIFIABLE: {s['by_verdict'].get('UNVERIFIABLE', 0)}")


if __name__ == "__main__":
    main()
