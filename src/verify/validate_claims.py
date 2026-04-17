# src/verify/validate_claims.py
"""
Claim validator (MVP).

Takes structured transcript claims and verifies them against SEC facts in SQLite.

Outputs:
- verdict: VERIFIED / INACCURATE / UNVERIFIABLE
- computed actual value
- cherry-pick flags (timeframe and segment)
- short explanation text
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from src.data.facts_repo import FactsRepo
from src.transcripts.extract_claims import Claim
from src.transcripts.transcript_mapping import TranscriptMeta


@dataclass(frozen=True)
class ValidationResult:
    claim: Dict[str, Any]
    verdict: str
    actual: Optional[float]
    delta: Optional[float]
    flags: List[str]
    explanation: str


# -----------------------------
# Tolerances (tweakable)
# -----------------------------
TOL_ABS_PCT = 0.005      # 0.5% for big amounts in millions
TOL_GROWTH_PP = 0.7      # percentage points tolerance for growth claims
TOL_EPS = 0.02           # $0.02 tolerance for EPS
TOL_MARGIN_PP = 0.3      # 0.3 pp for margin


def approx_equal_abs(claim_val: float, actual_val: float, tol_pct: float) -> bool:
    if actual_val == 0:
        return abs(claim_val) < 1e-9
    return abs(claim_val - actual_val) / abs(actual_val) <= tol_pct


def _normalize_sec_value_to_millions(value: float, unit: str) -> float:
    """
    Convert SEC stored value to USD millions for comparison.
    SEC XBRL can store revenue/income in full USD or thousands/millions.
    """
    u = (unit or "").strip().upper()
    if u == "USD":
        if value >= 1e9:
            return value / 1_000_000.0  # full dollars -> millions
        if value >= 1e6 and value < 1e9:
            return value / 1_000.0  # thousands -> millions
    return value  # assume already millions


def _normalize_claim_to_sec_units(claim: Claim, for_eps: bool = False) -> Optional[float]:
    """
    Convert claim.value from any extractor unit to SEC storage units.

    SEC stores: revenue/income in USD millions; eps_diluted in USD per share.
    Extractors may use: usd_millions, USD (full dollars), usd_per_share, USD/shares.
    """
    unit = (claim.unit or "").strip().lower()
    value = claim.value

    if for_eps:
        if unit == "usd" and value > 100:
            return None  # Likely full dollars mistaken for EPS
        return value

    if unit in {"usd_millions", "usd_m", "millions"}:
        return value
    if unit == "usd":
        if value >= 1_000_000_000:
            return value / 1_000_000.0
        if value >= 1_000_000:
            return value / 1_000_000.0
        return value
    if unit in {"usd_billion", "usd_b", "billions"}:
        return value * 1_000.0
    return value


def validate_one(
    repo: FactsRepo,
    meta: TranscriptMeta,
    claim: Claim,
    ticker: str = "AAPL",
) -> ValidationResult:
    fy = meta.fiscal_year
    fp = meta.fp if meta.fp != "Q4" else "Q4_DERIVED"

    flags: List[str] = []

    # --- Non-GAAP / unverifiable claims ---
    if getattr(claim, "force_unverifiable", False):
        return ValidationResult(
            asdict(claim), "UNVERIFIABLE", None, None, flags,
            "Claim references non-GAAP, adjusted, or constant-currency metrics; SEC data is GAAP.",
        )

    # --- Handle gross margin (computed) ---
    if claim.metric == "gross_margin":
        gm = repo.compute_gross_margin(ticker, fy, fp)
        if gm is None:
            return ValidationResult(asdict(claim), "UNVERIFIABLE", None, None, flags,
                                    "Gross margin not computable from stored facts.")
        actual_pct = gm * 100.0
        delta = claim.value - actual_pct
        ok = abs(delta) <= TOL_MARGIN_PP
        verdict = "VERIFIED" if ok else "INACCURATE"
        return ValidationResult(asdict(claim), verdict, actual_pct, delta, flags,
                                f"Gross margin computed from SEC facts is {actual_pct:.2f}%.")

    # --- Absolute claims ---
    if claim.kind == "absolute":
        if claim.metric in {"revenue_total", "revenue_services", "net_income", "cost_of_revenue", "gross_profit"}:
            pt = repo.get_point(ticker, claim.metric, fy, fp)
            if pt is None:
                return ValidationResult(asdict(claim), "UNVERIFIABLE", None, None, flags,
                                        "Metric not found in filtered SEC facts.")

            actual_millions = _normalize_sec_value_to_millions(pt.value, pt.unit)
            claim_val = _normalize_claim_to_sec_units(claim)
            if claim_val is None:
                return ValidationResult(asdict(claim), "UNVERIFIABLE", None, None, flags,
                                        "Could not normalize claim value to SEC units.")
            ok = approx_equal_abs(claim_val, actual_millions, TOL_ABS_PCT)
            verdict = "VERIFIED" if ok else "INACCURATE"
            delta = claim_val - actual_millions
            return ValidationResult(asdict(claim), verdict, actual_millions, delta, flags,
                                    f"Compared claim (USD millions) to SEC fact (USD millions).")

        if claim.metric == "eps_diluted":
            actual = repo.get_value(ticker, "eps_diluted", fy, fp)
            if actual is None:
                return ValidationResult(asdict(claim), "UNVERIFIABLE", None, None, flags,
                                        "EPS not found in filtered SEC facts.")
            claim_val = _normalize_claim_to_sec_units(claim, for_eps=True)
            if claim_val is None:
                return ValidationResult(asdict(claim), "UNVERIFIABLE", None, None, flags,
                                        "Could not normalize EPS claim to SEC units.")
            delta = claim_val - actual
            ok = abs(delta) <= TOL_EPS
            verdict = "VERIFIED" if ok else "INACCURATE"
            return ValidationResult(asdict(claim), verdict, actual, delta, flags,
                                    f"Diluted EPS per SEC facts is {actual:.2f}.")

    # --- Growth claims ---
    if claim.kind == "growth":
        # claim.value is percent (e.g., 15 means 15%)
        if claim.frame not in {"YoY", "QoQ"}:
            return ValidationResult(asdict(claim), "UNVERIFIABLE", None, None, flags,
                                    "Growth claim missing frame (YoY/QoQ).")

        if claim.frame == "YoY":
            g = repo.compute_yoy(ticker, claim.metric, fy, fp)
            alt = repo.compute_qoq(ticker, claim.metric, fy, fp)
        else:
            g = repo.compute_qoq(ticker, claim.metric, fy, fp)
            alt = repo.compute_yoy(ticker, claim.metric, fy, fp)

        if g is None:
            return ValidationResult(asdict(claim), "UNVERIFIABLE", None, None, flags,
                                    "Not enough data to compute growth from SEC facts.")

        actual_pct = g * 100.0
        delta_pp = claim.value - actual_pct
        ok = abs(delta_pp) <= TOL_GROWTH_PP
        verdict = "VERIFIED" if ok else "INACCURATE"

        # --- Cherry-pick flags ---
        # 1) Timeframe cherry-pick: verified growth but alt frame materially opposite
        if alt is not None:
            alt_pct = alt * 100.0
            # material opposite: one positive, other negative and large enough
            if (actual_pct >= 5 and alt_pct <= -2) or (actual_pct <= -5 and alt_pct >= 2):
                flags.append("CHERRY_PICK_TIMEFRAME")

        # 2) Segment cherry-pick: Services strong, Total weak (only for services metric)
        if claim.metric == "revenue_services":
            svc_yoy = repo.compute_yoy(ticker, "revenue_services", fy, fp)
            tot_yoy = repo.compute_yoy(ticker, "revenue_total", fy, fp)
            if svc_yoy is not None and tot_yoy is not None:
                svc_pct = svc_yoy * 100.0
                tot_pct = tot_yoy * 100.0
                if svc_pct >= 10 and tot_pct <= 2:
                    flags.append("CHERRY_PICK_SEGMENT_SERVICES_VS_TOTAL")

        explanation = f"{claim.frame} growth computed from SEC facts is {actual_pct:.2f}%."
        if flags:
            explanation += f" Flags: {', '.join(flags)}."
        return ValidationResult(asdict(claim), verdict, actual_pct, delta_pp, flags, explanation)

    return ValidationResult(asdict(claim), "UNVERIFIABLE", None, None, flags, "Unsupported claim type/kind.")


def validate_claims(
    repo: FactsRepo,
    meta: TranscriptMeta,
    claims: List[Claim],
    ticker: str = "AAPL",
) -> List[ValidationResult]:
    return [validate_one(repo, meta, c, ticker) for c in claims]
