"""
Extract claims from Apple FY2017 transcripts for ALL financial metrics present in
the SEC filtered JSON files (from batch_process_filter.py).

Purpose:
- Process all 4 quarters of 2017 transcripts (Jan, May, Aug, Nov).
- Extract quantitative claims for every metric that has values in the 5 JSON files.
- Output structured JSON for later verification against SEC data.

Metrics source: data/raw/sec/AAPL_10K_10Q_merged_filtered_endyear.json (or any of
the per-transcript filtered JSONs - they share the same metric set).

Output: data/raw/sec/transcript_claims_by_sec_metric_2017.json
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.transcripts.extract_claim_hybrid import GroqClient
from src.transcripts.transcript_mapping import detect_transcript_meta

# -----------------------------
# SEC metric config
# -----------------------------

TRANSCRIPT_FILES_2017 = [
    "2017-Jan-31-AAPL.txt",
    "2017-May-02-AAPL.txt",
    "2017-Aug-01-AAPL.txt",
    "2017-Nov-02-AAPL.txt",
]

DEFAULT_TRANSCRIPTS_DIR = Path("data/raw/transcripts/AAPL")
DEFAULT_SEC_JSON_PATH = Path("data/raw/sec/AAPL_10K_10Q_merged_filtered_endyear.json")
DEFAULT_OUTPUT_PATH = Path("data/raw/sec/transcript_claims_by_sec_metric_2017.json")
DEFAULT_CACHE_DIR = "data/cache/groq_sec_claims"


@dataclass(frozen=True)
class SecClaim:
    """A claim with sec_metric for later verification against SEC JSON."""

    sec_metric: str  # e.g. "us-gaap:Revenues", "dei:EntityCommonStockSharesOutstanding"
    kind: str  # absolute | growth | margin
    value: float
    unit: str  # USD, shares, percent, USD/shares
    frame: Optional[str] = None  # YoY | QoQ | None
    raw_sentence: str = ""
    confidence: str = "medium"
    evidence_start: Optional[int] = None
    evidence_end: Optional[int] = None
    flags: List[str] = field(default_factory=list)


# Priority metrics for LLM prompt (keeps prompt under Groq token limit).
# Full metric set used for filtering; only these appear in the prompt.
PRIORITY_SEC_METRICS = frozenset([
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
    "us-gaap:NetIncomeLoss",
    "us-gaap:EarningsPerShareDiluted",
    "us-gaap:EarningsPerShareBasic",
    "us-gaap:GrossProfit",
    "us-gaap:CostOfGoodsAndServicesSold",
    "us-gaap:CostOfRevenue",
    "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "us-gaap:Assets",
    "us-gaap:AssetsCurrent",
    "us-gaap:Liabilities",
    "us-gaap:LiabilitiesCurrent",
    "us-gaap:DebtInstrumentCarryingAmount",
    "dei:EntityCommonStockSharesOutstanding",
    "us-gaap:CommonStockSharesOutstanding",
    "us-gaap:Dividends",
    "us-gaap:DepreciationAndAmortization",
    "us-gaap:OperatingIncomeLoss",
    "us-gaap:AccountsPayableCurrent",
    "us-gaap:AccountsReceivableNetCurrent",
    "us-gaap:InventoryNet",
    "us-gaap:DeferredRevenueCurrent",
    "us-gaap:DeferredRevenueNoncurrent",
    "us-gaap:SalesRevenueServicesNet",
    "us-gaap:SalesRevenueServicesGross",
    "us-gaap:AvailableForSaleSecuritiesCurrent",
])

# Derived metrics: computed as sum of SEC metrics (verified in verifier)
DERIVED_SEC_METRICS = frozenset([
    "derived:CashPlusMarketableSecurities",
])


def load_sec_metrics_from_json(json_path: Path) -> List[Tuple[str, str, List[str]]]:
    """
    Load (sec_metric, label, units) from the merged/filtered SEC JSON.

    Returns list of (taxonomy:fact_name, human label, unit types).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics: List[Tuple[str, str, List[str]]] = []
    for taxonomy, tax_facts in data.get("facts", {}).items():
        for fact_name, fact_obj in tax_facts.items():
            sec_metric = f"{taxonomy}:{fact_name}"
            label = fact_obj.get("label", fact_name)
            units = list(fact_obj.get("units", {}).keys())
            metrics.append((sec_metric, label, units))
    return metrics


def get_prompt_metrics(
    all_metrics: List[Tuple[str, str, List[str]]],
) -> List[Tuple[str, str, List[str]]]:
    """
    Return priority metrics for LLM prompt (subset to stay under Groq token limit).
    """
    by_sec = {m[0]: m for m in all_metrics}
    return [by_sec[sec] for sec in PRIORITY_SEC_METRICS if sec in by_sec]


def build_metric_reference(metrics: List[Tuple[str, str, List[str]]]) -> str:
    """Build compact reference for LLM prompt."""
    lines = []
    for sec_metric, label, units in metrics:
        unit_str = ",".join(units) if units else "?"
        lines.append(f"  - {sec_metric} | {label} | units: {unit_str}")
    return "\n".join(lines)


# -----------------------------
# Parsing (reused/adapted from hybrid)
# -----------------------------

MONEY_RE = re.compile(
    r"(?:(\$)\s*)?([0-9]+(?:\.[0-9]+)?)\s*(billion|million|bn|mm)?",
    re.IGNORECASE,
)
PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
EPS_RE = re.compile(
    r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+share|EPS)\b",
    re.IGNORECASE,
)
SHARES_RE = re.compile(
    r"\b([0-9]+(?:\.[0-9]+)?)\s*(million|bn|billion|mm)?\s*(?:shares?)\b",
    re.IGNORECASE,
)


def normalize_money_to_usd(amount: float, scale_word: Optional[str]) -> float:
    if not scale_word:
        return amount
    s = scale_word.lower()
    if s in {"billion", "bn"}:
        return amount * 1_000_000_000.0
    if s in {"million", "mm"}:
        return amount * 1_000_000.0
    return amount


def parse_any_money(evidence: str) -> Optional[Tuple[float, str]]:
    """Parse first money amount from evidence. Returns (value_usd, "USD")."""
    candidates = []
    for m in MONEY_RE.finditer(evidence):
        dollar_sign = m.group(1)
        amt = float(m.group(2))
        scale = m.group(3)
        if dollar_sign or scale:
            usd = normalize_money_to_usd(amt, scale)
            score = 2 if scale else 1
            candidates.append((score, usd))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], "USD"


def parse_percent(evidence: str) -> Optional[float]:
    m = PCT_RE.search(evidence)
    return float(m.group(1)) if m else None


def parse_eps(evidence: str) -> Optional[float]:
    m = re.search(
        r"\b(diluted\s+)?eps\b.*?\$?\s*([0-9]+(?:\.[0-9]+)?)",
        evidence,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(2))
    m2 = EPS_RE.search(evidence)
    return float(m2.group(1)) if m2 else None


def parse_shares(evidence: str) -> Optional[float]:
    """Parse share count (e.g. 5.2 billion shares, 44.3m shares)."""
    m = SHARES_RE.search(evidence)
    if m:
        amt = float(m.group(1))
        scale = m.group(2)
        if scale:
            s = scale.lower()
            if s in {"billion", "bn"}:
                return amt * 1e9
            if s in {"million", "mm"}:
                return amt * 1e6
        return amt
    return None


def infer_unit_from_sec_metric(
    sec_metric: str,
    metrics_config: Dict[str, List[str]],
) -> str:
    """Infer expected unit for parsing. Returns USD, shares, percent, or USD/shares."""
    units = metrics_config.get(sec_metric, [])
    if "USD/shares" in units or "usd/shares" in [u.lower() for u in units]:
        return "USD/shares"
    if "shares" in units:
        return "shares"
    if "percent" in units or "pure" in units:
        return "percent"
    return "USD"


def parse_value_from_evidence(
    evidence: str,
    sec_metric: str,
    kind: str,
    metrics_config: Dict[str, List[str]],
) -> Optional[Tuple[float, str]]:
    """
    Parse numeric value from evidence based on metric type.
    Returns (value, unit) or None.
    """
    ev_lower = evidence.lower()

    # EPS metrics
    if "EarningsPerShare" in sec_metric or "eps" in ev_lower and "per share" in ev_lower:
        eps = parse_eps(evidence)
        if eps is not None:
            return eps, "USD/shares"

    # Shares metrics
    if "shares" in ev_lower or "EntityCommonStockSharesOutstanding" in sec_metric:
        sh = parse_shares(evidence)
        if sh is not None:
            return sh, "shares"

    # Percent (growth, margin)
    if kind in {"growth", "margin"}:
        p = parse_percent(evidence)
        if p is not None:
            return p, "percent"

    # USD (absolute)
    if kind == "absolute":
        parsed = parse_any_money(evidence)
        if parsed:
            return parsed
        p = parse_percent(evidence)
        if p is not None and "margin" in ev_lower:
            return p, "percent"

    return None


# -----------------------------
# LLM prompt
# -----------------------------


def build_prompt(transcript_chunk: str, metric_reference: str) -> str:
    return f"""
Return valid json only. You are extracting quantitative claims from an earnings call transcript.

For each claim that states a numeric value for a financial metric, output:
- sec_metric: MUST be one of the metric IDs below (format: taxonomy:factname)
- kind: "absolute" (single value like $X or Y shares) | "growth" (YoY/QoQ % change) | "margin" (% margin)
- frame: "YoY" | "QoQ" | null
- evidence_text: EXACT verbatim quote from the transcript

Rules:
- evidence_text must be copied exactly from the transcript.
- Only include claims where a numeric value is clearly stated.
- Map to the best-matching sec_metric from the list below.
- If unsure which metric, use the closest match or omit.

SEC metrics (format: sec_metric | label | units):
{metric_reference}

Transcript chunk:
\"\"\"{transcript_chunk}\"\"\"

Output JSON only:
{{
  "claims": [
    {{
      "sec_metric": "us-gaap:Revenues",
      "kind": "absolute",
      "frame": "YoY|QoQ|null",
      "evidence_text": "verbatim quote"
    }}
  ]
}}
""".strip()


# -----------------------------
# Chunking + caching
# -----------------------------


def split_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.\?!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(sentences: List[str], max_chars: int = 3500) -> List[str]:
    chunks = []
    buf = []
    size = 0
    for s in sentences:
        if size + len(s) + 1 > max_chars and buf:
            chunks.append(" ".join(buf))
            buf = []
            size = 0
        buf.append(s)
        size += len(s) + 1
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def cache_path_for(text: str, base: str, cache_dir: Path) -> Path:
    h = sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"sec_{base}_{h}.json"


# -----------------------------
# Classification & Reclassification (company-agnostic)
# -----------------------------
#
# STRATEGY: Prevent INACCURATE verdicts caused by claim-to-SEC-metric mismatches.
#
# 1. REASSIGN: When LLM misclassifies, map to correct sec_metric if one exists.
# 2. DROP: When no suitable SEC metric exists (market share, ASP, etc.).
# 3. DERIVED METRICS: Map "cash + marketable" -> derived:CashPlusMarketableSecurities
#    (verifier sums Cash + AvailableForSaleSecuritiesCurrent + Noncurrent).
# 4. SCOPE AWARENESS: Drop/don't verify claims that compare wrong scopes:
#    - Partial repurchases ("44.3m shares for $5b") vs SEC total quarterly
#    - Program-level ("authorization", "capital return program") vs quarterly spend
# 5. All patterns are evidence-based; no company-specific values.

# Patterns for repurchase scope (partial/cumulative = not verifiable vs quarterly total)
_REPURCHASE_PARTIAL_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:m|million|mm)\s*(?:shares?|AAPL|co\.)",
    re.IGNORECASE,
)  # "44.3m shares", "31.1m Co. shares"
_REPURCHASE_PROGRAM_RE = re.compile(
    r"authorization|capital return program|completed \$\d+\.?\d*b\s+of\s+\$\d+",
    re.IGNORECASE,
)  # "$222.9b of $300b", "authorization", "capital return program"


def reclassify_claim(
    sec_metric: str,
    kind: str,
    value: float,
    unit: str,
    raw_sentence: str,
    allowed_sec_metrics: Set[str],
) -> Optional[str]:
    """
    Reclassify misclassified claims to the correct sec_metric.
    Returns the correct sec_metric to use, or None if claim should be dropped.
    Company-agnostic: uses evidence patterns only (no company-specific values).
    """
    t = raw_sentence.lower()

    # ---- Drop: no SEC metric exists for these
    if "share" in t and "year ago" in t:
        return None  # Market share
    if "pc market" in t or "idc" in t:
        return None  # Third-party / industry
    if "visitors" in t:
        return None  # Visitor count
    if "year ago" in t and ("one-off" in t or "one off" in t or "benefit" in t):
        return None  # Prior year one-off
    if "asp" in t or "average selling" in t:
        return None  # ASP (unit-level KPI)
    if "GrossProfit" in sec_metric and unit == "percent":
        return None  # Gross margin % (derived, not in SEC)
    if "OperatingIncomeLoss" in sec_metric and unit == "percent":
        return None  # Operating margin %

    # ---- Repurchase scope: drop partial tranche or program-level (not verifiable vs quarterly total)
    def _is_repurchase_verifiable() -> bool:
        if _REPURCHASE_PROGRAM_RE.search(t):
            return False  # "authorization", "capital return program", cumulative
        if _REPURCHASE_PARTIAL_RE.search(t):
            return False  # "44.3m shares for $5b" = one tranche, not total quarterly
        # "$144b in share repurchases" = cumulative; "$35b authorization" = not a spend
        if re.search(r"\$\d+\.?\d*b\s+in\s+share\s+repurchase", t):
            return False
        return True

    # ---- Shares outstanding: repurchase $ -> PaymentsForRepurchaseOfCommonStock
    if ("CommonStockSharesOutstanding" in sec_metric or "EntityCommonStockSharesOutstanding" in sec_metric):
        if ("repurchase" in t or "for $" in t or "spent $" in t) or unit == "USD":
            if not _is_repurchase_verifiable():
                return None
            for c in ["us-gaap:PaymentsForRepurchaseOfCommonStock", "us-gaap:StockRepurchasedAndRetiredDuringPeriodValue"]:
                if c in allowed_sec_metrics:
                    return c
            return None

    # ---- OperatingIncomeLoss: OpEx or OI&E -> correct metric
    if "OperatingIncomeLoss" in sec_metric:
        if "opex" in t or "op ex" in t or "operating expense" in t:
            for c in ["us-gaap:OperatingExpenses"]:
                if c in allowed_sec_metrics:
                    return c
            return None
        if "oi&e" in t or "o i & e" in t or ("other income" in t and "expense" in t):
            for c in ["us-gaap:NonoperatingIncomeExpense", "us-gaap:OtherNonoperatingIncomeExpense"]:
                if c in allowed_sec_metrics:
                    return c
            return None

    # ---- Cash metrics: debt/repurchase/dividends/cash flow -> correct metric
    if "CashAndCashEquivalents" in sec_metric or "CashCashEquivalentsRestrictedCash" in sec_metric:
        # "Cash plus marketable securities" = Cash + AvailableForSaleSecuritiesCurrent (derived)
        if ("cash plus marketable" in t or "cash and marketable" in t
                or "marketable securities" in t and ("cash" in t or "end" in t)):
            if "derived:CashPlusMarketableSecurities" in allowed_sec_metrics:
                return "derived:CashPlusMarketableSecurities"
            return sec_metric  # fallback to cash-only if derived not available
        if "term debt" in t or "long term debt" in t or " in debt" in t:
            for c in ["us-gaap:DebtInstrumentCarryingAmount", "us-gaap:LongTermDebtNoncurrent"]:
                if c in allowed_sec_metrics:
                    return c
            return None
        if "commercial paper" in t or "paper outstanding" in t:
            if "us-gaap:CommercialPaper" in allowed_sec_metrics:
                return "us-gaap:CommercialPaper"
            return None
        if "dividends" in t and ("paid" in t or " in " in t):
            for c in ["us-gaap:PaymentsOfDividends", "us-gaap:PaymentsOfDividendsCommonStock", "us-gaap:Dividends"]:
                if c in allowed_sec_metrics:
                    return c
            return None
        if "repurchase" in t or "repurchases" in t or "capital return" in t or "completed $" in t:
            if not _is_repurchase_verifiable():
                return None
            for c in ["us-gaap:PaymentsForRepurchaseOfCommonStock", "us-gaap:StockRepurchasedAndRetiredDuringPeriodValue"]:
                if c in allowed_sec_metrics:
                    return c
            return None
        if "cash flow from operations" in t or "cash flow from operating" in t:
            if "us-gaap:NetCashProvidedByUsedInOperatingActivities" in allowed_sec_metrics:
                return "us-gaap:NetCashProvidedByUsedInOperatingActivities"
            return None

    # ---- Revenue metrics: cost claims -> cost metrics (common LLM confusion)
    if sec_metric in ("us-gaap:Revenues", "us-gaap:SalesRevenueNet", "us-gaap:SalesRevenueServicesNet"):
        if (
            "cost of revenue" in t or "cost of revenues" in t or "cor " in t
            or "total cost" in t and "revenue" in t
            or "other cost of revenue" in t
            or "cost of goods" in t
        ):
            for c in ["us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsAndServicesSold"]:
                if c in allowed_sec_metrics:
                    return c

    # ---- CapEx / capital expenditure -> PPE spend (not AccountsPayable)
    if "AccountsPayableCurrent" in sec_metric or "AccountsPayable" in sec_metric:
        if (
            "capex" in t or "cap ex" in t or "capital expenditure" in t
            or "capital spending" in t or "investments in production" in t
            or "facilities and data center" in t
        ):
            for c in [
                "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
                "us-gaap:PaymentsToAcquirePropertyPlantAndEquipmentAndOtherProductiveAssets",
            ]:
                if c in allowed_sec_metrics:
                    return c

    # ---- Total Revenues: segment-specific -> SalesRevenueServices when evidence says services
    if sec_metric in ("us-gaap:Revenues", "us-gaap:SalesRevenueNet"):
        if (
            "services" in t
            or "revenue almost" in t
            or ("revenue" in t and "record" in t and "total" not in t and "company" not in t)
        ):
            for c in ["us-gaap:SalesRevenueServicesNet", "us-gaap:SalesRevenueServicesGross"]:
                if c in allowed_sec_metrics:
                    return c

    # ---- No reclassification needed
    return sec_metric


# -----------------------------
# Extraction pipeline
# -----------------------------


def verify_evidence_in_text(evidence: str, transcript_text: str) -> bool:
    return evidence in transcript_text


def find_evidence_span(evidence: str, transcript_text: str) -> Optional[Tuple[int, int]]:
    idx = transcript_text.find(evidence)
    if idx == -1:
        return None
    return idx, idx + len(evidence)


def extract_claims_for_transcript(
    transcript_text: str,
    transcript_filename: str,
    client: GroqClient,
    all_metrics: List[Tuple[str, str, List[str]]],
    allowed_sec_metrics: Set[str],
    cache_dir: Path,
    max_chunk_chars: int = 2500,
    sleep_between_chunks: float = 2.0,
) -> List[SecClaim]:
    """
    Extract claims from a single transcript for SEC metrics.
    Uses priority metric subset in prompt to stay under token limits.
    """
    prompt_metrics = get_prompt_metrics(all_metrics)
    metric_reference = build_metric_reference(prompt_metrics)
    metrics_config = {m[0]: m[2] for m in all_metrics}

    cache_key = Path(transcript_filename).stem
    cache_file = cache_path_for(transcript_text, cache_key, cache_dir)

    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        sentences = split_sentences(transcript_text)
        chunks = chunk_sentences(sentences, max_chars=max_chunk_chars)
        all_raw: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            prompt = build_prompt(chunk, metric_reference)
            resp = client.generate_json(prompt)
            all_raw.extend(resp.get("claims", []))
            if i < len(chunks) - 1:
                time.sleep(sleep_between_chunks)

        payload = {"claims": all_raw}
        cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    raw_claims = payload.get("claims", [])
    out: List[SecClaim] = []
    for c in raw_claims:
        sec_metric = (c.get("sec_metric") or "").strip()
        kind = (c.get("kind") or "absolute").strip().lower()
        frame = c.get("frame")
        evidence_text = (c.get("evidence_text") or "").strip()

        if not sec_metric or not evidence_text:
            continue
        if sec_metric not in allowed_sec_metrics:
            continue
        if kind not in {"absolute", "growth", "margin"}:
            continue
        if frame == "null":
            frame = None
        if frame and frame not in {"YoY", "QoQ"}:
            frame = None

        if not verify_evidence_in_text(evidence_text, transcript_text):
            continue

        parsed = parse_value_from_evidence(
            evidence_text, sec_metric, kind, metrics_config
        )
        if parsed is None:
            continue

        value, unit = parsed

        # Reclassify misclassified claims to correct metric (or drop if no mapping)
        resolved_metric = reclassify_claim(
            sec_metric, kind, value, unit, evidence_text, allowed_sec_metrics
        )
        if resolved_metric is None:
            continue  # Drop: no valid SEC metric to map to

        span = find_evidence_span(evidence_text, transcript_text)
        start_idx, end_idx = span if span else (None, None)

        out.append(
            SecClaim(
                sec_metric=resolved_metric,
                kind=kind,
                value=value,
                unit=unit,
                frame=frame,
                raw_sentence=evidence_text,
                confidence="medium",
                evidence_start=start_idx,
                evidence_end=end_idx,
                flags=[],
            )
        )

    # Deduplicate
    seen = set()
    deduped: List[SecClaim] = []
    for cl in out:
        key = (cl.sec_metric, cl.kind, round(float(cl.value), 6), cl.raw_sentence[:100])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cl)

    return deduped


# -----------------------------
# Main
# -----------------------------


def _discover_transcript_files(
    transcripts_dir: Path,
    pattern: Optional[str] = None,
) -> List[str]:
    """Discover transcript filenames. If pattern given, glob; else use default Apple list."""
    if pattern:
        matches = sorted(transcripts_dir.glob(pattern))
        return [p.name for p in matches if p.is_file()]
    return list(TRANSCRIPT_FILES_2017)


def extract_claims_from_transcript(
    transcripts_dir: Path = DEFAULT_TRANSCRIPTS_DIR,
    sec_json_path: Path = DEFAULT_SEC_JSON_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    cache_dir: str = DEFAULT_CACHE_DIR,
    ticker: str = "AAPL",
    fiscal_year: Optional[int] = None,
    transcript_pattern: Optional[str] = None,
    fiscal_year_end: str = "sep",
) -> Dict[str, Any]:
    """
    Extract claims from transcripts for all SEC metrics.
    transcript_pattern: glob pattern to discover files (e.g., "*.txt", "*-AMZN.txt"); if None, uses default Apple list.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY must be set for claim extraction.")

    if not sec_json_path.exists():
        raise FileNotFoundError(f"SEC JSON not found: {sec_json_path}")

    metrics = load_sec_metrics_from_json(sec_json_path)
    allowed_sec_metrics = {m[0] for m in metrics} | DERIVED_SEC_METRICS

    transcript_files = _discover_transcript_files(Path(transcripts_dir), pattern=transcript_pattern)
    print(f"Loaded {len(metrics)} SEC metrics from {sec_json_path.name}")
    print(f"Processing {len(transcript_files)} transcripts from {transcripts_dir}")

    client = GroqClient(api_key=api_key)
    cache_dir_path = Path(cache_dir)

    result: Dict[str, Any] = {
        "ticker": ticker,
        "fiscal_year": fiscal_year or 2017,
        "source": "transcripts",
        "sec_metrics_source": str(sec_json_path.name),
        "quarters": {},
        "all_claims": [],
        "summary": {
            "total_claims": 0,
            "by_sec_metric": {},
            "by_quarter": {},
        },
    }

    for filename in transcript_files:
        transcript_path = transcripts_dir / filename
        if not transcript_path.exists():
            print(f"[SKIP] {filename} not found at {transcript_path}")
            result["quarters"][filename] = {
                "error": "file_not_found",
                "claims": [],
                "claim_count": 0,
            }
            continue

        text = transcript_path.read_text(encoding="utf-8", errors="ignore")
        meta = detect_transcript_meta(text, filename=filename, fiscal_year_end=fiscal_year_end)

        print(f"\n[{filename}] Extracting (mapped to {meta.fp})...")
        claims = extract_claims_for_transcript(
            transcript_text=text,
            transcript_filename=filename,
            client=client,
            all_metrics=metrics,
            allowed_sec_metrics=allowed_sec_metrics,
            cache_dir=cache_dir_path,
        )

        def claim_to_dict(c: SecClaim) -> Dict[str, Any]:
            return {
                "sec_metric": c.sec_metric,
                "kind": c.kind,
                "value": c.value,
                "unit": c.unit,
                "frame": c.frame,
                "raw_sentence": c.raw_sentence[:500],
                "confidence": c.confidence,
            }

        quarter_key = meta.fp if meta.fp != "UNKNOWN" else filename
        claims_dict = [claim_to_dict(c) for c in claims]

        result["quarters"][quarter_key] = {
            "transcript_file": filename,
            "fiscal_year": meta.fiscal_year,
            "fp": meta.fp,
            "call_date": meta.call_date,
            "claims": claims_dict,
            "claim_count": len(claims_dict),
        }
        result["all_claims"].extend(
            [{"_quarter": quarter_key, **claim_to_dict(c)} for c in claims]
        )
        result["summary"]["total_claims"] += len(claims)
        result["summary"]["by_quarter"][quarter_key] = (
            result["summary"]["by_quarter"].get(quarter_key, 0) + len(claims)
        )
        for c in claims:
            result["summary"]["by_sec_metric"][c.sec_metric] = (
                result["summary"]["by_sec_metric"].get(c.sec_metric, 0) + 1
            )

        print(f"      → {len(claims)} claims extracted")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[OK] Saved to {output_path}")

    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract claims from FY2017 transcripts for all SEC metrics"
    )
    parser.add_argument(
        "--transcripts-dir",
        default=str(DEFAULT_TRANSCRIPTS_DIR),
        help="Directory containing transcript .txt files",
    )
    parser.add_argument(
        "--sec-json",
        default=str(DEFAULT_SEC_JSON_PATH),
        help="Path to merged/filtered SEC JSON (metric source)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON path",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Cache directory for Groq responses",
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker symbol (for output metadata)",
    )
    parser.add_argument(
        "--fiscal-year",
        type=int,
        default=None,
        help="Fiscal year (inferred from transcripts if not set)",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help='Glob pattern to discover transcripts (e.g., "*.txt", "*-AMZN.txt"); if not set, uses default Apple list',
    )
    parser.add_argument(
        "--fiscal-year-end",
        choices=["sep", "dec"],
        default="sep",
        help="Fiscal year end: sep (Apple) or dec (Google, Microsoft)",
    )
    args = parser.parse_args()

    extract_claims_from_transcript(
        transcripts_dir=Path(args.transcripts_dir),
        sec_json_path=Path(args.sec_json),
        output_path=Path(args.output),
        cache_dir=args.cache_dir,
        ticker=args.ticker,
        fiscal_year=args.fiscal_year,
        transcript_pattern=args.pattern,
        fiscal_year_end=args.fiscal_year_end,
    )


if __name__ == "__main__":
    main()
