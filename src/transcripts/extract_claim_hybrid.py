# src/transcripts/hybrid_claim_extractor.py
"""
Hybrid claim extraction (LLM + deterministic parsing).

Design:
1) LLM proposes *evidence spans* (exact quotes from transcript) and metadata:
   - metric (revenue_total, revenue_services, net_income, eps_diluted, gross_margin, guidance_revenue, guidance_gross_margin)
   - kind (absolute, growth, margin, guidance)
   - frame (YoY, QoQ, None)
   - evidence_text (must be verbatim from transcript)
2) We verify evidence_text is present in transcript.
3) We deterministically parse numeric value(s) and units from evidence_text.
4) We emit Claim objects compatible with your validator.

Uses Groq API (free tier) with Llama models.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.transcripts.extract_claims import Claim  # reuse your Claim dataclass


# -----------------------------
# Config
# -----------------------------

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# -----------------------------
# Deterministic parsers
# -----------------------------

MONEY_STRICT_RE = re.compile(
    r"(?:(\$)\s*)?([0-9]+(?:\.[0-9]+)?)\s*(billion|million|bn|mm)?",
    re.IGNORECASE,
)

PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
EPS_RE = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+share|EPS)\b", re.IGNORECASE)

ANCHORS = {
    "revenue_total": ["revenue", "net sales", "total net sales", "sales"],
    "revenue_services": ["services revenue", "services", "service revenue"],
    "net_income": ["net income", "net profit", "profit", "earnings"],
    "gross_margin": ["gross margin", "gross profit margin", "gm"],
    "eps_diluted": ["diluted eps", "earnings per share", "eps"],
    "guidance_revenue": ["we expect", "outlook", "guidance", "forecast", "between", "range"],
    "guidance_gross_margin": ["we expect", "outlook", "guidance", "forecast", "between", "range"],
}

BALANCE_SHEET_BLOCK = [
    "cash", "marketable securities", "securities", "assets", "liabilities", "debt", "leverage",
    "inventory", "receivables", "payables", "working capital", "free cash flow", "fcf"
]

OPERATING_KPI_BLOCK = [
    "users", "subscribers", "mau", "dau", "active users", "installed base", "devices",
    "units", "shipments", "downloads", "streams", "views", "engagement"
]

THIRD_PARTY_OR_COMP_BLOCK = [
    "according to", "report", "survey", "estimate", "third-party", "industry", "market share",
    "vs ", " versus", "compared to"
]

MULTI_PERIOD_BLOCK = [
    "year-to-date", "ytd", "first half", "second half", "six months", "nine months",
    "full year", "fiscal year", "ttm", "trailing twelve"
]

QUARTER_HINTS = ["for the quarter", "this quarter", "in the quarter", "quarter ended", "q1", "q2", "q3", "q4"]
YOY_HINTS = ["year over year", "yoy", "from last year", "year-ago", "year ago", "prior year"]
QOQ_HINTS = ["quarter over quarter", "qoq", "sequential", "last quarter", "prior quarter"]

GUIDANCE_HINTS = [
    "we expect", "we anticipate", "we forecast", "we project",
    "outlook", "guidance", "forecast", "expected", "expectations",
    "range", "between"
]

COMPANY_LEVEL_REVENUE_HINTS = [
    "total revenue", "net sales", "total net sales", "consolidated",
    "company revenue", "overall revenue", "group revenue",
    "revenue for the quarter", "revenues for the quarter"
]

SEGMENT_LEVEL_HINTS = [
    "segment", "division", "business", "product", "category", "region",
    "international", "domestic", "consumer", "enterprise"
]

ADJUSTMENT_HINTS = [
    "non-gaap", "adjusted", "excluding", "excl", "before special items",
    "one-time", "special items", "restructuring", "impairment",
    "constant currency", "fx-neutral", "currency-neutral"
]

INCOME_STATEMENT_METRICS = {
    "revenue_total", "revenue_services", "net_income", "gross_margin", "eps_diluted"
}


def _contains_any(t: str, words: list[str]) -> bool:
    return any(w in t for w in words)


def score_evidence_v2(
    evidence: str,
    metric: str,
    kind: str,
    frame: Optional[str],
) -> Tuple[int, List[str], bool]:
    """
    Returns: (score, flags, force_unverifiable)

    force_unverifiable=True means:
      - keep the claim for transparency
      - but avoid strict SEC validation (likely apples-to-oranges)
    """
    t = evidence.lower()
    score = 0
    flags: List[str] = []
    force_unverifiable = False

    def has_any(words: List[str]) -> bool:
        return any(w in t for w in words)

    has_money = ("$" in evidence) or any(w in t for w in ["billion", "million", "bn", "mm"])
    has_pct = "%" in evidence

    # 1) Metric anchor quality
    anchors = ANCHORS.get(metric, [])
    if anchors and has_any(anchors):
        score += 5
    elif metric in {"revenue_total", "net_income", "gross_margin", "eps_diluted"}:
        score -= 4
        flags.append("weak_metric_anchor")

    # 2) Kind-specific cues
    if kind == "absolute":
        if metric in {"revenue_total", "revenue_services", "net_income"}:
            if has_money:
                score += 2
            else:
                score -= 3
                flags.append("missing_money_signal")
        if metric == "eps_diluted":
            if ("eps" in t) or ("per share" in t):
                score += 2
            else:
                score -= 3
                flags.append("missing_eps_signal")

    if kind in {"margin", "growth"}:
        if has_pct:
            score += 2
        else:
            score -= 3
            flags.append("missing_percent_signal")

    # 3) Frame sanity for growth
    if kind == "growth":
        if frame == "YoY":
            if has_any(YOY_HINTS):
                score += 2
            else:
                score -= 2
                flags.append("weak_yoy_frame")
        elif frame == "QoQ":
            if has_any(QOQ_HINTS):
                score += 2
            else:
                score -= 2
                flags.append("weak_qoq_frame")
        else:
            score -= 3
            flags.append("missing_growth_frame")

    # 4) Quarter context helps for single-quarter reported numbers
    if has_any(QUARTER_HINTS):
        score += 1

    # 5) Categories to keep but force UNVERIFIABLE for strict SEC validation
    if has_any(BALANCE_SHEET_BLOCK):
        flags.append("balance_sheet_or_cashflow")
        if metric in INCOME_STATEMENT_METRICS:
            force_unverifiable = True
            score -= 6

    if has_any(OPERATING_KPI_BLOCK):
        flags.append("operating_kpi")
        if metric in INCOME_STATEMENT_METRICS:
            force_unverifiable = True
            score -= 6

    if has_any(THIRD_PARTY_OR_COMP_BLOCK):
        flags.append("third_party_or_comparison")
        force_unverifiable = True
        score -= 4

    if has_any(MULTI_PERIOD_BLOCK) and kind != "guidance":
        flags.append("multi_period_statement")
        force_unverifiable = True
        score -= 5

    if has_any(ADJUSTMENT_HINTS):
        flags.append("adjusted_or_non_gaap")
        force_unverifiable = True
        score -= 3

    if has_any(GUIDANCE_HINTS):
        flags.append("guidance_language")
        # Only force_unverifiable when evidence is primarily ABOUT future guidance (e.g. "we expect X"),
        # not when stating actuals that mention guidance in passing (e.g. "was $50.6B, within our guidance range")
        is_future_guidance = has_any(["we expect", "we anticipate", "we forecast", "between", "outlook"])
        has_past_actual = any(phrase in t for phrase in ["was $", "were $", "was ", "revenue for the quarter was", "reached ", "came in at"])
        if kind != "guidance" and not metric.startswith("guidance_") and is_future_guidance and not has_past_actual:
            force_unverifiable = True
            score -= 2

    # 6) revenue_total special handling
    if metric == "revenue_total":
        if has_any(SEGMENT_LEVEL_HINTS) and not has_any(COMPANY_LEVEL_REVENUE_HINTS):
            flags.append("possible_segment_or_region")
            force_unverifiable = True
            score -= 3
        elif has_any(COMPANY_LEVEL_REVENUE_HINTS):
            score += 2

    return score, flags, force_unverifiable


def score_to_confidence(score: int) -> str:
    if score >= 9:
        return "high"
    if score >= 5:
        return "medium"
    return "low"

def normalize_money_to_usd(amount: float, scale_word: Optional[str]) -> float:
    """Normalize money to plain USD."""
    if not scale_word:
        return amount
    s = scale_word.lower()
    if s in {"billion", "bn"}:
        return amount * 1_000_000_000.0
    if s in {"million", "mm"}:
        return amount * 1_000_000.0
    return amount


def parse_money_after_keyword(evidence: str, keyword: str) -> Optional[Tuple[float, str]]:
    """
    Robustly parse a money value that appears AFTER a keyword.

    Fixes the bug where sentences like:
      "revenue jumped 20% to $6 billion"
    incorrectly return 20 as dollars.

    Priority:
    1) Prefer "$" amounts (optionally with billion/million scale)
    2) Else prefer amounts with an explicit scale word (billion/million/bn/mm)
    3) Else return None (never treat bare numbers as money)
    """
    # Work only on the substring after the keyword to avoid unrelated earlier numbers.
    m_kw = re.search(re.escape(keyword), evidence, re.IGNORECASE)
    if not m_kw:
        return None
    tail = evidence[m_kw.end():]

    # 1) Strong pattern: "to $6 billion" / "to $6.0bn"
    m_to = re.search(
        r"\bto\s*\$\s*([0-9]+(?:\.[0-9]+)?)\s*(billion|million|bn|mm)?\b",
        tail,
        re.IGNORECASE,
    )
    if m_to:
        amt = float(m_to.group(1))
        scale = m_to.group(2)
        return normalize_money_to_usd(amt, scale), "USD"

    # 2) Any explicit $ amount after keyword
    m_dollar = re.search(
        r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(billion|million|bn|mm)?\b",
        tail,
        re.IGNORECASE,
    )
    if m_dollar:
        amt = float(m_dollar.group(1))
        scale = m_dollar.group(2)
        return normalize_money_to_usd(amt, scale), "USD"

    # 3) Amount with scale word (even if no $), e.g. "6 billion"
    #    NOTE: This will correctly pick "6 billion" and ignore "20" from "20%"
    m_scaled = re.search(
        r"\b([0-9]+(?:\.[0-9]+)?)\s*(billion|million|bn|mm)\b",
        tail,
        re.IGNORECASE,
    )
    if m_scaled:
        amt = float(m_scaled.group(1))
        scale = m_scaled.group(2)
        return normalize_money_to_usd(amt, scale), "USD"

    # 4) No safe money value found
    return None



def parse_any_money(evidence: str) -> Optional[Tuple[float, str]]:
    candidates = []
    for m in MONEY_STRICT_RE.finditer(evidence):
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
    m = re.search(r"\b(diluted\s+)?eps\b.*?\$?\s*([0-9]+(?:\.[0-9]+)?)", evidence, re.IGNORECASE)
    if m:
        return float(m.group(2))
    m2 = EPS_RE.search(evidence)
    return float(m2.group(1)) if m2 else None


# -----------------------------
# Groq client
# -----------------------------

class GroqClient:
    """Client using the Groq API (free tier with Llama models)."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: int = 60):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Export it or pass api_key explicitly.")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key, timeout=timeout)
        except ImportError:
            raise ImportError(
                "groq is required for hybrid extraction. Install with: pip install groq"
            )

    def generate_json(
        self,
        prompt: str,
        max_retries: int = 5,
        initial_backoff: float = 5.0,
        backoff_multiplier: float = 2.0,
    ) -> Dict[str, Any]:
        """Call Groq and parse JSON from response. Retries on rate limit (429)."""
        model_name = os.getenv("GROQ_MODEL") or self.model

        # Truncate very long prompts (Llama context ~128k tokens, ~4 chars/token)
        max_chars = 30000
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars] + "\n\n[Transcript truncated...]"

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content
                if not text:
                    raise RuntimeError("Groq returned empty response.")

                text = text.strip()
                # Strip code fences if present
                if text.startswith("```"):
                    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
                    text = re.sub(r"\n?```$", "", text).strip()

                return json.loads(text)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "limit" in err_str:
                    if attempt < max_retries - 1:
                        wait_time = initial_backoff * (backoff_multiplier**attempt)
                        time.sleep(wait_time)
                        continue
                raise


# -----------------------------
# Chunking + caching
# -----------------------------

def split_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(sentences: List[str], max_chars: int = 4000) -> List[str]:
    """Chunk transcript into pieces within token budget."""
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


def cache_path_for(text: str, cache_dir: Path) -> Path:
    h = sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"groq_claims_{h}.json"


# -----------------------------
# Prompt + schema
# -----------------------------

def build_prompt(transcript_chunk: str) -> str:
    return f"""
You are extracting verifiable quantitative claims from an earnings call transcript chunk.

Return ONLY valid JSON (no markdown, no commentary).

Rules:
- Each claim MUST include evidence_text that is an EXACT verbatim quote from the transcript chunk.
- Do NOT invent numbers.
- Focus ONLY on these metrics:
  - revenue_total (total company revenue)
  - revenue_services (services revenue)
  - net_income
  - eps_diluted
  - gross_margin
  - guidance_revenue (forward-looking revenue range)
  - guidance_gross_margin (forward-looking gross margin range)
- kind:
  - "absolute" for single quarter values ($ or EPS)
  - "growth" for percent change (YoY/QoQ)
  - "margin" for margin percent
  - "guidance" for forward-looking ranges
- frame:
  - "YoY" only if evidence indicates year-over-year
  - "QoQ" only if evidence indicates quarter-over-quarter
  - null otherwise

Transcript chunk:
\"\"\"{transcript_chunk}\"\"\"

Output JSON schema:
{{
  "claims": [
    {{
      "metric": "revenue_total|revenue_services|net_income|eps_diluted|gross_margin|guidance_revenue|guidance_gross_margin",
      "kind": "absolute|growth|margin|guidance",
      "frame": "YoY|QoQ|null",
      "evidence_text": "verbatim quote from transcript chunk"
    }}
  ]
}}
""".strip()


# -----------------------------
# Main extraction pipeline
# -----------------------------

@dataclass(frozen=True)
class ProposedClaim:
    metric: str
    kind: str
    frame: Optional[str]
    evidence_text: str
    flags: Tuple[str, ...] = ()
    force_unverifiable: bool = False


def verify_evidence_in_text(evidence: str, transcript_text: str) -> bool:
    return evidence in transcript_text


def is_valid_claim(claim: Claim) -> bool:
    """
    Drop claims that fail sanity checks (wrong metric, unit mismatch, unit-sales-as-revenue, etc.).
    Returns False if the claim should be dropped.
    """
    ev = claim.raw_sentence.lower()
    metric = claim.metric
    kind = claim.kind
    unit = claim.unit
    value = claim.value

    # 1) guidance_revenue must be USD (money guidance), not percent
    if metric == "guidance_revenue" and unit == "percent":
        return False
    # 2) guidance_gross_margin must be percent, not USD
    if metric == "guidance_gross_margin" and unit == "USD":
        return False

    # 3) Revenue/margin guidance: evidence must mention the right concept
    if metric == "guidance_revenue" and "revenue" not in ev and "billion" not in ev and "million" not in ev:
        return False
    if metric == "guidance_gross_margin" and "margin" not in ev and "gross" not in ev:
        return False

    # 4) Net income: evidence should mention net income, profit, or earnings (not just "revenue")
    if metric == "net_income" and kind == "absolute":
        if "net income" not in ev and "net profit" not in ev and "earnings" not in ev and "profit" not in ev:
            # "Revenue for the quarter was $50.6B" wrongly labeled as net_income
            if "revenue" in ev and "net" not in ev and "profit" not in ev and "earnings" not in ev:
                return False

    # 5) Unit sales mistaken for revenue: "sold X million iPhones" / "X million units"
    if metric in {"revenue_total", "revenue_services"} and kind == "absolute" and unit == "USD":
        if value < 1_000_000_000:  # Less than $1B is suspicious for Apple revenue
            if re.search(r"\bsold\b.*\b(million|bn)\b", ev) or re.search(r"\b\d+\.?\d*\s*million\s*(iphone|unit|device)s?", ev):
                return False
        # "greater China revenue was down 26%" → wrong if we got value=26, unit=USD
        if value < 1000 and ("down" in ev or "up" in ev) and "%" in ev:
            return False

    # 6) gross_margin: avoid E2R/expense-to-revenue (not gross margin)
    if metric == "gross_margin" and ("e2r" in ev or "expense" in ev and "revenue" in ev and "margin" not in ev):
        return False

    # 7) Growth-rate ranges mislabeled as revenue/margin guidance
    # "10% year-over-year increase... range of 7% to 9% up" = opex growth, not revenue/margin guidance
    if metric in {"guidance_revenue", "guidance_gross_margin"} and "increase" in ev and "year-over-year" in ev:
        if "revenue" not in ev and "margin" not in ev:
            return False

    # 8) revenue_total and net_income from same sentence with same value → drop net_income
    # (handled implicitly: we drop net_income when evidence doesn't mention net income; same sentence
    # "Revenue for the quarter was $50.6B" gets both - rule 4 catches net_income)

    return True


def find_evidence_span(evidence: str, transcript_text: str) -> Optional[Tuple[int, int]]:
    """Return (start, end) character indices of evidence in transcript, or None."""
    idx = transcript_text.find(evidence)
    if idx == -1:
        return None
    return idx, idx + len(evidence)


def proposed_to_claims(pc: ProposedClaim, transcript_text: str) -> List[Claim]:
    ev = pc.evidence_text.strip()
    t = ev.lower()

    span = find_evidence_span(ev, transcript_text)
    start_idx, end_idx = span if span else (None, None)

    # Score gate: only drop very low-confidence extractions (score < -5)
    # Relaxed from s < 0 to keep valid claims like "$50.6 billion", "$10.5 billion", "39.4%"
    s, score_flags, score_force_unverifiable = score_evidence_v2(ev, pc.metric, pc.kind, pc.frame)
    if s < -5:
        print("[DROP]", s, pc.metric, pc.kind, pc.frame, ev[:120])
        return []
    conf = score_to_confidence(s)

    # Keep claims, attach explainability flags, and force UNVERIFIABLE when needed.
    flags = list(getattr(pc, "flags", ()) or ())
    flags.extend(score_flags)
    flags = list(dict.fromkeys(flags))
    force_unverifiable = bool(getattr(pc, "force_unverifiable", False)) or score_force_unverifiable

    def make_claim(metric: str, kind: str, value: float, unit: str, frame: Optional[str], confidence: str) -> Claim:
        return Claim(
            metric=metric,
            kind=kind,
            value=value,
            unit=unit,
            frame=frame,
            raw_sentence=ev,
            confidence=confidence,
            evidence_start=start_idx,
            evidence_end=end_idx,
            flags=list(flags),
            force_unverifiable=force_unverifiable,
        )

    # Guidance handling
    if pc.kind == "guidance":
        if pc.metric not in {"guidance_revenue", "guidance_gross_margin"}:
            return []

        if pc.metric == "guidance_revenue":
            nums = []
            for m in MONEY_STRICT_RE.finditer(ev):
                if m.group(1) or m.group(3):
                    amt = float(m.group(2))
                    scale = m.group(3)
                    nums.append(normalize_money_to_usd(amt, scale))
            if len(nums) >= 2:
                lo, hi = nums[0], nums[1]
                return [
                    make_claim("guidance_revenue", "guidance", lo, "USD", None, conf),
                    make_claim("guidance_revenue", "guidance", hi, "USD", None, conf),
                ]
            return []

        if pc.metric == "guidance_gross_margin":
            pcts = [float(m.group(1)) for m in PCT_RE.finditer(ev)]
            if len(pcts) >= 2:
                lo, hi = pcts[0], pcts[1]
                return [
                    make_claim("guidance_gross_margin", "guidance", lo, "percent", None, conf),
                    make_claim("guidance_gross_margin", "guidance", hi, "percent", None, conf),
                ]
            return []

        return []

    # Margin
    if pc.kind == "margin":
        p = parse_percent(ev)
        if p is None:
            return []
        return [make_claim(pc.metric, "margin", p, "percent", None, conf)]

    # Growth
    if pc.kind == "growth":
        p = parse_percent(ev)
        if p is None:
            return []
        return [make_claim(pc.metric, "growth", p, "percent", pc.frame, conf)]

    # EPS
    if pc.metric == "eps_diluted":
        eps = parse_eps(ev)
        if eps is None:
            return []
        return [make_claim("eps_diluted", "absolute", eps, "USD/shares", None, conf)]

    # Net income
    if pc.metric == "net_income":
        parsed = parse_money_after_keyword(ev, "net income") or parse_any_money(ev)
        if not parsed:
            return []
        val, unit = parsed
        return [make_claim("net_income", "absolute", val, unit, None, conf)]

    # Revenue (absolute)
    if pc.metric in {"revenue_total", "revenue_services"}:
        # revenue_total should have company-level or quarter context; allow bare $ amounts from earnings context
        if pc.metric == "revenue_total" and not _contains_any(t, COMPANY_LEVEL_REVENUE_HINTS):
            if not _contains_any(t, QUARTER_HINTS) and not ("$" in ev and any(w in t for w in ["billion", "million", "bn", "mm"])):
                return []  # drop only if no revenue/quarter context AND no clear money amount

        parsed = parse_money_after_keyword(ev, "revenue") or parse_any_money(ev)
        if not parsed:
            return []
        val, unit = parsed
        return [make_claim(pc.metric, "absolute", val, unit, None, conf)]

    # Gross margin
    if pc.metric == "gross_margin":
        p = parse_percent(ev)
        if p is None:
            return []
        return [make_claim("gross_margin", "margin", p, "percent", None, conf)]

    return []




def extract_claims_hybrid(
    transcript_text: str,
    client: GroqClient,
    cache_dir: str = "data/cache/gemini",
    max_chunk_chars: int = 4000,
    sleep_between_calls: float = 2.0,
) -> List[Claim]:
    """Main entry point: transcript text -> Claim list."""
    print("[1/7] Starting hybrid claim extraction...")
    cache_dir_path = Path(cache_dir)
    cache_file = cache_path_for(transcript_text, cache_dir_path)

    if cache_file.exists():
        print("[2/7] Loading from cache...")
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        print(f"      Cached {len(payload.get('claims', []))} raw claims from LLM.")
    else:
        print("[2/7] Cache miss — calling Groq API...")
        sentences = split_sentences(transcript_text)
        chunks = chunk_sentences(sentences, max_chars=max_chunk_chars)
        print(f"      Split into {len(chunks)} chunk(s).")

        all_proposed: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            print(f"      Processing chunk {i + 1}/{len(chunks)}...")
            prompt = build_prompt(chunk)
            resp = client.generate_json(prompt)
            count = len(resp.get("claims", []))
            all_proposed.extend(resp.get("claims", []))
            print(f"         → Extracted {count} claims from chunk.")
            time.sleep(sleep_between_calls)

        payload = {"claims": all_proposed}
        cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"      Total: {len(all_proposed)} raw claims. Cached.")

    print("[3/7] Verifying evidence spans in transcript...")
    proposed_claims: List[ProposedClaim] = []
    for c in payload.get("claims", []):
        metric = c.get("metric")
        kind = c.get("kind")
        frame = c.get("frame")
        evidence_text = (c.get("evidence_text") or "").strip()

        if not metric or not kind or not evidence_text:
            continue
        if frame == "null":
            frame = None
        if not verify_evidence_in_text(evidence_text, transcript_text):
            continue

        proposed_claims.append(ProposedClaim(metric=metric, kind=kind, frame=frame, evidence_text=evidence_text))

    print(f"      Passed verification: {len(proposed_claims)} proposed claims.")
    print("[4/7] Parsing values and converting to Claim objects...")
    out: List[Claim] = []
    for pc in proposed_claims:
        out.extend(proposed_to_claims(pc, transcript_text))
    print(f"      Parsed into {len(out)} claims.")

    print("[5/7] Deduplicating...")
    seen = set()
    after_dedup: List[Claim] = []
    for cl in out:
        key = (cl.metric, cl.kind, round(float(cl.value), 6), cl.raw_sentence[:120])
        if key in seen:
            continue
        seen.add(key)
        after_dedup.append(cl)
    print(f"      Removed {len(out) - len(after_dedup)} duplicates.")

    print("[6/7] Validating (dropping bad extractions)...")
    uniq = [cl for cl in after_dedup if is_valid_claim(cl)]
    dropped = len(after_dedup) - len(uniq)
    if dropped > 0:
        print(f"      Dropped {dropped} invalid claims. Kept: {len(uniq)}.")
    else:
        print(f"      All {len(uniq)} claims passed validation.")
    print("[7/7] Done.")
    return uniq