# src/transcripts/extract_claims.py
"""
Hybrid transcript claim extractor.

Flow:
1) Gemini proposes claim candidates with metric/kind/frame + evidence span.
2) Deterministic code verifies evidence text exists in transcript.
3) Deterministic parser extracts numeric value + unit from evidence text only.
4) Returns structured Claim objects for downstream SEC validation.

Environment:
- GEMINI_API_KEY must be set.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request


@dataclass(frozen=True)
class Claim:
    metric: str                 # revenue_total, revenue_services, net_income, eps_diluted, gross_margin
    kind: str                   # absolute | growth | margin
    value: float                # numeric value (for growth this is percent, e.g., 15 for 15%)
    unit: str                   # usd_millions, percent, usd_per_share
    frame: Optional[str] = None # YoY | QoQ | None
    raw_sentence: str = ""      # snippet for UI
    confidence: str = "medium"  # low/medium/high
    evidence_start: Optional[int] = None
    evidence_end: Optional[int] = None
    flags: List[str] = field(default_factory=list)
    force_unverifiable: bool = False


@dataclass(frozen=True)
class ProposedClaim:
    metric: str
    kind: str
    frame: Optional[str]
    evidence: str
    confidence: str
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None


MONEY_RE = re.compile(
    r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(billion|million|bn|mm)?",
    re.IGNORECASE,
)
PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
EPS_RE = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+share|EPS)\b", re.IGNORECASE)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL_CANDIDATES = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]

ALLOWED_METRICS = {"revenue_total", "revenue_services", "net_income", "eps_diluted", "gross_margin"}
ALLOWED_KINDS = {"absolute", "growth", "margin"}
ALLOWED_FRAMES = {"YoY", "QoQ", None}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}

METRIC_KEYWORDS = {
    "revenue_total": ["total revenue", "revenue", "net sales"],
    "revenue_services": ["services revenue", "services"],
    "net_income": ["net income", "profit"],
    "eps_diluted": ["diluted eps", "eps", "per share"],
    "gross_margin": ["gross margin"],
}

PROMPT = """
Extract quantitative claims from this earnings transcript.

Return JSON only: an array of objects with keys:
- metric: one of ["revenue_total", "revenue_services", "net_income", "eps_diluted", "gross_margin"]
- kind: one of ["absolute", "growth", "margin"]
- frame: "YoY" | "QoQ" | null
- evidence: exact quote from transcript text that supports this claim
- confidence: "low" | "medium" | "high"
- start_idx: integer start index of evidence in transcript, if known, else null
- end_idx: integer end index (exclusive) of evidence in transcript, if known, else null

Rules:
- Evidence must be copied exactly from transcript.
- Do not output values or units; only classify and cite evidence.
- Include all relevant claims for the allowed metrics.
- If no claims found, return [] only.
""".strip()

def normalize_money(amount: float, scale_word: Optional[str]) -> Tuple[float, str]:
    """Normalize money to USD millions."""
    if not scale_word:
        # If no scale word is present, treat as already in millions.
        return amount, "usd_millions"

    s = scale_word.lower()
    if s in {"billion", "bn"}:
        return amount * 1_000.0, "usd_millions"
    if s in {"million", "mm"}:
        return amount, "usd_millions"
    return amount, "usd_millions"


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def _list_generate_content_models(api_key: str) -> List[str]:
    req = request.Request(f"{GEMINI_API_BASE}/models?key={api_key}", method="GET")
    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    models = data.get("models") or []
    available: List[str] = []
    for item in models:
        name = item.get("name")
        methods = item.get("supportedGenerationMethods") or []
        if isinstance(name, str) and "generateContent" in methods:
            available.append(name)
    return available


def _choose_model_name(api_key: str) -> str:
    env_model = os.getenv("GEMINI_MODEL")
    if env_model:
        return env_model

    available = _list_generate_content_models(api_key)
    if available:
        available_short = [m.split("/")[-1] for m in available]
        for candidate in DEFAULT_MODEL_CANDIDATES:
            if candidate in available_short:
                return candidate
        return available_short[0]

    return DEFAULT_MODEL_CANDIDATES[0]


def _build_generate_url(model_name: str, api_key: str) -> str:
    normalized = model_name if model_name.startswith("models/") else f"models/{model_name}"
    return f"{GEMINI_API_BASE}/{normalized}:generateContent?key={api_key}"


def _post_generate_content(api_key: str, model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    req = request.Request(
        _build_generate_url(model_name, api_key),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_gemini(text: str) -> List[Dict[str, Any]]:
    api_key = 'AIzaSyDve9OQaXUgjcHqAH8pENWi1j6WLCNHscg'
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    payload = {
        "contents": [{"parts": [{"text": f"{PROMPT}\n\nTranscript:\n{text}"}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    model_name = _choose_model_name(api_key)

    try:
        data = _post_generate_content(api_key, model_name, payload)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if exc.code == 404 and not os.getenv("GEMINI_MODEL"):
            available = _list_generate_content_models(api_key)
            if available:
                fallback_model = available[0]
                try:
                    data = _post_generate_content(api_key, fallback_model, payload)
                except error.HTTPError as exc2:
                    detail2 = exc2.read().decode("utf-8", errors="ignore")
                    raise RuntimeError(
                        f"Gemini API HTTP error {exc2.code}: {detail2}"
                    ) from exc2
                except error.URLError as exc2:
                    raise RuntimeError(f"Gemini API network error: {exc2}") from exc2
            else:
                raise RuntimeError(
                    "Gemini model not found for generateContent and no fallback model available. "
                    "Set GEMINI_MODEL to a supported model for your account."
                ) from exc
        else:
            raise RuntimeError(f"Gemini API HTTP error {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Gemini API network error: {exc}") from exc

    candidates = data.get("candidates") or []
    if not candidates:
        return []

    parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
    if not parts:
        return []

    text_payload = parts[0].get("text", "[]")
    parsed = json.loads(_extract_json_text(text_payload))
    if not isinstance(parsed, list):
        return []
    return parsed


def _to_proposed(item: Dict[str, Any]) -> Optional[ProposedClaim]:
    metric = item.get("metric")
    kind = item.get("kind")
    frame = item.get("frame")
    evidence = item.get("evidence")
    confidence = item.get("confidence", "medium")
    start_idx = item.get("start_idx")
    end_idx = item.get("end_idx")

    if metric not in ALLOWED_METRICS or kind not in ALLOWED_KINDS:
        return None
    if frame not in ALLOWED_FRAMES:
        frame = None
    if confidence not in ALLOWED_CONFIDENCE:
        confidence = "medium"
    if not isinstance(evidence, str) or not evidence.strip():
        return None

    if not isinstance(start_idx, int):
        start_idx = None
    if not isinstance(end_idx, int):
        end_idx = None

    return ProposedClaim(
        metric=metric,
        kind=kind,
        frame=frame,
        evidence=evidence,
        confidence=confidence,
        start_idx=start_idx,
        end_idx=end_idx,
    )


def _verify_evidence_span(text: str, proposal: ProposedClaim) -> Optional[Tuple[int, int]]:
    evidence = proposal.evidence

    if proposal.start_idx is not None and proposal.end_idx is not None:
        if 0 <= proposal.start_idx < proposal.end_idx <= len(text):
            if text[proposal.start_idx:proposal.end_idx] == evidence:
                return proposal.start_idx, proposal.end_idx

    idx = text.find(evidence)
    if idx == -1:
        return None
    return idx, idx + len(evidence)


def _nearest_match(matches: List[re.Match[str]], sentence: str, metric: str) -> Optional[re.Match[str]]:
    if not matches:
        return None

    keywords = METRIC_KEYWORDS.get(metric, [])
    positions: List[int] = []
    low = sentence.lower()
    for keyword in keywords:
        start = low.find(keyword)
        if start != -1:
            positions.append(start)

    if not positions:
        return matches[0]

    def distance(match: re.Match[str]) -> int:
        center = (match.start() + match.end()) // 2
        return min(abs(center - pos) for pos in positions)

    return min(matches, key=distance)


def _parse_money_from_evidence(evidence: str, metric: str) -> Optional[Tuple[float, str]]:
    matches = list(MONEY_RE.finditer(evidence))
    if not matches:
        return None

    match = _nearest_match(matches, evidence, metric)
    if match is None:
        return None

    amount_raw = match.group(1).replace(",", "")
    scale = match.group(2)

    try:
        amount = float(amount_raw)
    except ValueError:
        return None

    # Guardrail: avoid treating percentages like 27.7 as dollars when no scale is present.
    if scale is None and "%" in evidence and amount < 1000:
        return None

    value, unit = normalize_money(amount, scale)
    return value, unit


def _parse_percent_from_evidence(evidence: str, metric: str) -> Optional[float]:
    matches = list(PCT_RE.finditer(evidence))
    if not matches:
        return None

    match = _nearest_match(matches, evidence, metric)
    if match is None:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_eps_from_evidence(evidence: str) -> Optional[float]:
    match = EPS_RE.search(evidence)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    money_matches = list(MONEY_RE.finditer(evidence))
    if not money_matches:
        return None
    amount_raw = money_matches[0].group(1).replace(",", "")
    try:
        return float(amount_raw)
    except ValueError:
        return None


def _proposal_to_claim(text: str, proposal: ProposedClaim) -> Optional[Claim]:
    span = _verify_evidence_span(text, proposal)
    if span is None:
        return None

    start_idx, end_idx = span
    evidence = proposal.evidence

    if proposal.kind == "absolute":
        if proposal.metric == "eps_diluted":
            eps = _parse_eps_from_evidence(evidence)
            if eps is None:
                return None
            return Claim(
                metric=proposal.metric,
                kind=proposal.kind,
                value=eps,
                unit="usd_per_share",
                frame=None,
                raw_sentence=evidence,
                confidence=proposal.confidence,
                evidence_start=start_idx,
                evidence_end=end_idx,
            )

        parsed = _parse_money_from_evidence(evidence, proposal.metric)
        if parsed is None:
            return None
        value, unit = parsed
        return Claim(
            metric=proposal.metric,
            kind=proposal.kind,
            value=value,
            unit=unit,
            frame=None,
            raw_sentence=evidence,
            confidence=proposal.confidence,
            evidence_start=start_idx,
            evidence_end=end_idx,
        )

    if proposal.kind in {"growth", "margin"}:
        pct = _parse_percent_from_evidence(evidence, proposal.metric)
        if pct is None:
            return None
        return Claim(
            metric=proposal.metric,
            kind=proposal.kind,
            value=pct,
            unit="percent",
            frame=proposal.frame,
            raw_sentence=evidence,
            confidence=proposal.confidence,
            evidence_start=start_idx,
            evidence_end=end_idx,
        )

    return None


def _dedupe(claims: List[Claim]) -> List[Claim]:
    seen = set()
    out: List[Claim] = []
    for claim in claims:
        key = (claim.metric, claim.kind, claim.frame, round(claim.value, 6), claim.raw_sentence[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(claim)
    return out

def extract_claims(text: str) -> List[Claim]:
    raw = _call_gemini(text)
    proposals: List[ProposedClaim] = []

    for item in raw:
        if not isinstance(item, dict):
            continue
        proposal = _to_proposed(item)
        if proposal is not None:
            proposals.append(proposal)

    claims: List[Claim] = []
    for proposal in proposals:
        claim = _proposal_to_claim(text, proposal)
        if claim is not None:
            claims.append(claim)

    return _dedupe(claims)