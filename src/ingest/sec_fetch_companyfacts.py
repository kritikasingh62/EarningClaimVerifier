"""
SEC Company Facts ingestion (ground truth).

- Fetches SEC companyfacts JSON once per company and caches it locally.
- Skips download if the raw file already exists unless --force is provided.
- Accepts either a ticker (e.g., AAPL) or a numeric CIK (e.g., 320193).
- Uses a separate ticker->CIK resolver backed by SEC company_tickers.json (also cached).

Outputs:
  data/raw/sec/companyfacts/{TICKER}.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import requests

RAW_DIR = Path("data/raw/sec")
COMPANYFACTS_DIR = RAW_DIR / "companyfacts"
TICKERS_CACHE_PATH = RAW_DIR / "company_tickers.json"


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sec_headers(user_agent: str) -> Dict[str, str]:
    # SEC strongly prefers a descriptive UA; gzip helps; Host is optional but harmless.
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


def fetch_json(url: str, headers: Dict[str, str], timeout: int = 30) -> Dict[str, Any]:
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# -----------------------------
# Ticker -> CIK resolver (separate)
# -----------------------------
def get_company_tickers_map(user_agent: str, force: bool = False) -> Dict[str, str]:
    """
    Returns dict: TICKER -> zero-padded 10-digit CIK string
    Cached at: data/raw/sec/company_tickers.json
    """
    ensure_dir(RAW_DIR)

    if TICKERS_CACHE_PATH.exists() and not force:
        data = load_json(TICKERS_CACHE_PATH)
    else:
        url = "https://www.sec.gov/files/company_tickers.json"
        data = fetch_json(url, headers=sec_headers(user_agent))
        save_json(TICKERS_CACHE_PATH, data)

    # SEC file is a dict keyed by row index. Normalize into ticker->cik
    out: Dict[str, str] = {}
    for _, row in data.items():
        t = str(row.get("ticker", "")).upper()
        cik_int = row.get("cik_str")
        if t and cik_int is not None:
            out[t] = str(cik_int).zfill(10)
    return out


def resolve_cik(identifier: str, user_agent: str, force_tickers_cache: bool = False) -> str:
    """
    identifier: either numeric CIK or ticker symbol
    returns: zero-padded 10-digit CIK string
    """
    ident = identifier.strip().upper()
    if ident.isdigit():
        return ident.zfill(10)

    tickers_map = get_company_tickers_map(user_agent=user_agent, force=force_tickers_cache)
    if ident not in tickers_map:
        raise ValueError(f"Could not resolve ticker to CIK: {ident}")
    return tickers_map[ident]


# -----------------------------
# Companyfacts fetch + cache
# -----------------------------
def companyfacts_path(tag: str) -> Path:
    # Store by ticker if ticker provided, else by CIK
    return COMPANYFACTS_DIR / f"{tag.upper()}.json"


def fetch_companyfacts(cik10: str, user_agent: str) -> Dict[str, Any]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    return fetch_json(url, headers=sec_headers(user_agent))


def download_companyfacts(
    identifier: str,
    user_agent: str,
    out_tag: Optional[str] = None,
    force: bool = False,
    force_tickers_cache: bool = False,
) -> Path:
    """
    identifier: ticker or numeric CIK
    out_tag: filename tag to save under (defaults to ticker if provided, else CIK)
    """
    ident = identifier.strip().upper()
    tag = (out_tag or ident).upper()

    out_path = companyfacts_path(tag)
    ensure_dir(out_path.parent)

    if out_path.exists() and not force:
        return out_path

    cik10 = resolve_cik(ident, user_agent=user_agent, force_tickers_cache=force_tickers_cache)
    payload = fetch_companyfacts(cik10, user_agent=user_agent)
    save_json(out_path, payload)
    return out_path


# -----------------------------
# CLI
# -----------------------------
def load_tickers_from_config(config_path: str) -> list[str]:
    obj = json.loads(Path(config_path).read_text(encoding="utf-8"))
    tickers = obj.get("tickers", [])
    return [str(t).upper().strip() for t in tickers]


def main():
    parser = argparse.ArgumentParser(description="Fetch & cache SEC companyfacts JSON (ticker or CIK).")
    parser.add_argument("--user-agent", required=True, help="Descriptive User-Agent per SEC policy (Name email).")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", help="Ticker symbol (e.g., AAPL)")
    group.add_argument("--cik", help="Numeric CIK (e.g., 320193)")
    group.add_argument("--config", help="Path to config/tickers.json to fetch multiple tickers")

    parser.add_argument("--force", action="store_true", help="Re-download companyfacts even if cached.")
    parser.add_argument(
        "--force-tickers-cache",
        action="store_true",
        help="Re-download SEC company_tickers.json mapping even if cached.",
    )
    parser.add_argument(
        "--out-tag",
        help="Filename tag to save as (e.g., GOOGL). If not set, uses ticker or CIK.",
    )

    args = parser.parse_args()

    if args.config:
        tickers = load_tickers_from_config(args.config)
        for t in tickers:
            path = download_companyfacts(
                identifier=t,
                user_agent=args.user_agent,
                out_tag=t,
                force=args.force,
                force_tickers_cache=args.force_tickers_cache,
            )
            print(f"{t}: {path}")
        return

    identifier = args.ticker or args.cik
    out_tag = args.out_tag or identifier
    path = download_companyfacts(
        identifier=identifier,
        user_agent=args.user_agent,
        out_tag=out_tag,
        force=args.force,
        force_tickers_cache=args.force_tickers_cache,
    )
    print(path)


if __name__ == "__main__":
    main()
