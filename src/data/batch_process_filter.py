#!/usr/bin/env python3
"""
Batch filter SEC companyfacts JSON (e.g., AAPL.json) using multiple transcript files.

What it does:
- For each transcript, determines transcript_year from the transcript text (fallback: filename)
- Keeps only datapoints where:
    ✅ form in {10-Q, 10-K} (optionally includes 10-Q/A and 10-K/A)
    ✅ end.year ∈ {transcript_year, transcript_year - 1}
- Writes:
    1) One filtered JSON per transcript
    2) (Optional) One merged JSON across all transcripts found

Why filter by end.year?
- 'end' is the date the number is ABOUT (the “snapshot date” for balance sheet, or period end).
- This avoids pulling older comparative snapshots (e.g., 2015 end dates) that appear in 2016/2017 filings.

Usage examples:

# 1) Folder scan (recommended)
python batch_filter_companyfacts_endyear.py \
  --companyfacts /mnt/data/AAPL.json \
  --transcripts-dir /mnt/data \
  --pattern "2017-*-AAPL.txt" \
  --out-dir /mnt/data/filtered \
  --include-amendments \
  --write-merged

# 2) Explicit transcripts list
python batch_filter_companyfacts_endyear.py \
  --companyfacts /mnt/data/AAPL.json \
  --transcripts /mnt/data/2017-Jan-31-AAPL.txt /mnt/data/2017-May-02-AAPL.txt \
  --out-dir /mnt/data/filtered \
  --write-merged
"""

import argparse
import glob
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple


# -----------------------------
# Transcript year extraction
# -----------------------------
def extract_year_from_transcript_text(text: str) -> int | None:
    # Example: "AUGUST 01, 2017"
    m = re.search(
        r"\b(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(\d{1,2}),\s+(\d{4})\b",
        text.upper(),
    )
    if m:
        return int(m.group(3))

    # Fallback: any year near "DATE"
    m2 = re.search(r"\bDATE\b.*?\b(20\d{2}|19\d{2})\b", text.upper(), flags=re.DOTALL)
    if m2:
        return int(m2.group(1))

    return None


def extract_year_from_filename(path: str) -> int | None:
    base = os.path.basename(path)
    m = re.search(r"(19\d{2}|20\d{2})", base)
    return int(m.group(1)) if m else None


def get_transcript_year(transcript_path: str) -> int:
    with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    year = extract_year_from_transcript_text(text)
    if year is not None:
        return year

    year = extract_year_from_filename(transcript_path)
    if year is not None:
        return year

    raise ValueError(f"Could not determine transcript year from: {transcript_path}")


# -----------------------------
# Date helpers
# -----------------------------
def parse_iso_date_year(value: str) -> int | None:
    """Generic ISO date string -> year."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).year
    except Exception:
        return None


def end_year(point: Dict[str, Any]) -> int | None:
    """SEC companyfacts datapoint 'end' field -> year."""
    return parse_iso_date_year(point.get("end"))


# -----------------------------
# Filtering by end.year + form
# -----------------------------
def filter_companyfacts_by_endyear(
    companyfacts: Dict[str, Any],
    years: Set[int],
    forms: Tuple[str, ...],
    include_amendments: bool = False,
) -> Dict[str, Any]:
    allowed_forms = set(forms)
    if include_amendments:
        allowed_forms |= {f"{f}/A" for f in forms}  # 10-K/A, 10-Q/A

    out: Dict[str, Any] = {
        "cik": companyfacts.get("cik"),
        "entityName": companyfacts.get("entityName"),
        "filter": {
            "end_years": sorted(years),
            "forms": sorted(allowed_forms),
            "filter_basis": "end.year",
        },
        "facts": {},
    }

    for taxonomy, tax_facts in companyfacts.get("facts", {}).items():
        kept_tax: Dict[str, Any] = {}

        for fact_name, fact_obj in tax_facts.items():
            units = fact_obj.get("units", {})
            new_units: Dict[str, Any] = {}

            for unit, points in units.items():
                kept_points = []
                for p in points:
                    form = p.get("form")
                    if form not in allowed_forms:
                        continue

                    ey = end_year(p)
                    if ey not in years:
                        continue

                    kept_points.append(p)

                if kept_points:
                    new_units[unit] = kept_points

            if new_units:
                kept_fact = {k: v for k, v in fact_obj.items() if k != "units"}
                kept_fact["units"] = new_units
                kept_tax[fact_name] = kept_fact

        if kept_tax:
            out["facts"][taxonomy] = kept_tax

    return out


# -----------------------------
# Output helpers
# -----------------------------
def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def merge_filtered_companyfacts(filtered_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge filtered outputs by unioning fact datapoints.
    Keeps companyfacts-like structure; de-dupes by (accn, filed, fy, fp, end, form, val, frame).
    NOTE: This is NOT the same as "choose best" dedupe. It just avoids exact duplicates.
    """
    if not filtered_list:
        raise ValueError("No filtered JSON objects to merge.")

    base = {
        "cik": filtered_list[0].get("cik"),
        "entityName": filtered_list[0].get("entityName"),
        "filter": {
            "merged_from": len(filtered_list),
            "end_years": sorted({y for f in filtered_list for y in f.get("filter", {}).get("end_years", [])}),
            "forms": sorted({frm for f in filtered_list for frm in f.get("filter", {}).get("forms", [])}),
            "filter_basis": "end.year",
        },
        "facts": {},
    }

    def point_key(p: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            p.get("accn"),
            p.get("filed"),
            p.get("fy"),
            p.get("fp"),
            p.get("end"),
            p.get("form"),
            p.get("val"),
            p.get("frame"),
        )

    for fjson in filtered_list:
        for taxonomy, tax_facts in fjson.get("facts", {}).items():
            base["facts"].setdefault(taxonomy, {})
            for fact_name, fact_obj in tax_facts.items():
                base["facts"][taxonomy].setdefault(
                    fact_name, {k: v for k, v in fact_obj.items() if k != "units"}
                )
                base["facts"][taxonomy][fact_name].setdefault("units", {})

                for unit, points in fact_obj.get("units", {}).items():
                    base_points = base["facts"][taxonomy][fact_name]["units"].setdefault(unit, [])
                    seen = {point_key(p) for p in base_points}
                    for p in points:
                        k = point_key(p)
                        if k not in seen:
                            base_points.append(p)
                            seen.add(k)

    return base


def discover_transcripts(args) -> List[str]:
    transcripts: List[str] = []
    if args.transcripts:
        transcripts.extend(args.transcripts)

    if args.transcripts_dir:
        pattern = args.pattern or "*.txt"
        transcripts.extend(glob.glob(os.path.join(args.transcripts_dir, pattern)))

    # de-dupe, keep stable order
    seen = set()
    uniq = []
    for t in transcripts:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--companyfacts", required=True, help="Path to SEC companyfacts JSON (e.g., AAPL.json)")
    parser.add_argument("--transcripts-dir", help="Directory containing transcript .txt files")
    parser.add_argument("--pattern", default=None, help='Glob pattern inside transcripts-dir (e.g., "2017-*-AAPL.txt")')
    parser.add_argument("--transcripts", nargs="*", help="Explicit transcript paths (space-separated)")
    parser.add_argument("--out-dir", required=True, help="Output directory for filtered JSON files")

    parser.add_argument("--forms", default="10-K,10-Q", help="Comma-separated forms to keep (default: 10-K,10-Q)")
    parser.add_argument("--include-amendments", action="store_true", help="Include 10-K/A and 10-Q/A")

    parser.add_argument("--write-merged", action="store_true", help="Also write one merged JSON across all transcript years")
    parser.add_argument("--merged-name", default="AAPL_10K_10Q_merged_filtered_endyear.json", help="Filename for merged JSON")
    parser.add_argument(
        "--filter-year",
        type=int,
        default=None,
        help="Restrict to transcripts & SEC data for this year. Uses {year-2, year-1, year} for SEC (prior years needed for YoY, including Q4-of-prior-FY from Jan transcripts).",
    )

    args = parser.parse_args()

    transcripts = discover_transcripts(args)
    if not transcripts:
        raise SystemExit("No transcripts found. Use --transcripts or --transcripts-dir + --pattern.")

    os.makedirs(args.out_dir, exist_ok=True)

    forms = tuple([s.strip() for s in args.forms.split(",") if s.strip()])

    with open(args.companyfacts, "r", encoding="utf-8") as f:
        companyfacts = json.load(f)

    filtered_outputs: List[Dict[str, Any]] = []
    end_years_seen: Set[int] = set()

    for tpath in sorted(transcripts):
        transcript_year = get_transcript_year(tpath)
        if args.filter_year is not None:
            # Include year-2: Jan/Feb transcripts report Q4 of prior FY (e.g. Q4 FY2016);
            # YoY growth needs prior-prior year (2015 for Q4 FY2016 vs Q4 FY2015).
            years = {args.filter_year - 2, args.filter_year - 1, args.filter_year}
        else:
            years = {transcript_year, transcript_year - 1}
        end_years_seen |= years

        filtered = filter_companyfacts_by_endyear(
            companyfacts=companyfacts,
            years=years,
            forms=forms,
            include_amendments=args.include_amendments,
        )
        filtered_outputs.append(filtered)

        base = os.path.basename(tpath)
        base_no_ext = os.path.splitext(base)[0]
        yr_min, yr_max = min(years), max(years)
        out_name = sanitize_filename(f"{base_no_ext}_10K_10Q_ENDYEAR_{yr_min}_{yr_max}.json")
        out_path = os.path.join(args.out_dir, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2)

        print(f"[OK] {base} -> end_years={sorted(years)}  saved: {out_path}")

    if args.write_merged:
        merged = merge_filtered_companyfacts(filtered_outputs)
        merged["filter"]["note"] = "Union of per-transcript filters (de-duped exact duplicates only)."
        merged["filter"]["all_end_years_seen"] = sorted(end_years_seen)

        merged_path = os.path.join(args.out_dir, args.merged_name)
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

        print(f"[OK] Merged saved: {merged_path}")


if __name__ == "__main__":
    main()
