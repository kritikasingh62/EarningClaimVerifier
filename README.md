Project ingest utilities

This workspace includes simple ingestion skeletons for:

- FMP transcripts: `src/ingest/fmp_transcripts.py`
- SEC companyfacts: `src/ingest/sec_facts.py`
- XBRL extraction helpers: `src/ingest/xbrl_extract.py`

Config

- Tickers list: [config/tickers.json](config/tickers.json)

Quick usage examples

1) Ingest transcripts for a ticker (requires FMP API key):

```bash
python src/ingest/fmp_transcripts.py AAPL --api-key YOUR_FMP_KEY
```

2) Download SEC companyfacts by CIK (supply `--user-agent` per SEC policy):

```bash
python src/ingest/sec_facts.py 0000320193 --user-agent "Your Name contact@domain.com"
```

3) Extract facts from a saved companyfacts JSON:

```bash
python src/ingest/xbrl_extract.py data/raw/sec_companyfacts/0000320193.json --out-parquet data/normalized/financial_facts.parquet
```

Dependencies

See `requirements.txt`.
