from pathlib import Path
from src.transcripts.transcript_mapping import detect_transcript_meta

p = Path("data/raw/transcripts/AAPL/2016-Apr-26-AAPL.txt")

print("File exists:", p.exists())
print("Absolute path:", p.resolve())

text = p.read_text(encoding="utf-8", errors="ignore")

print("Text length:", len(text))
print("First 200 chars:")
print(text[:200])

meta = detect_transcript_meta(text, filename=p.name)

print("Detected Transcript Meta:")
print(meta)
