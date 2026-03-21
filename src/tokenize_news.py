"""
tokenize_news.py

Reads market_news_*.csv files from the data directory (picks the most recent
one), tokenizes each article, extracts tickers from the `related` field, and
writes one JSON object per line to news_tokenized.jsonl.

Output schema per line:
{
  "id":       "news_<original_id>",
  "date":     "YYYY-MM-DD",
  "source":   "news",
  "raw_text": "<headline> <summary>",
  "tokens":   ["token1", "token2", ...],   # lowercased, stemmed, no stopwords/punct
  "ticker":   ["AAPL", ...],               # detected via extract_tickers; falls back to `related` field
  "url":      "https://...",
  "metadata": {
    "headline":       "...",
    "source_outlet":  "CNBC",
    "category":       "top news"
  }
}
"""

import csv
import json
import string
import sys
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

sys.path.insert(0, str(Path(__file__).parent))
from ticker_detection import extract_tickers

# ---------------------------------------------------------------------------
# Paths — pick the most recently modified market_news_*.csv
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_JSONL = DATA_DIR / "news_tokenized.jsonl"


def find_news_csv() -> Path:
    candidates = sorted(DATA_DIR.glob("market_news_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No market_news_*.csv found in {DATA_DIR}")
    return candidates[0]


# ---------------------------------------------------------------------------
# NLP setup (same pipeline as tokenize_reddit.py)
# ---------------------------------------------------------------------------
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def tokenize(text: str) -> list[str]:
    """Lowercase → word_tokenize → remove stopwords & non-alpha → stem."""
    text = text.lower()
    tokens = word_tokenize(text)
    result = []
    for token in tokens:
        token = token.translate(PUNCT_TABLE)
        if not token or not token.isalpha():
            continue
        if token in STOP_WORDS:
            continue
        result.append(STEMMER.stem(token))
    return result


def parse_tickers(related: str) -> list[str]:
    """Split comma-separated ticker string, strip whitespace, drop empty."""
    if not related or not related.strip():
        return []
    return [t.strip() for t in related.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    input_csv = find_news_csv()
    print(f"Reading: {input_csv}")

    written = 0
    skipped = 0

    with input_csv.open(newline="", encoding="utf-8") as infile, \
         OUTPUT_JSONL.open("w", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        for row in reader:
            headline = row.get("headline", "").strip()
            summary = row.get("summary", "").strip()
            raw_text = f"{headline} {summary}".strip()

            if not raw_text:
                skipped += 1
                continue

            date_str = row.get("date", "")
            if date_str:
                date_str = date_str[:10]  # keep YYYY-MM-DD only

            record = {
                "id":       f"news_{row['id']}",
                "date":     date_str,
                "source":   "news",
                "raw_text": raw_text,
                "tokens":   tokenize(raw_text),
                "ticker":   extract_tickers(
                            raw_text,
                            labeled_tickers=parse_tickers(row.get("related", "")),
                        ),
                "url":      row.get("url", ""),
                "metadata": {
                    "headline":      headline,
                    "source_outlet": row.get("source", ""),
                    "category":      row.get("category", ""),
                },
            }
            outfile.write(json.dumps(record) + "\n")
            written += 1

    print(f"Written : {written:,} records → {OUTPUT_JSONL}")
    print(f"Skipped : {skipped} (empty text)")


if __name__ == "__main__":
    main()
