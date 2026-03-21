"""
corpus.py

Merges reddit_tokenized.jsonl and news_tokenized.jsonl into a single
corpus.jsonl file, and exposes load/lookup helpers for the IR pipeline.

Usage:
    python -m src.corpus          # rebuild corpus.jsonl
    from src.corpus import load_corpus, get_doc_by_id
"""

import json
from pathlib import Path

from src.ticker_detection import extract_tickers

DATA_DIR = Path(__file__).parent.parent / "data"
REDDIT_JSONL = DATA_DIR / "reddit_tokenized.jsonl"
NEWS_JSONL = DATA_DIR / "news_tokenized.jsonl"
CORPUS_JSONL = DATA_DIR / "corpus.jsonl"


def build_corpus() -> int:
    """
    Merge reddit_tokenized.jsonl and news_tokenized.jsonl into corpus.jsonl.
    Returns total number of documents written.
    """
    sources = [REDDIT_JSONL, NEWS_JSONL]
    for src in sources:
        if not src.exists():
            raise FileNotFoundError(
                f"{src} not found. Run data/tokenize_reddit.py and "
                "data/tokenize_news.py first."
            )

    total = 0
    with CORPUS_JSONL.open("w", encoding="utf-8") as out:
        for src in sources:
            with src.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        total += 1

    return total


def enrich_corpus() -> int:
    """
    Run ticker detection over every document in corpus.jsonl and rewrite
    the ticker field with the regex-detected result (falling back to the
    labeled field if detection yields nothing). Overwrites corpus.jsonl.

    Returns total number of documents written.
    """
    corpus = load_corpus()

    for doc in corpus:
        doc["ticker"] = extract_tickers(
            doc.get("raw_text", ""),
            labeled_tickers=doc.get("ticker", []),
        )

    with CORPUS_JSONL.open("w", encoding="utf-8") as out:
        for doc in corpus:
            out.write(json.dumps(doc) + "\n")

    return len(corpus)


def load_corpus() -> list[dict]:
    """Load corpus.jsonl into memory as a list of document dicts."""
    if not CORPUS_JSONL.exists():
        raise FileNotFoundError(
            f"{CORPUS_JSONL} not found. Call build_corpus() first."
        )
    docs = []
    with CORPUS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def get_doc_by_id(doc_id: str, corpus: list[dict] | None = None) -> dict | None:
    """
    Look up a single document by its id field.
    Pass in a pre-loaded corpus list to avoid re-reading from disk.
    """
    if corpus is not None:
        for doc in corpus:
            if doc["id"] == doc_id:
                return doc
        return None

    with CORPUS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if doc["id"] == doc_id:
                return doc
    return None


if __name__ == "__main__":
    n = build_corpus()
    print(f"corpus.jsonl written: {n:,} documents")

    print("Enriching ticker fields...")
    n = enrich_corpus()
    print(f"Ticker enrichment done: {n:,} documents updated")

    corpus = load_corpus()
    reddit_count = sum(1 for d in corpus if d["source"] == "reddit")
    news_count = sum(1 for d in corpus if d["source"] == "news")
    with_tickers = sum(1 for d in corpus if d["ticker"])
    print(f"  reddit      : {reddit_count:,}")
    print(f"  news        : {news_count:,}")
    print(f"  total       : {len(corpus):,}")
    print(f"  with tickers: {with_tickers:,}")

    sample = corpus[0]
    print(f"\nSample doc:")
    print(f"  id     : {sample['id']}")
    print(f"  date   : {sample['date']}")
    print(f"  source : {sample['source']}")
    print(f"  ticker : {sample['ticker']}")
    print(f"  tokens : {sample['tokens'][:10]}")
