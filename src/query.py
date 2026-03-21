"""
query.py

Public API for the RiskLens IR pipeline.

Typical frontend usage:
    from src.query import search

    results = search("war in the middle east")
    for r in results:
        print(r.score, r.tickers, r.title, r.url)

CLI usage (unchanged):
    python3 src/query.py
    python3 src/query.py "war in the middle east"
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Allow running as `python3 src/query.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus import load_corpus
from src.retrieval import build_index, query as tfidf_query, Index
from src.lsa import load_lsa, hybrid_retrieve, lsa_retrieve, LsaIndex

TOP_K = 10

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    score: float
    tickers: list[str]
    title: str
    url: str


# ---------------------------------------------------------------------------
# Lazy-loaded global indices (initialized once per process)
# ---------------------------------------------------------------------------

_tfidf_idx: Optional[Index] = None
_lsa_idx: Optional[LsaIndex] = None


def _ensure_indices() -> tuple[Index, LsaIndex]:
    """Load and cache indices on first call."""
    global _tfidf_idx, _lsa_idx
    if _tfidf_idx is None or _lsa_idx is None:
        corpus = load_corpus()
        _tfidf_idx = build_index(corpus)
        _lsa_idx = load_lsa(corpus)
    return _tfidf_idx, _lsa_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(user_query: str, top_k: int = TOP_K) -> list[QueryResult]:
    """
    Run the hybrid IR pipeline for a user query and return ranked results.

    Args:
        user_query: Free-form text query from the frontend.
        top_k:      Maximum number of results to return (default 10).

    Returns:
        List of QueryResult objects sorted by descending score.
    """
    tfidf_idx, lsa_idx = _ensure_indices()
    raw_results = hybrid_retrieve(lsa_idx, tfidf_idx, user_query, top_k=top_k)
    return [_to_result(r) for r in raw_results]


def _to_result(r: dict) -> QueryResult:
    doc = r["doc"]
    tickers: list[str] = doc.get("ticker") or []
    if doc.get("source") == "news":
        title = (doc.get("metadata") or {}).get("headline", doc.get("raw_text", ""))
    else:
        text = doc.get("raw_text", "")
        title = text[:150].rsplit(" ", 1)[0] + "…" if len(text) > 150 else text
    return QueryResult(
        score=round(float(r["score"]), 4),
        tickers=tickers,
        title=title,
        url=doc.get("url", ""),
    )


# ---------------------------------------------------------------------------
# CLI helper (kept for debugging / manual testing)
# ---------------------------------------------------------------------------

def _run_query_verbose(query_text: str, tfidf_idx: Index, lsa_idx: LsaIndex) -> None:
    """Print TF-IDF / LSA / Hybrid results side-by-side (CLI use only)."""
    print(f"\n{'='*70}")
    print(f"Query: '{query_text}'")
    print(f"{'='*70}")

    tfidf_results = tfidf_query(tfidf_idx, query_text, top_k=TOP_K)
    lsa_results = lsa_retrieve(lsa_idx, query_text, top_k=TOP_K)
    hybrid_results = hybrid_retrieve(lsa_idx, tfidf_idx, query_text, top_k=TOP_K)

    def fmt_doc(r):
        doc = r["doc"]
        ticker_str = ",".join(doc.get("ticker", [])) if doc.get("ticker") else "—"
        if doc.get("source") == "news":
            headline = (doc.get("metadata") or {}).get("headline", doc.get("raw_text", ""))
        else:
            headline = doc.get("raw_text", "")[:150].rsplit(" ", 1)[0] + "…"
        url = doc.get("url", "")
        return (
            f"    [{r['score']:.4f}] {doc['source']:6s}\n"
            f"             ticker: {ticker_str}\n"
            f"             title : {headline}\n"
            f"             url   : {url}"
        )

    def print_results(label, results):
        print(f"\n  [{label}] — {len(results)} results")
        if not results:
            print("    (no results)")
            return
        for r in results:
            print(fmt_doc(r))

    print_results("TF-IDF", tfidf_results)
    print_results("LSA   ", lsa_results)
    print_results("Hybrid", hybrid_results)

    tfidf_ids = {r["doc"]["id"] for r in tfidf_results}
    lsa_only = [r for r in lsa_results if r["doc"]["id"] not in tfidf_ids]
    if lsa_only:
        print(f"\n  [LSA-only — semantic matches TF-IDF missed]")
        for r in lsa_only:
            print(fmt_doc(r))


def main() -> None:
    print("Loading corpus...")
    corpus = load_corpus()
    print(f"  {len(corpus):,} documents")

    print("Building TF-IDF index...")
    tfidf_idx = build_index(corpus)

    print("Loading LSA models from disk...")
    lsa_idx = load_lsa(corpus)

    print("Ready.\n")

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        _run_query_verbose(q, tfidf_idx, lsa_idx)
        return

    print("Enter a query (or 'quit' to exit):")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        _run_query_verbose(q, tfidf_idx, lsa_idx)


if __name__ == "__main__":
    main()
