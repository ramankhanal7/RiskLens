"""
test-query.py

Interactive query tester for the RiskLens IR pipeline.
Runs TF-IDF, LSA, and Hybrid retrieval side by side for any query.

Usage:
    python3 src/test-query.py
    python3 src/test-query.py "war in the middle east"
"""

import sys
from pathlib import Path

# Allow running as `python3 src/test-query.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus import load_corpus
from src.retrieval import build_index, query as tfidf_query
from src.lsa import load_lsa, hybrid_retrieve, lsa_retrieve

TOP_K = 10


def run_query(query_text: str, tfidf_idx, lsa_idx) -> None:
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

    # Highlight docs LSA found that TF-IDF missed
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

    # If a query was passed as a CLI argument, run it and exit
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        run_query(q, tfidf_idx, lsa_idx)
        return

    # Otherwise enter interactive loop
    print("Enter a query (or 'quit' to exit):")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        run_query(q, tfidf_idx, lsa_idx)


if __name__ == "__main__":
    main()
