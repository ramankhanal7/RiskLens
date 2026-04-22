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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Allow running as `python3 src/query.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus import load_corpus
from src.retrieval import build_index, query as tfidf_query, Index, _preprocess
from src.lsa import load_lsa, hybrid_retrieve, lsa_retrieve, LsaIndex, explain_query

VALID_MODES = {"hybrid", "tfidf", "lsa"}

TOP_K = 10

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    score: float
    tickers: list[str]
    title: str
    url: str
    snippet: str
    date: str
    source: str


@dataclass
class DimensionInfo:
    index: int
    weight: float
    signed_weight: float
    top_terms: list[str]
    label: str


# Common stemmed-term to readable-word mapping for topic labels
_UNSTEM_MAP = {
    "nvda": "NVDA", "appl": "Apple", "aapl": "AAPL", "tsla": "Tesla",
    "googl": "Google", "amzn": "Amazon", "msft": "Microsoft", "amd": "AMD",
    "nvidia": "Nvidia", "intel": "Intel", "meta": "Meta",
    "tariff": "Tariffs", "trade": "Trade", "china": "China", "export": "Exports",
    "import": "Imports", "semiconductor": "Semiconductors", "chip": "Chips",
    "war": "War", "oil": "Oil", "gold": "Gold", "crypto": "Crypto",
    "bitcoin": "Bitcoin", "inflat": "Inflation", "rate": "Rates",
    "interest": "Interest", "fed": "Fed", "bank": "Banking", "market": "Markets",
    "stock": "Stocks", "earn": "Earnings", "revenu": "Revenue",
    "profit": "Profits", "growth": "Growth", "debt": "Debt", "bond": "Bonds",
    "yield": "Yields", "suppli": "Supply", "chain": "Chain",
    "disrupt": "Disruption", "tech": "Tech", "ai": "AI",
    "cybersecur": "Cybersecurity", "breach": "Data Breach",
    "billion": "Billions", "million": "Millions", "price": "Pricing",
    "invest": "Investment", "fund": "Funds", "etf": "ETFs",
    "regul": "Regulation", "polici": "Policy", "risk": "Risk",
    "energi": "Energy", "climat": "Climate", "green": "Green",
    "reddit": "Reddit", "post": "Posts", "comment": "Comments",
}


def _generate_topic_label(top_terms: list[str]) -> str:
    """Generate a human-readable topic label from the top stemmed terms."""
    readable = []
    for term in top_terms[:4]:
        t = term.lower()
        if t in _UNSTEM_MAP:
            readable.append(_UNSTEM_MAP[t])
        elif len(t) <= 5 and t.isupper():
            readable.append(t)  # likely a ticker
        else:
            readable.append(t.capitalize())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for r in readable:
        if r.lower() not in seen:
            seen.add(r.lower())
            unique.append(r)
    return " & ".join(unique[:3]) if unique else "General"


@dataclass
class SearchResponse:
    results: list[QueryResult]
    preprocessed_query: str
    query_dimensions: list[DimensionInfo] = field(default_factory=list)
    svd_n_components: int = 0
    svd_explained_variance: float = 0.0


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

def search(user_query: str, top_k: int = TOP_K, mode: str = "hybrid") -> SearchResponse:
    """
    Run the IR pipeline for a user query and return ranked results
    along with SVD explainability data.

    Args:
        user_query: Free-form text query from the frontend.
        top_k:      Maximum number of results to return (default 10).
        mode:       Retrieval method — "hybrid", "tfidf", or "lsa".

    Returns:
        SearchResponse with results and SVD explainability metadata.
    """
    if mode not in VALID_MODES:
        mode = "hybrid"
    tfidf_idx, lsa_idx = _ensure_indices()

    if mode == "tfidf":
        raw_results = tfidf_query(tfidf_idx, user_query, top_k=top_k)
    elif mode == "lsa":
        raw_results = lsa_retrieve(lsa_idx, user_query, top_k=top_k)
    else:
        raw_results = hybrid_retrieve(lsa_idx, tfidf_idx, user_query, top_k=top_k)

    # Preprocess query and get SVD explainability (only for modes that use SVD)
    tokens = _preprocess(user_query)
    preprocessed = " ".join(tokens)

    query_dims: list[DimensionInfo] = []
    svd_n = 0
    svd_var = 0.0

    if tokens and mode in ("hybrid", "lsa"):
        explanation = explain_query(lsa_idx, preprocessed, top_k_dims=5)
        query_dims = [
            DimensionInfo(
                index=d["index"],
                weight=d["weight"],
                signed_weight=d["signed_weight"],
                top_terms=d["top_terms"],
                label=_generate_topic_label(d["top_terms"]),
            )
            for d in explanation["active_dimensions"]
        ]
        svd_n = explanation["svd_info"]["n_components"]
        svd_var = explanation["svd_info"]["explained_variance"]

    return SearchResponse(
        results=[_to_result(r) for r in raw_results],
        preprocessed_query=preprocessed,
        query_dimensions=query_dims,
        svd_n_components=svd_n,
        svd_explained_variance=svd_var,
    )


def _to_result(r: dict) -> QueryResult:
    doc = r["doc"]
    tickers: list[str] = doc.get("ticker") or []
    raw = doc.get("raw_text", "")
    if doc.get("source") == "news":
        title = (doc.get("metadata") or {}).get("headline", raw)
    else:
        title = raw[:150].rsplit(" ", 1)[0] + "…" if len(raw) > 150 else raw

    snippet = raw[:200].rsplit(" ", 1)[0] + "…" if len(raw) > 200 else raw
    if snippet == title:
        snippet = ""

    return QueryResult(
        score=round(float(r["score"]), 4),
        tickers=tickers,
        title=title,
        url=doc.get("url", ""),
        snippet=snippet,
        date=doc.get("date", ""),
        source=doc.get("source", ""),
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
