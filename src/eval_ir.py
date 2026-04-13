"""
eval_ir.py

Evaluation script for the RiskLens IR pipeline. Provides:

  1. Precision@K and Recall@K comparison across TF-IDF, LSA, and Hybrid
     retrieval for 10 ground-truth query sets.
  2. Latent dimension analysis — top terms and explained variance for each
     SVD component, showing what semantic themes the dimensions capture.

Usage:
    python -m src.eval_ir
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus import load_corpus
from src.retrieval import build_index, query as tfidf_query
from src.lsa import build_lsa, load_lsa, lsa_retrieve, hybrid_retrieve

# ---------------------------------------------------------------------------
# Ground-truth query sets
# Each entry: (query_string, relevance_function applied to raw_text.lower())
# ---------------------------------------------------------------------------

GROUND_TRUTH: list[tuple[str, str, callable]] = [
    (
        "tariff trade war impact on stocks",
        "tariff + trade/stock",
        lambda t: "tariff" in t and ("trade" in t or "stock" in t),
    ),
    (
        "semiconductor chip export restrictions",
        "semiconductor OR chip+export",
        lambda t: "semiconductor" in t or ("chip" in t and "export" in t),
    ),
    (
        "nvidia gpu artificial intelligence",
        "nvidia/nvda mention",
        lambda t: "nvidia" in t or "NVDA" in t.upper(),
    ),
    (
        "boeing 737 safety concerns",
        "boeing OR 737",
        lambda t: "boeing" in t or "737" in t,
    ),
    (
        "federal reserve interest rate decision",
        "fed/federal reserve + rate/interest",
        lambda t: ("fed" in t or "federal reserve" in t)
        and ("rate" in t or "interest" in t),
    ),
    (
        "oil prices energy sector",
        "oil + price/energy/barrel",
        lambda t: "oil" in t
        and ("price" in t or "energy" in t or "barrel" in t),
    ),
    (
        "tesla electric vehicle sales",
        "tesla + car/vehicle/ev/sale",
        lambda t: "tesla" in t
        and ("car" in t or "vehicle" in t or "ev " in t or "sale" in t),
    ),
    (
        "apple iphone revenue",
        "apple + iphone/revenue/mac",
        lambda t: "apple" in t
        and ("iphone" in t or "revenue" in t or "mac" in t),
    ),
    (
        "banking crisis recession fears",
        "bank + crisis/recession/fail",
        lambda t: "bank" in t
        and ("crisis" in t or "recession" in t or "fail" in t),
    ),
    (
        "inflation consumer spending",
        "inflation + consumer/spend/price",
        lambda t: "inflation" in t
        and ("consumer" in t or "spend" in t or "price" in t),
    ),
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / len(relevant_ids)


# ---------------------------------------------------------------------------
# Latent dimension analysis
# ---------------------------------------------------------------------------

def inspect_latent_dimensions(lsa_index, n_dims: int = 10, n_terms: int = 10):
    """Print the top terms and explained variance for the first n_dims SVD components."""
    feature_names = lsa_index.vectorizer.get_feature_names_out()
    components = lsa_index.svd.components_
    variance = lsa_index.svd.explained_variance_ratio_

    print(f"\n{'='*70}")
    print(f"LATENT DIMENSION ANALYSIS  (top {n_dims} of {components.shape[0]} dimensions)")
    print(f"Total explained variance: {variance.sum():.4f}")
    print(f"{'='*70}")

    for i in range(min(n_dims, components.shape[0])):
        top_indices = np.argsort(np.abs(components[i]))[::-1][:n_terms]
        top_terms = [feature_names[j] for j in top_indices]
        weights = [components[i][j] for j in top_indices]

        term_strs = [f"{t} ({w:+.3f})" for t, w in zip(top_terms, weights)]

        print(f"\nDimension {i}  |  Explained variance: {variance[i]:.4f}")
        print(f"  Top terms: {', '.join(term_strs)}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    print("Loading corpus...")
    corpus = load_corpus()
    id_to_idx = {d["id"]: i for i, d in enumerate(corpus)}
    print(f"  {len(corpus):,} documents\n")

    print("Building TF-IDF index...")
    tfidf_idx = build_index(corpus)

    print("Loading LSA models...")
    try:
        lsa_idx = load_lsa(corpus)
    except FileNotFoundError:
        print("  LSA models not found, building from scratch...")
        lsa_idx = build_lsa(corpus, save=True)

    # -- Build ground-truth relevant sets --
    queries: list[tuple[str, str, set[str]]] = []
    for query_text, label, fn in GROUND_TRUTH:
        relevant = {
            d["id"] for d in corpus if fn(d.get("raw_text", "").lower())
        }
        queries.append((query_text, label, relevant))

    # -- Run retrieval for each method --
    methods = {
        "TF-IDF": lambda q, k: tfidf_query(tfidf_idx, q, top_k=k),
        "LSA": lambda q, k: lsa_retrieve(lsa_idx, q, top_k=k),
        "Hybrid": lambda q, k: hybrid_retrieve(lsa_idx, tfidf_idx, q, top_k=k),
    }

    K_VALUES = [5, 10]

    # Collect results: method -> metric_name -> list of per-query values
    results: dict[str, dict[str, list[float]]] = {
        m: {f"P@{k}": [] for k in K_VALUES} | {f"R@{k}": [] for k in K_VALUES}
        for m in methods
    }

    print(f"\n{'='*70}")
    print("RETRIEVAL COMPARISON  (10 queries)")
    print(f"{'='*70}")

    for query_text, label, relevant_ids in queries:
        print(f"\nQuery: \"{query_text}\"  ({len(relevant_ids)} relevant docs)")

        for method_name, retrieve_fn in methods.items():
            raw = retrieve_fn(query_text, max(K_VALUES))
            retrieved_ids = [r["doc"]["id"] for r in raw]

            row_parts = [f"  {method_name:8s}"]
            for k in K_VALUES:
                p = precision_at_k(retrieved_ids, relevant_ids, k)
                r = recall_at_k(retrieved_ids, relevant_ids, k)
                results[method_name][f"P@{k}"].append(p)
                results[method_name][f"R@{k}"].append(r)
                row_parts.append(f"P@{k}={p:.2f}  R@{k}={r:.4f}")
            print("  |  ".join(row_parts))

    # -- Summary table --
    print(f"\n{'='*70}")
    print("AVERAGE METRICS ACROSS ALL QUERIES")
    print(f"{'='*70}")

    header = f"{'Method':10s}"
    for k in K_VALUES:
        header += f"  {'P@'+str(k):>8s}  {'R@'+str(k):>8s}"
    print(header)
    print("-" * len(header))

    for method_name in methods:
        row = f"{method_name:10s}"
        for k in K_VALUES:
            avg_p = np.mean(results[method_name][f"P@{k}"])
            avg_r = np.mean(results[method_name][f"R@{k}"])
            row += f"  {avg_p:8.4f}  {avg_r:8.4f}"
        print(row)

    # -- Latent dimension analysis --
    inspect_latent_dimensions(lsa_idx, n_dims=10, n_terms=10)


if __name__ == "__main__":
    main()
