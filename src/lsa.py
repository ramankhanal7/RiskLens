"""
lsa.py

Builds an LSA (Latent Semantic Analysis) model over the unified corpus using
TruncatedSVD on a TF-IDF matrix. Exposes three retrieval functions:

    lsa_retrieve()    — pure LSA ranked results
    hybrid_retrieve() — weighted combination of TF-IDF and LSA scores

Usage:
    python -m src.lsa              # build, save models, run smoke test
    from src.lsa import build_lsa, lsa_retrieve, hybrid_retrieve
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
SVD_PATH = MODELS_DIR / "svd.pkl"
DOC_VECTORS_PATH = MODELS_DIR / "doc_vectors.npy"

N_COMPONENTS = 200  # 100–300 is standard for LSA


# ---------------------------------------------------------------------------
# Index dataclass
# ---------------------------------------------------------------------------
@dataclass
class LsaIndex:
    docs: list[dict]                  # ordered corpus records
    vectorizer: TfidfVectorizer       # fitted vectorizer (vocab up to 20k)
    svd: TruncatedSVD                 # fitted SVD model
    doc_vectors: np.ndarray           # L2-normalized (n_docs × N_COMPONENTS)


# ---------------------------------------------------------------------------
# Build + save
# ---------------------------------------------------------------------------
def build_lsa(corpus: list[dict], save: bool = True) -> LsaIndex:
    """
    Fit a TF-IDF vectorizer and TruncatedSVD over the corpus, L2-normalize
    the resulting document vectors, optionally save models to disk, and return
    an LsaIndex ready for querying.

    Uses the pre-stemmed `tokens` field from each corpus record (same token
    space as retrieval.py) joined into a whitespace string.

    Args:
        corpus: Loaded corpus list from src.corpus.load_corpus().
        save:   If True, pickle vectorizer + SVD and save doc_vectors.npy.
    """
    # Step 1 — build corpus text strings from pre-stemmed tokens
    corpus_texts = [" ".join(doc.get("tokens", [])) for doc in corpus]

    # Step 2 — fit TF-IDF matrix
    vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)
    print(f"TF-IDF matrix : {tfidf_matrix.shape}")

    # Step 3 — apply TruncatedSVD
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    doc_vectors = svd.fit_transform(tfidf_matrix)
    doc_vectors = normalize(doc_vectors)  # L2 normalize for cosine similarity
    print(f"Doc vectors   : {doc_vectors.shape}")
    print(f"Explained var : {svd.explained_variance_ratio_.sum():.3f}")

    # Step 4 — save to disk
    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        with VECTORIZER_PATH.open("wb") as f:
            pickle.dump(vectorizer, f)
        with SVD_PATH.open("wb") as f:
            pickle.dump(svd, f)
        np.save(DOC_VECTORS_PATH, doc_vectors)
        print(f"Models saved to {MODELS_DIR}/")

    return LsaIndex(
        docs=corpus,
        vectorizer=vectorizer,
        svd=svd,
        doc_vectors=doc_vectors,
    )


def load_lsa(corpus: list[dict]) -> LsaIndex:
    """
    Load pre-fitted models from disk instead of refitting.
    Call this at Flask startup after build_lsa() has been run at least once.
    """
    for path in (VECTORIZER_PATH, SVD_PATH, DOC_VECTORS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.lsa` to build models first."
            )

    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)
    with SVD_PATH.open("rb") as f:
        svd = pickle.load(f)
    doc_vectors = np.load(DOC_VECTORS_PATH)

    return LsaIndex(docs=corpus, vectorizer=vectorizer, svd=svd, doc_vectors=doc_vectors)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def _project_query(lsa_index: LsaIndex, preprocessed_text: str) -> np.ndarray:
    """Transform a pre-stemmed query string into the LSA latent space (L2-normalized)."""
    query_tfidf = lsa_index.vectorizer.transform([preprocessed_text])  # (1, vocab)
    query_vec = lsa_index.svd.transform(query_tfidf)                   # (1, N_COMPONENTS)
    query_vec = normalize(query_vec)
    return query_vec.squeeze()                                          # (N_COMPONENTS,)


def lsa_retrieve(
    lsa_index: LsaIndex,
    query_text: str,
    top_k: int = 20,
) -> list[dict]:
    """
    Retrieve and rank documents using LSA cosine similarity.

    Returns:
        [{"doc": {...}, "score": float, "method": "lsa"}, ...]
    """
    from src.retrieval import _preprocess
    tokens = _preprocess(query_text)
    if not tokens:
        return []
    query_vec = _project_query(lsa_index, " ".join(tokens))

    # Cosine similarity = dot product because both sides are L2-normalized
    scores = lsa_index.doc_vectors @ query_vec  # (n_docs,)

    top_idx = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "doc": lsa_index.docs[int(idx)],
            "score": round(float(scores[idx]), 4),
            "method": "lsa",
        }
        for idx in top_idx
        if scores[idx] > 0
    ]


def hybrid_retrieve(
    lsa_index: LsaIndex,
    tfidf_index,           # src.retrieval.Index — passed in, not imported at module level
    query_text: str,
    top_k: int = 20,
    tfidf_weight: float = 0.5,
    lsa_weight: float = 0.5,
) -> list[dict]:
    """
    Combine TF-IDF and LSA scores into a single ranked list.

    Both score arrays are normalized to [0, 1] before weighting so they are
    on the same scale. Documents scoring well in both methods receive a
    natural boost from the combined score.

    Args:
        lsa_index:     Fitted LsaIndex from build_lsa() / load_lsa().
        tfidf_index:   Fitted Index from src.retrieval.build_index().
        query_text:    Raw user query string.
        top_k:         Number of results to return.
        tfidf_weight:  Weight for TF-IDF component (default 0.5).
        lsa_weight:    Weight for LSA component (default 0.5).

    Returns:
        [{"doc": {...}, "score": float, "method": "hybrid"}, ...]
    """
    n_docs = len(lsa_index.docs)

    # --- TF-IDF scores (all documents, not just top-K) ---
    # Re-use the fitted vectorizer + matrix from tfidf_index directly
    # so we don't need to modify retrieval.py at all.
    from src.retrieval import _preprocess
    tokens = _preprocess(query_text)
    if not tokens:
        return []

    stemmed_query = " ".join(tokens)
    query_tfidf_vec = tfidf_index.vectorizer.transform([stemmed_query])
    tfidf_scores = cosine_similarity(query_tfidf_vec, tfidf_index.tfidf_matrix).flatten()

    # --- LSA scores (all documents) ---
    query_lsa_vec = _project_query(lsa_index, stemmed_query)
    lsa_scores = lsa_index.doc_vectors @ query_lsa_vec  # (n_docs,)

    # --- Normalize both to [0, 1] ---
    tfidf_max = tfidf_scores.max()
    lsa_max = lsa_scores.max()

    tfidf_norm = tfidf_scores / tfidf_max if tfidf_max > 0 else tfidf_scores
    lsa_norm = lsa_scores / lsa_max if lsa_max > 0 else lsa_scores

    # --- Combine ---
    combined = (tfidf_weight * tfidf_norm) + (lsa_weight * lsa_norm)

    top_idx = np.argsort(combined)[::-1][:top_k]

    return [
        {
            "doc": lsa_index.docs[int(idx)],
            "score": round(float(combined[idx]), 4),
            "method": "hybrid",
        }
        for idx in top_idx
        if combined[idx] > 0
    ]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.corpus import load_corpus
    from src.retrieval import build_index, query as tfidf_query

    print("Loading corpus...")
    corpus = load_corpus()
    print(f"  {len(corpus):,} documents")

    print("\nBuilding LSA index...")
    lsa_idx = build_lsa(corpus, save=True)

    print("\nBuilding TF-IDF index (for hybrid)...")
    tfidf_idx = build_index(corpus)

    test_queries = [
        "semiconductor export restrictions",
        "war in the middle east",
        "supply chain disruption",
        "interest rate inflation",
    ]

    for q in test_queries:
        tfidf_results = tfidf_query(tfidf_idx, q, top_k=5)
        lsa_results = lsa_retrieve(lsa_idx, q, top_k=5)
        hybrid_results = hybrid_retrieve(lsa_idx, tfidf_idx, q, top_k=5)

        print(f"\n{'='*70}")
        print(f"Query: '{q}'")

        print(f"\n  TF-IDF ({len(tfidf_results)} results):")
        for r in tfidf_results:
            print(f"    [{r['score']:.4f}] {r['doc']['ticker']}  {r['doc']['raw_text'][:60]!r}")

        print(f"\n  LSA ({len(lsa_results)} results):")
        for r in lsa_results:
            print(f"    [{r['score']:.4f}] {r['doc']['ticker']}  {r['doc']['raw_text'][:60]!r}")

        print(f"\n  Hybrid ({len(hybrid_results)} results):")
        for r in hybrid_results:
            print(f"    [{r['score']:.4f}] {r['doc']['ticker']}  {r['doc']['raw_text'][:60]!r}")

        # Show docs LSA found that TF-IDF missed
        tfidf_ids = {r["doc"]["id"] for r in tfidf_results}
        lsa_only = [r for r in lsa_results if r["doc"]["id"] not in tfidf_ids]
        if lsa_only:
            print(f"\n  LSA-only (not in TF-IDF top-5):")
            for r in lsa_only:
                print(f"    [{r['score']:.4f}] {r['doc']['ticker']}  {r['doc']['raw_text'][:60]!r}")
