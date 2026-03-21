"""
retrieval.py

Builds an inverted index and TF-IDF scorer over the unified corpus, and
exposes a query function for ranked document retrieval.

Usage:
    python -m src.retrieval               # smoke test
    from src.retrieval import build_index, query
"""

import string
from collections import defaultdict
from dataclasses import dataclass, field

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# NLP setup — same pipeline as tokenize_*.py so query and docs share a space
# ---------------------------------------------------------------------------
_STEMMER = PorterStemmer()
_STOP_WORDS = set(stopwords.words("english"))
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _preprocess(text: str) -> list[str]:
    """Lowercase → word_tokenize → remove stopwords & non-alpha → stem."""
    tokens = []
    for token in word_tokenize(text.lower()):
        token = token.translate(_PUNCT_TABLE)
        if not token or not token.isalpha():
            continue
        if token in _STOP_WORDS:
            continue
        tokens.append(_STEMMER.stem(token))
    return tokens


# ---------------------------------------------------------------------------
# Index dataclass
# ---------------------------------------------------------------------------
@dataclass
class Index:
    docs: list[dict]
    inverted_index: dict[str, list[int]]          # token → [doc indices]
    vectorizer: TfidfVectorizer
    tfidf_matrix: object                           # scipy sparse (n_docs × vocab)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_index(corpus: list[dict]) -> Index:
    """
    Build an inverted index and TF-IDF matrix from a loaded corpus.

    Each document's `tokens` list (pre-stemmed) is joined into a whitespace
    string and fed to TfidfVectorizer with analyzer='word' so no further
    tokenization is applied — preserving the stemming done at ingest time.
    """
    # --- inverted index ---
    inv: dict[str, list[int]] = defaultdict(list)
    for idx, doc in enumerate(corpus):
        seen = set()
        for token in doc.get("tokens", []):
            if token not in seen:
                inv[token].append(idx)
                seen.add(token)

    # --- TF-IDF matrix ---
    # Join pre-stemmed tokens so the vectorizer does a simple whitespace split
    token_strings = [" ".join(doc.get("tokens", [])) for doc in corpus]

    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=lambda x: x.split(),   # tokens already clean
        token_pattern=None,
        sublinear_tf=True,                # log(1 + tf) dampens high-freq terms
        min_df=2,                         # ignore tokens appearing in only 1 doc
    )
    tfidf_matrix = vectorizer.fit_transform(token_strings)

    return Index(
        docs=corpus,
        inverted_index=dict(inv),
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def query(index: Index, query_text: str, top_k: int = 20) -> list[dict]:
    """
    Retrieve and rank documents for a free-form query.

    Steps:
      1. Preprocess query with the same stem/stopword pipeline.
      2. Transform to TF-IDF vector via the fitted vectorizer.
      3. Compute cosine similarity against all document vectors.
      4. Return top-K results (score > 0) sorted descending.

    Returns a list of dicts:
      [{"doc": {...corpus record...}, "score": float}, ...]
    """
    tokens = _preprocess(query_text)
    if not tokens:
        return []

    query_str = " ".join(tokens)
    query_vec = index.vectorizer.transform([query_str])

    sims = cosine_similarity(query_vec, index.tfidf_matrix).flatten()

    # Collect (score, doc_idx) for docs with non-zero similarity
    ranked = sorted(
        ((float(sims[i]), i) for i in range(len(sims)) if sims[i] > 0),
        key=lambda x: x[0],
        reverse=True,
    )[:top_k]

    return [{"doc": index.docs[i], "score": score} for score, i in ranked]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.corpus import load_corpus

    print("Loading corpus...")
    corpus = load_corpus()
    print(f"  {len(corpus):,} documents")

    print("Building index...")
    idx = build_index(corpus)
    vocab_size = len(idx.vectorizer.vocabulary_)
    print(f"  vocab size : {vocab_size:,}")
    print(f"  matrix     : {idx.tfidf_matrix.shape}")

    test_queries = [
        "semiconductor export restrictions",
        "supply chain disruption",
        "interest rate inflation",
        "cyberattack data breach",
    ]

    for q in test_queries:
        results = query(idx, q, top_k=5)
        print(f"\nQuery: '{q}'  ({len(results)} results)")
        for r in results:
            doc = r["doc"]
            print(
                f"  [{r['score']:.4f}] {doc['id']:30s} "
                f"tickers={doc['ticker']}  "
                f"text={doc['raw_text'][:60]!r}"
            )
