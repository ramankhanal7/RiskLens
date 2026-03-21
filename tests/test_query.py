"""
tests/test_query.py

Unit tests for src.query — verifies the QueryResult dataclass and search()
function without requiring the full corpus/models on disk.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query import QueryResult, _to_result, search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(source="news", ticker=None, headline=None, raw_text=None, url=""):
    doc = {
        "id": "doc-1",
        "source": source,
        "ticker": ticker or [],
        "raw_text": raw_text or "",
        "url": url,
        "metadata": {"headline": headline} if headline else {},
    }
    return {"score": 0.75, "doc": doc, "method": "hybrid"}


# ---------------------------------------------------------------------------
# QueryResult dataclass
# ---------------------------------------------------------------------------

def test_query_result_fields():
    r = QueryResult(score=0.9, tickers=["AAPL"], title="Apple selloff", url="https://example.com")
    assert r.score == 0.9
    assert r.tickers == ["AAPL"]
    assert r.title == "Apple selloff"
    assert r.url == "https://example.com"


def test_query_result_empty_tickers():
    r = QueryResult(score=0.5, tickers=[], title="No ticker", url="")
    assert r.tickers == []


# ---------------------------------------------------------------------------
# _to_result conversion
# ---------------------------------------------------------------------------

def test_to_result_news_uses_headline():
    raw = _make_raw(source="news", ticker=["TSLA"], headline="Tesla earnings beat", url="https://news.example.com")
    result = _to_result(raw)
    assert isinstance(result, QueryResult)
    assert result.score == 0.75
    assert result.tickers == ["TSLA"]
    assert result.title == "Tesla earnings beat"
    assert result.url == "https://news.example.com"


def test_to_result_reddit_truncates_text():
    long_text = "word " * 60  # 300 chars
    raw = _make_raw(source="reddit", ticker=["GME"], raw_text=long_text, url="https://reddit.com/r/wsb")
    result = _to_result(raw)
    assert result.tickers == ["GME"]
    assert result.title.endswith("…")
    assert len(result.title) <= 155  # 150 chars + "…"


def test_to_result_short_reddit_text_no_truncation():
    raw = _make_raw(source="reddit", raw_text="Short post", url="")
    result = _to_result(raw)
    assert result.title == "Short post"


def test_to_result_missing_ticker_defaults_to_empty():
    doc = {"id": "x", "source": "news", "raw_text": "", "url": "", "metadata": {}}
    result = _to_result({"score": 0.1, "doc": doc, "method": "hybrid"})
    assert result.tickers == []


def test_to_result_none_ticker_defaults_to_empty():
    doc = {"id": "x", "source": "news", "ticker": None, "raw_text": "", "url": "", "metadata": {}}
    result = _to_result({"score": 0.1, "doc": doc, "method": "hybrid"})
    assert result.tickers == []


# ---------------------------------------------------------------------------
# search() — mocked indices
# ---------------------------------------------------------------------------

_MOCK_HYBRID_RESULTS = [
    {
        "score": 0.82,
        "method": "hybrid",
        "doc": {
            "id": "doc-a",
            "source": "news",
            "ticker": ["NVDA"],
            "raw_text": "",
            "url": "https://example.com/nvda",
            "metadata": {"headline": "Nvidia chip export ban"},
        },
    },
    {
        "score": 0.61,
        "method": "hybrid",
        "doc": {
            "id": "doc-b",
            "source": "reddit",
            "ticker": [],
            "raw_text": "supply chain issues are hitting everyone hard",
            "url": "",
            "metadata": {},
        },
    },
]


def test_search_returns_query_results():
    with (
        patch("src.query._ensure_indices") as mock_indices,
        patch("src.query.hybrid_retrieve", return_value=_MOCK_HYBRID_RESULTS),
    ):
        mock_indices.return_value = (MagicMock(), MagicMock())
        results = search("semiconductor export restrictions")

    assert len(results) == 2
    assert all(isinstance(r, QueryResult) for r in results)


def test_search_result_values():
    with (
        patch("src.query._ensure_indices") as mock_indices,
        patch("src.query.hybrid_retrieve", return_value=_MOCK_HYBRID_RESULTS),
    ):
        mock_indices.return_value = (MagicMock(), MagicMock())
        results = search("chip ban")

    assert results[0].score == 0.82
    assert results[0].tickers == ["NVDA"]
    assert results[0].title == "Nvidia chip export ban"
    assert results[0].url == "https://example.com/nvda"

    assert results[1].score == 0.61
    assert results[1].tickers == []


def test_search_empty_query_returns_empty():
    """hybrid_retrieve returns [] for empty/meaningless queries."""
    with (
        patch("src.query._ensure_indices") as mock_indices,
        patch("src.query.hybrid_retrieve", return_value=[]),
    ):
        mock_indices.return_value = (MagicMock(), MagicMock())
        results = search("")

    assert results == []


def test_search_respects_top_k():
    with (
        patch("src.query._ensure_indices") as mock_indices,
        patch("src.query.hybrid_retrieve", return_value=_MOCK_HYBRID_RESULTS) as mock_hr,
    ):
        mock_indices.return_value = (MagicMock(), MagicMock())
        search("test", top_k=5)

    _, call_kwargs = mock_hr.call_args
    assert call_kwargs.get("top_k") == 5 or mock_hr.call_args[0][3] == 5


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [
        test_query_result_fields,
        test_query_result_empty_tickers,
        test_to_result_news_uses_headline,
        test_to_result_reddit_truncates_text,
        test_to_result_short_reddit_text_no_truncation,
        test_to_result_missing_ticker_defaults_to_empty,
        test_to_result_none_ticker_defaults_to_empty,
        test_search_returns_query_results,
        test_search_result_values,
        test_search_empty_query_returns_empty,
        test_search_respects_top_k,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()

    print(f"\n{passed}/{len(tests)} tests passed")
