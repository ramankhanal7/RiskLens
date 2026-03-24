import re
import json
import finnhub
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
finnhub_api_key = os.getenv("FINNHUB_API_KEY")
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SENTIMENT_LEXICON_PATH = DATA_DIR / "sentiment_lexicon.json"

def load_sentiment_lexicon(filepath: str | Path = SENTIMENT_LEXICON_PATH) -> tuple[set[str], set[str]]:
    """
    Loads sentiment_lexicon.json with shape:
        { "positive": [...], "negative": [...] }
    Returns (positive_words, negative_words) as sets for O(1) lookup.
    """
    with Path(filepath).open("r", encoding="utf-8") as f:
        lexicon = json.load(f)
    return set(w.lower() for w in lexicon["positive"]), \
           set(w.lower() for w in lexicon["negative"])


def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphabetic tokens."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def article_sentiment(article: dict, positive_words: set, negative_words: set) -> float:
    """
    Computes a sentiment score in [-1, 1] for a single article
    using its headline and summary.
    """
    text = article.get("headline", "") + " " + article.get("summary", "")
    tokens = tokenize(text)
    pos = sum(1 for t in tokens if t in positive_words)
    neg = sum(1 for t in tokens if t in negative_words)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def fetch_company_news(
    client: finnhub.Client,
    ticker: str,
    from_date: str,
    to_date: str,
) -> list[dict]:
    """
    Fetches news articles for a specific ticker using Finnhub's
    company_news endpoint.

    Parameters
    ----------
    client    : Finnhub client instance
    ticker    : Stock ticker symbol, e.g. "AAPL"
    from_date : Start date in YYYY-MM-DD format
    to_date   : End date in YYYY-MM-DD format
    """
    return client.company_news(ticker, _from=from_date, to=to_date)


def score_portfolio(
    tickers: list[str],
    from_date: str,
    to_date: str,
    lexicon_path: str | Path = SENTIMENT_LEXICON_PATH,
) -> dict[str, float]:
    """
    Fetches company-specific news from Finnhub for each ticker and
    returns a sentiment score in [-5, 5] per ticker.

    Parameters
    ----------
    tickers      : List of stock ticker symbols, e.g. ["AAPL", "MSFT"]
    from_date    : Start date in YYYY-MM-DD format, e.g. "2025-06-01"
    to_date      : End date in YYYY-MM-DD format, e.g. "2025-06-20"
    lexicon_path : Path to sentiment_lexicon.json

    Returns
    -------
    dict mapping ticker -> float score in [-5.0, 5.0]
      Positive = net positive news sentiment
      Negative = net negative news sentiment
      0.0      = neutral or no relevant news found
    """
    # company_news articles are pre-filtered to the ticker, so category
    # weight just reflects how directly market-relevant the article type is
    CATEGORY_WEIGHTS = {
        "company news": 1.0,
        "business": 1.0,
        "technology": 0.8,
        "top news": 0.6,
    }

    tickers = [t.upper() for t in tickers]
    positive_words, negative_words = load_sentiment_lexicon(lexicon_path)
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)

    results = {}

    for ticker in tickers:
        articles = fetch_company_news(finnhub_client, ticker, from_date, to_date)

        if not articles:
            results[ticker] = 0.0
            continue

        entries: list[tuple[float, float]] = []  # (sentiment, weight)
        for article in articles:
            sentiment = article_sentiment(article, positive_words, negative_words)
            category = article.get("category", "").lower()
            cat_weight = CATEGORY_WEIGHTS.get(category, 0.5)
            entries.append((sentiment, cat_weight))

        weighted_sum = sum(s * w for s, w in entries)
        total_weight = sum(w for _, w in entries)
        raw = weighted_sum / total_weight   # in [-1, 1]
        score = raw * 5                     # scale to [-5, 5]
        results[ticker] = round(max(-5.0, min(5.0, score)), 4)

    return results


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
    scores = score_portfolio(tickers, from_date="2025-01-01", to_date="2025-03-31")
    for ticker, score in scores.items():
        print(f"{ticker}: {score:+.4f}")