"""
ticker_validation.py

Loads valid US-listed ticker symbols from three data sources at import time
and exposes lookup helpers for the API layer.

Data sources:
    nasdaqlisted.txt        — ~5,400 NASDAQ symbols (pipe-delimited)
    company_tickers.json    — ~10,400 SEC-registered tickers (all exchanges)
    data/ticker_associations.json — ~490 S&P 500 tickers
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_NASDAQ_PATH = ROOT / "nasdaqlisted.txt"
_SEC_PATH = ROOT / "company_tickers.json"
_ASSOC_PATH = ROOT / "data" / "ticker_associations.json"

VALID_TICKERS: set[str] = set()
TICKER_TO_NAME: dict[str, str] = {}


def _load() -> None:
    # SEC company_tickers.json — most comprehensive, includes name
    if _SEC_PATH.exists():
        with _SEC_PATH.open(encoding="utf-8") as f:
            for entry in json.load(f).values():
                ticker = entry.get("ticker", "").upper()
                title = entry.get("title", "")
                if ticker:
                    VALID_TICKERS.add(ticker)
                    if ticker not in TICKER_TO_NAME and title:
                        TICKER_TO_NAME[ticker] = title.title()

    # nasdaqlisted.txt — pipe-delimited, skip header and test issues
    if _NASDAQ_PATH.exists():
        with _NASDAQ_PATH.open(encoding="utf-8") as f:
            next(f, None)  # skip header
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 4 and parts[3] == "N":  # Test Issue = N
                    symbol = parts[0].strip().upper()
                    name = parts[1].strip() if len(parts) >= 2 else ""
                    if symbol:
                        VALID_TICKERS.add(symbol)
                        if symbol not in TICKER_TO_NAME and name:
                            TICKER_TO_NAME[symbol] = name

    # ticker_associations.json — S&P 500 tickers with names
    if _ASSOC_PATH.exists():
        with _ASSOC_PATH.open(encoding="utf-8") as f:
            for ticker, data in json.load(f).items():
                ticker = ticker.upper()
                VALID_TICKERS.add(ticker)
                if ticker not in TICKER_TO_NAME:
                    TICKER_TO_NAME[ticker] = data.get("name", "")


_load()


def is_valid_ticker(ticker: str) -> bool:
    return ticker.upper() in VALID_TICKERS


def get_company_name(ticker: str) -> str | None:
    return TICKER_TO_NAME.get(ticker.upper()) or None
