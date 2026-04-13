"""
ticker_detection.py

Extracts S&P 500 ticker symbols from document text using a four-pass strategy:

  Pass 1 — $TICKER pattern  (e.g. "$NVDA", "$AAPL")
  Pass 2 — bare uppercase words against S&P 500 whitelist  (e.g. "AMD", "TSLA")
  Pass 3 — company name substring matching  (e.g. "Nvidia", "Super Micro")
  Pass 3b — ticker_association.json keyword matching  (e.g. "737 max" → BA,
             "jensen huang" → NVDA, "f-35" → LMT)

Keywords shared by more than MAX_KEYWORD_TICKERS tickers are considered too
generic and skipped to avoid false positives (e.g. "natural gas" alone should
not tag every energy company).

If none of the passes yield results, falls back to the pre-labeled ticker field
already stored in the corpus record.

Exposed API:
    from src.ticker_detection import extract_tickers, enrich_corpus_tickers
"""

import json
import re
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# S&P 500 ticker whitelist — union of hardcoded list + ticker_association.json
# ---------------------------------------------------------------------------
SP500_TICKERS: set[str] = {
    "MMM", "ABBV", "ACN", "ADBE", "AMD", "A", "APD", "ABNB", "ARE", "ALGN",
    "GOOGL", "MO", "AMZN", "AEP", "AXP", "AIG", "AWK", "AMGN", "APO", "AAPL",
    "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ADSK", "ADP",
    "AZO", "BKR", "BAC", "BRK", "BBY", "BIIB", "BLK", "BA", "BKNG", "BSX",
    "AVGO", "BR", "BLDR", "CDNS", "COF", "CARR", "CBOE", "CNP", "SCHW", "CMG",
    "CSCO", "C", "CFG", "CME", "KO", "CEG", "COO", "COST", "CRWD", "CMI",
    "CVS", "DHI", "DRI", "DE", "DXCM", "DG", "DLTR", "DASH", "DOW", "EMN",
    "EA", "EFX", "EXE", "EXR", "FDS", "FRT", "FDX", "FIS", "FE", "F", "GD",
    "GM", "GPC", "GPN", "GDDY", "GS", "HD", "HON", "IBM", "ITW", "INTC",
    "ICE", "INTU", "IVZ", "JKHY", "J", "JNJ", "JPM", "KHC", "KR", "LVS",
    "LLY", "LYV", "LMT", "LOW", "MTB", "MCD", "META", "MSFT", "MRNA", "TAP",
    "MPWR", "MS", "NDAQ", "NTAP", "NFLX", "NEM", "NCLH", "NVDA", "ON",
    "PLTR", "PYPL", "PEP", "PFE", "PNC", "PFG", "PG", "PEG", "RJF", "O",
    "RF", "SPGI", "CRM", "NOW", "SPG", "SNA", "SWK", "SBUX", "STT", "TROW",
    "TGT", "TSLA", "TKO", "TRV", "UBER", "UAL", "UPS", "UNH", "UHS", "VMC",
    "WRB", "GWW", "WMT", "WBD", "WM", "WFC",
}

# ---------------------------------------------------------------------------
# Company name → ticker  (for Pass 3 substring matching)
# Ordered longest-first so "super micro" matches before "micro"
# ---------------------------------------------------------------------------
_COMPANY_NAME_MAP_RAW: list[tuple[str, str]] = [
    ("super micro",                  "SMCI"),
    ("supermicro",                   "SMCI"),
    ("jpmorgan chase",               "JPM"),
    ("jpmorgan",                     "JPM"),
    ("goldman sachs",                "GS"),
    ("morgan stanley",               "MS"),
    ("bank of america",              "BAC"),
    ("wells fargo",                  "WFC"),
    ("american express",             "AXP"),
    ("american international group", "AIG"),
    ("american electric power",      "AEP"),
    ("american water works",         "AWK"),
    ("apollo global",                "APO"),
    ("applied materials",            "AMAT"),
    ("arch capital",                 "ACGL"),
    ("archer daniels midland",       "ADM"),
    ("arista networks",              "ANET"),
    ("arthur j gallagher",           "AJG"),
    ("automatic data processing",    "ADP"),
    ("baker hughes",                 "BKR"),
    ("berkshire hathaway",           "BRK"),
    ("best buy",                     "BBY"),
    ("booking holdings",             "BKNG"),
    ("boston scientific",            "BSX"),
    ("broadridge financial",         "BR"),
    ("builders firstsource",         "BLDR"),
    ("cadence design",               "CDNS"),
    ("capital one",                  "COF"),
    ("carrier global",               "CARR"),
    ("cboe global",                  "CBOE"),
    ("centerpoint energy",           "CNP"),
    ("charles schwab",               "SCHW"),
    ("chipotle",                     "CMG"),
    ("citigroup",                    "C"),
    ("citizens financial",           "CFG"),
    ("coca cola",                    "KO"),
    ("coca-cola",                    "KO"),
    ("constellation energy",         "CEG"),
    ("cooper companies",             "COO"),
    ("crowdstrike",                  "CRWD"),
    ("cvs health",                   "CVS"),
    ("d r horton",                   "DHI"),
    ("darden restaurants",           "DRI"),
    ("deere",                        "DE"),
    ("dollar general",               "DG"),
    ("dollar tree",                  "DLTR"),
    ("doordash",                     "DASH"),
    ("eastman chemical",             "EMN"),
    ("electronic arts",              "EA"),
    ("equifax",                      "EFX"),
    ("expand energy",                "EXE"),
    ("extra space storage",          "EXR"),
    ("factset",                      "FDS"),
    ("federal realty",               "FRT"),
    ("fedex",                        "FDX"),
    ("fidelity national information","FIS"),
    ("firstenergy",                  "FE"),
    ("ford motor",                   "F"),
    ("ford",                         "F"),
    ("general dynamics",             "GD"),
    ("general motors",               "GM"),
    ("genuine parts",                "GPC"),
    ("global payments",              "GPN"),
    ("godaddy",                      "GDDY"),
    ("home depot",                   "HD"),
    ("honeywell",                    "HON"),
    ("illinois tool works",          "ITW"),
    ("intercontinental exchange",    "ICE"),
    ("invesco",                      "IVZ"),
    ("jack henry",                   "JKHY"),
    ("jacobs solutions",             "J"),
    ("johnson & johnson",            "JNJ"),
    ("johnson and johnson",          "JNJ"),
    ("kraft heinz",                  "KHC"),
    ("kroger",                       "KR"),
    ("las vegas sands",              "LVS"),
    ("eli lilly",                    "LLY"),
    ("live nation",                  "LYV"),
    ("lockheed martin",              "LMT"),
    ("lowe's",                       "LOW"),
    ("lowes",                        "LOW"),
    ("m&t bank",                     "MTB"),
    ("mcdonald's",                   "MCD"),
    ("mcdonalds",                    "MCD"),
    ("molson coors",                 "TAP"),
    ("monolithic power",             "MPWR"),
    ("nasdaq inc",                   "NDAQ"),
    ("netapp",                       "NTAP"),
    ("newmont",                      "NEM"),
    ("norwegian cruise",             "NCLH"),
    ("on semiconductor",             "ON"),
    ("palantir",                     "PLTR"),
    ("paypal",                       "PYPL"),
    ("pepsico",                      "PEP"),
    ("pfizer",                       "PFE"),
    ("pnc financial",                "PNC"),
    ("principal financial",          "PFG"),
    ("procter & gamble",             "PG"),
    ("procter and gamble",           "PG"),
    ("public service enterprise",    "PEG"),
    ("raymond james",                "RJF"),
    ("realty income",                "O"),
    ("regions financial",            "RF"),
    ("s&p global",                   "SPGI"),
    ("servicenow",                   "NOW"),
    ("simon property",               "SPG"),
    ("snap-on",                      "SNA"),
    ("stanley black & decker",       "SWK"),
    ("starbucks",                    "SBUX"),
    ("state street",                 "STT"),
    ("t. rowe price",                "TROW"),
    ("t rowe price",                 "TROW"),
    ("tko group",                    "TKO"),
    ("travelers companies",          "TRV"),
    ("united airlines",              "UAL"),
    ("united parcel service",        "UPS"),
    ("unitedhealth",                 "UNH"),
    ("universal health",             "UHS"),
    ("vulcan materials",             "VMC"),
    ("w r berkley",                  "WRB"),
    ("w w grainger",                 "GWW"),
    ("warner bros",                  "WBD"),
    ("waste management",             "WM"),
    ("abbvie",                       "ABBV"),
    ("accenture",                    "ACN"),
    ("adobe",                        "ADBE"),
    ("agilent",                      "A"),
    ("air products",                 "APD"),
    ("airbnb",                       "ABNB"),
    ("alexandria real estate",       "ARE"),
    ("align technology",             "ALGN"),
    ("alphabet",                     "GOOGL"),
    ("altria",                       "MO"),
    ("amazon",                       "AMZN"),
    ("amgen",                        "AMGN"),
    ("apple",                        "AAPL"),
    ("aptiv",                        "APTV"),
    ("assurant",                     "AIZ"),
    ("at&t",                         "T"),
    ("autodesk",                     "ADSK"),
    ("autozone",                     "AZO"),
    ("biogen",                       "BIIB"),
    ("blackrock",                    "BLK"),
    ("boeing",                       "BA"),
    ("broadcom",                     "AVGO"),
    ("cme group",                    "CME"),
    ("cisco",                        "CSCO"),
    ("costco",                       "COST"),
    ("cummins",                      "CMI"),
    ("dexcom",                       "DXCM"),
    ("google",                       "GOOGL"),
    ("ibm",                          "IBM"),
    ("intel",                        "INTC"),
    ("intuit",                       "INTU"),
    ("meta platforms",               "META"),
    ("meta",                         "META"),
    ("microsoft",                    "MSFT"),
    ("moderna",                      "MRNA"),
    ("netflix",                      "NFLX"),
    ("nvidia",                       "NVDA"),
    ("salesforce",                   "CRM"),
    ("target",                       "TGT"),
    ("tesla",                        "TSLA"),
    ("uber",                         "UBER"),
    ("walmart",                      "WMT"),
    ("3m",                           "MMM"),
]

# Pre-compile: sort longest-first to avoid partial matches shadowing longer ones
COMPANY_NAME_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE), ticker)
    for name, ticker in sorted(_COMPANY_NAME_MAP_RAW, key=lambda x: -len(x[0]))
]

# ---------------------------------------------------------------------------
# Pass 3b — ticker_association.json keyword map
# ---------------------------------------------------------------------------
_ASSOC_PATH = Path(__file__).parent.parent / "data" / "ticker_associations.json"

# Maps compiled regex pattern → ticker.
# Also expands SP500_TICKERS with every ticker found in the association file.
KEYWORD_MAP: list[tuple[re.Pattern, str]] = []

_MIN_ALIAS_LEN = 3  # skip 1-2 char aliases (e.g. "GE", "IT") — already caught by Pass 2

def _load_association() -> None:
    """Load ticker_association.json and build KEYWORD_MAP + expand SP500_TICKERS."""
    if not _ASSOC_PATH.exists():
        return

    with _ASSOC_PATH.open(encoding="utf-8") as f:
        assoc: dict = json.load(f)

    for ticker in assoc:
        SP500_TICKERS.add(ticker)

    entries: list[tuple[str, str]] = []
    for ticker, data in assoc.items():
        for alias in data.get("aliases", []):
            if len(alias) < _MIN_ALIAS_LEN:
                continue
            entries.append((alias, ticker))

    entries.sort(key=lambda x: -len(x[0]))

    for kw, ticker in entries:
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
        KEYWORD_MAP.append((pattern, ticker))

_load_association()

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
_DOLLAR_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")
_BARE_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------
def extract_tickers(
    text: str,
    labeled_tickers: Sequence[str] | None = None,
) -> list[str]:
    """
    Return a deduplicated list of S&P 500 ticker symbols found in `text`.

    Detection order (highest → lowest precision):
      1. $TICKER mentions       — e.g. "$NVDA"
      2. Bare uppercase words   — e.g. "AMD", "TSLA"
      3. Company name substrings — e.g. "Nvidia", "Super Micro"
      3b. Association keywords  — e.g. "737 max" → BA, "jensen huang" → NVDA
      4. Fallback to `labeled_tickers` if passes 1–3b yield nothing

    Args:
        text:            Raw document text to scan.
        labeled_tickers: Pre-labeled tickers from the corpus record (fallback).

    Returns:
        Ordered list of unique ticker strings (insertion-order preserved).
    """
    seen: dict[str, None] = {}  # ordered set via dict keys

    # Pass 1 — $TICKER
    for ticker in _DOLLAR_TICKER_RE.findall(text):
        if ticker in SP500_TICKERS:
            seen[ticker] = None

    # Pass 2 — bare uppercase words
    for ticker in _BARE_TICKER_RE.findall(text):
        if ticker in SP500_TICKERS:
            seen[ticker] = None

    # Pass 3 — company name substrings
    for pattern, ticker in COMPANY_NAME_MAP:
        if pattern.search(text):
            seen[ticker] = None

    # Pass 3b — association keyword matching
    for pattern, ticker in KEYWORD_MAP:
        if pattern.search(text):
            seen[ticker] = None

    # Pass 4 — fallback to labeled field
    if not seen and labeled_tickers:
        for ticker in labeled_tickers:
            if ticker:
                seen[ticker] = None

    return list(seen.keys())


# ---------------------------------------------------------------------------
# Batch helper: enrich an already-loaded corpus in place
# ---------------------------------------------------------------------------
def enrich_corpus_tickers(corpus: list[dict]) -> list[dict]:
    """
    Re-run ticker detection over every document in `corpus` and overwrite
    the `ticker` field with the regex-detected result (falling back to the
    labeled field if detection yields nothing).

    Modifies corpus in place and also returns it.
    """
    for doc in corpus:
        doc["ticker"] = extract_tickers(
            doc.get("raw_text", ""),
            labeled_tickers=doc.get("ticker", []),
        )
    return corpus


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.corpus import load_corpus

    corpus = load_corpus()
    print(f"Loaded {len(corpus):,} documents")

    enrich_corpus_tickers(corpus)

    # Stats
    with_tickers = [d for d in corpus if d["ticker"]]
    print(f"Docs with ≥1 ticker : {len(with_tickers):,} / {len(corpus):,}")

    # Spot checks
    spot_check_ids = [
        "reddit_1hww0t1",   # Biden/Nvidia/AMD — labeled ALGN (wrong)
        "reddit_1ie1y1s",   # DeepSeek/Nvidia — labeled NVDA (correct)
        "reddit_1jmx1cb",   # SMCI and DELL — labeled AAPL (wrong)
        "news_7671440",     # Super Micro / Nvidia — labeled [] (missing)
        "reddit_1joi1j2",   # $SPY / $PG article — labeled FDS
    ]
    print()
    for doc in corpus:
        if doc["id"] not in spot_check_ids:
            continue
        print(f"{doc['id']}")
        print(f"  tickers  : {doc['ticker']}")
        print(f"  text[:80]: {doc['raw_text'][:80]}")
        print()
