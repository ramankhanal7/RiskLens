"""
tokenize_reddit.py

Reads reddit_finance_sp500_final.csv, tokenizes each post, maps the company
field to an S&P 500 ticker symbol, and writes one JSON object per line to
reddit_tokenized.jsonl.

Output schema per line:
{
  "id":       "reddit_<original_id>",
  "date":     "YYYY-MM-DD",
  "source":   "reddit",
  "raw_text": "<original full_text>",
  "tokens":   ["token1", "token2", ...],   # lowercased, stemmed, no stopwords/punct
  "ticker":   ["AAPL", ...],               # may be empty list if no mapping found
  "url":      "https://...",
  "metadata": {
    "subreddit":   "stocks",
    "score":       150,
    "num_comments": 42,
    "company":     "apple_inc."
  }
}
"""

import csv
import json
import re
import string
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_CSV = DATA_DIR / "reddit_finance_sp500_final.csv"
OUTPUT_JSONL = DATA_DIR / "reddit_tokenized.jsonl"

# ---------------------------------------------------------------------------
# Company name → ticker mapping (all 158 unique values in the dataset)
# ---------------------------------------------------------------------------
COMPANY_TO_TICKER: dict[str, str] = {
    "3m":                                    "MMM",
    "abbvie":                                "ABBV",
    "accenture":                             "ACN",
    "adobe_inc.":                            "ADBE",
    "advanced_micro_devices":                "AMD",
    "agilent_technologies":                  "A",
    "air_products":                          "APD",
    "airbnb":                                "ABNB",
    "alexandria_real_estate_equities":       "ARE",
    "align_technology":                      "ALGN",
    "alphabet_inc._(class_a)":               "GOOGL",
    "altria":                                "MO",
    "amazon":                                "AMZN",
    "american_electric_power":               "AEP",
    "american_express":                      "AXP",
    "american_international_group":          "AIG",
    "american_water_works":                  "AWK",
    "amgen":                                 "AMGN",
    "apollo_global_management":              "APO",
    "apple":                                 "AAPL",
    "apple_inc.":                            "AAPL",
    "applied_materials":                     "AMAT",
    "aptiv":                                 "APTV",
    "arch_capital_group":                    "ACGL",
    "archer_daniels_midland":                "ADM",
    "arista_networks":                       "ANET",
    "arthur_j._gallagher_&_co.":             "AJG",
    "assurant":                              "AIZ",
    "at&t":                                  "T",
    "autodesk":                              "ADSK",
    "automatic_data_processing":             "ADP",
    "autozone":                              "AZO",
    "baker_hughes":                          "BKR",
    "bank_of_america":                       "BAC",
    "berkshire_hathaway":                    "BRK.B",
    "best_buy":                              "BBY",
    "biogen":                                "BIIB",
    "blackrock":                             "BLK",
    "boeing":                                "BA",
    "booking_holdings":                      "BKNG",
    "boston_scientific":                     "BSX",
    "broadcom":                              "AVGO",
    "broadridge_financial_solutions":        "BR",
    "builders_firstsource":                  "BLDR",
    "cadence_design_systems":                "CDNS",
    "capital_one":                           "COF",
    "carrier_global":                        "CARR",
    "cboe_global_markets":                   "CBOE",
    "centerpoint_energy":                    "CNP",
    "charles_schwab_corporation":            "SCHW",
    "chipotle_mexican_grill":                "CMG",
    "cisco":                                 "CSCO",
    "citigroup":                             "C",
    "citizens_financial_group":              "CFG",
    "cme_group":                             "CME",
    "coca-cola_company_(the)":               "KO",
    "constellation_energy":                  "CEG",
    "cooper_companies_(the)":                "COO",
    "costco":                                "COST",
    "crowdstrike":                           "CRWD",
    "cummins":                               "CMI",
    "cvs_health":                            "CVS",
    "d._r._horton":                          "DHI",
    "darden_restaurants":                    "DRI",
    "deere_&_company":                       "DE",
    "dexcom":                                "DXCM",
    "dollar_general":                        "DG",
    "dollar_tree":                           "DLTR",
    "doordash":                              "DASH",
    "dow_inc.":                              "DOW",
    "eastman_chemical_company":              "EMN",
    "electronic_arts":                       "EA",
    "equifax":                               "EFX",
    "expand_energy":                         "EXE",
    "extra_space_storage":                   "EXR",
    "factset":                               "FDS",
    "federal_realty_investment_trust":       "FRT",
    "fedex":                                 "FDX",
    "fidelity_national_information_services": "FIS",
    "firstenergy":                           "FE",
    "ford_motor_company":                    "F",
    "general_dynamics":                      "GD",
    "general_motors":                        "GM",
    "genuine_parts_company":                 "GPC",
    "global_payments":                       "GPN",
    "godaddy":                               "GDDY",
    "goldman_sachs":                         "GS",
    "home_depot_(the)":                      "HD",
    "honeywell":                             "HON",
    "ibm":                                   "IBM",
    "illinois_tool_works":                   "ITW",
    "intel":                                 "INTC",
    "intercontinental_exchange":             "ICE",
    "intuit":                                "INTU",
    "invesco":                               "IVZ",
    "jack_henry_&_associates":               "JKHY",
    "jacobs_solutions":                      "J",
    "johnson_&_johnson":                     "JNJ",
    "jpmorgan_chase":                        "JPM",
    "kraft_heinz":                           "KHC",
    "kroger":                                "KR",
    "las_vegas_sands":                       "LVS",
    "lilly_(eli)":                           "LLY",
    "live_nation_entertainment":             "LYV",
    "lockheed_martin":                       "LMT",
    "lowe's":                                "LOW",
    "m&t_bank":                              "MTB",
    "mcdonald's":                            "MCD",
    "meta_platforms":                        "META",
    "microsoft":                             "MSFT",
    "moderna":                               "MRNA",
    "molson_coors_beverage_company":         "TAP",
    "monolithic_power_systems":              "MPWR",
    "morgan_stanley":                        "MS",
    "nasdaq,_inc.":                          "NDAQ",
    "netapp":                                "NTAP",
    "netflix":                               "NFLX",
    "newmont":                               "NEM",
    "norwegian_cruise_line_holdings":        "NCLH",
    "nvidia":                                "NVDA",
    "on_semiconductor":                      "ON",
    "palantir_technologies":                 "PLTR",
    "paypal":                                "PYPL",
    "pepsico":                               "PEP",
    "pfizer":                                "PFE",
    "pnc_financial_services":               "PNC",
    "principal_financial_group":             "PFG",
    "procter_&_gamble":                      "PG",
    "public_service_enterprise_group":       "PEG",
    "raymond_james_financial":               "RJF",
    "realty_income":                         "O",
    "regions_financial_corporation":         "RF",
    "s&p_global":                            "SPGI",
    "salesforce":                            "CRM",
    "servicenow":                            "NOW",
    "simon_property_group":                  "SPG",
    "snap-on":                               "SNA",
    "stanley_black_&_decker":               "SWK",
    "starbucks":                             "SBUX",
    "state_street_corporation":              "STT",
    "t._rowe_price":                         "TROW",
    "target_corporation":                    "TGT",
    "tesla":                                 "TSLA",
    "tesla,_inc.":                           "TSLA",
    "tko_group_holdings":                    "TKO",
    "travelers_companies_(the)":             "TRV",
    "uber":                                  "UBER",
    "united_airlines_holdings":              "UAL",
    "united_parcel_service":                 "UPS",
    "unitedhealth_group":                    "UNH",
    "universal_health_services":             "UHS",
    "vulcan_materials_company":              "VMC",
    "w._r._berkley_corporation":             "WRB",
    "w._w._grainger":                        "GWW",
    "walmart":                               "WMT",
    "warner_bros._discovery":               "WBD",
    "waste_management":                      "WM",
    "wells_fargo":                           "WFC",
}

# ---------------------------------------------------------------------------
# NLP setup
# ---------------------------------------------------------------------------
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def tokenize(text: str) -> list[str]:
    """Lowercase → word_tokenize → remove stopwords & non-alpha → stem."""
    text = text.lower()
    tokens = word_tokenize(text)
    result = []
    for token in tokens:
        token = token.translate(PUNCT_TABLE)
        if not token or not token.isalpha():
            continue
        if token in STOP_WORDS:
            continue
        result.append(STEMMER.stem(token))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    written = 0
    skipped = 0
    unmapped = set()

    with INPUT_CSV.open(newline="", encoding="utf-8") as infile, \
         OUTPUT_JSONL.open("w", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        for row in reader:
            raw_text = row.get("full_text", "").strip()
            if not raw_text:
                skipped += 1
                continue

            company_key = row.get("company", "").strip().lower()
            ticker = COMPANY_TO_TICKER.get(company_key)
            if ticker is None:
                unmapped.add(company_key)
                tickers = []
            else:
                tickers = [ticker]

            date_str = row.get("created_datetime", "")
            if date_str:
                date_str = date_str[:10]  # keep YYYY-MM-DD only

            record = {
                "id":       f"reddit_{row['id']}",
                "date":     date_str,
                "source":   "reddit",
                "raw_text": raw_text,
                "tokens":   tokenize(raw_text),
                "ticker":   tickers,
                "url":      row.get("url", ""),
                "metadata": {
                    "subreddit":    row.get("subreddit", ""),
                    "score":        int(row["score"]) if row.get("score") else 0,
                    "num_comments": int(row["num_comments"]) if row.get("num_comments") else 0,
                    "company":      company_key,
                },
            }
            outfile.write(json.dumps(record) + "\n")
            written += 1

    print(f"Written : {written:,} records → {OUTPUT_JSONL}")
    print(f"Skipped : {skipped} (empty text)")
    if unmapped:
        print(f"Unmapped companies ({len(unmapped)}):")
        for c in sorted(unmapped):
            print(f"  {c}")


if __name__ == "__main__":
    main()
