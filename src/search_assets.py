from pathlib import Path
import sys
import nltk

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def ensure_nltk() -> None:
    resources = [
        ("stopwords", "corpora/stopwords"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]
    for name, lookup_path in resources:
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            nltk.download(name, quiet=False)

def main() -> None:
    ensure_nltk()

    from src.tokenize_reddit import main as tokenize_reddit
    from src.tokenize_news import main as tokenize_news
    from src.corpus import build_corpus, enrich_corpus, load_corpus
    from src.lsa import build_lsa

    tokenize_reddit()
    tokenize_news()
    build_corpus()
    enrich_corpus()
    corpus = load_corpus()
    build_lsa(corpus, save=True)

if __name__ == "__main__":
    main()
