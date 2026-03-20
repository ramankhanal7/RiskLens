from convokit import Corpus, download

wsb_corpus       = Corpus(filename=download("subreddit-wallstreetbets"))
stocks_corpus    = Corpus(filename=download("subreddit-stocks"))
investing_corpus = Corpus(filename=download("subreddit-investing"))

# Alternative merge using corpus list
merged_corpus = Corpus.merge(wsb_corpus, stocks_corpus)
merged_corpus = Corpus.merge(merged_corpus, investing_corpus)

merged_corpus.print_summary_stats()