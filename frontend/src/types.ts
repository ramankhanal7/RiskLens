export interface WatchlistItem {
  ticker: string;
  score: number | null;
  status: 'safe' | 'neutral' | 'caution' | 'loading' | 'error';
}

export interface SearchResult {
  score: number;
  tickers: string[];
  title: string;
  url: string;
}
