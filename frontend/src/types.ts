export interface WatchlistItem {
  ticker: string;
  name?: string;
  score: number | null;
  status: 'safe' | 'neutral' | 'caution' | 'loading' | 'error';
  error?: string;
}

export interface SearchResult {
  score: number;
  tickers: string[];
  title: string;
  url: string;
  snippet: string;
  date: string;
  source: string;
}
