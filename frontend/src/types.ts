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
  sentiment: number;
}

export interface DimensionInfo {
  index: number;
  weight: number;
  signed_weight: number;
  top_terms: string[];
  label: string;
}

export interface QueryInfo {
  preprocessed_query: string;
  active_dimensions: DimensionInfo[];
}

export interface SvdInfo {
  n_components: number;
  explained_variance: number;
}

export interface SearchApiResponse {
  results: SearchResult[];
  query_info: QueryInfo;
  svd_info: SvdInfo;
  error?: string;
}

export interface RagResponse {
  original_query: string;
  improved_query: string;
  ir_results: SearchResult[];
  query_info: QueryInfo;
  svd_info: SvdInfo;
  llm_summary: string;
  relevant_tickers: RelevantTicker[];
  error?: string;
}

export interface RelevantTicker {
  ticker: string;
  reason: string;
}
