import { useState, useEffect, useRef } from 'react'
import './App.css'
import { WatchlistItem, SearchResult } from './types'

const STORAGE_KEY = 'risklens-watchlist'

type AppTab = 'portfolio' | 'research'
type RetrievalMode = 'hybrid' | 'tfidf' | 'lsa'
type SourceFilter = '' | 'news' | 'reddit'
type SortMode = 'insertion' | 'score'

function scoreToStatus(score: number): 'safe' | 'neutral' | 'caution' {
  if (score > 0.5) return 'safe'
  if (score < -0.5) return 'caution'
  return 'neutral'
}

function StatusBadge({ status }: { status: WatchlistItem['status'] }) {
  const labels: Record<WatchlistItem['status'], string> = {
    safe: 'Safe',
    neutral: 'Neutral',
    caution: 'Caution',
    loading: '...',
    error: 'Error',
  }
  return <span className={`badge badge-${status}`}>{labels[status]}</span>
}

function App(): JSX.Element {
  const [activeTab, setActiveTab] = useState<AppTab>('portfolio')

  const [watchlist, setWatchlist] = useState<WatchlistItem[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (!saved) return []
      const items: WatchlistItem[] = JSON.parse(saved)
      return items.map(w =>
        w.score !== null ? { ...w, status: scoreToStatus(w.score) } : w
      )
    } catch {
      return []
    }
  })
  const [tickerInput, setTickerInput] = useState('')
  const [tickerError, setTickerError] = useState('')
  const [addingTicker, setAddingTicker] = useState(false)
  const [sortMode, setSortMode] = useState<SortMode>('insertion')
  const [refreshing, setRefreshing] = useState(false)

  const [queryInput, setQueryInput] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryError, setQueryError] = useState('')
  const [hasSearched, setHasSearched] = useState(false)
  const [lastSearchedQuery, setLastSearchedQuery] = useState('')
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>('hybrid')
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('')
  const [showScoringInfo, setShowScoringInfo] = useState(false)

  const tickerRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(watchlist))
  }, [watchlist])

  const fetchSentiment = async (ticker: string): Promise<{ score: number } | { error: string }> => {
    const res = await fetch(`/api/sentiment?ticker=${encodeURIComponent(ticker)}`)
    return res.json()
  }

  const addTicker = async (ticker?: string) => {
    const t = (ticker ?? tickerInput.trim()).toUpperCase()
    if (!t) return
    if (watchlist.some(w => w.ticker === t)) {
      if (!ticker) setTickerError(`${t} is already in your watchlist`)
      return
    }
    const fromInput = !ticker
    if (fromInput) {
      setTickerError('')
      setTickerInput('')
    }
    setAddingTicker(true)

    let companyName: string | undefined
    try {
      const vRes = await fetch(`/api/validate-ticker?ticker=${encodeURIComponent(t)}`)
      const vData = await vRes.json()
      if (!vData.valid) {
        if (fromInput) setTickerError(`"${t}" is not a recognized ticker`)
        setAddingTicker(false)
        return
      }
      companyName = vData.name ?? undefined
    } catch {
      // If validation endpoint is unreachable, allow the add anyway
    }

    setWatchlist(prev => [...prev, { ticker: t, name: companyName, score: null, status: 'loading' }])
    setAddingTicker(false)

    try {
      const data = await fetchSentiment(t)
      if ('error' in data) throw new Error(data.error)
      const score: number = data.score
      setWatchlist(prev =>
        prev.map(w =>
          w.ticker === t
            ? { ticker: t, name: companyName, score, status: scoreToStatus(score) }
            : w
        )
      )
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to fetch sentiment'
      setWatchlist(prev =>
        prev.map(w => (w.ticker === t ? { ...w, status: 'error', error: msg } : w))
      )
    }
  }

  const removeTicker = (ticker: string) => {
    setWatchlist(prev => prev.filter(w => w.ticker !== ticker))
  }

  const refreshAll = async () => {
    if (refreshing || watchlist.length === 0) return
    setRefreshing(true)
    setWatchlist(prev => prev.map(w => ({ ...w, status: 'loading' as const })))
    const updated = await Promise.all(
      watchlist.map(async (w) => {
        try {
          const data = await fetchSentiment(w.ticker)
          if ('error' in data) return { ...w, status: 'error' as const, error: data.error }
          return { ...w, score: data.score, status: scoreToStatus(data.score), error: undefined }
        } catch {
          return { ...w, status: 'error' as const, error: 'Network error' }
        }
      })
    )
    setWatchlist(updated)
    setRefreshing(false)
  }

  const handleQuery = async () => {
    const q = queryInput.trim()
    if (!q) return
    setQueryLoading(true)
    setQueryError('')
    setHasSearched(true)
    setLastSearchedQuery(q)
    try {
      let url = `/api/search?q=${encodeURIComponent(q)}&mode=${retrievalMode}`
      if (sourceFilter) url += `&source=${sourceFilter}`
      const res = await fetch(url)
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setSearchResults(data)
    } catch (e: unknown) {
      setQueryError(e instanceof Error ? e.message : 'Search failed')
      setSearchResults([])
    } finally {
      setQueryLoading(false)
    }
  }

  const watchlistTickers = new Set(watchlist.map(w => w.ticker))
  const relatedTickers = Array.from(
    new Set(searchResults.flatMap(r => r.tickers))
  ).filter(t => !watchlistTickers.has(t))

  const displayWatchlist = sortMode === 'score'
    ? [...watchlist].sort((a, b) => (a.score ?? 99) - (b.score ?? 99))
    : watchlist

  const modeLabels: Record<RetrievalMode, string> = {
    hybrid: 'Hybrid (TF-IDF + SVD)',
    tfidf: 'TF-IDF Only',
    lsa: 'SVD / LSA Only',
  }

  const sourceLabels: Record<SourceFilter, string> = {
    '': 'All Sources',
    news: 'News Only',
    reddit: 'Reddit Only',
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo-group">
            <span className="logo-icon">◈</span>
            <span className="logo-text">RiskLens</span>
          </div>
          <p className="tagline">Financial Sentiment &amp; Market Intelligence</p>
        </div>
      </header>

      <nav className="tab-bar">
        <button
          className={`tab-btn ${activeTab === 'portfolio' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('portfolio')}
        >
          Portfolio Monitor
        </button>
        <button
          className={`tab-btn ${activeTab === 'research' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('research')}
        >
          Research
        </button>
      </nav>

      <main className="main single-col">
        {activeTab === 'portfolio' && (
          <section className="panel watchlist-panel">
            <div className="panel-header-row">
              <h2 className="panel-title">Portfolio Monitor</h2>
              <div className="panel-actions">
                <select
                  className="sort-select"
                  value={sortMode}
                  onChange={e => setSortMode(e.target.value as SortMode)}
                >
                  <option value="insertion">Sort: Added</option>
                  <option value="score">Sort: Score</option>
                </select>
                <button
                  className="btn-refresh"
                  onClick={refreshAll}
                  disabled={refreshing || watchlist.length === 0}
                  title="Refresh all scores"
                >
                  {refreshing ? '⟳' : '↻'}
                </button>
              </div>
            </div>
            <p className="panel-subtitle">
              Real-time sentiment from live financial news (last 30 days)
            </p>

            <div className="add-row">
              <input
                ref={tickerRef}
                className="ticker-input"
                placeholder="Ticker symbol, e.g. AAPL"
                value={tickerInput}
                onChange={e => {
                  setTickerInput(e.target.value.toUpperCase())
                  setTickerError('')
                }}
                onKeyDown={e => e.key === 'Enter' && addTicker()}
                maxLength={10}
                disabled={addingTicker}
              />
              <button
                className="btn-primary"
                onClick={() => addTicker()}
                disabled={addingTicker}
              >
                {addingTicker ? '…' : 'Add'}
              </button>
            </div>
            {tickerError && <p className="field-error">{tickerError}</p>}

            <ul className="watchlist-list">
              {watchlist.length === 0 && (
                <li className="empty-state">
                  No securities added yet. Type a ticker above to get started.
                </li>
              )}
              {displayWatchlist.map(item => (
                <li key={item.ticker} className="watchlist-item">
                  <div className="item-left">
                    <span className="item-ticker">{item.ticker}</span>
                    {item.name && (
                      <span className="item-name">{item.name}</span>
                    )}
                    {item.score !== null && (
                      <span className={`item-score score-${item.status}`}>
                        {item.score >= 0 ? '+' : ''}{item.score.toFixed(2)}
                      </span>
                    )}
                  </div>
                  <div className="item-right">
                    <StatusBadge status={item.status} />
                    {item.status === 'error' && item.error && (
                      <span className="item-error-detail" title={item.error}>!</span>
                    )}
                    <button
                      className="btn-remove"
                      onClick={() => removeTicker(item.ticker)}
                      aria-label={`Remove ${item.ticker}`}
                    >
                      ✕
                    </button>
                  </div>
                </li>
              ))}
            </ul>

            <div className="legend">
              <span className="legend-item"><span className="dot dot-safe" />Safe (&gt;+0.5)</span>
              <span className="legend-item"><span className="dot dot-neutral" />Neutral</span>
              <span className="legend-item"><span className="dot dot-caution" />Caution (&lt;-0.5)</span>
              <button
                className="info-toggle"
                onClick={() => setShowScoringInfo(s => !s)}
                aria-label="Scoring info"
              >
                &#9432;
              </button>
            </div>
            {showScoringInfo && (
              <p className="scoring-info">
                Scores are computed from Finnhub company news over the past 30 days
                using a sentiment lexicon. Positive/negative word counts produce a
                score in [-5, +5]. Safe &gt; +0.5, Caution &lt; -0.5, otherwise Neutral.
              </p>
            )}
          </section>
        )}

        {activeTab === 'research' && (
          <div className="research-layout">
            <section className="panel query-panel">
              <h2 className="panel-title">Research</h2>
              <p className="panel-subtitle">
                Search historical news &amp; Reddit corpus for market insights
              </p>

              <div className="query-row">
                <input
                  className="query-input"
                  placeholder="e.g. semiconductor export restrictions, tariff trade war..."
                  value={queryInput}
                  onChange={e => setQueryInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleQuery()}
                />
                <button
                  className="btn-primary"
                  onClick={handleQuery}
                  disabled={queryLoading}
                >
                  {queryLoading ? 'Searching…' : 'Search'}
                </button>
              </div>

              <div className="filter-row">
                <div className="mode-selector">
                  <span className="mode-label">Method:</span>
                  {(['hybrid', 'tfidf', 'lsa'] as RetrievalMode[]).map(m => (
                    <label key={m} className="mode-option">
                      <input
                        type="radio"
                        name="retrieval-mode"
                        value={m}
                        checked={retrievalMode === m}
                        onChange={() => setRetrievalMode(m)}
                      />
                      <span>{modeLabels[m]}</span>
                    </label>
                  ))}
                </div>
                <div className="mode-selector">
                  <span className="mode-label">Source:</span>
                  {(['', 'news', 'reddit'] as SourceFilter[]).map(s => (
                    <label key={s || 'all'} className="mode-option">
                      <input
                        type="radio"
                        name="source-filter"
                        value={s}
                        checked={sourceFilter === s}
                        onChange={() => setSourceFilter(s)}
                      />
                      <span>{sourceLabels[s]}</span>
                    </label>
                  ))}
                </div>
              </div>

              {queryLoading && (
                <div className="loading-row">
                  <span className="spinner" />
                  <span className="loading-text">
                    Running {modeLabels[retrievalMode].toLowerCase()} pipeline…
                  </span>
                </div>
              )}

              {queryError && <p className="field-error">{queryError}</p>}

              {!queryLoading && hasSearched && searchResults.length === 0 && !queryError && (
                <p className="empty-state">No results found for your query.</p>
              )}

              {!queryLoading && hasSearched && searchResults.length > 0 && (
                <p className="result-count">
                  {searchResults.length} result{searchResults.length !== 1 ? 's' : ''} for &ldquo;{lastSearchedQuery}&rdquo;
                </p>
              )}

              <div className="results-list">
                {searchResults.map((r, i) => (
                  <div key={i} className="result-card">
                    <div className="result-header">
                      <span className="result-score">Score: {r.score.toFixed(4)}</span>
                      {r.source && (
                        <span className={`source-badge source-${r.source}`}>
                          {r.source === 'news' ? 'News' : 'Reddit'}
                        </span>
                      )}
                      {r.date && <span className="result-date">{r.date}</span>}
                      {r.tickers.length > 0 && (
                        <div className="result-tickers">
                          {r.tickers.map(t => (
                            <span key={t} className="ticker-tag">{t}</span>
                          ))}
                        </div>
                      )}
                    </div>
                    <p className="result-title">
                      {r.url ? (
                        <a href={r.url} target="_blank" rel="noopener noreferrer">
                          {r.title}
                        </a>
                      ) : (
                        r.title
                      )}
                    </p>
                    {r.snippet && (
                      <p className="result-snippet">{r.snippet}</p>
                    )}
                  </div>
                ))}
              </div>
            </section>

            {relatedTickers.length > 0 && (
              <aside className="panel related-panel">
                <h3 className="panel-title">Related Stocks</h3>
                <p className="panel-subtitle">Tickers from results not in your portfolio</p>
                <ul className="related-list">
                  {relatedTickers.map(t => (
                    <li key={t} className="related-item">
                      <span className="related-ticker">{t}</span>
                      <button
                        className="btn-add-small"
                        onClick={() => { addTicker(t); setActiveTab('portfolio') }}
                      >
                        + Add
                      </button>
                    </li>
                  ))}
                </ul>
              </aside>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
