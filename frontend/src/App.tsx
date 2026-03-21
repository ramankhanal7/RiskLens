import { useState, useEffect, useRef } from 'react'
import './App.css'
import { WatchlistItem, SearchResult } from './types'

const STORAGE_KEY = 'risklens-watchlist'

function scoreToStatus(score: number): 'safe' | 'neutral' | 'caution' {
  if (score > 1.0) return 'safe'
  if (score < -1.0) return 'caution'
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
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })
  const [tickerInput, setTickerInput] = useState('')
  const [tickerError, setTickerError] = useState('')

  const [queryInput, setQueryInput] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryError, setQueryError] = useState('')
  const [hasSearched, setHasSearched] = useState(false)

  const tickerRef = useRef<HTMLInputElement>(null)

  // Persist watchlist (scores + statuses) to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(watchlist))
  }, [watchlist])

  const addTicker = async () => {
    const ticker = tickerInput.trim().toUpperCase()
    if (!ticker) return
    if (watchlist.some(w => w.ticker === ticker)) {
      setTickerError(`${ticker} is already in your watchlist`)
      return
    }
    setTickerError('')
    setTickerInput('')

    // Optimistically add with loading state
    setWatchlist(prev => [...prev, { ticker, score: null, status: 'loading' }])

    try {
      const res = await fetch(`/api/sentiment?ticker=${encodeURIComponent(ticker)}`)
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      const score: number = data.score
      setWatchlist(prev =>
        prev.map(w =>
          w.ticker === ticker
            ? { ticker, score, status: scoreToStatus(score) }
            : w
        )
      )
    } catch {
      setWatchlist(prev =>
        prev.map(w => (w.ticker === ticker ? { ...w, status: 'error' } : w))
      )
    }
  }

  const removeTicker = (ticker: string) => {
    setWatchlist(prev => prev.filter(w => w.ticker !== ticker))
  }

  const handleQuery = async () => {
    const q = queryInput.trim()
    if (!q) return
    setQueryLoading(true)
    setQueryError('')
    setHasSearched(true)
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`)
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

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo-group">
            <span className="logo-icon">◈</span>
            <span className="logo-text">RiskLens</span>
          </div>
          <p className="tagline">Financial Sentiment &amp; Market Intelligence</p>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="main">
        {/* ── Watchlist panel ── */}
        <section className="panel watchlist-panel">
          <h2 className="panel-title">Watchlist</h2>

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
            />
            <button className="btn-primary" onClick={addTicker}>
              Add
            </button>
          </div>
          {tickerError && <p className="field-error">{tickerError}</p>}

          <ul className="watchlist-list">
            {watchlist.length === 0 && (
              <li className="empty-state">
                No securities added yet. Type a ticker above to get started.
              </li>
            )}
            {watchlist.map(item => (
              <li key={item.ticker} className="watchlist-item">
                <div className="item-left">
                  <span className="item-ticker">{item.ticker}</span>
                  {item.score !== null && (
                    <span className={`item-score score-${item.status}`}>
                      {item.score >= 0 ? '+' : ''}{item.score.toFixed(2)}
                    </span>
                  )}
                </div>
                <div className="item-right">
                  <StatusBadge status={item.status} />
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

          {/* Legend */}
          <div className="legend">
            <span className="legend-item"><span className="dot dot-safe" />Safe (&gt;+1.0)</span>
            <span className="legend-item"><span className="dot dot-neutral" />Neutral</span>
            <span className="legend-item"><span className="dot dot-caution" />Caution (&lt;-1.0)</span>
          </div>
        </section>

        {/* ── Market query panel ── */}
        <section className="panel query-panel">
          <h2 className="panel-title">Market Outlook</h2>

          <div className="query-row">
            <input
              className="query-input"
              placeholder="e.g. war in the middle east, tech sector outlook..."
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

          {queryLoading && (
            <div className="loading-row">
              <span className="spinner" />
              <span className="loading-text">Running hybrid IR pipeline…</span>
            </div>
          )}

          {queryError && <p className="field-error">{queryError}</p>}

          {!queryLoading && hasSearched && searchResults.length === 0 && !queryError && (
            <p className="empty-state">No results found for your query.</p>
          )}

          <div className="results-list">
            {searchResults.map((r, i) => (
              <div key={i} className="result-card">
                <div className="result-header">
                  <span className="result-score">Score: {r.score.toFixed(4)}</span>
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
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
