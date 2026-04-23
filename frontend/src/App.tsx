import { useState, useEffect, useRef } from 'react'
import './App.css'
import {
  WatchlistItem, SearchResult, DimensionInfo, QueryInfo,
  SvdInfo, SearchApiResponse, RagResponse, RelevantTicker,
} from './types'

const STORAGE_KEY = 'risklens-watchlist'

/** Lightweight markdown → HTML for LLM output (bold, italic, headers, lists, citations). */
function renderMarkdown(md: string): string {
  return md
    .split('\n\n')
    .map(block => {
      block = block.trim()
      if (!block) return ''

      // Heading lines
      if (block.startsWith('### ')) return `<h4>${block.slice(4)}</h4>`
      if (block.startsWith('## '))  return `<h3>${block.slice(3)}</h3>`
      if (block.startsWith('# '))   return `<h3>${block.slice(2)}</h3>`

      // Bullet list block
      const lines = block.split('\n')
      if (lines.every(l => /^[\s]*[\*\-]\s/.test(l))) {
        const items = lines.map(l => {
          let text = l.replace(/^[\s]*[\*\-]\s/, '')
          text = inlineFormat(text)
          return `<li>${text}</li>`
        }).join('')
        return `<ul>${items}</ul>`
      }

      // Regular paragraph(s)
      return lines.map(l => `<p>${inlineFormat(l)}</p>`).join('')
    })
    .filter(Boolean)
    .join('')
}

function inlineFormat(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')  // bold
    .replace(/\*(.+?)\*/g, '<em>$1</em>')              // italic
    .replace(/\[([\d,\s]+)\]/g, '<span class="ai-cite">[$1]</span>')  // citations
}

type AppTab = 'research' | 'portfolio'
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

/* ── Dimension bar component ──────────────────────────────────────────────── */

/* ── Topic theme card component ───────────────────────────────────────── */

function TopicThemeCard({ dim, maxWeight }: { dim: DimensionInfo; maxWeight: number }) {
  const relevancePct = maxWeight > 0 ? Math.round((dim.weight / maxWeight) * 100) : 0
  return (
    <div className="theme-card">
      <div className="theme-card-header">
        <span className="theme-card-label">{dim.label}</span>
        <span className="theme-card-pct">{relevancePct}% relevance</span>
      </div>
      <div className="theme-bar-track">
        <div
          className="theme-bar-fill"
          style={{ width: `${Math.min(relevancePct, 100)}%` }}
        />
      </div>
      <div className="theme-keywords">
        {dim.top_terms.slice(0, 5).map((term, i) => (
          <span key={i} className="theme-keyword">{term}</span>
        ))}
      </div>
    </div>
  )
}

/* ── Main App ────────────────────────────────────────────────────────────── */

function App(): JSX.Element {
  // Research is now the default/first tab per TA feedback
  const [activeTab, setActiveTab] = useState<AppTab>('research')

  // Portfolio state
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

  // Research state
  const [queryInput, setQueryInput] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryError, setQueryError] = useState('')
  const [hasSearched, setHasSearched] = useState(false)
  const [lastSearchedQuery, setLastSearchedQuery] = useState('')
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>('hybrid')
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('')
  const [showScoringInfo, setShowScoringInfo] = useState(false)

  // SVD explainability state
  const [queryInfo, setQueryInfo] = useState<QueryInfo | null>(null)
  const [svdInfo, setSvdInfo] = useState<SvdInfo | null>(null)

  // RAG / AI state
  const [aiMode, setAiMode] = useState(false)
  const [ragLoading, setRagLoading] = useState(false)
  const [improvedQuery, setImprovedQuery] = useState('')
  const [llmSummary, setLlmSummary] = useState('')
  const [originalQuery, setOriginalQuery] = useState('')
  const [relevantTickers, setRelevantTickers] = useState<RelevantTicker[]>([])

  // Insights sidebar collapse state
  const [showStocks, setShowStocks] = useState(true)
  const [showTopics, setShowTopics] = useState(false)
  const [showExposure, setShowExposure] = useState(true)
  const [showTimeline, setShowTimeline] = useState(false)

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

  /* ── Standard IR search ─────────────────────────────────────────────────── */
  const handleQuery = async () => {
    const q = queryInput.trim()
    if (!q) return
    setQueryLoading(true)
    setQueryError('')
    setHasSearched(true)
    setLastSearchedQuery(q)
    // Clear RAG state when doing standard search
    setImprovedQuery('')
    setLlmSummary('')
    setOriginalQuery('')
    try {
      let url = `/api/search?q=${encodeURIComponent(q)}&mode=${retrievalMode}`
      if (sourceFilter) url += `&source=${sourceFilter}`
      const res = await fetch(url)
      const data: SearchApiResponse = await res.json()
      if (data.error) throw new Error(data.error)
      setSearchResults(data.results || [])
      setQueryInfo(data.query_info || null)
      setSvdInfo(data.svd_info || null)
      // Fetch ticker descriptions in background
      const tickers = Array.from(new Set((data.results || []).flatMap((r: SearchResult) => r.tickers)))
      if (tickers.length > 0) {
        fetch('/api/ticker-context', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, tickers }),
        }).then(r => r.json()).then(d => {
          if (Array.isArray(d) && d.length > 0) setRelevantTickers(d)
        }).catch(() => {})
      }
    } catch (e: unknown) {
      setQueryError(e instanceof Error ? e.message : 'Search failed')
      setSearchResults([])
      setQueryInfo(null)
      setSvdInfo(null)
    } finally {
      setQueryLoading(false)
    }
  }

  /* ── RAG search ────────────────────────────────────────────────────────── */
  const handleRagSearch = async () => {
    const q = queryInput.trim()
    if (!q) return
    setRagLoading(true)
    setQueryLoading(true)
    setQueryError('')
    setHasSearched(true)
    setLastSearchedQuery(q)
    try {
      const res = await fetch('/api/rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          mode: retrievalMode,
          source: sourceFilter,
        }),
      })
      const data: RagResponse = await res.json()
      if (data.error) throw new Error(data.error)
      setOriginalQuery(data.original_query)
      setImprovedQuery(data.improved_query)
      setSearchResults(data.ir_results || [])
      setQueryInfo(data.query_info || null)
      setSvdInfo(data.svd_info || null)
      setLlmSummary(data.llm_summary || '')
      setRelevantTickers(data.relevant_tickers || [])
    } catch (e: unknown) {
      setQueryError(e instanceof Error ? e.message : 'RAG search failed')
      setSearchResults([])
      setQueryInfo(null)
      setSvdInfo(null)
      setImprovedQuery('')
      setLlmSummary('')
      setRelevantTickers([])
    } finally {
      setRagLoading(false)
      setQueryLoading(false)
    }
  }

  const handleSearch = () => {
    if (aiMode) {
      handleRagSearch()
    } else {
      handleQuery()
    }
  }

  const watchlistTickers = new Set(watchlist.map(w => w.ticker))

  // For display: use LLM-curated tickers when available (from either AI or standard search)
  const allResultTickers = Array.from(new Set(searchResults.flatMap(r => r.tickers)))
  const displayTickers: { ticker: string; reason: string }[] = relevantTickers.length > 0
    ? relevantTickers
    : allResultTickers.map(t => ({ ticker: t, reason: '' }))

  // Ticker exposure chart data
  const tickerCounts: { ticker: string; count: number }[] = (() => {
    const counts: Record<string, number> = {}
    searchResults.forEach(r => r.tickers.forEach(t => { counts[t] = (counts[t] || 0) + 1 }))
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([ticker, count]) => ({ ticker, count }))
  })()
  const maxTickerCount = Math.max(...tickerCounts.map(t => t.count), 1)

  // Sentiment timeline data (group results by date, average sentiment)
  const timelineData: { date: string; avgSentiment: number; count: number }[] = (() => {
    const byDate: Record<string, { total: number; count: number }> = {}
    searchResults.forEach(r => {
      if (!r.date) return
      const d = r.date.split('T')[0]
      if (!byDate[d]) byDate[d] = { total: 0, count: 0 }
      byDate[d].total += r.sentiment
      byDate[d].count += 1
    })
    return Object.entries(byDate)
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([date, v]) => ({ date, avgSentiment: v.total / v.count, count: v.count }))
  })()

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

  const maxDimWeight = queryInfo?.active_dimensions
    ? Math.max(...queryInfo.active_dimensions.map(d => d.weight), 0.001)
    : 0

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

      {/* Tab bar — Research first per TA feedback */}
      <nav className="tab-bar">
        <button
          className={`tab-btn ${activeTab === 'research' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('research')}
        >
          Research
        </button>
        <button
          className={`tab-btn ${activeTab === 'portfolio' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('portfolio')}
        >
          Portfolio Monitor
        </button>
      </nav>

      <main className="main single-col">
        {/* ── Research Tab ──────────────────────────────────────────────────── */}
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
                  onKeyDown={e => e.key === 'Enter' && handleSearch()}
                />
                <button
                  className="btn-primary"
                  onClick={handleSearch}
                  disabled={queryLoading}
                >
                  {queryLoading ? (ragLoading ? 'AI Searching…' : 'Searching…') : 'Search'}
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
                {/* AI Mode toggle */}
                <label className="ai-toggle">
                  <input
                    type="checkbox"
                    checked={aiMode}
                    onChange={e => setAiMode(e.target.checked)}
                  />
                  <span className="ai-toggle-label">✦ AI-Powered</span>
                </label>
              </div>

              {queryLoading && (
                <div className="loading-row">
                  <span className="spinner" />
                  <span className="loading-text">
                    {ragLoading
                      ? 'Running AI-powered RAG pipeline…'
                      : `Running ${modeLabels[retrievalMode].toLowerCase()} pipeline…`}
                  </span>
                </div>
              )}

              {queryError && <p className="field-error">{queryError}</p>}

              {!queryLoading && hasSearched && searchResults.length === 0 && !queryError && (
                <p className="empty-state">No results found for your query.</p>
              )}

              {/* ── Query Enhancement (RAG mode) ─────────────────────────────── */}
              {!queryLoading && improvedQuery && (
                <div className="query-enhancement">
                  <div className="qe-header">
                    <span className="qe-icon">✦</span>
                    <span className="qe-title">Query Enhancement</span>
                  </div>
                  <div className="qe-body">
                    <div className="qe-row">
                      <span className="qe-label">Your query</span>
                      <span className="qe-value">{originalQuery}</span>
                    </div>
                    <div className="qe-arrow">↓ AI optimized ↓</div>
                    <div className="qe-row">
                      <span className="qe-label">Optimized</span>
                      <span className="qe-value qe-improved">{improvedQuery}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* ── AI Summary (RAG mode) ────────────────────────────────────── */}
              {!queryLoading && llmSummary && (
                <div className="ai-summary-panel">
                  <div className="ai-summary-header">
                    <span className="ai-summary-icon">✦</span>
                    <span className="ai-summary-title">AI Summary</span>
                  </div>
                  <div
                    className="ai-summary-body"
                    dangerouslySetInnerHTML={{ __html: renderMarkdown(llmSummary) }}
                  />
                </div>
              )}



              {/* ── Result count ─────────────────────────────────────────────── */}
              {!queryLoading && hasSearched && searchResults.length > 0 && (
                <p className="result-count">
                  {searchResults.length} result{searchResults.length !== 1 ? 's' : ''} for &ldquo;{lastSearchedQuery}&rdquo;
                  {improvedQuery && <span className="result-count-note"> (via optimized query)</span>}
                </p>
              )}

              {/* ── Results list ─────────────────────────────────────────────── */}
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

            {/* ── Insights Sidebar ──────────────────────────────────────────── */}
            {hasSearched && searchResults.length > 0 && (
              <aside className="insights-panel">
                <h3 className="insights-title">Insights</h3>

                {/* ── Related Stocks ──────────────────────────────────────── */}
                {displayTickers.length > 0 && (
                  <div className="insight-section">
                    <button className="insight-toggle" onClick={() => setShowStocks(v => !v)}>
                      <span>Related Stocks</span>
                      <span className="insight-chevron">{showStocks ? '▾' : '▸'}</span>
                    </button>
                    {showStocks && (
                      <div className="insight-body">
                        <ul className="related-list">
                          {displayTickers.map(({ ticker, reason }) => (
                            <li key={ticker} className="related-item">
                              <span className="related-ticker">{ticker}</span>
                              {watchlistTickers.has(ticker) && (
                                <span className="portfolio-tag">In Portfolio</span>
                              )}
                              {reason && <span className="related-reason">{reason}</span>}
                              {!watchlistTickers.has(ticker) && (
                                <button
                                  className="btn-add-small"
                                  onClick={() => { addTicker(ticker); setActiveTab('portfolio') }}
                                >
                                  + Add
                                </button>
                              )}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                {/* ── Ticker Exposure Chart ───────────────────────────────── */}
                {tickerCounts.length > 0 && (
                  <div className="insight-section">
                    <button className="insight-toggle" onClick={() => setShowExposure(v => !v)}>
                      <span>Ticker Exposure</span>
                      <span className="insight-chevron">{showExposure ? '▾' : '▸'}</span>
                    </button>
                    {showExposure && (
                      <div className="insight-body">
                        <p className="insight-description">How often each ticker appears across results</p>
                        <div className="exposure-chart">
                          {tickerCounts.map(({ ticker, count }) => (
                            <div key={ticker} className="exposure-row">
                              <span className="exposure-label">{ticker}</span>
                              <div className="exposure-bar-track">
                                <div
                                  className="exposure-bar-fill"
                                  style={{ width: `${(count / maxTickerCount) * 100}%` }}
                                />
                              </div>
                              <span className="exposure-count">{count}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* ── Semantic Topic Themes ───────────────────────────────── */}
                {queryInfo && queryInfo.active_dimensions.length > 0 && retrievalMode !== 'tfidf' && (
                  <div className="insight-section">
                    <button className="insight-toggle" onClick={() => setShowTopics(v => !v)}>
                      <span>Semantic Topics</span>
                      <span className="insight-chevron">{showTopics ? '▾' : '▸'}</span>
                    </button>
                    {showTopics && (
                      <div className="insight-body">
                        <p className="insight-description">
                          SVD-detected topic themes ({svdInfo ? `${svdInfo.n_components} dimensions, ${(svdInfo.explained_variance * 100).toFixed(1)}% variance` : ''})
                        </p>
                        <div className="theme-list">
                          {queryInfo.active_dimensions.map(dim => (
                            <TopicThemeCard key={dim.index} dim={dim} maxWeight={maxDimWeight} />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* ── Sentiment Timeline ──────────────────────────────────── */}
                {timelineData.length > 1 && (
                  <div className="insight-section">
                    <button className="insight-toggle" onClick={() => setShowTimeline(v => !v)}>
                      <span>Sentiment Timeline</span>
                      <span className="insight-chevron">{showTimeline ? '▾' : '▸'}</span>
                    </button>
                    {showTimeline && (
                      <div className="insight-body">
                        <p className="insight-description">Average document sentiment over time (positive ↑ / negative ↓)</p>
                        <div className="timeline-chart">
                          <svg viewBox={`0 0 ${timelineData.length * 40} 70`} className="timeline-svg">
                            {(() => {
                              const maxAbs = Math.max(
                                ...timelineData.map(d => Math.abs(d.avgSentiment)),
                                0.01
                              )
                              const midY = 35
                              return (
                                <>
                                  {/* Zero baseline */}
                                  <line
                                    x1="0" y1={midY}
                                    x2={timelineData.length * 40} y2={midY}
                                    stroke="#2d3748" strokeWidth="1" strokeDasharray="3,3"
                                  />
                                  {/* Sentiment line */}
                                  <polyline
                                    points={timelineData.map((d, i) => {
                                      const x = i * 40 + 20
                                      const y = midY - (d.avgSentiment / maxAbs) * 30
                                      return `${x},${y}`
                                    }).join(' ')}
                                    fill="none"
                                    stroke="#94a3b8"
                                    strokeWidth="2"
                                    strokeLinejoin="round"
                                  />
                                  {/* Data points — green if positive, red if negative */}
                                  {timelineData.map((d, i) => {
                                    const x = i * 40 + 20
                                    const y = midY - (d.avgSentiment / maxAbs) * 30
                                    const color = d.avgSentiment > 0.01 ? '#22c55e'
                                      : d.avgSentiment < -0.01 ? '#ef4444' : '#94a3b8'
                                    return <circle key={i} cx={x} cy={y} r="4" fill={color} />
                                  })}
                                </>
                              )
                            })()}
                          </svg>
                          <div className="timeline-legend">
                            <span className="tl-pos">● Positive</span>
                            <span className="tl-neu">● Neutral</span>
                            <span className="tl-neg">● Negative</span>
                          </div>
                          <div className="timeline-labels">
                            {timelineData.map((d, i) => (
                              <span key={i} className="timeline-label">
                                {d.date.slice(5)}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </aside>
            )}
          </div>
        )}

        {/* ── Portfolio Tab ─────────────────────────────────────────────────── */}
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
      </main>
    </div>
  )
}

export default App
