import { useState, useEffect, useRef } from 'react'
import './App.css'
import {
  WatchlistItem, SearchResult, DimensionInfo, QueryInfo,
  SvdInfo, SearchApiResponse, RagResponse, RelevantTicker,
  StockQuote, RecommendationTrend,
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
      if (block.startsWith('## ')) return `<h3>${block.slice(3)}</h3>`
      if (block.startsWith('# ')) return `<h3>${block.slice(2)}</h3>`

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

  // SVD explainability state
  const [queryInfo, setQueryInfo] = useState<QueryInfo | null>(null)
  const [svdInfo, setSvdInfo] = useState<SvdInfo | null>(null)

  // RAG / AI state
  const [aiMode, setAiMode] = useState(true)
  const [ragLoading, setRagLoading] = useState(false)
  const [improvedQuery, setImprovedQuery] = useState('')
  const [llmSummary, setLlmSummary] = useState('')
  const [originalQuery, setOriginalQuery] = useState('')
  const [relevantTickers, setRelevantTickers] = useState<RelevantTicker[]>([])

  // Insights sidebar collapse state
  const [showStocks, setShowStocks] = useState(true)
  const [showTopics, setShowTopics] = useState(true)
  const [showExposure, setShowExposure] = useState(true)
  const [showTimeline, setShowTimeline] = useState(true)

  // Portfolio dashboard state
  const [quotes, setQuotes] = useState<Record<string, StockQuote>>({})
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null)
  const [recommendations, setRecommendations] = useState<RecommendationTrend[]>([])
  const [recsLoading, setRecsLoading] = useState(false)

  const [briefing, setBriefing] = useState('')
  const [briefingLoading, setBriefingLoading] = useState(false)

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
      fetchQuote(t)
      if (!selectedTicker) selectHolding(t)
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
    // Also refresh quotes
    fetchAllQuotes(updated.map(w => w.ticker))
  }

  /* ── Portfolio dashboard helpers ────────────────────────────────────────── */

  const fetchQuote = async (ticker: string) => {
    try {
      const res = await fetch(`/api/quote?ticker=${encodeURIComponent(ticker)}`)
      const data: StockQuote = await res.json()
      if (data.price !== undefined) {
        setQuotes(prev => ({ ...prev, [ticker]: data }))
      }
    } catch { /* ignore */ }
  }

  const fetchAllQuotes = (tickers: string[]) => {
    tickers.forEach(t => fetchQuote(t))
  }

  const selectHolding = async (ticker: string) => {
    setSelectedTicker(ticker)
    // Fetch recommendations
    setRecsLoading(true)
    try {
      const res = await fetch(`/api/recommendation?ticker=${encodeURIComponent(ticker)}`)
      const data: RecommendationTrend[] = await res.json()
      setRecommendations(Array.isArray(data) ? data : [])
    } catch { setRecommendations([]) }
    setRecsLoading(false)

  }

  const generateBriefing = async () => {
    setBriefingLoading(true)
    try {
      const holdings = watchlist.map(w => ({
        ticker: w.ticker,
        name: w.name,
        score: w.score,
        price: quotes[w.ticker]?.price,
        changePct: quotes[w.ticker]?.changePct,
      }))
      const res = await fetch('/api/portfolio-briefing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ holdings }),
      })
      const data = await res.json()
      setBriefing(data.briefing || data.error || 'Failed to generate briefing.')
    } catch {
      setBriefing('Failed to generate briefing.')
    }
    setBriefingLoading(false)
  }

  // Auto-select first holding when switching to portfolio tab
  useEffect(() => {
    if (activeTab === 'portfolio' && watchlist.length > 0 && !selectedTicker) {
      selectHolding(watchlist[0].ticker)
      fetchAllQuotes(watchlist.map(w => w.ticker))
    }
  }, [activeTab]) // eslint-disable-line react-hooks/exhaustive-deps

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
        }).catch(() => { })
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
            <aside className="insights-panel">
              <h3 className="insights-title">Insights</h3>

              {!(hasSearched && searchResults.length > 0) && (
                <div className="insights-empty">
                  <span className="insights-empty-icon">◇</span>
                  <p>Run a search to see related stocks, topic analysis, ticker exposure, and sentiment trends.</p>
                </div>
              )}

              {hasSearched && searchResults.length > 0 && (<>
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

              </>)}
            </aside>
          </div>
        )}

        {/* ── Portfolio Dashboard ──────────────────────────────────────────── */}
        {activeTab === 'portfolio' && (() => {
          const scored = watchlist.filter(w => w.score !== null)
          const avgSent = scored.length > 0
            ? scored.reduce((s, w) => s + (w.score ?? 0), 0) / scored.length : 0
          const mostAtRisk = scored.length > 0
            ? scored.reduce((a, b) => ((a.score ?? 0) < (b.score ?? 0) ? a : b)) : null
          const bestMom = scored.length > 0
            ? scored.reduce((a, b) => ((a.score ?? 0) > (b.score ?? 0) ? a : b)) : null
          const selItem = watchlist.find(w => w.ticker === selectedTicker) || null
          const selQuote = selectedTicker ? quotes[selectedTicker] : null
          return (
            <div className="dash">
              {/* ── Header ─────────────────────────────────────────────────── */}
              <div className="dash-header">
                <div>
                  <h2 className="dash-title">Portfolio Monitoring Dashboard</h2>
                  <p className="dash-subtitle">Real-time sentiment &amp; risk overview from live financial news</p>
                </div>
                <div className="dash-actions">
                  <div className="dash-add-row">
                    <input
                      ref={tickerRef}
                      className="ticker-input"
                      placeholder="Add ticker…"
                      value={tickerInput}
                      onChange={e => { setTickerInput(e.target.value.toUpperCase()); setTickerError('') }}
                      onKeyDown={e => e.key === 'Enter' && addTicker()}
                      maxLength={10}
                      disabled={addingTicker}
                    />
                    <button className="btn-primary btn-sm" onClick={() => addTicker()} disabled={addingTicker}>
                      {addingTicker ? '…' : '+ Add'}
                    </button>
                  </div>
                  <select
                    className="sort-select"
                    value={sortMode}
                    onChange={e => setSortMode(e.target.value as SortMode)}
                  >
                    <option value="insertion">Sort: Added</option>
                    <option value="score">Sort: Score</option>
                  </select>
                  <button
                    className="btn-secondary btn-sm"
                    onClick={refreshAll}
                    disabled={refreshing || watchlist.length === 0}
                  >
                    {refreshing ? '⟳ Refreshing…' : '↻ Refresh All'}
                  </button>
                </div>
              </div>
              {tickerError && <p className="field-error" style={{ marginTop: -8 }}>{tickerError}</p>}

              {/* ── Summary Cards ─────────────────────────────────────────── */}
              <div className="dash-cards">
                <div className="dash-card">
                  <span className="dash-card-label">Holdings</span>
                  <span className="dash-card-value">{watchlist.length}</span>
                </div>
                <div className="dash-card">
                  <span className="dash-card-label">Avg Sentiment</span>
                  <span className={`dash-card-value ${avgSent > 0.3 ? 'val-pos' : avgSent < -0.3 ? 'val-neg' : 'val-neu'}`}>
                    {avgSent >= 0 ? '+' : ''}{avgSent.toFixed(2)}
                  </span>
                </div>
                <div className="dash-card">
                  <span className="dash-card-label">Most At Risk</span>
                  <span className="dash-card-value val-neg">{mostAtRisk?.ticker ?? '—'}</span>
                  {mostAtRisk && <span className="dash-card-sub">{mostAtRisk.score?.toFixed(2)}</span>}
                </div>
                <div className="dash-card">
                  <span className="dash-card-label">Best Momentum</span>
                  <span className="dash-card-value val-pos">{bestMom?.ticker ?? '—'}</span>
                  {bestMom && <span className="dash-card-sub">+{bestMom.score?.toFixed(2)}</span>}
                </div>
              </div>

              {/* ── AI Portfolio Briefing ─────────────────────────────────── */}
              <div className="dash-briefing">
                <div className="dash-briefing-header">
                  <div>
                    <span className="dash-briefing-icon">✦</span>
                    <span className="dash-briefing-title">AI Portfolio Briefing</span>
                    <span className="ai-badge">AI</span>
                  </div>
                  <button
                    className="btn-primary btn-sm"
                    onClick={generateBriefing}
                    disabled={briefingLoading || watchlist.length === 0}
                  >
                    {briefingLoading ? '⟳ Generating…' : '✦ Generate Briefing'}
                  </button>
                </div>
                <div className="dash-briefing-body">
                  {briefing
                    ? <div dangerouslySetInnerHTML={{ __html: renderMarkdown(briefing) }} />
                    : <p className="dash-briefing-placeholder">
                      Press <strong>"Generate Briefing"</strong> to receive an AI-powered analysis of your portfolio's risk posture,
                      key concerns, and actionable recommendations.
                    </p>
                  }
                </div>
              </div>

              {/* ── Holdings Table + Detail Panel ────────────────────────── */}
              {watchlist.length === 0 ? (
                <div className="dash-empty">
                  <p>No holdings yet. Add a ticker above to get started.</p>
                </div>
              ) : (
                <div className="dash-main">
                  {/* Left: Holdings Table */}
                  <div className="dash-table-wrap">
                    <table className="dash-table">
                      <thead>
                        <tr>
                          <th>Ticker</th>
                          <th>Price</th>
                          <th>Day</th>
                          <th>Sentiment</th>
                          <th>Risk</th>
                          <th></th>
                        </tr>
                      </thead>
                      <tbody>
                        {displayWatchlist.map(item => {
                          const q = quotes[item.ticker]
                          return (
                            <tr
                              key={item.ticker}
                              className={`dash-row ${selectedTicker === item.ticker ? 'dash-row-selected' : ''}`}
                              onClick={() => selectHolding(item.ticker)}
                            >
                              <td>
                                <div className="dash-ticker-cell">
                                  <span className="dash-ticker">{item.ticker}</span>
                                  {item.name && <span className="dash-name">{item.name}</span>}
                                </div>
                              </td>
                              <td className="dash-mono">{q ? `$${q.price.toFixed(2)}` : '—'}</td>
                              <td>
                                {q ? (
                                  <span className={`dash-badge ${q.changePct >= 0 ? 'badge-green' : 'badge-red'}`}>
                                    {q.changePct >= 0 ? '+' : ''}{q.changePct.toFixed(2)}%
                                  </span>
                                ) : '—'}
                              </td>
                              <td>
                                {item.score !== null ? (
                                  <span className={`dash-badge ${item.score > 0.3 ? 'badge-green' : item.score < -0.3 ? 'badge-red' : 'badge-amber'}`}>
                                    {item.score >= 0 ? '+' : ''}{item.score.toFixed(2)}
                                  </span>
                                ) : (
                                  <span className="dash-badge badge-muted">{item.status === 'loading' ? '…' : '—'}</span>
                                )}
                              </td>
                              <td><StatusBadge status={item.status} /></td>
                              <td>
                                <button className="btn-remove" onClick={e => { e.stopPropagation(); removeTicker(item.ticker) }}>✕</button>
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Right: Selected Holding Detail */}
                  {selItem && (
                    <div className="dash-detail">
                      <div className="dash-detail-header">
                        <div>
                          <span className="dash-detail-ticker">{selItem.ticker}</span>
                          {selItem.name && <span className="dash-detail-name">{selItem.name}</span>}
                        </div>
                        <StatusBadge status={selItem.status} />
                      </div>

                      {/* Stat cards */}
                      <div className="dash-stats">
                        <div className="dash-stat">
                          <span className="dash-stat-label">Current Price</span>
                          <span className="dash-stat-value">{selQuote ? `$${selQuote.price.toFixed(2)}` : '—'}</span>
                        </div>
                        <div className="dash-stat">
                          <span className="dash-stat-label">Daily Move</span>
                          <span className={`dash-stat-value ${selQuote && selQuote.changePct >= 0 ? 'val-pos' : 'val-neg'}`}>
                            {selQuote ? `${selQuote.changePct >= 0 ? '+' : ''}${selQuote.changePct.toFixed(2)}%` : '—'}
                          </span>
                        </div>
                      </div>

                      {/* Analyst Recommendations */}
                      <div className="dash-section">
                        <h4 className="dash-section-title">Analyst Recommendations</h4>
                        {recsLoading ? (
                          <div className="dash-loading"><span className="spinner" /> Loading…</div>
                        ) : recommendations.length > 0 ? (
                          <div className="recs-chart">
                            {recommendations.slice().reverse().map((r, i) => {
                              const total = r.strongBuy + r.buy + r.hold + r.sell + r.strongSell
                              if (total === 0) return null
                              return (
                                <div key={i} className="recs-row">
                                  <span className="recs-period">{r.period.slice(0, 7)}</span>
                                  <div className="recs-bar">
                                    <div className="recs-seg recs-strong-buy" style={{ width: `${(r.strongBuy / total) * 100}%` }} title={`Strong Buy: ${r.strongBuy}`} />
                                    <div className="recs-seg recs-buy" style={{ width: `${(r.buy / total) * 100}%` }} title={`Buy: ${r.buy}`} />
                                    <div className="recs-seg recs-hold" style={{ width: `${(r.hold / total) * 100}%` }} title={`Hold: ${r.hold}`} />
                                    <div className="recs-seg recs-sell" style={{ width: `${(r.sell / total) * 100}%` }} title={`Sell: ${r.sell}`} />
                                    <div className="recs-seg recs-strong-sell" style={{ width: `${(r.strongSell / total) * 100}%` }} title={`Strong Sell: ${r.strongSell}`} />
                                  </div>
                                </div>
                              )
                            })}
                            <div className="recs-legend">
                              <span><span className="recs-dot recs-strong-buy" />Strong Buy</span>
                              <span><span className="recs-dot recs-buy" />Buy</span>
                              <span><span className="recs-dot recs-hold" />Hold</span>
                              <span><span className="recs-dot recs-sell" />Sell</span>
                            </div>
                          </div>
                        ) : (
                          <p className="dash-muted">No recommendation data available.</p>
                        )}
                      </div>



                      <button
                        className="btn-primary btn-full"
                        onClick={() => {
                          setQueryInput(selItem.ticker)
                          setActiveTab('research')
                        }}
                      >
                        Open Full Research View →
                      </button>
                    </div>
                  )}
                </div>
              )}

            </div>
          )
        })()}
      </main>
    </div>
  )
}

export default App
