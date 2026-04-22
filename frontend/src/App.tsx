import { useState, useEffect, useRef } from 'react'
import './App.css'
import {
  WatchlistItem, SearchResult, DimensionInfo, QueryInfo,
  SvdInfo, SearchApiResponse, RagResponse,
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
  const [showQueryAnalysis, setShowQueryAnalysis] = useState(true)

  // RAG / AI state
  const [aiMode, setAiMode] = useState(false)
  const [ragLoading, setRagLoading] = useState(false)
  const [improvedQuery, setImprovedQuery] = useState('')
  const [llmSummary, setLlmSummary] = useState('')
  const [originalQuery, setOriginalQuery] = useState('')

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
    } catch (e: unknown) {
      setQueryError(e instanceof Error ? e.message : 'RAG search failed')
      setSearchResults([])
      setQueryInfo(null)
      setSvdInfo(null)
      setImprovedQuery('')
      setLlmSummary('')
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

              {/* ── Semantic Topic Themes (SVD — only for hybrid/lsa) ──────────── */}
              {!queryLoading && hasSearched && queryInfo && queryInfo.active_dimensions.length > 0 && retrievalMode !== 'tfidf' && (
                <div className="query-analysis">
                  <button
                    className="qa-toggle"
                    onClick={() => setShowQueryAnalysis(v => !v)}
                  >
                    <span className="qa-icon">⬡</span>
                    <span className="qa-title">Semantic Topic Themes</span>
                    <span className="qa-chevron">{showQueryAnalysis ? '▾' : '▸'}</span>
                  </button>

                  {showQueryAnalysis && (
                    <div className="qa-body">
                      <p className="qa-description">
                        Our SVD model identified these topic themes as most relevant to your query.
                        Each theme represents a cluster of related concepts found across {svdInfo ? svdInfo.n_components : 200} learned
                        topics{svdInfo ? ` (capturing ${(svdInfo.explained_variance * 100).toFixed(1)}% of corpus patterns)` : ''}.
                        The bar shows how strongly your query relates to each theme.
                      </p>

                      <div className="qa-meta">
                        <span className="qa-meta-item">
                          Processed as: <code>{queryInfo.preprocessed_query}</code>
                        </span>
                      </div>

                      <div className="theme-list">
                        {queryInfo.active_dimensions.map(dim => (
                          <TopicThemeCard key={dim.index} dim={dim} maxWeight={maxDimWeight} />
                        ))}
                      </div>
                    </div>
                  )}
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
