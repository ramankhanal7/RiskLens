"""
LLM RAG route — only loaded when USE_LLM = True in routes.py.
Adds a POST /api/rag endpoint that implements the full RAG pipeline:

    1. User query  →  LLM improves/expands the query
    2. Improved query  →  IR system returns ranked documents
    3. IR results  →  LLM generates a synthesised financial summary

Setup:
  1. Add API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""
import json
import os
import logging
from dataclasses import asdict
from flask import request, jsonify
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Query Enhancement
# ---------------------------------------------------------------------------
def improve_query(client: LLMClient, user_query: str) -> str:
    """
    Ask the LLM to reformulate the user's query for better retrieval
    from a financial news + Reddit corpus.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a search-query optimizer for a financial information retrieval "
                "system. The corpus contains financial news articles and Reddit posts "
                "about stocks, markets, and the economy.\n\n"
                "Given a user's natural-language question, produce an IMPROVED search "
                "query that will retrieve the most relevant documents. Rules:\n"
                "- Add relevant financial keywords and synonyms\n"
                "- Include related ticker symbols if obvious (e.g., AAPL for Apple)\n"
                "- Expand abbreviations\n"
                "- Keep it concise — no more than 15 words\n"
                "- Return ONLY the improved query text, nothing else"
            ),
        },
        {"role": "user", "content": user_query},
    ]
    response = client.chat(messages)
    improved = (response.get("content") or user_query).strip().strip('"').strip("'")
    logger.info(f"Query improvement: '{user_query}' → '{improved}'")
    return improved


# ---------------------------------------------------------------------------
# Step 3 — RAG Synthesis
# ---------------------------------------------------------------------------
def synthesize_answer(
    client: LLMClient,
    user_query: str,
    ir_results: list[dict],
) -> str:
    """
    Given the original user query and the top IR results, ask the LLM
    to produce a synthesised financial summary with source citations.
    """
    # Build context from IR results
    context_parts = []
    for i, r in enumerate(ir_results, 1):
        source_label = "News" if r.get("source") == "news" else "Reddit"
        date_str = r.get("date", "unknown date")
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        tickers = ", ".join(r.get("tickers", [])) or "N/A"
        text = title
        if snippet and snippet != title:
            text += f"\n{snippet}"
        context_parts.append(
            f"[{i}] ({source_label}, {date_str}, Tickers: {tickers})\n{text}"
        )
    context_text = "\n\n---\n\n".join(context_parts) if context_parts else "No results found."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial research analyst assistant. The user asked a "
                "financial question and our information retrieval system found the "
                "following relevant documents from news articles and Reddit posts.\n\n"
                "Your job is to synthesise these results into a clear, informative "
                "summary that helps the user understand the current financial context "
                "related to their question. Rules:\n"
                "- Reference sources using [1], [2], etc. citations\n"
                "- Highlight key themes, trends, and risks\n"
                "- Mention relevant tickers when applicable\n"
                "- Be concise but thorough (3-5 paragraphs)\n"
                "- If the results don't contain enough information, say so\n"
                "- Do NOT invent information not present in the sources"
            ),
        },
        {
            "role": "user",
            "content": (
                f"My question: {user_query}\n\n"
                f"Retrieved documents:\n\n{context_text}"
            ),
        },
    ]
    response = client.chat(messages)
    return (response.get("content") or "Unable to generate summary.").strip()


def identify_relevant_tickers(
    client: LLMClient,
    user_query: str,
    ir_results: list[dict],
) -> list[dict]:
    """
    Ask the LLM to pick the tickers most affected by/relevant to the query,
    based on the IR results. Returns a list of {ticker, reason} dicts.
    """
    all_tickers = set()
    for r in ir_results:
        for t in r.get("tickers", []):
            all_tickers.add(t)
    if not all_tickers:
        return []

    ticker_list = ", ".join(sorted(all_tickers))
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst. Given a user query and a list of stock "
                "tickers that appeared in related news/Reddit articles, identify which "
                "tickers are MOST directly affected by or relevant to the query topic.\n\n"
                "Rules:\n"
                "- Select only genuinely relevant tickers (max 8)\n"
                "- For each, give a brief 5-10 word reason why it's relevant\n"
                "- Reply as a JSON array: [{\"ticker\": \"NVDA\", \"reason\": \"Major AI chip maker affected by export controls\"}]\n"
                "- Return ONLY the JSON array, no other text"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Query: {user_query}\n"
                f"Tickers found in articles: {ticker_list}"
            ),
        },
    ]
    try:
        resp = client.chat(messages)
        content = (resp.get("content") or "[]").strip()
        # Extract JSON array from response
        import re
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            import json as json_mod
            return json_mod.loads(match.group())
    except Exception as e:
        logger.warning(f"Ticker identification failed: {e}")
    return [{"ticker": t, "reason": ""} for t in sorted(all_tickers)[:8]]


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------
def register_chat_route(app):
    """Register the /api/rag endpoint. Called from routes.py when USE_LLM=True."""

    @app.route("/api/rag", methods=["POST"])
    def rag():
        data = request.get_json() or {}
        user_query = (data.get("query") or "").strip()
        mode = (data.get("mode") or "hybrid").strip().lower()
        source_filter = (data.get("source") or "").strip().lower()

        if not user_query:
            return jsonify({"error": "query is required"}), 400

        api_key = os.getenv("SPARK_API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY not set — add it to your .env file"}), 500

        try:
            client = LLMClient(api_key=api_key)

            # Step 1 — LLM improves the query
            improved_query = improve_query(client, user_query)

            # Step 2 — Run improved query through IR system
            from query import search
            response = search(improved_query, mode=mode)
            results_list = response.results
            if source_filter in ("news", "reddit"):
                results_list = [r for r in results_list if r.source == source_filter]

            # Serialise IR results
            ir_results = [
                {
                    "score": r.score, "tickers": r.tickers, "title": r.title,
                    "url": r.url, "snippet": r.snippet, "date": r.date,
                    "source": r.source, "sentiment": r.sentiment,
                }
                for r in results_list
            ]

            # Step 3 — Feed IR results to LLM for synthesis
            llm_summary = synthesize_answer(client, user_query, ir_results)

            # Step 4 — LLM identifies most relevant tickers
            relevant_tickers = identify_relevant_tickers(client, user_query, ir_results)

            return jsonify({
                "original_query": user_query,
                "improved_query": improved_query,
                "ir_results": ir_results,
                "query_info": {
                    "preprocessed_query": response.preprocessed_query,
                    "active_dimensions": [
                        asdict(d) for d in response.query_dimensions
                    ],
                },
                "svd_info": {
                    "n_components": response.svd_n_components,
                    "explained_variance": response.svd_explained_variance,
                },
                "llm_summary": llm_summary,
                "relevant_tickers": relevant_tickers,
            })

        except Exception as e:
            logger.error(f"RAG pipeline error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/research-chat", methods=["POST"])
    def research_chat():
        """Conversational research assistant grounded in search results."""
        data = request.get_json() or {}
        messages = data.get("messages") or []
        context = data.get("context") or {}
        search_results = context.get("results") or []
        original_query = context.get("query") or ""

        if not messages:
            return jsonify({"error": "messages required"}), 400

        api_key = os.getenv("SPARK_API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY not set"}), 500

        try:
            client = LLMClient(api_key=api_key)

            # Build context from search results
            context_parts = []
            for i, r in enumerate(search_results[:10], 1):
                source_label = "News" if r.get("source") == "news" else "Reddit"
                date_str = r.get("date", "unknown")
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                tickers = ", ".join(r.get("tickers", [])) or "N/A"
                text = title
                if snippet and snippet != title:
                    text += f"\n{snippet}"
                context_parts.append(
                    f"[{i}] ({source_label}, {date_str}, Tickers: {tickers})\n{text}"
                )
            context_text = "\n\n---\n\n".join(context_parts) if context_parts else "No search results available."

            system_msg = {
                "role": "system",
                "content": (
                    "You are a financial research assistant for the RiskLens platform. "
                    "The user has searched for financial information and you have access to "
                    "the search results below. Answer follow-up questions based on these results.\n\n"
                    "Rules:\n"
                    "- Reference sources using [1], [2], etc. citations\n"
                    "- Be concise but thorough\n"
                    "- If the results don't contain enough information, say so\n"
                    "- Do NOT invent information not present in the sources\n"
                    "- Use **bold** for tickers and key terms\n"
                    "- Keep responses focused and under 3 paragraphs\n\n"
                    f"Original search query: \"{original_query}\"\n\n"
                    f"Search results:\n\n{context_text}"
                ),
            }

            # Build full message list: system + conversation history
            llm_messages = [system_msg] + [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ]

            resp = client.chat(llm_messages)
            content = (resp.get("content") or "Unable to generate response.").strip()
            return jsonify({"content": content})

        except Exception as e:
            logger.error(f"Research chat error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
