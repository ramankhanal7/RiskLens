"""
Routes: React app serving and episode search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import json
import os
from dataclasses import asdict
from datetime import date, timedelta
from flask import send_from_directory, request, jsonify
from models import db, Episode, Review

# -- AI toggle ----------------------------------------------------------------
USE_LLM = True
# ------------------------------------------------------------------------------


def json_search(query):
    if not query or not query.strip():
        query = "Kardashian"
    results = db.session.query(Episode, Review).join(
        Review, Episode.id == Review.id
    ).filter(
        Episode.title.ilike(f'%{query}%')
    ).all()
    matches = []
    for episode, review in results:
        matches.append({
            'title': episode.title,
            'descr': episode.descr,
            'imdb_rating': review.imdb_rating
        })
    return matches


def register_routes(app):
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": USE_LLM})

    @app.route("/api/episodes")
    def episodes_search():
        text = request.args.get("title", "")
        return jsonify(json_search(text))

    @app.route("/api/sentiment")
    def sentiment():
        ticker = request.args.get("ticker", "").strip().upper()
        if not ticker:
            return jsonify({"error": "ticker required"}), 400
        today = date.today()
        from_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        try:
            from financial_data import score_portfolio
            scores = score_portfolio([ticker], from_date, to_date)
            return jsonify({"ticker": ticker, "score": scores.get(ticker, 0.0)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/validate-ticker")
    def validate_ticker():
        ticker = request.args.get("ticker", "").strip().upper()
        if not ticker:
            return jsonify({"ticker": "", "valid": False, "name": None})
        from ticker_validation import is_valid_ticker, get_company_name
        valid = is_valid_ticker(ticker)
        name = get_company_name(ticker) if valid else None
        return jsonify({"ticker": ticker, "valid": valid, "name": name})

    @app.route("/api/search")
    def market_search():
        q = request.args.get("q", "").strip()
        mode = request.args.get("mode", "hybrid").strip().lower()
        source_filter = request.args.get("source", "").strip().lower()
        if not q:
            return jsonify({"results": [], "query_info": {}, "svd_info": {}})
        try:
            from query import search
            response = search(q, mode=mode)
            results_list = response.results
            if source_filter in ("news", "reddit"):
                results_list = [r for r in results_list if r.source == source_filter]
            return jsonify({
                "results": [
                    {
                        "score": r.score, "tickers": r.tickers, "title": r.title,
                        "url": r.url, "snippet": r.snippet, "date": r.date,
                        "source": r.source, "sentiment": r.sentiment,
                    }
                    for r in results_list
                ],
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
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if USE_LLM:
        from llm_routes import register_chat_route, identify_relevant_tickers
        register_chat_route(app)

        @app.route("/api/ticker-context", methods=["POST"])
        def ticker_context():
            """Lightweight endpoint: given query + tickers, return LLM descriptions."""
            data = request.get_json() or {}
            query = (data.get("query") or "").strip()
            tickers = data.get("tickers") or []
            if not query or not tickers:
                return jsonify([])
            api_key = os.getenv("SPARK_API_KEY")
            if not api_key:
                return jsonify([])
            try:
                from infosci_spark_client import LLMClient
                client = LLMClient(api_key=api_key)
                fake_results = [{"tickers": tickers}]
                return jsonify(identify_relevant_tickers(client, query, fake_results))
            except Exception:
                return jsonify([{"ticker": t, "reason": ""} for t in tickers[:8]])
