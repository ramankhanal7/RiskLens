"""
Routes: React app serving and episode search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import json
import os
from datetime import date, timedelta
from flask import send_from_directory, request, jsonify
from models import db, Episode, Review

# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = False
# USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────


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

    @app.route("/api/search")
    def market_search():
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify([])
        try:
            from query import search
            results = search(q)
            return jsonify([
                {"score": r.score, "tickers": r.tickers, "title": r.title, "url": r.url}
                for r in results
            ])
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
