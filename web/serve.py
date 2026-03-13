"""Server-side search API for TGN RAG chatbot.

Handles embedding + hybrid vector/keyword search so the browser
doesn't need to download the 143MB database.

Usage:
    python web/serve.py [--db build/podcast.db] [--port 5555]
"""

import argparse
import json
import os
import re
import sqlite3
import struct
import sys
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests
import sqlite_vec

OLLAMA_URL = "http://127.0.0.1:11434"
TOP_K = 16

STOPWORDS = {
    "the", "be", "to", "of", "and", "in", "that", "have", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
    "his", "by", "from", "they", "we", "say", "her", "she", "or", "an",
    "will", "my", "one", "all", "would", "there", "their", "what", "so",
    "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only",
    "come", "its", "over", "think", "also", "back", "after", "use",
    "two", "how", "our", "work", "first", "well", "way", "even", "new",
    "want", "because", "any", "these", "give", "day", "most", "us",
    "tell", "does", "did", "been", "has", "had", "are", "was", "were",
    "being", "is", "am", "much", "many", "very", "own", "too", "here",
    "where", "why", "let", "may", "should", "more", "still", "find",
    "long", "thing", "said", "each", "got", "same", "name", "times",
    "brand", "watch", "watches", "episode", "show",
}

db = None
embedding_model = None
embedding_dim = None


def init_db(db_path):
    global db, embedding_model, embedding_dim
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)

    meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
    embedding_model = meta["embedding_model"]
    embedding_dim = int(meta["embedding_dim"])

    episodes = db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"DB loaded: {episodes} episodes, {chunks} chunks "
          f"(model: {embedding_model}, dim: {embedding_dim})", file=sys.stderr)


def get_embedding(text):
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": embedding_model, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def serialize_float32(vec):
    return struct.pack(f"{len(vec)}f", *vec)


def search_vec(query_vec, top_k):
    query_bytes = serialize_float32(query_vec)
    rows = db.execute("""
        SELECT
            c.content, c.chunk_type, c.speakers,
            e.number, e.title, e.episode_url,
            v.distance
        FROM chunks_vec v
        JOIN chunks c ON c.id = v.chunk_id
        JOIN episodes e ON e.id = c.episode_id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
    """, (query_bytes, top_k)).fetchall()

    return [
        {
            "content": r[0], "chunk_type": r[1], "speakers": r[2],
            "episode_number": r[3], "episode_title": r[4],
            "episode_url": r[5], "distance": r[6],
        }
        for r in rows
    ]


def search_fts(query_text, top_k):
    tokens = re.split(r"[\s\-–—]+", query_text.lower())
    tokens = [re.sub(r"[^\w]", "", t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
    if not tokens:
        return []

    fts_query = " OR ".join(f'"{t}"' for t in tokens)
    try:
        rows = db.execute("""
            SELECT
                c.content, c.chunk_type, c.speakers,
                e.number, e.title, e.episode_url,
                chunks_fts.rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN episodes e ON e.id = c.episode_id
            WHERE chunks_fts MATCH ?
            ORDER BY chunks_fts.rank
            LIMIT ?
        """, (fts_query, top_k)).fetchall()

        return [
            {
                "content": r[0], "chunk_type": r[1], "speakers": r[2],
                "episode_number": r[3], "episode_title": r[4],
                "episode_url": r[5], "distance": 0, "fts": True,
            }
            for r in rows
        ]
    except Exception as e:
        print(f"FTS error: {e}", file=sys.stderr)
        return []


def merge_results(vec_results, fts_results, top_k):
    seen = {}
    for i, r in enumerate(vec_results):
        seen[r["content"]] = {**r, "vec_rank": i}

    mid_distance = (
        vec_results[min(top_k // 2, len(vec_results) - 1)]["distance"]
        if vec_results else 0.5
    )

    for fts in fts_results:
        if fts["content"] in seen:
            seen[fts["content"]]["distance"] *= 0.5
        else:
            seen[fts["content"]] = {**fts, "distance": mid_distance * 0.8}

    merged = sorted(seen.values(), key=lambda x: x["distance"])
    return merged[:top_k]


def hybrid_search(query_text, top_k=TOP_K):
    query_vec = get_embedding(query_text)
    vec_results = search_vec(query_vec, top_k * 2)
    fts_results = search_fts(query_text, top_k)
    print(f"Search: {len(vec_results)} vector, {len(fts_results)} FTS", file=sys.stderr)
    return merge_results(vec_results, fts_results, top_k)


def get_db_info():
    meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
    return {
        "model": meta["embedding_model"],
        "dim": int(meta["embedding_dim"]),
        "episodes": db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
        "chunks": db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
    }


LOG_DIR = "logs"


def write_log(session_id, event):
    """Append a log event to the session's markdown file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    # Use first 8 chars of UUID for filename readability
    short_id = session_id[:8]
    now = datetime.now()
    ts = now.strftime("%H:%M:%S")

    event_type = event.get("type", "unknown")
    filepath = os.path.join(LOG_DIR, f"{now.strftime('%Y-%m-%d')}_{short_id}.md")
    is_new = not os.path.exists(filepath)

    with open(filepath, "a", encoding="utf-8") as f:
        if is_new:
            f.write(f"# Session {short_id}\n\n")
            f.write(f"**Started:** {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            ua = event.get("user_agent", "")
            if ua:
                f.write(f"**User-Agent:** {ua}\n")
            f.write("\n---\n\n")

        if event_type == "session_start":
            return  # Header already written above

        if event_type == "query":
            f.write(f"## {ts} — Query\n\n")
            f.write(f"**Model:** {event.get('model', '?')}\n\n")
            f.write(f"**Query:** {event.get('query', '')}\n\n")

            search_t = event.get("search_time_s")
            ttft = event.get("ttft_s")
            total_t = event.get("total_time_s")
            tokens = event.get("tokens")
            tok_s = event.get("tok_per_sec")
            eps = event.get("source_episodes", [])

            f.write("| Metric | Value |\n|---|---|\n")
            if search_t is not None:
                f.write(f"| Search | {search_t}s |\n")
            if ttft is not None:
                f.write(f"| Time to first token | {ttft}s |\n")
            if total_t is not None:
                f.write(f"| Total time | {total_t}s |\n")
            if tokens is not None:
                f.write(f"| Tokens | {tokens} |\n")
            if tok_s is not None:
                f.write(f"| Token rate | {tok_s} tok/s |\n")
            if eps:
                f.write(f"| Source episodes | {', '.join(str(e) for e in eps)} |\n")
            f.write("\n")

            response = event.get("response", "")
            if response:
                # Indent response as blockquote, truncate long ones
                lines = response.split("\n")
                f.write("**Response:**\n\n")
                for line in lines:
                    f.write(f"> {line}\n")
                f.write("\n")

            f.write("---\n\n")

        elif event_type == "error":
            f.write(f"## {ts} — Error\n\n")
            f.write(f"```\n{event.get('error', '')}\n```\n\n---\n\n")


class SearchHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/search":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            query = body.get("query", "")
            top_k = body.get("top_k", TOP_K)

            try:
                results = hybrid_search(query, top_k)
                clean = [
                    {k: v for k, v in r.items() if k not in ("fts", "vec_rank")}
                    for r in results
                ]
                self._json_response(200, {"results": clean})
            except Exception as e:
                print(f"Search error: {e}", file=sys.stderr)
                self._json_response(500, {"error": str(e)})

        elif self.path == "/log":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            session_id = body.pop("session_id", "unknown")
            try:
                write_log(session_id, body)
            except Exception as e:
                print(f"Log error: {e}", file=sys.stderr)
            self._json_response(200, {"ok": True})

        else:
            self._json_response(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/search/info":
            self._json_response(200, get_db_info())
        else:
            self._json_response(404, {"error": "not found"})

    def _json_response(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter logging
        print(f"  {args[0]}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="TGN search API server")
    parser.add_argument("--db", default="build/podcast.db", help="Database path")
    parser.add_argument("--port", type=int, default=5555, help="Listen port")
    args = parser.parse_args()

    init_db(args.db)
    server = HTTPServer(("127.0.0.1", args.port), SearchHandler)
    print(f"Search API listening on http://127.0.0.1:{args.port}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
