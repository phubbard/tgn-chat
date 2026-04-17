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

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234")
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
db_path = None
embedding_model = None
embedding_dim = None
APP_VERSION = None


def compute_app_version():
    """Short git SHA + commit date, or 'unknown' if not a git checkout."""
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        date = subprocess.check_output(
            ["git", "log", "-1", "--format=%cs", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return f"{sha} ({date})"
    except Exception:
        return "unknown"


def init_db(path):
    global db, db_path, embedding_model, embedding_dim
    db_path = path
    db = sqlite3.connect(path, check_same_thread=False)
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
        f"{LM_STUDIO_URL}/v1/embeddings",
        json={"model": embedding_model, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def serialize_float32(vec):
    return struct.pack(f"{len(vec)}f", *vec)


def search_vec(query_vec, top_k):
    query_bytes = serialize_float32(query_vec)
    rows = db.execute("""
        SELECT
            c.content, c.chunk_type, c.speakers,
            e.number, e.title, e.episode_url, e.pub_date, e.topics,
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
            "episode_url": r[5], "pub_date": r[6], "topics": r[7],
            "distance": r[8],
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
                e.number, e.title, e.episode_url, e.pub_date, e.topics,
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
                "episode_url": r[5], "pub_date": r[6], "topics": r[7],
                "distance": 0, "fts": True,
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
    latest_ep = db.execute("SELECT MAX(number) FROM episodes").fetchone()[0]
    built_at = meta.get("built_at")
    if not built_at:
        # Fallback for DBs built before built_at was tracked
        built_at = datetime.fromtimestamp(os.path.getmtime(db_path)).isoformat(timespec="seconds")
    return {
        "model": meta["embedding_model"],
        "dim": int(meta["embedding_dim"]),
        "episodes": db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
        "chunks": db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        "latest_episode": int(latest_ep) if latest_ep is not None else None,
        "built_at": built_at,
        "app_version": APP_VERSION,
    }


LOG_DIR = "logs"

# In-memory ring buffer of recent events for the monitoring dashboard
import collections
import threading

_events_lock = threading.Lock()
_recent_events = collections.deque(maxlen=200)


EVENTS_JSONL = os.path.join(LOG_DIR, "events.jsonl")
CHATS_DB_PATH = os.path.join(LOG_DIR, "chats.db")
chats_db = None


def init_chats_db():
    """Create the chats + messages tables used for shareable chat URLs."""
    global chats_db
    os.makedirs(LOG_DIR, exist_ok=True)
    chats_db = sqlite3.connect(CHATS_DB_PATH, check_same_thread=False)
    chats_db.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            title TEXT,
            session_id TEXT,
            user_agent TEXT,
            message_count INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updated_at DESC);

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT,
            query_id TEXT,
            source_episodes TEXT,
            tokens INTEGER,
            ttft_s REAL,
            total_time_s REAL,
            tok_per_sec REAL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
    """)
    chats_db.commit()


def mirror_query_to_chats_db(chat_id, session_id, user_agent, event):
    """Mirror a query event into the chats DB as user + assistant messages."""
    if not chat_id:
        return
    now = event.get("timestamp") or datetime.now().isoformat()
    query = event.get("query", "")
    response = event.get("response", "")
    title = (query[:80] + ("…" if len(query) > 80 else "")) if query else None

    # Upsert chat row
    row = chats_db.execute("SELECT id, title FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if row is None:
        chats_db.execute(
            "INSERT INTO chats (id, created_at, updated_at, title, session_id, user_agent, message_count) "
            "VALUES (?, ?, ?, ?, ?, ?, 0)",
            (chat_id, now, now, title, session_id, user_agent),
        )
    else:
        # Keep first-query title; just bump updated_at
        chats_db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))

    # Insert user + assistant messages
    chats_db.execute(
        "INSERT INTO messages (chat_id, created_at, role, content, model, query_id) "
        "VALUES (?, ?, 'user', ?, ?, ?)",
        (chat_id, now, query, event.get("model"), event.get("query_id")),
    )
    chats_db.execute(
        "INSERT INTO messages (chat_id, created_at, role, content, model, query_id, "
        "source_episodes, tokens, ttft_s, total_time_s, tok_per_sec) "
        "VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            chat_id, now, response, event.get("model"), event.get("query_id"),
            json.dumps(event.get("source_episodes") or []),
            event.get("tokens"), event.get("ttft_s"),
            event.get("total_time_s"), event.get("tok_per_sec"),
        ),
    )
    chats_db.execute(
        "UPDATE chats SET message_count = message_count + 2 WHERE id = ?", (chat_id,)
    )
    chats_db.commit()


def list_chats(limit, before):
    """Return chats ordered newest-first, cursor-paginated by updated_at."""
    if before:
        rows = chats_db.execute(
            "SELECT id, created_at, updated_at, title, message_count "
            "FROM chats WHERE updated_at < ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (before, limit),
        ).fetchall()
    else:
        rows = chats_db.execute(
            "SELECT id, created_at, updated_at, title, message_count "
            "FROM chats ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"id": r[0], "created_at": r[1], "updated_at": r[2],
         "title": r[3], "message_count": r[4]}
        for r in rows
    ]


def get_chat(chat_id):
    """Return a single chat + ordered messages, or None."""
    meta = chats_db.execute(
        "SELECT id, created_at, updated_at, title, message_count FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()
    if not meta:
        return None
    messages = chats_db.execute(
        "SELECT id, created_at, role, content, model, query_id, source_episodes, "
        "tokens, ttft_s, total_time_s, tok_per_sec FROM messages "
        "WHERE chat_id = ? ORDER BY id",
        (chat_id,),
    ).fetchall()

    # Collect all episode numbers referenced across messages, resolve once.
    all_eps = set()
    parsed_sources = []
    for m in messages:
        eps_json = m[6]
        eps = json.loads(eps_json) if eps_json else []
        parsed_sources.append(eps)
        all_eps.update(eps)
    ep_info = {}
    if all_eps:
        placeholders = ",".join("?" for _ in all_eps)
        rows = db.execute(
            f"SELECT number, title, episode_url FROM episodes WHERE number IN ({placeholders})",
            tuple(all_eps),
        ).fetchall()
        ep_info = {r[0]: {"number": r[0], "title": r[1], "url": r[2]} for r in rows}

    return {
        "id": meta[0], "created_at": meta[1], "updated_at": meta[2],
        "title": meta[3], "message_count": meta[4],
        "messages": [
            {
                "created_at": m[1], "role": m[2], "content": m[3],
                "model": m[4], "query_id": m[5],
                "sources": [ep_info[n] for n in parsed_sources[i] if n in ep_info],
                "tokens": m[7], "ttft_s": m[8],
                "total_time_s": m[9], "tok_per_sec": m[10],
            }
            for i, m in enumerate(messages)
        ],
    }


def _load_events_from_disk():
    """Load all events from the JSONL log file."""
    if not os.path.exists(EVENTS_JSONL):
        return []
    events = []
    with open(EVENTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def write_log(session_id, event):
    """Append a log event to the session's markdown file and JSONL log."""
    enriched = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        **event,
    }

    # Store in memory for dashboard
    with _events_lock:
        _recent_events.append(enriched)

    # Append to structured JSONL log
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(EVENTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    # Mirror query events into the chats DB so URLs can reload the thread
    if event.get("type") == "query":
        try:
            mirror_query_to_chats_db(
                event.get("chat_id"),
                session_id,
                event.get("user_agent"),
                enriched,
            )
        except Exception as e:
            print(f"chats.db mirror error: {e}", file=sys.stderr)
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
            query_id = event.get("query_id", "")
            qid_suffix = f" ({query_id})" if query_id else ""
            f.write(f"## {ts} — Query{qid_suffix}\n\n")
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

        elif event_type == "feedback":
            vote = event.get("vote", "?")
            icon = "\U0001f44d" if vote == "up" else "\U0001f44e"
            query_id = event.get("query_id", "?")
            f.write(f"### {ts} — Feedback {icon} (query {query_id})\n\n")
            f.write(f"**Query:** {event.get('query', '')}\n")
            f.write(f"**Model:** {event.get('model', '?')}\n\n---\n\n")

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
        from urllib.parse import urlparse, parse_qs

        if self.path == "/search/info":
            self._json_response(200, get_db_info())
        elif self.path.startswith("/monitor/events"):
            qs = parse_qs(urlparse(self.path).query)
            since = qs.get("since", [None])[0]
            history = qs.get("history", ["0"])[0] == "1"

            if history and not since:
                # Load full history from disk
                events = _load_events_from_disk()
            else:
                # Use in-memory buffer for live polling
                with _events_lock:
                    events = list(_recent_events)

            if since:
                events = [e for e in events if e["timestamp"] > since]
            self._json_response(200, {"events": events})
        elif self.path == "/monitor":
            self._serve_file("web/monitor.html", "text/html")
        elif self.path.startswith("/chats"):
            parsed = urlparse(self.path)
            # /chats/{id} — single thread
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) == 2:
                chat = get_chat(parts[1])
                if chat is None:
                    self._json_response(404, {"error": "not found"})
                else:
                    self._json_response(200, chat)
                return
            # /chats — paginated list
            qs = parse_qs(parsed.query)
            try:
                limit = min(int(qs.get("limit", ["20"])[0]), 100)
            except ValueError:
                limit = 20
            before = qs.get("before", [None])[0]
            self._json_response(200, {"chats": list_chats(limit, before)})
        else:
            self._json_response(404, {"error": "not found"})

    def _serve_file(self, path, content_type):
        try:
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
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

    global APP_VERSION
    APP_VERSION = compute_app_version()
    print(f"App version: {APP_VERSION}", file=sys.stderr)
    init_db(args.db)
    init_chats_db()
    server = HTTPServer(("0.0.0.0", args.port), SearchHandler)
    print(f"Search API listening on http://127.0.0.1:{args.port}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
