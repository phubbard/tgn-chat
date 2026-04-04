"""Build SQLite database with sqlite-vec from embedded chunks.

Reads embedded chunks JSONL and creates a SQLite database with text content
and vector embeddings for browser-side search.

Usage:
    python ingest/build_db.py --model bge-m3 [--input build/chunks.bge-m3.jsonl] [--output build/podcast.bge-m3.db]
"""

import argparse
import json
import os
import sqlite3
import struct
import sys

import sqlite_vec


def serialize_float32(vec):
    """Serialize a list of floats to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def create_schema(conn, embedding_dim):
    """Create database tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY,
            number REAL UNIQUE NOT NULL,
            title TEXT NOT NULL,
            pub_date TEXT,
            synopsis TEXT,
            mp3_url TEXT,
            episode_url TEXT,
            guests TEXT,
            topics TEXT
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            episode_id INTEGER NOT NULL REFERENCES episodes(id),
            chunk_type TEXT NOT NULL,
            content TEXT NOT NULL,
            speakers TEXT,
            start_turn INTEGER,
            end_turn INTEGER,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS chunks_emb (
            chunk_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL
        );
    """)

    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{embedding_dim}]
        );
    """)

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            content_rowid='id',
            tokenize='porter unicode61'
        );
    """)


def main():
    parser = argparse.ArgumentParser(description="Build SQLite + sqlite-vec database")
    parser.add_argument("--model", required=True, help="Model name (for file naming)")
    parser.add_argument("--input", default=None, help="Input JSONL with embeddings")
    parser.add_argument("--output", default=None, help="Output .db path")
    parser.add_argument("--episode-json-dir", default="data/inputs",
                        help="Directory with episode.json files for metadata")
    args = parser.parse_args()

    if args.input is None:
        args.input = f"build/chunks.{args.model}.jsonl"
    if args.output is None:
        args.output = f"build/podcast.{args.model}.db"

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run embed.py first.", file=sys.stderr)
        sys.exit(1)

    # Remove existing DB to rebuild
    if os.path.exists(args.output):
        os.remove(args.output)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Peek at first chunk to determine embedding dimension
    with open(args.input, "r") as f:
        first = json.loads(f.readline())
        embedding_dim = len(first["embedding"])
    print(f"Embedding dimension: {embedding_dim}", file=sys.stderr)

    conn = sqlite3.connect(args.output)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    create_schema(conn, embedding_dim)

    # Track episodes we've already inserted
    episodes_seen = {}
    chunk_count = 0

    with open(args.input, "r") as f:
        for line in f:
            chunk = json.loads(line)
            ep_num = chunk["episode_number"]

            # Insert episode if not seen
            if ep_num not in episodes_seen:
                # Try to load extra metadata from episode.json
                mp3_url = None
                episode_url = None
                ep_json_path = os.path.join(
                    args.episode_json_dir, str(ep_num), "episode.json"
                )
                if os.path.exists(ep_json_path):
                    with open(ep_json_path, "r") as ej:
                        ep_meta = json.load(ej)
                        mp3_url = ep_meta.get("mp3_url")
                        episode_url = ep_meta.get("episode_url")

                synopsis = chunk["content"] if chunk["chunk_type"] == "synopsis" else None
                guests = json.dumps(chunk["guests"]) if chunk.get("guests") else None

                conn.execute(
                    """INSERT OR IGNORE INTO episodes
                       (number, title, pub_date, synopsis, mp3_url, episode_url, guests)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (ep_num, chunk["episode_title"], chunk["pub_date"],
                     synopsis, mp3_url, episode_url, guests),
                )

                episodes_seen[ep_num] = conn.execute(
                    "SELECT id FROM episodes WHERE number = ?", (ep_num,)
                ).fetchone()[0]

            # Update synopsis if we encounter it after initial insert
            if chunk["chunk_type"] == "synopsis":
                conn.execute(
                    "UPDATE episodes SET synopsis = ? WHERE id = ?",
                    (chunk["content"], episodes_seen[ep_num]),
                )

            # Insert chunk
            conn.execute(
                """INSERT INTO chunks
                   (episode_id, chunk_type, content, speakers, start_turn, end_turn)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (episodes_seen[ep_num], chunk["chunk_type"], chunk["content"],
                 chunk.get("speakers"), chunk.get("start_turn"), chunk.get("end_turn")),
            )
            chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Insert into FTS5 index
            conn.execute(
                "INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)",
                (chunk_id, chunk["content"]),
            )

            # Insert embedding into both vec0 (for native sqlite-vec) and
            # regular table (for brute-force fallback in browsers without vec0)
            embedding_bytes = serialize_float32(chunk["embedding"])
            conn.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, embedding_bytes),
            )
            conn.execute(
                "INSERT INTO chunks_emb (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, embedding_bytes),
            )

            chunk_count += 1
            if chunk_count % 500 == 0:
                conn.commit()
                print(f"  {chunk_count} chunks inserted", file=sys.stderr)

    conn.commit()

    # Store model info
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)
    """)
    conn.execute(
        "INSERT OR REPLACE INTO meta VALUES (?, ?)",
        ("embedding_model", args.model),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta VALUES (?, ?)",
        ("embedding_dim", str(embedding_dim)),
    )
    conn.commit()
    conn.close()

    db_size = os.path.getsize(args.output) / (1024 * 1024)
    print(
        f"\nDone: {chunk_count} chunks, {len(episodes_seen)} episodes -> "
        f"{args.output} ({db_size:.1f} MB)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
