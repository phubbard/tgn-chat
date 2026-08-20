#!/usr/bin/env python3
"""One-command incremental rebuild of the podcast database.

Embeddings are cached by a hash of each chunk's *content text*, so only
chunks whose text actually changed are re-embedded. This makes rebuilds
truly incremental:

  - Cosmetic changes to episode.md (regenerated "## Show Notes"/"## Links"
    sections, frontmatter, whitespace) do NOT change the synopsis/transcript
    chunk text, so those chunks are reused from cache.
  - A shownotes.md change only alters the per-episode *link* chunk text, so
    only the ~1 link chunk per episode is re-embedded — not the thousands of
    transcript chunks.
  - New/edited episodes re-embed only their changed chunks.

The cache is simply the previous embedded JSONL (build/chunks.{model}.jsonl),
which is per-model, so cached vectors always match the active model.

Usage:
    python ingest/rebuild.py [--model bge-m3] [--data-dir data/inputs] [--force]
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import struct
import sys
import time
from datetime import datetime

from chunk import (
    parse_episode_md,
    parse_shownotes,
    extract_guests,
    make_synopsis_chunk,
    make_link_chunk,
    make_transcript_chunks,
)
from embed import get_embedding

import build_db
import sqlite_vec


def content_key(text):
    """Stable cache key for an embedding: sha256 of the chunk's content text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_embedding_cache(embedded_jsonl):
    """Return (cache, prev_episodes) from a previous embedded JSONL.

    cache maps content_key -> embedding vector; prev_episodes is the set of
    episode-number strings the previous build covered (for new/removed stats).
    """
    cache = {}
    prev_eps = set()
    if not os.path.exists(embedded_jsonl):
        return cache, prev_eps
    with open(embedded_jsonl, "r") as f:
        for line in f:
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            prev_eps.add(str(c["episode_number"]))
            emb = c.get("embedding")
            if emb:
                cache[content_key(c["content"])] = emb
    return cache, prev_eps


def scan_episodes(data_dir):
    """Return {episode_dir_name: episode.md_path} for all episodes on disk."""
    episodes = {}
    for name in os.listdir(data_dir):
        if not re.match(r"^\d+(\.\d+)?$", name):
            continue
        md_path = os.path.join(data_dir, name, "episode.md")
        if os.path.exists(md_path):
            episodes[name] = md_path
    return episodes


def chunk_episode(ep_num_str, data_dir, shownotes):
    """Chunk a single episode, return list of chunk dicts (no embeddings)."""
    md_path = os.path.join(data_dir, ep_num_str, "episode.md")
    episode_number = float(ep_num_str) if "." in ep_num_str else int(ep_num_str)
    episode = parse_episode_md(md_path)
    if episode["number"] is None:
        episode["number"] = episode_number

    guests = extract_guests(episode) or None
    chunks = []
    synopsis = make_synopsis_chunk(episode, episode_number, guests)
    if synopsis:
        chunks.append(synopsis)
    links = make_link_chunk(episode, episode_number, shownotes, guests)
    if links:
        chunks.append(links)
    chunks.extend(make_transcript_chunks(episode, episode_number, guests=guests))
    return chunks


def build_database(embedded_jsonl, db_output, model, data_dir):
    """Build SQLite + sqlite-vec + FTS5 database from an embedded JSONL."""
    if os.path.exists(db_output):
        os.remove(db_output)

    with open(embedded_jsonl, "r") as f:
        first = json.loads(f.readline())
        embedding_dim = len(first["embedding"])

    conn = sqlite3.connect(db_output)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    build_db.create_schema(conn, embedding_dim)

    episodes_seen = {}
    chunk_count = 0
    with open(embedded_jsonl, "r") as f:
        for line in f:
            chunk = json.loads(line)
            ep_num = chunk["episode_number"]

            if ep_num not in episodes_seen:
                mp3_url = episode_url = None
                ep_json_path = os.path.join(data_dir, str(ep_num), "episode.json")
                if os.path.exists(ep_json_path):
                    with open(ep_json_path, "r") as ej:
                        ep_meta = json.load(ej)
                        mp3_url = ep_meta.get("mp3_url")
                        episode_url = ep_meta.get("episode_url")

                synopsis = chunk["content"] if chunk["chunk_type"] == "synopsis" else None
                guests = json.dumps(chunk["guests"]) if chunk.get("guests") else None
                conn.execute(
                    "INSERT OR IGNORE INTO episodes (number, title, pub_date, synopsis, mp3_url, episode_url, guests) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ep_num, chunk["episode_title"], chunk["pub_date"], synopsis, mp3_url, episode_url, guests),
                )
                episodes_seen[ep_num] = conn.execute(
                    "SELECT id FROM episodes WHERE number = ?", (ep_num,)
                ).fetchone()[0]

            if chunk["chunk_type"] == "synopsis":
                conn.execute("UPDATE episodes SET synopsis = ? WHERE id = ?",
                             (chunk["content"], episodes_seen[ep_num]))

            conn.execute(
                "INSERT INTO chunks (episode_id, chunk_type, content, speakers, start_turn, end_turn) VALUES (?, ?, ?, ?, ?, ?)",
                (episodes_seen[ep_num], chunk["chunk_type"], chunk["content"],
                 chunk.get("speakers"), chunk.get("start_turn"), chunk.get("end_turn")),
            )
            chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            conn.execute("INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)",
                         (chunk_id, chunk["content"]))

            embedding_bytes = struct.pack(f"{len(chunk['embedding'])}f", *chunk["embedding"])
            conn.execute("INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                         (chunk_id, embedding_bytes))
            conn.execute("INSERT INTO chunks_emb (chunk_id, embedding) VALUES (?, ?)",
                         (chunk_id, embedding_bytes))

            chunk_count += 1
            if chunk_count % 500 == 0:
                conn.commit()

    conn.commit()
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", ("embedding_model", model))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", ("embedding_dim", str(embedding_dim)))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                 ("built_at", datetime.now().isoformat(timespec="seconds")))
    conn.commit()
    conn.close()
    return chunk_count, len(episodes_seen), embedding_dim


def main():
    parser = argparse.ArgumentParser(description="Incremental rebuild of podcast database")
    parser.add_argument("--model", default="bge-m3", help="Embedding model (default: bge-m3)")
    parser.add_argument("--data-dir", default="data/inputs", help="Episode data directory")
    parser.add_argument("--force", action="store_true", help="Ignore the embedding cache and re-embed everything")
    args = parser.parse_args()

    embedded_jsonl = f"build/chunks.{args.model}.jsonl"
    db_output = f"build/podcast.{args.model}.db"
    db_link = "build/podcast.db"

    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} not found", file=sys.stderr)
        sys.exit(1)

    os.makedirs("build", exist_ok=True)

    # Embedding cache (content-keyed) from the previous build.
    if args.force:
        cache, prev_eps = {}, set()
        print("Force: ignoring embedding cache — re-embedding everything", file=sys.stderr)
    else:
        cache, prev_eps = load_embedding_cache(embedded_jsonl)
        print(f"Cache: {len(cache)} embeddings from previous build "
              f"({len(prev_eps)} episodes)", file=sys.stderr)

    # Shownotes drive the per-episode link chunks.
    shownotes_path = os.path.join(args.data_dir, "shownotes.md")
    shownotes = None
    if os.path.exists(shownotes_path):
        shownotes = parse_shownotes(shownotes_path)
        print(f"Shownotes: {len(shownotes)} episodes with links", file=sys.stderr)
    else:
        print("Warning: shownotes.md not found, using episode.md links", file=sys.stderr)

    # Chunk every episode on disk (cheap; no LM Studio).
    on_disk = scan_episodes(args.data_dir)
    all_ep_strs = sorted(on_disk, key=float)
    cur_eps = set(all_ep_strs)
    new_eps = sorted(cur_eps - prev_eps, key=float)
    removed_eps = sorted(prev_eps - cur_eps, key=float)

    t0 = time.time()
    all_chunks = []
    for ep_str in all_ep_strs:
        all_chunks.extend(chunk_episode(ep_str, args.data_dir, shownotes))
    print(f"\nChunked {len(all_ep_strs)} episodes -> {len(all_chunks)} chunks", file=sys.stderr)
    if new_eps:
        print(f"  new episodes: {', '.join(new_eps)}", file=sys.stderr)
    if removed_eps:
        print(f"  removed episodes: {', '.join(removed_eps)}", file=sys.stderr)

    # Embed with cache: reuse unchanged text, embed only new/changed text.
    reused = embedded = 0
    for chunk in all_chunks:
        key = content_key(chunk["content"])
        cached = cache.get(key)
        if cached is not None:
            chunk["embedding"] = cached
            reused += 1
        else:
            chunk["embedding"] = get_embedding(chunk["content"], args.model)
            cache[key] = chunk["embedding"]
            embedded += 1
            if embedded % 50 == 0:
                elapsed = time.time() - t0
                rate = embedded / elapsed if elapsed > 0 else 0
                print(f"  embedded {embedded} new chunks ({rate:.1f}/s)", file=sys.stderr)

    print(f"\nEmbeddings: {reused} reused (cache hit), {embedded} newly embedded", file=sys.stderr)

    # Write the full embedded JSONL (this becomes the next build's cache).
    print(f"Writing {embedded_jsonl}...", file=sys.stderr)
    with open(embedded_jsonl, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Build the database.
    print(f"Building database ({len(all_chunks)} chunks)...", file=sys.stderr)
    chunk_count, ep_count, dim = build_database(embedded_jsonl, db_output, args.model, args.data_dir)

    # Point podcast.db -> podcast.{model}.db
    if os.path.exists(db_link):
        os.remove(db_link)
    os.link(db_output, db_link)

    elapsed = time.time() - t0
    db_size = os.path.getsize(db_output) / (1024 * 1024)
    print(f"\nDone in {elapsed:.1f}s: {chunk_count} chunks, {ep_count} episodes "
          f"-> {db_output} ({db_size:.1f} MB)", file=sys.stderr)
    print(f"  Reused {reused} embeddings, re-embedded {embedded} "
          f"({100 * reused / max(1, reused + embedded):.0f}% cache hit)", file=sys.stderr)
    if removed_eps:
        print(f"  Removed: episodes {', '.join(removed_eps)}", file=sys.stderr)


if __name__ == "__main__":
    main()
