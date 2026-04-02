#!/usr/bin/env python3
"""One-command incremental rebuild of the podcast database.

Detects new, changed, and deleted episodes by comparing content hashes
against the previous build. Only re-chunks and re-embeds what changed,
then rebuilds the full database.

Usage:
    python ingest/rebuild.py [--model bge-m3] [--data-dir data/inputs] [--force]
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

from chunk import (
    parse_episode_md,
    parse_shownotes,
    make_synopsis_chunk,
    make_link_chunk,
    make_transcript_chunks,
)
from embed import get_embedding

# build_db expects to be importable
import build_db


MANIFEST_FILE = "build/manifest.json"


def hash_file(path):
    """Return hex SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest():
    """Load the previous build manifest (episode hashes + shownotes hash)."""
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {"episodes": {}, "shownotes": None}


def save_manifest(manifest):
    """Save the build manifest."""
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def scan_episodes(data_dir):
    """Return dict of {episode_dir_name: episode.md_path} for all episodes on disk."""
    episodes = {}
    for name in os.listdir(data_dir):
        if not re.match(r"^\d+(\.\d+)?$", name):
            continue
        md_path = os.path.join(data_dir, name, "episode.md")
        if os.path.exists(md_path):
            episodes[name] = md_path
    return episodes


def diff_episodes(data_dir, old_manifest, force=False):
    """Compare disk state to manifest. Returns (added, changed, deleted, unchanged) sets of ep number strings."""
    on_disk = scan_episodes(data_dir)
    old_eps = old_manifest.get("episodes", {})

    added = set()
    changed = set()
    deleted = set()
    unchanged = set()

    for ep_str, md_path in on_disk.items():
        current_hash = hash_file(md_path)
        if force or ep_str not in old_eps:
            added.add(ep_str)
        elif old_eps[ep_str] != current_hash:
            changed.add(ep_str)
        else:
            unchanged.add(ep_str)

    for ep_str in old_eps:
        if ep_str not in on_disk:
            deleted.add(ep_str)

    return added, changed, deleted, unchanged


def chunk_episode(ep_num_str, data_dir, shownotes):
    """Chunk a single episode, return list of chunk dicts."""
    md_path = os.path.join(data_dir, ep_num_str, "episode.md")
    episode_number = float(ep_num_str) if "." in ep_num_str else int(ep_num_str)
    episode = parse_episode_md(md_path)
    if episode["number"] is None:
        episode["number"] = episode_number

    chunks = []
    synopsis = make_synopsis_chunk(episode, episode_number)
    if synopsis:
        chunks.append(synopsis)
    links = make_link_chunk(episode, episode_number, shownotes)
    if links:
        chunks.append(links)
    chunks.extend(make_transcript_chunks(episode, episode_number))
    return chunks


def embed_chunks(chunks, model):
    """Add embedding vectors to a list of chunks. Modifies in place."""
    for chunk in chunks:
        chunk["embedding"] = get_embedding(chunk["content"], model)


def main():
    parser = argparse.ArgumentParser(description="Incremental rebuild of podcast database")
    parser.add_argument("--model", default="bge-m3", help="Embedding model (default: bge-m3)")
    parser.add_argument("--data-dir", default="data/inputs", help="Episode data directory")
    parser.add_argument("--force", action="store_true", help="Force full rebuild ignoring manifest")
    args = parser.parse_args()

    embedded_jsonl = f"build/chunks.{args.model}.jsonl"
    db_output = f"build/podcast.{args.model}.db"
    db_link = "build/podcast.db"

    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} not found", file=sys.stderr)
        sys.exit(1)

    os.makedirs("build", exist_ok=True)

    # Load previous manifest
    old_manifest = load_manifest() if not args.force else {"episodes": {}, "shownotes": None}

    # Check shownotes
    shownotes_path = os.path.join(args.data_dir, "shownotes.md")
    shownotes = None
    shownotes_hash = None
    shownotes_changed = False
    if os.path.exists(shownotes_path):
        shownotes = parse_shownotes(shownotes_path)
        shownotes_hash = hash_file(shownotes_path)
        shownotes_changed = shownotes_hash != old_manifest.get("shownotes")
        print(f"Shownotes: {len(shownotes)} episodes with links" +
              (" (changed)" if shownotes_changed else " (unchanged)"), file=sys.stderr)

    # Diff episodes
    added, changed, deleted, unchanged = diff_episodes(args.data_dir, old_manifest, args.force)

    # If shownotes changed, all episodes with link chunks need re-chunking
    if shownotes_changed and not args.force:
        # Move unchanged episodes to changed since their link chunks may differ
        changed = changed | unchanged
        unchanged = set()
        print("Shownotes changed — re-chunking all episodes", file=sys.stderr)

    print(f"\nEpisodes: {len(added)} new, {len(changed)} changed, "
          f"{len(deleted)} deleted, {len(unchanged)} unchanged", file=sys.stderr)

    if not added and not changed and not deleted:
        print("Nothing to do.", file=sys.stderr)
        return

    # Load existing embedded chunks for unchanged episodes
    existing_chunks = {}  # ep_num_str -> list of embedded chunk dicts
    if os.path.exists(embedded_jsonl) and unchanged:
        print(f"Loading cached embeddings for {len(unchanged)} unchanged episodes...", file=sys.stderr)
        with open(embedded_jsonl, "r") as f:
            for line in f:
                chunk = json.loads(line)
                ep_str = str(chunk["episode_number"])
                if ep_str in unchanged:
                    existing_chunks.setdefault(ep_str, []).append(chunk)

    # Chunk and embed new/changed episodes
    to_process = sorted(added | changed, key=float)
    total = len(to_process)
    t0 = time.time()

    if total > 0:
        print(f"\nChunking and embedding {total} episodes...", file=sys.stderr)

    new_chunks = {}  # ep_num_str -> list of embedded chunk dicts
    chunks_embedded = 0
    for i, ep_str in enumerate(to_process):
        chunks = chunk_episode(ep_str, args.data_dir, shownotes)
        embed_chunks(chunks, args.model)
        new_chunks[ep_str] = chunks
        chunks_embedded += len(chunks)
        elapsed = time.time() - t0
        rate = chunks_embedded / elapsed if elapsed > 0 else 0
        print(f"  [{i+1}/{total}] Episode {ep_str}: {len(chunks)} chunks "
              f"({chunks_embedded} total, {rate:.1f} chunks/s)", file=sys.stderr)

    # Write full embedded JSONL (unchanged + new/changed, sorted by episode)
    all_ep_strs = sorted(
        (set(existing_chunks.keys()) | set(new_chunks.keys())),
        key=float,
    )
    print(f"\nWriting {embedded_jsonl}...", file=sys.stderr)
    total_chunks = 0
    with open(embedded_jsonl, "w") as f:
        for ep_str in all_ep_strs:
            ep_chunks = new_chunks.get(ep_str) or existing_chunks.get(ep_str, [])
            for chunk in ep_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

    # Build database using build_db
    print(f"Building database ({total_chunks} chunks)...", file=sys.stderr)

    # build_db.main() expects sys.argv — call its internals directly
    if os.path.exists(db_output):
        os.remove(db_output)

    import sqlite3
    import sqlite_vec
    import struct

    # Peek at dimension
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
                ep_json_path = os.path.join(args.data_dir, str(ep_num), "episode.json")
                if os.path.exists(ep_json_path):
                    with open(ep_json_path, "r") as ej:
                        ep_meta = json.load(ej)
                        mp3_url = ep_meta.get("mp3_url")
                        episode_url = ep_meta.get("episode_url")

                synopsis = chunk["content"] if chunk["chunk_type"] == "synopsis" else None
                conn.execute(
                    "INSERT OR IGNORE INTO episodes (number, title, pub_date, synopsis, mp3_url, episode_url) VALUES (?, ?, ?, ?, ?, ?)",
                    (ep_num, chunk["episode_title"], chunk["pub_date"], synopsis, mp3_url, episode_url),
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
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", ("embedding_model", args.model))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", ("embedding_dim", str(embedding_dim)))
    conn.commit()
    conn.close()

    # Symlink podcast.db -> podcast.{model}.db
    if os.path.exists(db_link):
        os.remove(db_link)
    os.link(db_output, db_link)

    # Update manifest
    new_manifest = {"episodes": {}, "shownotes": shownotes_hash}
    for ep_str in all_ep_strs:
        md_path = os.path.join(args.data_dir, ep_str, "episode.md")
        new_manifest["episodes"][ep_str] = hash_file(md_path)
    save_manifest(new_manifest)

    elapsed = time.time() - t0
    db_size = os.path.getsize(db_output) / (1024 * 1024)
    print(f"\nDone in {elapsed:.1f}s: {chunk_count} chunks, {len(episodes_seen)} episodes "
          f"-> {db_output} ({db_size:.1f} MB)", file=sys.stderr)
    if chunks_embedded:
        print(f"  Re-embedded: {chunks_embedded} chunks across {total} episodes", file=sys.stderr)
    if deleted:
        print(f"  Removed: episodes {', '.join(sorted(deleted, key=float))}", file=sys.stderr)


if __name__ == "__main__":
    main()
