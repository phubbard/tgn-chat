"""Evaluate retrieval quality across embedding models.

Runs a set of test queries against each model's database and reports
which episodes/chunks are retrieved, allowing comparison of recall.

Usage:
    python ingest/eval.py [--build-dir build] [--top-k 5]
"""

import argparse
import glob
import json
import os
import sqlite3
import struct
import sys

import requests
import sqlite_vec

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234")

# Test queries with expected episode numbers (ground truth).
# Add more as you discover good test cases.
TEST_QUERIES = [
    {
        "query": "Oris Divers Sixty-Five review",
        "expect_episodes": [1],
        "description": "Early episode reviewing the Oris 65",
    },
    {
        "query": "What dive watches were discussed at SIHH 2016?",
        "expect_episodes": [1],
        "description": "First episode SIHH coverage",
    },
    {
        "query": "sumo wrestling in Tokyo",
        "expect_episodes": [363],
        "description": "Latest episode, James visits sumo stables",
    },
    {
        "query": "WatchRecon website for buying watches",
        "expect_episodes": [1],
        "description": "Watch marketplace recommendation",
    },
]


def get_embedding(text, model):
    resp = requests.post(
        f"{LM_STUDIO_URL}/v1/embeddings",
        json={"model": model, "input": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def serialize_float32(vec):
    return struct.pack(f"{len(vec)}f", *vec)


def search_db(db_path, query_embedding, top_k=5):
    """Search a database and return top-k results."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    query_bytes = serialize_float32(query_embedding)

    results = conn.execute(
        """
        SELECT
            c.id, c.chunk_type, c.content, c.speakers,
            e.number, e.title,
            v.distance
        FROM chunks_vec v
        JOIN chunks c ON c.id = v.chunk_id
        JOIN episodes e ON e.id = c.episode_id
        WHERE v.embedding MATCH ?
        ORDER BY v.distance
        LIMIT ?
        """,
        (query_bytes, top_k),
    ).fetchall()

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval across models")
    parser.add_argument("--build-dir", default="build", help="Directory with .db files")
    parser.add_argument("--top-k", type=int, default=5, help="Results per query")
    args = parser.parse_args()

    db_files = sorted(glob.glob(os.path.join(args.build_dir, "podcast.*.db")))
    if not db_files:
        print(f"No database files found in {args.build_dir}/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(db_files)} model(s) to evaluate:")
    for db in db_files:
        print(f"  {os.path.basename(db)}")
    print()

    for test in TEST_QUERIES:
        print(f"Query: \"{test['query']}\"")
        print(f"  Expected episodes: {test['expect_episodes']}")
        print()

        for db_path in db_files:
            # Extract model name from filename
            model_name = os.path.basename(db_path).replace("podcast.", "").replace(".db", "")

            # Get embedding for this model
            try:
                embedding = get_embedding(test["query"], model_name)
            except Exception as e:
                print(f"  [{model_name}] Error getting embedding: {e}")
                continue

            results = search_db(db_path, embedding, args.top_k)

            retrieved_episodes = [r[4] for r in results]
            hits = [ep for ep in test["expect_episodes"] if ep in retrieved_episodes]
            recall = len(hits) / len(test["expect_episodes"]) if test["expect_episodes"] else 0

            print(f"  [{model_name}] Recall: {recall:.0%} | Episodes: {retrieved_episodes}")
            for r in results:
                print(f"    Ep {r[4]:3d} ({r[1]:10s}) dist={r[6]:.4f} | {r[2][:80]}...")
            print()

        print("-" * 80)
        print()


if __name__ == "__main__":
    main()
