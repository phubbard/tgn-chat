#!/usr/bin/env python3
"""Extract topics from episode synopses using a local LLM via LM Studio.

Reads synopses from the database, sends each to LM Studio for topic extraction,
and updates the episodes table with a JSON array of topics.

Usage:
    python ingest/extract_topics.py [--db build/podcast.db] [--model gemma-3-12b-it]
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time

import requests
import sqlite_vec

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234")

PROMPT = """Extract 3-8 topic tags from this podcast episode synopsis. Return ONLY a JSON array of short tags (1-3 words each). Tags should be specific and useful for search — prefer concrete topics like "Omega Speedmaster", "Baselworld 2016", "dive watches" over vague ones like "discussion", "opinions", "watches" (too generic for a watch podcast).

Good tags: brand names, specific watch models, events, places, activities, gear categories, named people/guests.
Bad tags: "watches" (too broad), "conversation", "discussion", "podcast", "episode".

Synopsis:
{synopsis}

JSON array:"""


def extract_topics(synopsis, model):
    """Send synopsis to LM Studio and parse topic tags."""
    resp = requests.post(
        f"{LM_STUDIO_URL}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT.format(synopsis=synopsis)}],
            "temperature": 0.3,
            "max_tokens": 200,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()

    # Extract JSON array from response (LLM may include extra text)
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        print(f"  Warning: no JSON array found in response: {text[:100]}", file=sys.stderr)
        return None

    try:
        topics = json.loads(match.group(0))
        # Ensure all items are strings
        topics = [str(t).strip() for t in topics if isinstance(t, str) and t.strip()]
        return topics if topics else None
    except json.JSONDecodeError:
        print(f"  Warning: invalid JSON: {match.group(0)[:100]}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract topics from episode synopses")
    parser.add_argument("--db", default="build/podcast.db", help="Database path")
    parser.add_argument("--model", default="gemma-3-12b-it", help="LM Studio model identifier for extraction")
    parser.add_argument("--force", action="store_true", help="Re-extract even if topics exist")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    # Add topics column if it doesn't exist
    cols = [r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()]
    if "topics" not in cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN topics TEXT")
        conn.commit()

    # Get episodes to process
    if args.force:
        rows = conn.execute(
            "SELECT id, number, synopsis FROM episodes WHERE synopsis IS NOT NULL ORDER BY number"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, number, synopsis FROM episodes WHERE synopsis IS NOT NULL AND topics IS NULL ORDER BY number"
        ).fetchall()

    print(f"Extracting topics for {len(rows)} episodes using {args.model}...", file=sys.stderr)

    t0 = time.time()
    success = 0
    for i, (ep_id, ep_num, synopsis) in enumerate(rows):
        topics = extract_topics(synopsis, args.model)
        if topics:
            conn.execute("UPDATE episodes SET topics = ? WHERE id = ?",
                         (json.dumps(topics), ep_id))
            if (i + 1) % 10 == 0:
                conn.commit()
            success += 1
            print(f"  [{i+1}/{len(rows)}] Ep {int(ep_num)}: {topics}", file=sys.stderr)
        else:
            print(f"  [{i+1}/{len(rows)}] Ep {int(ep_num)}: FAILED", file=sys.stderr)

    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s: {success}/{len(rows)} episodes tagged "
          f"({success / elapsed:.1f} eps/s)", file=sys.stderr)


if __name__ == "__main__":
    main()
