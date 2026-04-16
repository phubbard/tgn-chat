"""Embed chunks via LM Studio's OpenAI-compatible /v1/embeddings endpoint.

Reads chunks from JSONL, calls LM Studio for embeddings, writes augmented JSONL
with embedding vectors attached.

Usage:
    python ingest/embed.py --model bge-m3 [--input build/chunks.jsonl] [--output build/chunks.bge-m3.jsonl]
"""

import argparse
import json
import os
import sys
import time

import requests

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234")


def get_embedding(text, model):
    """Get embedding vector from LM Studio."""
    resp = requests.post(
        f"{LM_STUDIO_URL}/v1/embeddings",
        json={"model": model, "input": text},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]


def main():
    parser = argparse.ArgumentParser(description="Embed chunks via LM Studio")
    parser.add_argument("--model", required=True, help="LM Studio embedding model identifier")
    parser.add_argument("--input", default="build/chunks.jsonl", help="Input JSONL")
    parser.add_argument("--output", default=None, help="Output JSONL (default: build/chunks.{model}.jsonl)")
    parser.add_argument("--batch-size", type=int, default=1, help="Chunks per API call")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"build/chunks.{args.model}.jsonl"

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run chunk.py first.", file=sys.stderr)
        sys.exit(1)

    # Count total lines for progress
    with open(args.input, "r") as f:
        total = sum(1 for _ in f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Check if we can resume from a partial run
    done = 0
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            done = sum(1 for _ in f)
        print(f"Resuming from chunk {done}/{total}", file=sys.stderr)

    with open(args.input, "r") as fin, open(args.output, "a") as fout:
        for i, line in enumerate(fin):
            if i < done:
                continue

            chunk = json.loads(line)
            try:
                embedding = get_embedding(chunk["content"], args.model)
                chunk["embedding"] = embedding
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as e:
                print(f"Error on chunk {i}: {e}", file=sys.stderr)
                sys.exit(1)

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{total} chunks embedded", file=sys.stderr)

    print(f"\nDone: {total} chunks embedded with {args.model} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
