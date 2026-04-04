"""Parse episode.md files into structured chunks for embedding.

Reads data/inputs/{n}/episode.md and produces a JSON lines file with
three chunk types: synopsis, transcript, and links.

Usage:
    python ingest/chunk.py [--data-dir data/inputs] [--output build/chunks.jsonl]
"""

import argparse
import json
import os
import re
import sys


HOSTS = {
    "James Stacy", "Jason Heaton",
    # Common WhisperX diarization artifacts for the hosts
    "James Stacey", "James", "Jason", "Host", "Unknown",
}

# Speaker labels that are transcription artifacts, not people
_ARTIFACT_PATTERNS = re.compile(
    r"(?i)("
    r"^sound\b|^music\b|^filler\b|^affirmative\b|^no label|^speaker response"
    r"|^general agreement|^\[|^silence|^interviewer$|^co-host$"
    r"|^unknown|^host$|^pause|^break"
    r")"
)


def _is_artifact(name):
    """Return True if a speaker label is a transcription artifact."""
    return bool(_ARTIFACT_PATTERNS.search(name))


def parse_episode_md(filepath):
    """Parse a single episode.md into its sections.

    Returns dict with keys: number, title, pub_date, synopsis, links, transcript.
    Transcript is a list of (speaker, text) tuples.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    result = {
        "number": None,
        "title": None,
        "pub_date": None,
        "synopsis": None,
        "links": [],
        "transcript": [],
    }

    # Strip YAML frontmatter
    content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)

    # Parse title from first heading
    title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
    if title_match:
        result["title"] = title_match.group(1).strip()
        # Try to extract episode number from title
        num_match = re.search(
            r"Episode\s+(\d+)|[–-]\s*(\d+)\s*[–-]", result["title"]
        )
        if num_match:
            result["number"] = int(num_match.group(1) or num_match.group(2))

    # Parse pub_date
    date_match = re.search(
        r"Published on (.+?)$", content, re.MULTILINE
    )
    if date_match:
        result["pub_date"] = date_match.group(1).strip()

    # Split into sections by ## headings
    sections = re.split(r"^## (.+)$", content, flags=re.MULTILINE)
    # sections is: [preamble, heading1, body1, heading2, body2, ...]

    for i in range(1, len(sections) - 1, 2):
        heading = sections[i].strip().lower()
        body = sections[i + 1].strip()

        if heading == "synopsis":
            result["synopsis"] = body

        elif heading == "links":
            # Parse markdown links: - [text](url)
            for link_match in re.finditer(
                r"-\s*\[([^\]]+)\]\(([^)]+)\)", body
            ):
                result["links"].append(
                    {"text": link_match.group(1), "url": link_match.group(2)}
                )

        elif heading == "transcript":
            # Parse pipe-delimited table rows, skip header and separator
            for row_match in re.finditer(
                r"^\|([^|*][^|]*)\|(.+)\|$", body, re.MULTILINE
            ):
                speaker = row_match.group(1).strip()
                text = row_match.group(2).strip()
                if speaker and text and speaker != "----":
                    result["transcript"].append((speaker, text))

    return result


def extract_guests(episode):
    """Return sorted list of guest names from transcript speakers."""
    if not episode["transcript"]:
        return []
    speakers = {s for s, _ in episode["transcript"]}
    guests = sorted(
        s for s in speakers - HOSTS
        if not _is_artifact(s)
        # Filter host name variants (e.g. "James Stacey (co-host)", "Jason Heaton (Host)")
        and not any(h.split()[0] in s and ("host" in s.lower() or "co-host" in s.lower())
                    for h in {"James Stacy", "Jason Heaton"})
    )
    return guests


def make_synopsis_chunk(episode, episode_number, guests=None):
    """Create a synopsis chunk from parsed episode data."""
    if not episode["synopsis"]:
        return None
    return {
        "episode_number": episode_number,
        "episode_title": episode["title"],
        "pub_date": episode["pub_date"],
        "chunk_type": "synopsis",
        "content": episode["synopsis"],
        "speakers": None,
        "start_turn": None,
        "end_turn": None,
        "guests": guests,
    }


def parse_shownotes(filepath):
    """Parse shownotes.md into a dict mapping episode number to link lists.

    Returns {episode_number: [{"text": ..., "url": ...}, ...]}.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    shownotes = {}
    # Split on episode headings: ## [Title](url)
    entries = re.split(r"^## \[", content, flags=re.MULTILINE)

    for entry in entries[1:]:  # skip preamble
        # Extract episode number from title like "The Grey NATO – 363 – ..."
        title_match = re.match(r"[^]]+", entry)
        if not title_match:
            continue
        title = title_match.group(0)
        num_match = re.search(r"(?:[–-]\s*(?:Ep(?:isode)?\s+)?|Ep\s+)(\d+)\b", title, re.IGNORECASE)
        if not num_match:
            continue
        ep_num = int(num_match.group(1))

        links = []
        for link_match in re.finditer(r"^- \[([^\]]+)\]\(([^)]+)\)", entry, re.MULTILINE):
            links.append({"text": link_match.group(1), "url": link_match.group(2)})

        if links:
            shownotes[ep_num] = links

    return shownotes


def make_link_chunk(episode, episode_number, shownotes=None, guests=None):
    """Create a links chunk, preferring shownotes links over episode.md links."""
    links = None
    if shownotes and episode_number in shownotes:
        links = shownotes[episode_number]
    elif episode["links"]:
        links = episode["links"]

    if not links:
        return None
    lines = [f"- [{l['text']}]({l['url']})" for l in links]
    content = f"Episode links for {episode['title']}:\n" + "\n".join(lines)
    return {
        "episode_number": episode_number,
        "episode_title": episode["title"],
        "pub_date": episode["pub_date"],
        "chunk_type": "links",
        "content": content,
        "speakers": None,
        "start_turn": None,
        "end_turn": None,
        "guests": guests,
    }


def make_transcript_chunks(episode, episode_number, target_words=500, guests=None):
    """Group consecutive speaker turns into chunks of ~target_words words."""
    if not episode["transcript"]:
        return []

    chunks = []
    current_turns = []
    current_words = 0
    start_turn = 0

    for i, (speaker, text) in enumerate(episode["transcript"]):
        word_count = len(text.split())
        current_turns.append(f"**{speaker}:** {text}")
        current_words += word_count

        if current_words >= target_words:
            speakers = list(
                dict.fromkeys(
                    s for s, _ in episode["transcript"][start_turn : i + 1]
                )
            )
            chunks.append(
                {
                    "episode_number": episode_number,
                    "episode_title": episode["title"],
                    "pub_date": episode["pub_date"],
                    "chunk_type": "transcript",
                    "content": "\n\n".join(current_turns),
                    "speakers": ", ".join(speakers),
                    "start_turn": start_turn,
                    "end_turn": i,
                    "guests": guests,
                }
            )
            current_turns = []
            current_words = 0
            start_turn = i + 1

    # Remaining turns
    if current_turns:
        speakers = list(
            dict.fromkeys(
                s
                for s, _ in episode["transcript"][
                    start_turn : len(episode["transcript"])
                ]
            )
        )
        chunks.append(
            {
                "episode_number": episode_number,
                "episode_title": episode["title"],
                "pub_date": episode["pub_date"],
                "chunk_type": "transcript",
                "content": "\n\n".join(current_turns),
                "speakers": ", ".join(speakers),
                "start_turn": start_turn,
                "end_turn": len(episode["transcript"]) - 1,
                "guests": guests,
            }
        )

    return chunks


def process_all_episodes(data_dir, output_path):
    """Process all episodes and write chunks to JSONL."""
    # Load shownotes for real content links
    shownotes_path = os.path.join(data_dir, "shownotes.md")
    shownotes = None
    if os.path.exists(shownotes_path):
        shownotes = parse_shownotes(shownotes_path)
        print(f"Loaded shownotes: {len(shownotes)} episodes with links", file=sys.stderr)
    else:
        print("Warning: shownotes.md not found, using episode.md links", file=sys.stderr)

    episode_dirs = sorted(
        [
            d
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
            and re.match(r"^\d+(\.\d+)?$", d)
        ],
        key=float,
    )

    total_chunks = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for dirname in episode_dirs:
            md_path = os.path.join(data_dir, dirname, "episode.md")
            if not os.path.exists(md_path):
                print(f"  Skipping {dirname}: no episode.md", file=sys.stderr)
                continue

            episode_number = float(dirname) if "." in dirname else int(dirname)
            episode = parse_episode_md(md_path)

            # Fall back to directory name for episode number
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

            chunks.extend(
                make_transcript_chunks(episode, episode_number, guests=guests)
            )

            for chunk in chunks:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            total_chunks += len(chunks)
            print(
                f"  Episode {episode_number}: {len(chunks)} chunks",
                file=sys.stderr,
            )

    print(f"\nTotal: {total_chunks} chunks written to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Chunk podcast episodes for embedding")
    parser.add_argument(
        "--data-dir",
        default="data/inputs",
        help="Directory containing episode subdirectories",
    )
    parser.add_argument(
        "--output",
        default="build/chunks.jsonl",
        help="Output JSONL file path",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} not found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    process_all_episodes(args.data_dir, args.output)


if __name__ == "__main__":
    main()
