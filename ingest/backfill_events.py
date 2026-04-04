#!/usr/bin/env python3
"""Backfill events.jsonl from existing markdown session logs.

Parses the markdown log files in logs/ and writes structured events
to logs/events.jsonl for the monitoring dashboard.

Usage:
    python ingest/backfill_events.py [--logs-dir logs]
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime


def parse_session_log(filepath):
    """Parse a markdown session log file into a list of event dicts."""
    filename = os.path.basename(filepath)
    # Extract date and session ID from filename: 2026-04-01_25c6fbba.md
    match = re.match(r"(\d{4}-\d{2}-\d{2})_([a-f0-9]+)\.md", filename)
    if not match:
        return []
    file_date = match.group(1)
    session_id = match.group(2)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    events = []

    # Parse header for session_start
    started_match = re.search(r"\*\*Started:\*\* (.+)", content)
    ua_match = re.search(r"\*\*User-Agent:\*\* (.+)", content)
    if started_match:
        started_str = started_match.group(1).strip()
        try:
            ts = datetime.strptime(started_str, "%Y-%m-%d %H:%M:%S").isoformat()
        except ValueError:
            ts = f"{file_date}T00:00:00"
        event = {
            "session_id": session_id,
            "timestamp": ts,
            "type": "session_start",
        }
        if ua_match:
            event["user_agent"] = ua_match.group(1).strip()
        events.append(event)

    # Split on ## or ### headings for individual events
    sections = re.split(r"^(#{2,3}) (.+)$", content, flags=re.MULTILINE)
    # sections: [preamble, level, heading, body, level, heading, body, ...]

    for i in range(1, len(sections) - 2, 3):
        heading = sections[i + 1].strip()
        body = sections[i + 2]

        # Parse time and type from heading: "19:11:42 — Query (4611c6f7)"
        heading_match = re.match(r"(\d{2}:\d{2}:\d{2}) — (\w+)(.*)", heading)
        if not heading_match:
            continue

        time_str = heading_match.group(1)
        event_type = heading_match.group(2).lower()
        extra = heading_match.group(3).strip()

        ts = f"{file_date}T{time_str}"

        if event_type == "query":
            event = {
                "session_id": session_id,
                "timestamp": ts,
                "type": "query",
            }

            # Extract query_id from heading
            qid_match = re.search(r"\(([a-f0-9]+)\)", extra)
            if qid_match:
                event["query_id"] = qid_match.group(1)

            # Extract model
            model_match = re.search(r"\*\*Model:\*\* (.+)", body)
            if model_match:
                event["model"] = model_match.group(1).strip()

            # Extract query
            query_match = re.search(r"\*\*Query:\*\* (.+)", body)
            if query_match:
                event["query"] = query_match.group(1).strip()

            # Extract metrics from table
            for metric_match in re.finditer(r"\| (.+?) \| (.+?) \|", body):
                key = metric_match.group(1).strip()
                val = metric_match.group(2).strip()
                if key == "Search":
                    event["search_time_s"] = float(val.rstrip("s"))
                elif key == "Time to first token":
                    event["ttft_s"] = float(val.rstrip("s"))
                elif key == "Total time":
                    event["total_time_s"] = float(val.rstrip("s"))
                elif key == "Tokens":
                    event["tokens"] = int(val)
                elif key == "Token rate":
                    event["tok_per_sec"] = float(val.split()[0])
                elif key == "Source episodes":
                    event["source_episodes"] = [
                        float(e.strip()) if "." in e.strip() else int(e.strip())
                        for e in val.split(",") if e.strip()
                    ]

            # Extract response (blockquoted lines after **Response:**)
            resp_match = re.search(r"\*\*Response:\*\*\s*\n((?:>.*\n?)+)", body)
            if resp_match:
                lines = resp_match.group(1).strip().split("\n")
                response = "\n".join(
                    line[2:] if line.startswith("> ") else line[1:] if line.startswith(">") else line
                    for line in lines
                )
                event["response"] = response

            events.append(event)

        elif event_type == "feedback":
            event = {
                "session_id": session_id,
                "timestamp": ts,
                "type": "feedback",
            }
            # "Feedback 👍 (query abc123)"
            if "\U0001f44d" in extra:
                event["vote"] = "up"
            elif "\U0001f44e" in extra:
                event["vote"] = "down"
            qid_match = re.search(r"query\s+([a-f0-9]+)", extra)
            if qid_match:
                event["query_id"] = qid_match.group(1)

            query_match = re.search(r"\*\*Query:\*\* (.+)", body)
            if query_match:
                event["query"] = query_match.group(1).strip()
            model_match = re.search(r"\*\*Model:\*\* (.+)", body)
            if model_match:
                event["model"] = model_match.group(1).strip()

            events.append(event)

        elif event_type == "error":
            event = {
                "session_id": session_id,
                "timestamp": ts,
                "type": "error",
            }
            # Extract error from code block
            err_match = re.search(r"```\n(.+?)\n```", body, re.DOTALL)
            if err_match:
                event["error"] = err_match.group(1).strip()
            events.append(event)

    return events


def main():
    parser = argparse.ArgumentParser(description="Backfill events.jsonl from markdown logs")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing markdown logs")
    args = parser.parse_args()

    if not os.path.isdir(args.logs_dir):
        print(f"Error: {args.logs_dir} not found", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(
        f for f in os.listdir(args.logs_dir)
        if f.endswith(".md")
    )

    all_events = []
    for filename in md_files:
        filepath = os.path.join(args.logs_dir, filename)
        events = parse_session_log(filepath)
        all_events.extend(events)
        if events:
            queries = sum(1 for e in events if e["type"] == "query")
            print(f"  {filename}: {len(events)} events ({queries} queries)", file=sys.stderr)

    # Sort by timestamp
    all_events.sort(key=lambda e: e["timestamp"])

    # Write to events.jsonl
    output = os.path.join(args.logs_dir, "events.jsonl")
    with open(output, "w", encoding="utf-8") as f:
        for event in all_events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    queries = sum(1 for e in all_events if e["type"] == "query")
    feedback = sum(1 for e in all_events if e["type"] == "feedback")
    sessions = sum(1 for e in all_events if e["type"] == "session_start")
    print(f"\nBackfilled {len(all_events)} events from {len(md_files)} log files -> {output}", file=sys.stderr)
    print(f"  {sessions} sessions, {queries} queries, {feedback} feedback", file=sys.stderr)


if __name__ == "__main__":
    main()
