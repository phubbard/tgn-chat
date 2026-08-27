# TGN Chatbot

Self-hosted RAG chatbot over 380 episodes (10 years) of [The Grey NATO](https://thegreynato.com) podcast. Hybrid vector + keyword search across transcripts, synopses, and 7,663 curated episode links.

Live at [tgnchat.phfactor.net](https://tgnchat.phfactor.net/).

```
                        Browser (vanilla JS)
                              |
                    query     |     streamed response
                              v
    +----------------------------------------------------+
    |                    Caddy :8080                      |
    |                                                     |
    |   /search, /log ---------> serve.py :5555           |
    |                            - hybrid search          |
    |                            - sqlite-vec (KNN)       |
    |                            - FTS5 (keyword)         |
    |                            - session logging        |
    |                                                     |
    |   /v1/* -----------------> LM Studio :1234          |
    |                            - /v1/embeddings (bge-m3)|
    |                            - /v1/chat/completions   |
    |                              (streaming)            |
    |                                                     |
    |   /* --------------------> web/ (static files)      |
    +----------------------------------------------------+
                              |
                    +-------------------+
                    | build/podcast.db  |
                    | sqlite + vec0     |
                    | 8,105 chunks      |
                    | 1024-dim vectors  |
                    +-------------------+
```

📊 **Interactive diagrams** (open in a browser — light/dark, guided views, zoom, export):
[architecture](docs/architecture.html) ·
[ingest & rebuild](docs/ingest.html) ·
[query lifecycle](docs/sequence.html) ·
[data lineage](docs/dataflow.html). Generated with
[archify](https://github.com/tt-a1i/archify) from the `docs/*.json` specs. See also the
full [design doc](docs/DESIGN.md).

## How it works

1. User asks a question in the browser
2. `serve.py` embeds the query via LM Studio (`bge-m3`) and runs hybrid search:
   - **Vector**: sqlite-vec KNN over 1024-dim embeddings
   - **Keyword**: FTS5 full-text search with BM25 ranking
   - Results are merged and re-ranked
3. Browser sends retrieved chunks + question to LM Studio for generation
4. Response streams back token-by-token with inline episode citations

## Ingest pipeline

```
data/inputs/{n}/episode.md ----+
                               +--> chunk.py --> embed.py --> build_db.py
data/inputs/shownotes.md ------+
                                        |
                                        v
                               build/podcast.bge-m3.db
```

Each episode is split into three chunk types:
- **Synopsis** -- episode summary, good for "which episode discussed X?"
- **Transcript** -- ~500-word speaker-attributed segments
- **Links** -- curated shownotes (watches, books, gear, people)

## Setup

```bash
# Python dependencies
pip install -r ingest/requirements.txt

# Sync episode data from the transcription server
./ingest/sync.sh

# Build (or incrementally rebuild) the database
python ingest/rebuild.py --model bge-m3

# Run (dev)
python web/serve.py &        # search API on :5555
caddy run                    # reverse proxy on :8081

# Run (production): both services as launchd agents (auto-start + restart).
# See "Runtime infrastructure" below.
for p in net.phfactor.tgnchat net.phfactor.tgnchat-caddy; do
  cp deploy/$p.plist ~/Library/LaunchAgents/
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/$p.plist
done
```

## Project structure

```
ingest/
  chunk.py          parse episodes into structured chunks
  embed.py          embed chunks via LM Studio
  build_db.py       build SQLite DB with vectors + FTS5
  rebuild.py        incremental rebuild (content-addressed cache; only
                    re-embeds chunks whose text changed)
  eval.py           compare retrieval across embedding models
  sync.sh           rsync episode data from the transcription server
web/
  index.html        chat UI (water.css, no build step)
  app.js            chat orchestration, LM Studio streaming
  search.js         thin client for server-side search
  serve.py          search API server (hybrid search + logging)
Caddyfile           reverse proxy config (listens on :8081)
scripts/
  serve.sh          launchd wrapper: wait for LM Studio, then run serve.py
deploy/
  net.phfactor.tgnchat.plist        LaunchAgent for serve.py
  net.phfactor.tgnchat-caddy.plist  LaunchAgent for caddy
```

## Runtime infrastructure

Three long-running services on the Mac Studio, all in the user's login session:

```
public proxy ──▶ caddy :8081 ──┬──▶ serve.py :5555        (search API + logging)
                               ├──▶ LM Studio :1234       (/v1/embeddings, /v1/chat)
                               └──▶ web/ static files
```

Both `caddy` and `serve.py` run as **launchd LaunchAgents** (not Daemons — they
depend on LM Studio, which is a GUI app in the login session). Each has
`RunAtLoad` (start at login) and `KeepAlive` (auto-restart on crash). Plists
live in `deploy/`; installed copies go in `~/Library/LaunchAgents/`.

| Service | Label | Port | Notes |
|---|---|---|---|
| Search API | `net.phfactor.tgnchat` | 5555 | via `scripts/serve.sh` wrapper |
| Reverse proxy | `net.phfactor.tgnchat-caddy` | 8081 | serves static files + proxies `/v1`, `/search`, etc. |
| LM Studio | `ai.lmstudio.server` | 1234 | LM Studio's own agent; autostarts on login |

**Dependency on LM Studio.** launchd has no native inter-service dependency for
a GUI app, so `scripts/serve.sh` approximates it: it runs `lms server start`
(idempotent) and blocks until `:1234` answers before starting `serve.py`, so the
API never serves queries before embeddings are available.

**Port note.** Caddy listens on **:8081** (not :8080) because another local
service holds :8080. The public-facing upstream proxy must point at :8081.

Common operations (`gui/$(id -u)` = your user's launchd domain):

```bash
launchctl print gui/$(id -u)/net.phfactor.tgnchat        # status (or -caddy)

launchctl kickstart -k gui/$(id -u)/net.phfactor.tgnchat # restart serve.py
                                                         # (do this after rebuild.py)

caddy reload --config Caddyfile --adapter caddyfile      # apply Caddyfile edits (no restart)

launchctl bootout gui/$(id -u)/net.phfactor.tgnchat      # stop + disable auto-restart
```

Logs: `logs/serve.launchd.log` and `logs/caddy.launchd.log`.

## Requirements

- Python 3.10+ with `requests`, `sqlite-vec`
- [LM Studio](https://lmstudio.ai) with `bge-m3` (embedding) and a chat model loaded, server running on `127.0.0.1:1234`
- [Caddy](https://caddyserver.com) — reverse proxy on :8081 (see Runtime infrastructure)
- Source data in `data/inputs/` (not included, ~1GB)
