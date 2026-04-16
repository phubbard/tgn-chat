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

# Run
python web/serve.py &        # search API on :5555
caddy run                    # reverse proxy on :8080
```

## Project structure

```
ingest/
  chunk.py          parse episodes into structured chunks
  embed.py          embed chunks via LM Studio
  build_db.py       build SQLite DB with vectors + FTS5
  rebuild.py        incremental rebuild (only re-embeds changed episodes)
  eval.py           compare retrieval across embedding models
  sync.sh           rsync episode data from the transcription server
web/
  index.html        chat UI (water.css, no build step)
  app.js            chat orchestration, LM Studio streaming
  search.js         thin client for server-side search
  serve.py          search API server (hybrid search + logging)
Caddyfile           reverse proxy config
```

## Requirements

- Python 3.10+ with `requests`, `sqlite-vec`
- [LM Studio](https://lmstudio.ai) with `bge-m3` (embedding) and a chat model loaded, server running on `127.0.0.1:1234`
- [Caddy](https://caddyserver.com) (optional, for LAN access)
- Source data in `data/inputs/` (not included, ~1GB)
