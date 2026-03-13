# TGN Search

Self-hosted RAG chatbot over 363 episodes (10 years) of [The Grey NATO](https://thegreynato.com) podcast. Hybrid vector + keyword search across transcripts, synopses, and 7,663 curated episode links.

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
    |   /api/* ----------------> Ollama :11434            |
    |                            - /api/embed (bge-m3)    |
    |                            - /api/chat  (streaming) |
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
2. `serve.py` embeds the query via Ollama (`bge-m3`) and runs hybrid search:
   - **Vector**: sqlite-vec KNN over 1024-dim embeddings
   - **Keyword**: FTS5 full-text search with BM25 ranking
   - Results are merged and re-ranked
3. Browser sends retrieved chunks + question to Ollama for generation
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

# Build the database (requires Ollama with bge-m3)
python ingest/chunk.py
python ingest/embed.py --model bge-m3
python ingest/build_db.py --model bge-m3
cp build/podcast.bge-m3.db build/podcast.db

# Run
python web/serve.py &        # search API on :5555
caddy run                    # reverse proxy on :8080
```

## Project structure

```
ingest/
  chunk.py          parse episodes into structured chunks
  embed.py          embed chunks via Ollama
  build_db.py       build SQLite DB with vectors + FTS5
  eval.py           compare retrieval across embedding models
web/
  index.html        chat UI (water.css, no build step)
  app.js            chat orchestration, Ollama streaming
  search.js         thin client for server-side search
  serve.py          search API server (hybrid search + logging)
Caddyfile           reverse proxy config
```

## Requirements

- Python 3.10+ with `requests`, `sqlite-vec`
- [Ollama](https://ollama.com) with `bge-m3` (embedding) and a chat model
- [Caddy](https://caddyserver.com) (optional, for LAN access)
- Source data in `data/inputs/` (not included, ~1GB)
