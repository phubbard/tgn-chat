/**
 * search.js — SQLite + sqlite-vec WASM vector search
 *
 * Loads the pre-built .db file, performs vector similarity search
 * using embeddings from Ollama.
 */

const DB_URL = '/build/podcast.db';
const EMBED_URL = '/api/embed';
const TOP_K = 8;

let db = null;
let embeddingModel = null;

/**
 * Initialize sql.js and load the database file.
 */
async function initDB() {
  const SQL = await initSqlJs({
    locateFile: file => `https://cdn.jsdelivr.net/npm/sql.js@1.11.0/dist/${file}`
  });

  const resp = await fetch(DB_URL);
  if (!resp.ok) throw new Error(`Failed to load database: ${resp.status}`);

  const buf = await resp.arrayBuffer();
  db = new SQL.Database(new Uint8Array(buf));

  // Read embedding model and dimension from meta table
  const meta = {};
  const rows = db.exec("SELECT key, value FROM meta");
  if (rows.length > 0) {
    for (const row of rows[0].values) {
      meta[row[0]] = row[1];
    }
  }
  embeddingModel = meta.embedding_model;

  return {
    model: embeddingModel,
    dim: parseInt(meta.embedding_dim),
    episodes: db.exec("SELECT COUNT(*) FROM episodes")[0].values[0][0],
    chunks: db.exec("SELECT COUNT(*) FROM chunks")[0].values[0][0],
  };
}

/**
 * Get embedding vector from Ollama.
 */
async function getEmbedding(text) {
  const resp = await fetch(EMBED_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: embeddingModel, input: text }),
  });
  if (!resp.ok) throw new Error(`Embedding failed: ${resp.status}`);
  const data = await resp.json();
  return data.embeddings[0];
}

/**
 * Serialize float array to bytes for sqlite-vec query.
 */
function serializeFloat32(vec) {
  const buf = new ArrayBuffer(vec.length * 4);
  const view = new Float32Array(buf);
  for (let i = 0; i < vec.length; i++) view[i] = vec[i];
  return new Uint8Array(buf);
}

/**
 * Search for similar chunks given a query string.
 * Returns array of {content, chunk_type, speakers, episode_number, episode_title, distance, episode_url}
 *
 * Note: sqlite-vec WASM support is evolving. If the vec0 virtual table
 * isn't available in sql.js, we fall back to brute-force cosine similarity
 * computed in JS. This is acceptable for our corpus size (10K-30K chunks).
 */
async function search(queryText, topK = TOP_K) {
  const queryVec = await getEmbedding(queryText);

  // Try sqlite-vec first, fall back to brute force
  try {
    return searchVec(queryVec, topK);
  } catch (e) {
    console.warn('sqlite-vec not available in WASM, using JS fallback:', e.message);
    return searchBruteForce(queryVec, topK);
  }
}

function searchVec(queryVec, topK) {
  const queryBytes = serializeFloat32(queryVec);
  const stmt = db.prepare(`
    SELECT
      c.content, c.chunk_type, c.speakers,
      e.number, e.title, e.episode_url,
      v.distance
    FROM chunks_vec v
    JOIN chunks c ON c.id = v.chunk_id
    JOIN episodes e ON e.id = c.episode_id
    WHERE v.embedding MATCH ?
    ORDER BY v.distance
    LIMIT ?
  `);
  stmt.bind([queryBytes, topK]);

  const results = [];
  while (stmt.step()) {
    const row = stmt.getAsObject();
    results.push({
      content: row.content,
      chunk_type: row.chunk_type,
      speakers: row.speakers,
      episode_number: row.number,
      episode_title: row.title,
      episode_url: row.episode_url,
      distance: row.distance,
    });
  }
  stmt.free();
  return results;
}

/**
 * Brute-force cosine similarity fallback.
 * Reads all embeddings and computes similarity in JS.
 */
function searchBruteForce(queryVec, topK) {
  // Read all chunks with their embeddings
  const rows = db.exec(`
    SELECT c.id, c.content, c.chunk_type, c.speakers,
           e.number, e.title, e.episode_url
    FROM chunks c
    JOIN episodes e ON e.id = c.episode_id
  `);

  if (!rows.length) return [];

  // Read embeddings from vec table
  const vecRows = db.exec("SELECT chunk_id, embedding FROM chunks_vec");
  if (!vecRows.length) return [];

  // Build embedding map
  const embeddings = new Map();
  for (const row of vecRows[0].values) {
    const chunkId = row[0];
    const bytes = row[1];
    const vec = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    embeddings.set(chunkId, vec);
  }

  // Compute similarities
  const scored = [];
  for (const row of rows[0].values) {
    const chunkId = row[0];
    const emb = embeddings.get(chunkId);
    if (!emb) continue;

    const sim = cosineSimilarity(queryVec, emb);
    scored.push({
      content: row[1],
      chunk_type: row[2],
      speakers: row[3],
      episode_number: row[4],
      episode_title: row[5],
      episode_url: row[6],
      distance: 1 - sim,
    });
  }

  scored.sort((a, b) => a.distance - b.distance);
  return scored.slice(0, topK);
}

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
