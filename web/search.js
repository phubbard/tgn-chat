/**
 * search.js — Thin client for server-side hybrid search API
 *
 * All embedding, vector search, and keyword search now run server-side.
 * The browser just sends the query and receives ranked chunks.
 */

const SEARCH_URL = '/search';
const TOP_K = 16;

/**
 * Initialize by fetching DB info from the server.
 */
async function initDB() {
  const resp = await fetch(`${SEARCH_URL}/info`);
  if (!resp.ok) throw new Error(`Search API not available: ${resp.status}`);
  return await resp.json();
}

/**
 * Search for relevant chunks given a query string.
 * Returns array of {content, chunk_type, speakers, episode_number, episode_title, distance, episode_url}
 */
async function search(queryText, topK = TOP_K) {
  const resp = await fetch(SEARCH_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: queryText, top_k: topK }),
  });
  if (!resp.ok) throw new Error(`Search failed: ${resp.status}`);
  const data = await resp.json();
  return data.results;
}
