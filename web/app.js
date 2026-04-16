/**
 * app.js — Chat UI and LM Studio streaming (OpenAI-compatible API)
 *
 * Orchestrates: user query -> embedding -> vector search -> LLM generation
 */

const CHAT_URL = '/v1/chat/completions';
const MODELS_URL = '/v1/models';
const DEFAULT_MODEL = 'openai/gpt-oss-120b';
// Embedding model name fragments to exclude from the chat model dropdown
const EMBED_MODELS = ['bge-m3', 'mxbai-embed', 'nomic-embed', 'all-minilm', 'snowflake-arctic-embed', 'embedding'];

const chat = document.getElementById('chat');
const form = document.getElementById('ask-form');
const queryInput = document.getElementById('query');
const submitBtn = document.getElementById('submit-btn');
const status = document.getElementById('status');
const modelSelect = document.getElementById('model-select');
const modelLink = document.getElementById('model-link');

const conversationHistory = [];
const SESSION_ID = crypto.randomUUID();

function logEvent(event) {
  fetch('/log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: SESSION_ID, ...event }),
  }).catch(() => {});
}

// Log session start
logEvent({ type: 'session_start', user_agent: navigator.userAgent });

function getChatModel() {
  return modelSelect.value;
}

function updateModelLink() {
  const model = getChatModel();
  modelLink.href = `https://huggingface.co/models?search=${encodeURIComponent(model)}`;
}

// Initialize on load
(async () => {
  try {
    const info = await initDB();
    status.textContent = `Loaded: ${info.episodes} episodes, ${info.chunks} chunks (embed: ${info.model})`;

    // Populate model dropdown from LM Studio
    try {
      const resp = await fetch(MODELS_URL);
      const data = await resp.json();
      const models = (data.data || [])
        .map(m => m.id)
        .filter(n => !EMBED_MODELS.some(e => n.toLowerCase().includes(e)))
        .sort();
      for (const name of models) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        if (name === DEFAULT_MODEL) opt.selected = true;
        modelSelect.appendChild(opt);
      }
    } catch (e) {
      // Fallback: just add the default
      const opt = document.createElement('option');
      opt.value = DEFAULT_MODEL;
      opt.textContent = DEFAULT_MODEL;
      modelSelect.appendChild(opt);
    }

    modelSelect.disabled = false;
    modelSelect.addEventListener('change', updateModelLink);
    updateModelLink();
    queryInput.disabled = false;
    submitBtn.disabled = false;
  } catch (e) {
    status.textContent = `Error loading database: ${e.message}`;
    console.error(e);
  }
})();

// Sample query links
document.querySelectorAll('.sample-query').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    queryInput.value = link.textContent;
    form.requestSubmit();
  });
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  queryInput.value = '';
  submitBtn.disabled = true;
  queryInput.disabled = true;

  appendMessage('user', query);

  try {
    const t0 = performance.now();
    status.textContent = 'Searching...';
    const results = await search(query);
    const searchMs = performance.now() - t0;

    status.textContent = 'Generating response...';
    await generateResponse(query, results, searchMs);
  } catch (e) {
    appendMessage('assistant', `Error: ${e.message}`);
    logEvent({ type: 'error', error: e.message });
    console.error(e);
  } finally {
    submitBtn.disabled = false;
    queryInput.disabled = false;
    queryInput.focus();
    status.textContent = '';
  }
});

function appendMessage(role, content) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  if (role === 'user') {
    div.textContent = content;
  } else {
    div.innerHTML = marked.parse(content);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function buildContext(results) {
  const parts = [];
  for (const r of results) {
    const date = r.pub_date ? `, ${r.pub_date}` : '';
    const header = `[Episode ${r.episode_number}: ${r.episode_title}${date}]`;
    const speakers = r.speakers ? ` (${r.speakers})` : '';
    const topics = r.topics ? `\nTopics: ${JSON.parse(r.topics).join(', ')}` : '';
    parts.push(`${header}${speakers}${topics}\n${r.content}`);
  }
  return parts.join('\n\n---\n\n');
}

function buildSourcesHTML(results) {
  // Deduplicate by episode
  const seen = new Set();
  const episodes = [];
  for (const r of results) {
    if (!seen.has(r.episode_number)) {
      seen.add(r.episode_number);
      episodes.push(r);
    }
  }

  const items = episodes.map(r => {
    const url = r.episode_url || '#';
    return `<li><a href="${url}" target="_blank">Ep ${r.episode_number}: ${r.episode_title}</a></li>`;
  });

  return `
    <details class="sources">
      <summary>Sources (${episodes.length} episodes)</summary>
      <ul>${items.join('')}</ul>
    </details>
  `;
}

async function generateResponse(query, searchResults, searchMs = 0) {
  const context = buildContext(searchResults);

  const systemPrompt = `You are a knowledgeable assistant for The Grey NATO (TGN) podcast, hosted by James Stacy and Jason Heaton, covering watches, gear, travel, and adventure over 380 episodes since 2016.

Instructions:
- Give detailed, thorough answers. Include specific quotes, names, and details from the transcripts.
- Cite episode numbers and dates inline (e.g. "In Episode 206 (March 2021), Jason mentioned...").
- When multiple episodes are relevant, discuss each one and explain the context.
- Include direct quotes from the hosts when they add color or specificity.
- If the retrieved context doesn't cover the topic well, say so briefly — but do NOT speculate about which episodes are or aren't in the database. The full archive of all 380 episodes is searchable; you simply receive the most relevant excerpts.

Retrieved context:
${context}`;

  const messages = [
    { role: 'system', content: systemPrompt },
    ...conversationHistory.slice(-4),  // Keep last 2 exchanges for context
    { role: 'user', content: query },
  ];

  const resp = await fetch(CHAT_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: getChatModel(),
      messages: messages,
      stream: true,
      stream_options: { include_usage: true },
    }),
  });

  if (!resp.ok) throw new Error(`Chat API error: ${resp.status}`);

  const msgDiv = appendMessage('assistant', '');
  const reasoningDiv = document.createElement('details');
  reasoningDiv.className = 'reasoning';
  reasoningDiv.innerHTML = '<summary>Thinking...</summary><div class="reasoning-body"></div>';
  const contentDiv = document.createElement('div');
  contentDiv.className = 'answer';
  msgDiv.appendChild(reasoningDiv);
  msgDiv.appendChild(contentDiv);
  const reasoningBody = reasoningDiv.querySelector('.reasoning-body');

  let fullResponse = '';
  let reasoningText = '';
  let usage = null;
  let firstTokenTime = null;
  const startTime = performance.now();

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (const raw of lines) {
      const line = raw.trim();
      if (!line || !line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') continue;
      let data;
      try {
        data = JSON.parse(payload);
      } catch (e) {
        continue;  // Partial JSON, ignore
      }
      if (data.error) {
        const msg = data.error.message || JSON.stringify(data.error);
        throw new Error(`LM Studio: ${msg}`);
      }
      const delta = data.choices?.[0]?.delta || {};
      const reasoningChunk = delta.reasoning ?? delta.reasoning_content;
      if (reasoningChunk) {
        if (!firstTokenTime) firstTokenTime = performance.now();
        reasoningText += reasoningChunk;
        reasoningBody.textContent = reasoningText;
        chat.scrollTop = chat.scrollHeight;
      }
      if (delta.content) {
        if (!firstTokenTime) firstTokenTime = performance.now();
        fullResponse += delta.content;
        contentDiv.innerHTML = marked.parse(fullResponse);
        chat.scrollTop = chat.scrollHeight;
      }
      if (data.usage) {
        usage = data.usage;
      }
    }
  }

  if (reasoningText) {
    reasoningDiv.querySelector('summary').textContent = 'Thinking';
  } else {
    reasoningDiv.remove();
  }

  const elapsed = (performance.now() - startTime) / 1000;
  const ttft = firstTokenTime ? (firstTokenTime - startTime) / 1000 : null;

  // Build stats line
  let statsHTML = '';
  let tokPerSec = null;
  let tokenCount = null;
  if (usage && usage.completion_tokens) {
    tokenCount = usage.completion_tokens;
    tokPerSec = parseFloat((tokenCount / elapsed).toFixed(1));
    statsHTML = `<div class="gen-stats">${tokenCount} tokens in ${elapsed.toFixed(1)}s (${tokPerSec} tok/s) · ${getChatModel()}</div>`;
  } else {
    const wordCount = fullResponse.split(/\s+/).length;
    tokenCount = Math.round(wordCount * 1.3);
    statsHTML = `<div class="gen-stats">~${tokenCount} tokens in ${elapsed.toFixed(1)}s · ${getChatModel()}</div>`;
  }

  // Add sources, stats, and feedback buttons
  const queryId = crypto.randomUUID().slice(0, 8);
  const feedbackHTML = `<span class="feedback" data-query-id="${queryId}">` +
    `<button data-vote="up" title="Good response">&#x1f44d;</button>` +
    `<button data-vote="down" title="Poor response">&#x1f44e;</button></span>`;
  const trailer = document.createElement('div');
  trailer.innerHTML = buildSourcesHTML(searchResults) + statsHTML + feedbackHTML;
  msgDiv.appendChild(trailer);

  // Wire up feedback buttons
  msgDiv.querySelectorAll('.feedback button').forEach(btn => {
    btn.addEventListener('click', () => {
      const vote = btn.dataset.vote;
      msgDiv.querySelectorAll('.feedback button').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      logEvent({ type: 'feedback', query_id: queryId, vote, query, model: getChatModel() });
    });
  });

  chat.scrollTop = chat.scrollHeight;

  // Log the interaction
  const sourceEps = [...new Set(searchResults.map(r => r.episode_number))];
  logEvent({
    type: 'query',
    query_id: queryId,
    model: getChatModel(),
    query,
    response: fullResponse,
    source_episodes: sourceEps,
    search_time_s: parseFloat((searchMs / 1000).toFixed(2)),
    ttft_s: ttft ? parseFloat(ttft.toFixed(2)) : null,
    total_time_s: parseFloat(elapsed.toFixed(2)),
    tokens: tokenCount,
    tok_per_sec: tokPerSec,
  });

  // Track conversation
  conversationHistory.push({ role: 'user', content: query });
  conversationHistory.push({ role: 'assistant', content: fullResponse });
}
