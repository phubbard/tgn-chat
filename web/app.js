/**
 * app.js — Chat UI and LM Studio streaming (OpenAI-compatible API)
 *
 * Orchestrates: user query -> embedding -> vector search -> LLM generation
 */

const CHAT_URL = '/v1/chat/completions';
const MODELS_URL = '/v1/models';
const LM_MODELS_STATE_URL = '/api/v0/models';  // LM Studio native API: returns per-model state
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

// Mobile menubar toggle (CSS handles responsive layout; JS just opens/closes).
const menuToggle = document.getElementById('menu-toggle');
const siteNav = document.getElementById('sitenav');
if (menuToggle && siteNav) {
  menuToggle.addEventListener('click', () => {
    const open = siteNav.classList.toggle('open');
    menuToggle.setAttribute('aria-expanded', String(open));
  });
  // Close the drawer after tapping a plain nav link on mobile.
  siteNav.querySelectorAll(':scope > a').forEach(a => {
    a.addEventListener('click', () => {
      siteNav.classList.remove('open');
      menuToggle.setAttribute('aria-expanded', 'false');
    });
  });
  // Close if the viewport grows past the mobile breakpoint while open.
  const mq = window.matchMedia('(min-width: 721px)');
  const sync = () => { if (mq.matches) siteNav.classList.remove('open'); };
  mq.addEventListener('change', sync);
}

// Navbar dropdowns (About, Samples). Click-to-open, click-outside/Escape to close.
function closeAllDropdowns(except = null) {
  document.querySelectorAll('[data-dropdown].open').forEach(dd => {
    if (dd === except) return;
    dd.classList.remove('open');
    const t = dd.querySelector('.dropdown-toggle');
    if (t) t.setAttribute('aria-expanded', 'false');
  });
}
document.querySelectorAll('[data-dropdown]').forEach(dd => {
  const toggle = dd.querySelector('.dropdown-toggle');
  if (!toggle) return;
  toggle.addEventListener('click', (e) => {
    e.stopPropagation();
    const willOpen = !dd.classList.contains('open');
    closeAllDropdowns(dd);
    dd.classList.toggle('open', willOpen);
    toggle.setAttribute('aria-expanded', String(willOpen));
  });
});
document.addEventListener('click', (e) => {
  // Close any open dropdown whose root doesn't contain the click target.
  document.querySelectorAll('[data-dropdown].open').forEach(dd => {
    if (!dd.contains(e.target)) {
      dd.classList.remove('open');
      const t = dd.querySelector('.dropdown-toggle');
      if (t) t.setAttribute('aria-expanded', 'false');
    }
  });
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeAllDropdowns();
});

const conversationHistory = [];
const SESSION_ID = crypto.randomUUID();
const CHAT_ID_RE = /^\/c\/([0-9a-f-]{36})$/i;
const urlMatch = location.pathname.match(CHAT_ID_RE);
let chatId = urlMatch ? urlMatch[1] : null;

function logEvent(event) {
  fetch('/log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: SESSION_ID, chat_id: chatId, ...event }),
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
    console.log('[init] fetching /search/info');
    status.textContent = 'Loading database...';
    const info = await initDB();
    console.log('[init] /search/info ok', info);
    status.textContent = '';

    // Drop the "Loaded: ..." line into the scrollable chat area so it scrolls
    // out of the way as the conversation grows instead of eating fixed pixels.
    const banner = document.createElement('div');
    banner.className = 'chat-banner';
    banner.textContent = `Loaded: ${info.episodes} episodes, ${info.chunks} chunks (embed: ${info.model})`;
    chat.appendChild(banner);

    const indexInfo = document.getElementById('index-info');
    if (indexInfo) {
      const parts = [];
      if (info.latest_episode != null) parts.push(`Latest episode indexed: <strong>#${info.latest_episode}</strong>`);
      if (info.built_at) {
        const d = new Date(info.built_at);
        const dateStr = Number.isNaN(d.getTime())
          ? info.built_at
          : d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
        parts.push(`indexed on ${dateStr}`);
      }
      if (info.app_version) parts.push(`app ${info.app_version}`);
      indexInfo.innerHTML = parts.join(' · ');
    }

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

    // If this URL points at an existing chat, reload its messages.
    if (chatId) {
      console.log('[init] restoring chat', chatId);
      await restoreChat(chatId);
      console.log('[init] chat restored');
    }
    console.log('[init] done');
  } catch (e) {
    status.textContent = `Error during init: ${e.message}`;
    console.error('[init] failed at some stage:', e);
  }
})();

async function restoreChat(id) {
  try {
    const resp = await fetch(`/chats/${encodeURIComponent(id)}`);
    if (!resp.ok) {
      appendMessage('assistant', `*Chat ${id} not found.*`);
      return;
    }
    const thread = await resp.json();
    document.title = thread.title ? `${thread.title} — TGN Chatbot` : 'TGN Chatbot';
    for (const m of thread.messages) {
      if (m.role === 'user') {
        appendMessage('user', m.content);
        conversationHistory.push({ role: 'user', content: m.content });
      } else {
        renderRestoredAssistantMessage(m);
        conversationHistory.push({ role: 'assistant', content: m.content });
      }
    }
  } catch (e) {
    console.error('restore failed', e);
    appendMessage('assistant', `*Failed to load chat: ${e.message}*`);
  }
}

function renderRestoredAssistantMessage(m) {
  const div = document.createElement('div');
  div.className = 'message assistant';
  const answer = document.createElement('div');
  answer.className = 'answer';
  answer.innerHTML = marked.parse(m.content || '');
  div.appendChild(answer);

  if (m.sources && m.sources.length) {
    const items = m.sources.map(s =>
      `<li><a href="${s.url || '#'}" target="_blank">Ep ${s.number}: ${s.title}</a></li>`
    ).join('');
    const details = document.createElement('details');
    details.className = 'sources';
    details.innerHTML = `<summary>Sources (${m.sources.length} episodes)</summary><ul>${items}</ul>`;
    div.appendChild(details);
  }

  if (m.tokens != null && m.total_time_s != null) {
    const rate = m.tok_per_sec ? ` (${m.tok_per_sec} tok/s)` : '';
    const stats = document.createElement('div');
    stats.className = 'gen-stats';
    stats.textContent = `${m.tokens} tokens in ${m.total_time_s}s${rate} · ${m.model || ''}`;
    div.appendChild(stats);
  }

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function ensureChatId() {
  if (!chatId) {
    chatId = crypto.randomUUID();
    history.replaceState({}, '', `/c/${chatId}`);
  }
  return chatId;
}

// Sample query links
document.querySelectorAll('.sample-query').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    queryInput.value = link.textContent;
    closeAllDropdowns();
    if (siteNav) {
      siteNav.classList.remove('open');
      menuToggle && menuToggle.setAttribute('aria-expanded', 'false');
    }
    form.requestSubmit();
  });
});

function createLoadingBubble() {
  const div = document.createElement('div');
  div.className = 'message assistant';

  const header = document.createElement('div');
  header.className = 'skeleton-header';
  div.appendChild(header);

  const bars = document.createElement('div');
  bars.className = 'skeleton-bars';
  for (const w of [100, 94, 82, 66]) {
    const bar = document.createElement('div');
    bar.className = 'skeleton-bar';
    bar.style.width = `${w}%`;
    bars.appendChild(bar);
  }
  div.appendChild(bars);

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;

  const labels = {
    searching: 'Searching episodes',
    loading_model: 'Loading model into memory',
    reading: 'Reading context',
  };
  let phase = 'searching';
  const startTime = performance.now();
  function render() {
    const s = Math.round((performance.now() - startTime) / 1000);
    header.textContent = `${labels[phase] || phase}… · ${s}s`;
  }
  render();
  const tick = setInterval(render, 1000);

  return {
    bubble: div,
    setPhase(next) { phase = next; render(); },
    remove() {
      clearInterval(tick);
      div.remove();
    },
  };
}

async function isModelLoaded(modelId) {
  // Returns true if LM Studio reports the model as loaded; null on unknown/error.
  try {
    const resp = await fetch(LM_MODELS_STATE_URL);
    if (!resp.ok) return null;
    const data = await resp.json();
    const rows = data.data || [];
    const match = rows.find(m => m.id === modelId);
    return match ? match.state === 'loaded' : null;
  } catch (e) {
    return null;
  }
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  ensureChatId();

  queryInput.value = '';
  submitBtn.disabled = true;
  queryInput.disabled = true;

  appendMessage('user', query);

  // Show shimmer skeleton immediately so there's no dead air
  const loader = createLoadingBubble();

  try {
    // Stage 1: pre-flight model-load check
    const modelId = getChatModel();
    const loaded = await isModelLoaded(modelId);
    if (loaded === false) {
      loader.setPhase('loading_model');
      status.textContent = `Loading ${modelId} into memory…`;
    }

    // Stage 2: embedding + search
    const t0 = performance.now();
    loader.setPhase('searching');
    status.textContent = 'Searching episodes…';
    const results = await search(query);
    const searchMs = performance.now() - t0;

    // Stage 3: prompt prefill and streaming — loader.setPhase('reading') is
    // called from generateResponse once the fetch returns headers.
    status.textContent = 'Generating response…';
    await generateResponse(query, results, searchMs, loader);
  } catch (e) {
    loader.remove();
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

async function generateResponse(query, searchResults, searchMs = 0, loader = null) {
  const context = buildContext(searchResults);

  const systemPrompt = `You are a knowledgeable assistant for The Grey NATO (TGN) podcast, hosted by James Stacy and Jason Heaton, covering watches, gear, travel, and adventure over 380 episodes since 2016.

Instructions:
- Give detailed, thorough answers. Include specific quotes, names, and details from the transcripts.
- Cite episode numbers and dates inline. Render each episode number as a markdown link to https://tgn.phfactor.net/{number}/episode/ — e.g. "In Episode [206](https://tgn.phfactor.net/206/episode/) (March 2021), Jason mentioned...". Use the bare episode number as the link text (no "Ep" or "#" prefix inside the brackets).
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

  if (!resp.ok) {
    if (loader) loader.remove();
    throw new Error(`Chat API error: ${resp.status}`);
  }

  // Fetch headers are back; we're now waiting for the LLM to produce tokens.
  if (loader) loader.setPhase('reading');

  // The real assistant bubble is only created on first token — until then,
  // the shimmer skeleton (loader) stands in for it.
  let msgDiv = null;
  let reasoningDiv = null;
  let reasoningBody = null;
  let contentDiv = null;

  function ensureBubble() {
    if (msgDiv) return;
    if (loader) loader.remove();
    msgDiv = appendMessage('assistant', '');
    reasoningDiv = document.createElement('details');
    reasoningDiv.className = 'reasoning';
    reasoningDiv.innerHTML = '<summary>Thinking…</summary><div class="reasoning-body"></div>';
    contentDiv = document.createElement('div');
    contentDiv.className = 'answer';
    msgDiv.appendChild(reasoningDiv);
    msgDiv.appendChild(contentDiv);
    reasoningBody = reasoningDiv.querySelector('.reasoning-body');
  }

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
        if (!firstTokenTime) { firstTokenTime = performance.now(); ensureBubble(); }
        reasoningText += reasoningChunk;
        reasoningBody.textContent = reasoningText;
        chat.scrollTop = chat.scrollHeight;
      }
      if (delta.content) {
        if (!firstTokenTime) { firstTokenTime = performance.now(); ensureBubble(); }
        fullResponse += delta.content;
        contentDiv.innerHTML = marked.parse(fullResponse);
        chat.scrollTop = chat.scrollHeight;
      }
      if (data.usage) {
        usage = data.usage;
      }
    }
  }

  // If the stream ended without producing any tokens (unusual), make sure the
  // skeleton is torn down and we still render a (possibly empty) assistant
  // bubble so the trailer/stats can attach somewhere.
  ensureBubble();

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
