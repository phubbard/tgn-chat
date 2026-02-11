/**
 * app.js — Chat UI and Ollama streaming
 *
 * Orchestrates: user query -> embedding -> vector search -> LLM generation
 */

const CHAT_URL = '/api/chat';
const CHAT_MODEL = 'llama3.1';  // Change to your preferred Ollama chat model

const chat = document.getElementById('chat');
const form = document.getElementById('ask-form');
const queryInput = document.getElementById('query');
const submitBtn = document.getElementById('submit-btn');
const status = document.getElementById('status');

const conversationHistory = [];

// Initialize on load
(async () => {
  try {
    const info = await initDB();
    status.textContent = `Loaded: ${info.episodes} episodes, ${info.chunks} chunks (${info.model})`;
    queryInput.disabled = false;
    submitBtn.disabled = false;
  } catch (e) {
    status.textContent = `Error loading database: ${e.message}`;
    console.error(e);
  }
})();

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  queryInput.value = '';
  submitBtn.disabled = true;
  queryInput.disabled = true;

  appendMessage('user', query);

  try {
    status.textContent = 'Searching...';
    const results = await search(query);

    status.textContent = 'Generating response...';
    await generateResponse(query, results);
  } catch (e) {
    appendMessage('assistant', `Error: ${e.message}`);
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
    const header = `[Episode ${r.episode_number}: ${r.episode_title}]`;
    const speakers = r.speakers ? ` (${r.speakers})` : '';
    parts.push(`${header}${speakers}\n${r.content}`);
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

async function generateResponse(query, searchResults) {
  const context = buildContext(searchResults);

  const systemPrompt = `You are a knowledgeable assistant for The Grey NATO podcast, a show about watches, gear, travel, and adventure hosted by James Stacy and Jason Heaton. Answer questions using the retrieved transcript excerpts and episode information provided below. Cite specific episode numbers when possible. If the context doesn't contain enough information to answer, say so.

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
      model: CHAT_MODEL,
      messages: messages,
      stream: true,
    }),
  });

  if (!resp.ok) throw new Error(`Chat API error: ${resp.status}`);

  const msgDiv = appendMessage('assistant', '');
  let fullResponse = '';

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value, { stream: true });
    for (const line of text.split('\n')) {
      if (!line.trim()) continue;
      try {
        const data = JSON.parse(line);
        if (data.message?.content) {
          fullResponse += data.message.content;
          msgDiv.innerHTML = marked.parse(fullResponse);
          chat.scrollTop = chat.scrollHeight;
        }
      } catch (e) {
        // Partial JSON line, ignore
      }
    }
  }

  // Add sources
  msgDiv.innerHTML = marked.parse(fullResponse) + buildSourcesHTML(searchResults);
  chat.scrollTop = chat.scrollHeight;

  // Track conversation
  conversationHistory.push({ role: 'user', content: query });
  conversationHistory.push({ role: 'assistant', content: fullResponse });
}
