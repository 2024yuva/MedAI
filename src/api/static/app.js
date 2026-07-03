/* ── Constants ── */
const STORAGE_KEY = "medai_chats_v2";
const THEME_KEY   = "medai_theme";

const CHIPS = [
  { label: "Symptoms" },
  { label: "Diagnosis" },
  { label: "Treatments" },
  { label: "Drug Info" },
  { label: "Lab Reports" },
];

const TOPICS = [
  { title: "Fever with chills",  prompt: "I have fever with chills. What could it mean?" },
  { title: "Chest pain causes",  prompt: "What are possible causes of chest pain?" },
  { title: "Medicine safety",    prompt: "What should I ask before taking a new medicine?" },
  { title: "CBC report help",    prompt: "Help me understand common CBC lab report findings." },
];

/* ── State ── */
const state = {
  chats: [],
  activeId: null,
  loading: false,
  search: "",
  dark: true,
};

/* ── DOM ── */
const $ = id => document.getElementById(id);
const els = {
  groups:    $("chat-groups"),
  messages:  $("messages"),
  form:      $("ask-form"),
  input:     $("q-input"),
  send:      $("send-btn"),
  newChat:   $("new-chat-btn"),
  search:    $("chat-search"),
  dialog:    $("del-dialog"),
  themeBtn:  $("theme-btn"),
  themeIcon: $("theme-icon"),
  sidebar:   $("sidebar"),
  overlay:   $("sb-overlay"),
  hamburger: $("hamburger"),
  sbClose:   $("sb-close"),
};

/* ── Theme ── */
function loadTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  state.dark = saved !== "light";
  applyTheme();
}
function applyTheme() {
  document.documentElement.setAttribute("data-theme", state.dark ? "dark" : "light");
  els.themeIcon.textContent = state.dark ? "Dark" : "Light";
}

/* ── Persistence ── */
function save() { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.chats)); }
function loadChats() {
  try { state.chats = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); }
  catch { state.chats = []; }
  if (!state.chats.length) state.chats = [newChat_()];
  state.chats.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
  state.activeId = state.chats[0].id;
}

/* ── Chat helpers ── */
function newChat_() {
  const now = new Date().toISOString();
  return { id: crypto.randomUUID(), title: "New chat", createdAt: now, updatedAt: now, messages: [] };
}
function active() { return state.chats.find(c => c.id === state.activeId); }
function bucket(iso) {
  const d = new Date(iso), now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yest  = new Date(today); yest.setDate(yest.getDate() - 1);
  if (d >= today) return "Today";
  if (d >= yest)  return "Yesterday";
  return "Previous";
}
function titleFrom(q) {
  const s = q.trim().replace(/\s+/g, " ");
  return s.length > 52 ? s.slice(0, 52) + "…" : s;
}
function esc(t) {
  return (t || "").replace(/[&<>"]/g, m => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[m]));
}

/* ── Sidebar ── */
function renderSidebar() {
  const groups = { Today: [], Yesterday: [], Previous: [] };
  const q = state.search.trim().toLowerCase();
  state.chats.filter(c => c.title.toLowerCase().includes(q))
    .forEach(c => groups[bucket(c.updatedAt)].push(c));

  els.groups.innerHTML = "";
  for (const [label, chats] of Object.entries(groups)) {
    if (!chats.length) continue;
    const sec = document.createElement("section");
    sec.innerHTML = `<div class="group-label">${label}</div>`;
    chats.forEach(chat => {
      const item = document.createElement("div");
      item.className = `chat-item${chat.id === state.activeId ? " active" : ""}`;
      item.innerHTML = `
        <div class="chat-title" title="${esc(chat.title)}">${esc(chat.title)}</div>
        <div class="chat-menu-wrap">
          <button class="chat-menu-btn" data-id="${chat.id}" aria-label="Options" title="Options">
            <span></span><span></span><span></span>
          </button>
          <div class="chat-dropdown" id="dd-${chat.id}">
            <button class="dd-item" data-action="rename" data-id="${chat.id}">Rename</button>
            <button class="dd-item danger" data-action="delete" data-id="${chat.id}">Delete</button>
          </div>
        </div>`;
      item.querySelector(".chat-title").addEventListener("click", () => {
        state.activeId = chat.id;
        closeAllDropdowns();
        closeSidebar();
        render();
      });
      item.querySelector(".chat-menu-btn").addEventListener("click", e => {
        e.stopPropagation();
        const dd = document.getElementById(`dd-${chat.id}`);
        const isOpen = dd.classList.contains("open");
        closeAllDropdowns();
        if (!isOpen) dd.classList.add("open");
      });
      sec.appendChild(item);
    });
    els.groups.appendChild(sec);
  }
}

function closeAllDropdowns() {
  document.querySelectorAll(".chat-dropdown.open").forEach(d => d.classList.remove("open"));
}

/* ── Message rendering ── */
function cleanAnswer(text) {
  return (text || "")
    .replace(/^\s*\d+\.\s*.+$/gm, "")
    .replace(/Final Answer:\s*/i, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function srcCard(s, i) {
  return `<div class="src-card">
    <div class="src-file">${i+1}. ${esc(s.sourceFile)}</div>
    <div class="src-meta">Page ${s.pageNumber} · ${Number(s.similarityScore).toFixed(2)}</div>
    <div class="src-excerpt">${esc(s.excerpt)}</div>
  </div>`;
}

function renderPayload(p) {
  const conf = Math.max(0, Math.min(1, Number(p.confidenceScore || 0)));
  const raw  = p.finalAnswer || p.answer || "";
  const text = cleanAnswer(raw) || raw || "No answer returned.";
  const safety = p.blocked
    ? `<span class="badge blocked">Safety blocked</span>`
    : `<span class="badge safe">Safety passed</span>`;
  const reasoning = p.reasoningSteps?.length ? `
    <details class="reasoning">
      <summary>Reasoning steps <span class="toggle-icon">+</span></summary>
      <ol>${p.reasoningSteps.map(r => `<li>${esc(r)}</li>`).join("")}</ol>
    </details>` : "";
  const sources = p.sources?.length ? `
    <div class="sources">
      <div class="sources-lbl">Sources</div>
      <div class="src-grid">${p.sources.map(srcCard).join("")}</div>
    </div>` : "";
  return `
    <div class="ans-head"><div class="ai-avatar">MA</div><span class="ans-title">MedAI</span></div>
    <div class="ans-body">${esc(text)}</div>
    <div class="meta-row">${safety}${p.blockReason ? `<span class="badge">${esc(p.blockReason)}</span>` : ""}</div>
    <div class="conf-card">
      <span class="conf-label">Confidence</span>
      <div class="conf-bar"><div class="conf-fill" style="width:${(conf*100).toFixed(0)}%"></div></div>
      <span class="conf-val">${(conf*100).toFixed(0)}%</span>
    </div>
    ${reasoning}${sources}`;
}

function typingHTML() {
  return `<div class="msg-row assistant">
    <div class="typing-card">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    </div>
  </div>`;
}

/* ── Welcome screen ── */
function welcomeHTML() {
  const chips = CHIPS.map(c =>
    `<button type="button" class="chip" data-prompt="${esc(c.label)}">${c.label}</button>`
  ).join("");
  const topics = TOPICS.map(t =>
    `<button type="button" class="topic-card" data-prompt="${esc(t.prompt)}">
      <span>${esc(t.title)}</span>
    </button>`).join("");
  return `
    <div class="empty-state">
      <h1 class="welcome-heading">Hi, how can I<br>help you today?</h1>
      <p class="welcome-sub">Your medical AI assistant is ready</p>
      <div class="chips">${chips}</div>
      <div class="topics-head">
        <span>Popular topics</span>
        <button class="topics-see-all">See all</button>
      </div>
      <div class="topics-grid">${topics}</div>
    </div>`;
}

/* ── Render messages ── */
function renderMessages() {
  const chat = active();
  els.messages.innerHTML = "";

  if (!chat.messages.length && !state.loading) {
    els.messages.innerHTML = welcomeHTML();
    bindShortcuts();
    return;
  }

  for (const msg of chat.messages) {
    const row = document.createElement("div");
    row.className = `msg-row ${msg.role}`;
    const bubble = document.createElement("article");
    bubble.className = `bubble ${msg.role}`;
    bubble.innerHTML = msg.role === "assistant" ? renderPayload(msg.payload) : esc(msg.content);
    row.appendChild(bubble);
    els.messages.appendChild(row);
  }

  if (state.loading) els.messages.insertAdjacentHTML("beforeend", typingHTML());
  requestAnimationFrame(() => {
    const wrap = els.messages.parentElement;
    wrap.scrollTop = wrap.scrollHeight;
  });
}

/* ── Full render ── */
function render() {
  state.chats.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
  renderSidebar();
  renderMessages();
  save();
}

/* ── API ── */
async function ask(question) {
  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error((await res.text()) || "Request failed");
  return res.json();
}

/* ── Submit ── */
async function submit(question) {
  if (state.loading) return;
  const q = question.trim();
  if (!q) return;

  const chat = active();
  if (!chat.messages.filter(m => m.role === "user").length) chat.title = titleFrom(q);

  chat.messages.push({ role: "user", content: q });
  
  const assistantMsg = { 
    role: "assistant", 
    payload: { answer: "", reasoningSteps: [], sources: [], confidenceScore: 0 }, 
    content: "" 
  };
  chat.messages.push(assistantMsg);
  
  chat.updatedAt = new Date().toISOString();
  state.loading = true;
  els.send.disabled = true;
  els.input.value = "";
  resize();
  render();

  try {
    const res = await fetch("/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    
    if (!res.ok) throw new Error("Request failed");

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    
    state.loading = false;
    renderMessages(); // clear typing indicator

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let lines = buffer.split("

");
      buffer = lines.pop(); // keep incomplete event

      for (const eventStr of lines) {
        const line = eventStr.trim();
        if (line.startsWith("data: ")) {
          const dataStr = line.slice(6);
          if (dataStr === "[DONE]") {
             // finished
          } else {
             const data = JSON.parse(dataStr);
             if (data.type === "metadata") {
                assistantMsg.payload.sources = data.sources;
                assistantMsg.payload.confidenceScore = data.confidenceScore;
                renderMessages();
             } else if (data.type === "chunk") {
                assistantMsg.payload.answer += data.text;
                assistantMsg.content += data.text;
                renderMessages();
             }
          }
        }
      }
    }
    chat.updatedAt = new Date().toISOString();
  } catch (err) {
    assistantMsg.payload = { answer: `Error: ${err.message}`, reasoningSteps: [], sources: [], confidenceScore: 0, blocked: true, blockReason: "request_error" };
    assistantMsg.content = `Error: ${err.message}`;
  } finally {
    state.loading = false;
    els.send.disabled = false;
    render(); // Full render saves to localStorage
  }
}

/* ── New chat ── */
function startNewChat() {
  const chat = newChat_();
  state.chats.unshift(chat);
  state.activeId = chat.id;
  render();
  els.input.focus();
}

/* ── Sidebar open/close ── */
function openSidebar()  { els.sidebar.classList.add("open"); els.overlay.classList.add("open"); }
function closeSidebar() { els.sidebar.classList.remove("open"); els.overlay.classList.remove("open"); }

/* ── Sidebar actions ── */
function handleSidebarClick(e) {
  const btn = e.target.closest("button[data-action]");
  if (!btn) return;
  const { id, action } = btn.dataset;
  const chat = state.chats.find(c => c.id === id);
  if (!chat) return;
  closeAllDropdowns();

  if (action === "rename") {
    const next = prompt("Rename chat", chat.title);
    if (next?.trim()) { chat.title = next.trim(); render(); }
    return;
  }
  if (action === "delete") {
    els.dialog.showModal();
    els.dialog.addEventListener("close", () => {
      if (els.dialog.returnValue !== "confirm") return;
      state.chats = state.chats.filter(c => c.id !== id);
      if (!state.chats.length) state.chats.push(newChat_());
      if (!state.chats.find(c => c.id === state.activeId)) state.activeId = state.chats[0].id;
      render();
    }, { once: true });
  }
}

/* ── Prompt shortcuts ── */
function bindShortcuts() {
  els.messages.querySelectorAll("[data-prompt]").forEach(btn => {
    btn.addEventListener("click", () => {
      els.input.value = btn.dataset.prompt;
      resize();
      els.input.focus();
    });
  });
}

/* ── Textarea auto-resize ── */
function resize() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(140, els.input.scrollHeight) + "px";
}

/* ── Events ── */
els.form.addEventListener("submit", e => { e.preventDefault(); submit(els.input.value); });
els.newChat.addEventListener("click", startNewChat);
els.groups.addEventListener("click", handleSidebarClick);
els.input.addEventListener("input", resize);
els.search.addEventListener("input", e => { state.search = e.target.value; renderSidebar(); });
els.hamburger.addEventListener("click", openSidebar);
els.sbClose.addEventListener("click", closeSidebar);
els.overlay.addEventListener("click", closeSidebar);
els.themeBtn.addEventListener("click", () => {
  state.dark = !state.dark;
  localStorage.setItem(THEME_KEY, state.dark ? "dark" : "light");
  applyTheme();
});
document.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && document.activeElement === els.input) {
    e.preventDefault(); els.form.requestSubmit();
  }
  if (e.key === "Escape") { closeSidebar(); closeAllDropdowns(); }
});
document.addEventListener("click", e => {
  if (!e.target.closest(".chat-menu-wrap")) closeAllDropdowns();
});

/* ── Init ── */
loadChats();
loadTheme();
render();
