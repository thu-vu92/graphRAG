/**
 * app.js — topic selection, build management, and chat interface.
 */

let currentTopic = null;
let pollInterval = null;
let chatHistory = [];

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  await refreshTopics();

  document.getElementById("topic-select").addEventListener("change", onTopicChange);
  document.getElementById("btn-build").addEventListener("click", onBuild);
  document.getElementById("chat-input").addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  document.getElementById("btn-send").addEventListener("click", sendMessage);
});

// ── Topic management ──────────────────────────────────────────────────────────

async function refreshTopics() {
  try {
    const res  = await fetch("/api/topics");
    const data = await res.json();

    const select = document.getElementById("topic-select");
    const currentVal = select.value;
    select.innerHTML = '<option value="">— Select a topic —</option>';
    data.forEach(t => {
      const opt = document.createElement("option");
      opt.value = t.topic;
      opt.textContent = t.has_graph
        ? `${t.topic} (${t.node_count || "?"} nodes)`
        : `${t.topic} (not built)`;
      select.appendChild(opt);
    });
    // Restore selection
    if (currentVal) select.value = currentVal;
  } catch (err) {
    console.error("Failed to load topics:", err);
  }
}

async function onTopicChange() {
  const topic = document.getElementById("topic-select").value;
  if (!topic) {
    currentTopic = null;
    clearGraph();
    clearChat();
    setStatus("", "");
    return;
  }

  currentTopic = topic;
  clearChat();
  stopPolling();

  // Load status
  const status = await fetchTopicStatus(topic);
  applyStatus(status);

  if (status.has_graph) {
    await loadGraph(topic);
  } else {
    clearGraph();
    setStatus("idle", "No graph built yet — click Build Graph");
  }
}

async function fetchTopicStatus(topic) {
  const res = await fetch(`/api/topics/${encodeURIComponent(topic)}/status`);
  return await res.json();
}

function applyStatus(status) {
  if (status.build_status === "building") {
    setStatus("building", status.build_progress || "Building...");
    startPolling(status.topic);
  } else if (status.build_status === "complete") {
    setStatus("complete", `${status.node_count || "?"} nodes · ${status.edge_count || "?"} edges · ${status.community_count || "?"} communities`);
  } else if (status.build_status === "error") {
    setStatus("error", status.build_error || "Build failed");
  } else {
    setStatus("idle", "Ready to build");
  }
}

// ── Graph build ───────────────────────────────────────────────────────────────

async function onBuild() {
  if (!currentTopic) {
    alert("Please select a topic first.");
    return;
  }

  const btn = document.getElementById("btn-build");
  btn.disabled = true;

  try {
    const res = await fetch(`/api/topics/${encodeURIComponent(currentTopic)}/build`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });

    if (res.status === 409) {
      const data = await res.json();
      setStatus("building", data.detail || "Already building...");
    } else if (!res.ok) {
      const data = await res.json();
      setStatus("error", data.detail || "Build request failed");
      btn.disabled = false;
      return;
    } else {
      setStatus("building", "Starting...");
    }

    startPolling(currentTopic);
  } catch (err) {
    setStatus("error", "Network error: " + err.message);
    btn.disabled = false;
  }
}

function startPolling(topic) {
  stopPolling();
  pollInterval = setInterval(async () => {
    if (!currentTopic || currentTopic !== topic) { stopPolling(); return; }
    try {
      const status = await fetchTopicStatus(topic);
      applyStatus(status);
      if (status.build_status === "complete") {
        stopPolling();
        document.getElementById("btn-build").disabled = false;
        await loadGraph(topic);
        await refreshTopics();
      } else if (status.build_status === "error") {
        stopPolling();
        document.getElementById("btn-build").disabled = false;
      }
    } catch (err) {
      console.error("Poll error:", err);
    }
  }, 2500);
}

function stopPolling() {
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

// ── Graph visualization ───────────────────────────────────────────────────────

async function loadGraph(topic) {
  try {
    const res = await fetch(`/api/topics/${encodeURIComponent(topic)}/graph`);
    if (!res.ok) { clearGraph(); return; }
    const data = await res.json();
    initGraph(data);  // from graph.js
  } catch (err) {
    console.error("Failed to load graph:", err);
    clearGraph();
  }
}

// ── Status badge ──────────────────────────────────────────────────────────────

function setStatus(state, message) {
  const badge = document.getElementById("status-badge");
  const text  = document.getElementById("status-text");
  badge.className = "status-badge " + (state || "");
  text.textContent = message || "";
}

// ── Chat ──────────────────────────────────────────────────────────────────────

function clearChat() {
  chatHistory = [];
  const messages = document.getElementById("chat-messages");
  messages.innerHTML = `
    <div class="chat-welcome">
      <div class="chat-welcome-title">GraphRAG Explorer</div>
      <div class="chat-welcome-sub">Select a topic and ask anything about its knowledge graph.</div>
    </div>`;
}

async function sendMessage() {
  if (!currentTopic) {
    appendMessage("system", "Please select a topic first.");
    return;
  }

  const input = document.getElementById("chat-input");
  const query = input.value.trim();
  if (!query) return;

  input.value = "";
  appendMessage("user", query);

  const loadingId = appendMessage("assistant", null, true);

  try {
    const res = await fetch(`/api/topics/${encodeURIComponent(currentTopic)}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const data = await res.json();
    removeMessage(loadingId);

    if (!res.ok) {
      appendMessage("system", data.detail || "Query failed.");
    } else {
      appendMessage("assistant", data.answer, false, {
        communities_checked: data.communities_checked,
        relevant_communities: data.relevant_communities,
      });
    }
  } catch (err) {
    removeMessage(loadingId);
    appendMessage("system", "Network error: " + err.message);
  }
}

let _msgId = 0;

function appendMessage(role, content, loading = false, meta = null) {
  const id = "msg-" + (++_msgId);
  const messages = document.getElementById("chat-messages");

  const div = document.createElement("div");
  div.id = id;
  div.className = `chat-message chat-${role}`;

  if (loading) {
    div.innerHTML = `<div class="chat-loading"><span></span><span></span><span></span></div>`;
  } else {
    // Convert newlines to <br> and basic markdown bold
    const html = (content || "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
    div.innerHTML = `<div class="chat-bubble">${html}</div>`;

    if (meta && role === "assistant") {
      div.innerHTML += `
        <div class="chat-meta">
          Checked ${meta.communities_checked} communities ·
          ${meta.relevant_communities} relevant
        </div>`;
    }
  }

  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return id;
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}
