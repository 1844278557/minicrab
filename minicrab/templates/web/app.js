const messagesEl = document.querySelector("#messages");
const statusEl = document.querySelector("#connectionStatus");
const formEl = document.querySelector("#composer");
const inputEl = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");

let socket;
let pendingStatus;
let connected = false;
let historyLoaded = false;
let approvalModal;

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderInlineMarkdown(value) {
  const codeSpans = [];
  let html = escapeHtml(value);
  html = html.replace(/`([^`]+)`/g, (_match, code) => {
    const token = `@@CODE_SPAN_${codeSpans.length}@@`;
    codeSpans.push(`<code>${code}</code>`);
    return token;
  });
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_match, label, url) => {
    return `<a href="${url}" target="_blank" rel="noreferrer">${label}</a>`;
  });
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  codeSpans.forEach((code, index) => {
    html = html.replace(`@@CODE_SPAN_${index}@@`, code);
  });
  return html;
}

function renderMarkdown(content) {
  const lines = String(content ?? "").replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let listItems = [];
  let listType = "";
  let codeLines = [];
  let inCodeBlock = false;

  function flushParagraph() {
    if (!paragraph.length) {
      return;
    }
    html.push(`<p>${renderInlineMarkdown(paragraph.join("\n"))}</p>`);
    paragraph = [];
  }

  function flushList() {
    if (!listItems.length) {
      return;
    }
    html.push(`<${listType}>${listItems.join("")}</${listType}>`);
    listItems = [];
    listType = "";
  }

  function closeBlocks() {
    flushParagraph();
    flushList();
  }

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (trimmed.startsWith("```")) {
      if (inCodeBlock) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
        inCodeBlock = false;
      } else {
        closeBlocks();
        inCodeBlock = true;
      }
      return;
    }
    if (inCodeBlock) {
      codeLines.push(line);
      return;
    }
    if (!trimmed) {
      closeBlocks();
      return;
    }

    const heading = /^(#{1,3})\s+(.+)$/.exec(trimmed);
    if (heading) {
      closeBlocks();
      const level = heading[1].length + 2;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      return;
    }

    const unordered = /^[-*]\s+(.+)$/.exec(trimmed);
    const ordered = /^\d+\.\s+(.+)$/.exec(trimmed);
    if (unordered || ordered) {
      flushParagraph();
      const nextType = unordered ? "ul" : "ol";
      if (listType && listType !== nextType) {
        flushList();
      }
      listType = nextType;
      listItems.push(`<li>${renderInlineMarkdown((unordered || ordered)[1])}</li>`);
      return;
    }

    const quote = /^>\s?(.+)$/.exec(trimmed);
    if (quote) {
      closeBlocks();
      html.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
      return;
    }

    flushList();
    paragraph.push(line);
  });

  if (inCodeBlock) {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }
  closeBlocks();
  return html.join("") || escapeHtml(content);
}

function appendMessage(role, content) {
  const normalizedRole = String(role || "assistant").toLowerCase();
  const node = document.createElement("div");
  node.className = `message ${normalizedRole}`;
  const text = document.createElement("div");
  text.className = "message-content";
  if (["user", "status", "error"].includes(normalizedRole)) {
    text.textContent = content;
  } else {
    text.innerHTML = renderMarkdown(content);
  }
  node.appendChild(text);
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
}

function ensureApprovalModal() {
  if (approvalModal) {
    return approvalModal;
  }
  const backdrop = document.createElement("div");
  backdrop.className = "approval-backdrop";
  backdrop.hidden = true;

  const panel = document.createElement("div");
  panel.className = "approval-modal";

  const title = document.createElement("h2");
  title.textContent = "需要确认";
  const summary = document.createElement("p");
  summary.className = "approval-summary";
  const reason = document.createElement("p");
  reason.className = "approval-reason";

  const actions = document.createElement("div");
  actions.className = "approval-modal-actions";
  const reject = document.createElement("button");
  reject.type = "button";
  reject.className = "secondary";
  reject.textContent = "拒绝";
  const temporary = document.createElement("button");
  temporary.type = "button";
  temporary.className = "secondary";
  temporary.textContent = "允许30分钟";
  const approve = document.createElement("button");
  approve.type = "button";
  approve.textContent = "允许一次";
  actions.append(reject, temporary, approve);
  panel.append(title, summary, reason, actions);
  backdrop.appendChild(panel);
  document.body.appendChild(backdrop);

  approvalModal = { backdrop, summary, reason, approve, temporary, reject };
  return approvalModal;
}

function showApprovalModal(approval) {
  if (!approval || !approval.id) {
    return;
  }
  const modal = ensureApprovalModal();
  modal.summary.textContent = approval.summary || "执行需要人工确认的操作";
  modal.reason.textContent = approval.reason || "";
  modal.approve.disabled = false;
  modal.temporary.disabled = false;
  modal.reject.disabled = false;
  modal.approve.onclick = () => sendApproval("approved", approval.id);
  modal.temporary.onclick = () => sendApproval("approved_temporary", approval.id);
  modal.reject.onclick = () => sendApproval("rejected", approval.id);
  modal.backdrop.hidden = false;
}

function closeApprovalModal() {
  if (approvalModal) {
    approvalModal.backdrop.hidden = true;
  }
}

function sendApproval(decision, id) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  const modal = ensureApprovalModal();
  modal.approve.disabled = true;
  modal.temporary.disabled = true;
  modal.reject.disabled = true;
  closeApprovalModal();
  socket.send(JSON.stringify({ type: "approval_response", decision, id }));
  setPending("处理中");
}

function setPending(text) {
  clearPending();
  pendingStatus = appendMessage("status", text);
}

function clearPending() {
  if (pendingStatus) {
    pendingStatus.remove();
    pendingStatus = null;
  }
}

function setConnected(isConnected) {
  connected = isConnected;
  statusEl.textContent = connected ? "已连接" : "连接已断开";
  if (!connected) {
    historyLoaded = false;
  }
  updateComposerState();
}

function updateComposerState() {
  const ready = connected && historyLoaded;
  inputEl.disabled = !ready;
  sendButton.disabled = !ready;
}

function requestHistory() {
  historyLoaded = false;
  updateComposerState();
  socket.send(JSON.stringify({ type: "history", limit: 100 }));
}

function renderHistory(messages) {
  clearPending();
  messagesEl.replaceChildren();
  messages.forEach((message) => {
    appendMessage(message.role || "assistant", message.content || "");
  });
  messagesEl.scrollTop = messagesEl.scrollHeight;
  historyLoaded = true;
  updateComposerState();
}

function connect() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${window.location.host}/web/ws`);

  socket.addEventListener("open", () => {
    setConnected(true);
    requestHistory();
  });

  socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "status") {
      if (data.content === "processing") {
        setPending("处理中");
      }
      return;
    }
    if (data.type === "history") {
      renderHistory(Array.isArray(data.messages) ? data.messages : []);
      return;
    }
    if (data.type === "approval_required") {
      clearPending();
      showApprovalModal(data.approval);
      return;
    }
    clearPending();
    if (data.type === "message") {
      appendMessage(data.role || "assistant", data.content || "");
    } else if (data.type === "error") {
      appendMessage("error", data.content || "处理失败，请稍后重试。");
    }
  });

  socket.addEventListener("close", () => {
    clearPending();
    setConnected(false);
  });

  socket.addEventListener("error", () => {
    clearPending();
    appendMessage("error", "连接异常。");
  });
}

function sendMessage() {
  const content = inputEl.value.trim();
  if (!content || !socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  appendMessage("user", content);
  socket.send(JSON.stringify({ type: "message", content }));
  inputEl.value = "";
  inputEl.style.height = "auto";
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = `${inputEl.scrollHeight}px`;
});

setConnected(false);
connect();
