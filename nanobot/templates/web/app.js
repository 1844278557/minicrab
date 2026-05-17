const messagesEl = document.querySelector("#messages");
const statusEl = document.querySelector("#connectionStatus");
const formEl = document.querySelector("#composer");
const inputEl = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");

let socket;
let pendingStatus;
let connected = false;
let historyLoaded = false;

function appendMessage(role, content, approval) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  const text = document.createElement("div");
  text.textContent = content;
  node.appendChild(text);
  if (approval && approval.id) {
    node.appendChild(createApprovalActions(approval));
  }
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
}

function createApprovalActions(approval) {
  const actions = document.createElement("div");
  actions.className = "approval-actions";
  const approve = document.createElement("button");
  approve.type = "button";
  approve.textContent = "同意";
  approve.addEventListener("click", () => sendApproval("同意", approval.id, actions));
  const reject = document.createElement("button");
  reject.type = "button";
  reject.textContent = "拒绝";
  reject.className = "secondary";
  reject.addEventListener("click", () => sendApproval("拒绝", approval.id, actions));
  actions.append(approve, reject);
  return actions;
}

function sendApproval(action, id, actions) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  actions.querySelectorAll("button").forEach((button) => {
    button.disabled = true;
  });
  const content = `${action} ${id}`;
  appendMessage("user", content);
  socket.send(JSON.stringify({ type: "message", content }));
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
    clearPending();
    if (data.type === "message") {
      appendMessage(data.role || "assistant", data.content || "", data.approval);
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
