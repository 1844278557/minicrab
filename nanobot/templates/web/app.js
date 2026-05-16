const messagesEl = document.querySelector("#messages");
const statusEl = document.querySelector("#connectionStatus");
const formEl = document.querySelector("#composer");
const inputEl = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");

let socket;
let pendingStatus;

function appendMessage(role, content) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = content;
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
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

function setConnected(connected) {
  statusEl.textContent = connected ? "已连接" : "连接已断开";
  sendButton.disabled = !connected;
}

function connect() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${window.location.host}/web/ws`);

  socket.addEventListener("open", () => {
    setConnected(true);
  });

  socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "status") {
      if (data.content === "processing") {
        setPending("处理中");
      }
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
