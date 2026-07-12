// app.js — 电商智能客服 Web Client 前端逻辑
(function () {
  "use strict";

  // ==============================================================
  // State
  // ==============================================================
  const TOKEN_KEY = "ec_cs_token";
  const USER_KEY  = "ec_cs_user";
  const MSG_PREFIX = "ec_cs_msgs_";

  let token = localStorage.getItem(TOKEN_KEY) || "";
  let user  = null;
  try { user = JSON.parse(localStorage.getItem(USER_KEY)); } catch (_) {}

  // ==============================================================
  // DOM refs
  // ==============================================================
  const $  = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const authView      = $("#authView");
  const chatView      = $("#chatView");
  const loginForm     = $("#loginForm");
  const registerForm  = $("#registerForm");
  const tabLogin      = $("#tabLogin");
  const tabRegister   = $("#tabRegister");
  const loginError    = $("#loginError");
  const regError      = $("#regError");
  const loginBtn      = $("#loginBtn");
  const regBtn        = $("#regBtn");

  const messageList   = $("#messageList");
  const messageInput  = $("#messageInput");
  const btnSend       = $("#btnSend");
  const btnClearChat  = $("#btnClearChat");
  const btnLogout     = $("#btnLogout");
  const loadingOverlay = $("#loadingOverlay");
  const chatStatus    = $("#chatStatus");
  const sidebarUsername = $("#sidebarUsername");
  const sidebarAvatar = $("#sidebarAvatar");

  // ==============================================================
  // Helpers
  // ==============================================================
  function showLoading(msg) {
    if (loadingOverlay) {
      loadingOverlay.classList.remove("hidden");
      const p = loadingOverlay.querySelector("p");
      if (p) p.textContent = msg || "加载中…";
    }
  }

  function hideLoading() {
    if (loadingOverlay) loadingOverlay.classList.add("hidden");
  }

  function formatTime() {
    const d = new Date();
    return d.getHours().toString().padStart(2, "0") + ":" +
           d.getMinutes().toString().padStart(2, "0");
  }

  function getMsgKey() {
    return MSG_PREFIX + (user ? user.id : "anon");
  }

  function loadMessages() {
    try {
      return JSON.parse(localStorage.getItem(getMsgKey()) || "[]");
    } catch (_) { return []; }
  }

  function saveMessages(msgs) {
    // keep at most 100 messages in localStorage
    const trimmed = msgs.slice(-100);
    localStorage.setItem(getMsgKey(), JSON.stringify(trimmed));
    return trimmed;
  }

  function clearMessages() {
    localStorage.removeItem(getMsgKey());
  }

  // Convert our local message format to the API format
  function messagesToApi(msgs) {
    return msgs.map(function (m) {
      return { role: m.role, content: m.content };
    });
  }

  // ==============================================================
  // API calls
  // ==============================================================
  function api(path, method, body) {
    var opts = {
      method: method || "GET",
      headers: { "Content-Type": "application/json" },
    };
    if (token) opts.headers["Authorization"] = "Bearer " + token;
    if (body) opts.body = JSON.stringify(body);

    return fetch(path, opts).then(function (resp) {
      return resp.json().then(function (data) {
        if (!resp.ok) {
          var err = new Error(data.detail || "请求失败");
          err.status = resp.status;
          throw err;
        }
        return data;
      });
    });
  }

  // ==============================================================
  // Auth
  // ==============================================================
  function switchTab(tab) {
    var isLogin = tab === "login";
    loginForm.classList.toggle("hidden", !isLogin);
    registerForm.classList.toggle("hidden", isLogin);
    tabLogin.classList.toggle("active", isLogin);
    tabRegister.classList.toggle("active", !isLogin);
    loginError.classList.add("hidden");
    regError.classList.add("hidden");
  }

  function handleAuthSuccess(data) {
    token = data.token;
    user  = data.user;
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, JSON.stringify(user));
    showChat();
  }

  function doLogin(e) {
    e.preventDefault();
    var username = $("#loginUsername").value.trim();
    var password = $("#loginPassword").value.trim();
    if (!username || !password) {
      loginError.textContent = "请填写用户名和密码";
      loginError.classList.remove("hidden");
      return;
    }
    loginBtn.disabled = true;
    loginError.classList.add("hidden");

    api("/api/auth/login", "POST", { username: username, password: password })
      .then(handleAuthSuccess)
      .catch(function (err) {
        loginError.textContent = err.message;
        loginError.classList.remove("hidden");
      })
      .finally(function () { loginBtn.disabled = false; });
  }

  function doRegister(e) {
    e.preventDefault();
    var username = $("#regUsername").value.trim();
    var password = $("#regPassword").value.trim();
    var password2 = $("#regPassword2").value.trim();

    if (!username || !password) {
      regError.textContent = "请填写用户名和密码";
      regError.classList.remove("hidden");
      return;
    }
    if (password !== password2) {
      regError.textContent = "两次输入的密码不一致";
      regError.classList.remove("hidden");
      return;
    }

    regBtn.disabled = true;
    regError.classList.add("hidden");

    api("/api/auth/register", "POST", { username: username, password: password })
      .then(handleAuthSuccess)
      .catch(function (err) {
        regError.textContent = err.message;
        regError.classList.remove("hidden");
      })
      .finally(function () { regBtn.disabled = false; });
  }

  function doLogout() {
    token = "";
    user  = null;
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    showAuth();
  }

  // ==============================================================
  // Chat
  // ==============================================================
  function showAuth() {
    authView.classList.remove("hidden");
    chatView.classList.add("hidden");
    $("#loginUsername").value = "";
    $("#loginPassword").value = "";
    $("#regUsername").value = "";
    $("#regPassword").value = "";
    $("#regPassword2").value = "";
    loginError.classList.add("hidden");
    regError.classList.add("hidden");
    switchTab("login");
  }

  function showChat() {
    authView.classList.add("hidden");
    chatView.classList.remove("hidden");

    if (user) {
      sidebarUsername.textContent = user.username;
      sidebarAvatar.textContent = user.username.charAt(0).toUpperCase();
    }

    // Render saved messages
    renderAllMessages();
    messageInput.focus();
  }

  function addMessageBubble(role, content, time) {
    var div = document.createElement("div");
    div.className = "message " + (role === "human" ? "message-user" : "message-ai");

    var avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "human" ? (user ? user.username.charAt(0).toUpperCase() : "U") : "🤖";

    var body = document.createElement("div");
    body.className = "message-body";

    var bubble = document.createElement("div");
    bubble.className = "message-content";
    bubble.textContent = content;

    var timeEl = document.createElement("div");
    timeEl.className = "message-time";
    timeEl.textContent = time || formatTime();

    body.appendChild(bubble);
    body.appendChild(timeEl);
    div.appendChild(avatar);
    div.appendChild(body);

    messageList.appendChild(div);
    scrollToBottom();
  }

  function addTypingBubble() {
    var div = document.createElement("div");
    div.className = "message message-ai message-typing";
    div.id = "typingBubble";

    var avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "🤖";

    var body = document.createElement("div");
    body.className = "message-body";

    var bubble = document.createElement("div");
    bubble.className = "message-content";
    bubble.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

    body.appendChild(bubble);
    div.appendChild(avatar);
    div.appendChild(body);

    messageList.appendChild(div);
    scrollToBottom();
  }

  function removeTypingBubble() {
    var el = $("#typingBubble");
    if (el) el.remove();
  }

  function scrollToBottom() {
    messageList.scrollTop = messageList.scrollHeight;
  }

  function renderAllMessages() {
    // Clear existing messages (keep the last render)
    messageList.querySelectorAll(".message").forEach(function (el) { el.remove(); });

    // Add welcome if empty
    var msgs = loadMessages();
    if (msgs.length === 0) {
      var welcome = document.createElement("div");
      welcome.className = "message message-ai";
      welcome.innerHTML =
        '<div class="message-avatar">🤖</div>' +
        '<div class="message-body">' +
        '<div class="message-content">你好呀！我是你的专属购物助手 <b>小智</b> 🎉<br>不管是想了解商品、查订单，还是退换货，我都能帮你搞定～<br><br>有什么可以帮你的吗？😊</div>' +
        '<div class="message-time">刚刚</div>' +
        '</div>';
      messageList.appendChild(welcome);
      return;
    }

    msgs.forEach(function (m) {
      addMessageBubble(m.role, m.content, m.time);
    });
  }

  // 创建一个空的 AI 消息气泡（用于流式填充）
  function createStreamingBubble() {
    var div = document.createElement("div");
    div.className = "message message-ai";
    div.id = "streamingBubble";

    var avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "🤖";

    var body = document.createElement("div");
    body.className = "message-body";

    var bubble = document.createElement("div");
    bubble.className = "message-content";
    bubble.id = "streamingContent";
    bubble.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

    var timeEl = document.createElement("div");
    timeEl.className = "message-time";
    timeEl.id = "streamingTime";
    timeEl.textContent = formatTime();

    body.appendChild(bubble);
    body.appendChild(timeEl);
    div.appendChild(avatar);
    div.appendChild(body);

    messageList.appendChild(div);
    scrollToBottom();
    return div;
  }

  function finalizeStreamingBubble(responseText) {
    var bubble = $("#streamingBubble");
    if (!bubble) return;
    bubble.removeAttribute("id");
    bubble.classList.remove("message-typing");
    var content = $("#streamingContent");
    if (content) {
      content.removeAttribute("id");
      content.textContent = responseText;
    }
    var timeEl = $("#streamingTime");
    if (timeEl) timeEl.removeAttribute("id");
    // Remove typing dots if still present
    var dots = bubble.querySelectorAll(".typing-dot");
    dots.forEach(function (d) { d.remove(); });
  }

  function sendMessage() {
    var text = messageInput.value.trim();
    if (!text) return;

    // Disable input while waiting
    messageInput.disabled = true;
    btnSend.disabled = true;
    chatStatus.textContent = "思考中…";
    chatStatus.style.color = "#f59e0b";

    var localMsgs = loadMessages();

    // Add user message to UI and local storage
    var time = formatTime();
    addMessageBubble("human", text, time);
    localMsgs.push({ role: "human", content: text, time: time });
    saveMessages(localMsgs);

    messageInput.value = "";
    messageInput.style.height = "auto";

    // Show streaming bubble (typing indicator inside)
    createStreamingBubble();

    // Use SSE streaming endpoint
    var apiMsgs = messagesToApi(localMsgs.slice(0, -1));
    fetch("/api/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token,
      },
      body: JSON.stringify({
        user_input: text,
        messages: apiMsgs,
      }),
    }).then(function (response) {
      if (!response.ok) {
        return response.json().then(function (err) {
          throw new Error(err.detail || "请求失败");
        });
      }

      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      var fullResponse = "";

      function processChunk() {
        return reader.read().then(function (result) {
          if (result.done) return;

          buffer += decoder.decode(result.value, { stream: true });

          // Parse SSE events from buffer
          var lines = buffer.split("\n");
          buffer = lines.pop() || ""; // keep incomplete line in buffer

          for (var i = 0; i < lines.length; i++) {
            var line = lines[i].trim();
            if (!line || line.indexOf("data: ") !== 0) continue;

            try {
              var eventData = JSON.parse(line.substring(6));
              var contentType = $("#streamingContent");

              if (eventData.type === "token") {
                // First token: remove typing dots
                if (!fullResponse) {
                  if (contentType) contentType.innerHTML = "";
                }
                fullResponse += eventData.content;
                if (contentType) contentType.textContent = fullResponse;
                chatStatus.textContent = "回复中…";
                chatStatus.style.color = "#f59e0b";
                scrollToBottom();
              } else if (eventData.type === "status") {
                if (contentType) contentType.textContent = eventData.content || "处理中…";
                chatStatus.textContent = eventData.content || "处理中…";
              } else if (eventData.type === "done") {
                finalizeStreamingBubble(fullResponse || eventData.response || "");
                localMsgs.push({ role: "ai", content: fullResponse || eventData.response || "", time: formatTime() });
                saveMessages(localMsgs);
                chatStatus.textContent = "在线";
                chatStatus.style.color = "#22c55e";
              } else if (eventData.type === "error") {
                finalizeStreamingBubble("抱歉，" + (eventData.content || "服务异常") + " 😥");
                chatStatus.textContent = "异常";
                chatStatus.style.color = "#ef4444";
                setTimeout(function () {
                  chatStatus.textContent = "在线";
                  chatStatus.style.color = "#22c55e";
                }, 3000);
              }
            } catch (e) {
              // ignore parse errors for partial SSE lines
            }
          }

          return processChunk(); // continue reading
        });
      }

      return processChunk();
    }).catch(function (err) {
      finalizeStreamingBubble("抱歉，" + (err.message || "服务异常，请稍后重试") + " 😥");
      chatStatus.textContent = "异常";
      chatStatus.style.color = "#ef4444";
      setTimeout(function () {
        chatStatus.textContent = "在线";
        chatStatus.style.color = "#22c55e";
      }, 3000);
    }).finally(function () {
      messageInput.disabled = false;
      btnSend.disabled = false;
      messageInput.focus();
    });
  }

  function clearChat() {
    if (confirm("确定要清空当前对话记录吗？")) {
      clearMessages();
      renderAllMessages();
    }
  }

  // ==============================================================
  // Event bindings
  // ==============================================================
  tabLogin.addEventListener("click", function () { switchTab("login"); });
  tabRegister.addEventListener("click", function () { switchTab("register"); });
  loginForm.addEventListener("submit", doLogin);
  registerForm.addEventListener("submit", doRegister);
  btnSend.addEventListener("click", sendMessage);
  btnClearChat.addEventListener("click", clearChat);
  btnLogout.addEventListener("click", doLogout);

  // Enter to send, Shift+Enter for newline
  messageInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-resize textarea if we switch to textarea later
  messageInput.addEventListener("input", function () {
    if (messageInput.tagName === "TEXTAREA") {
      messageInput.style.height = "auto";
      messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
    }
  });

  // ==============================================================
  // Init — check if already logged in
  // ==============================================================
  if (token && user) {
    // Verify token is still valid
    showLoading("验证登录状态…");
    api("/api/auth/me", "GET")
      .then(function (data) {
        user = data;
        localStorage.setItem(USER_KEY, JSON.stringify(user));
        hideLoading();
        showChat();
      })
      .catch(function () {
        // Token invalid, clear and show login
        token = "";
        user  = null;
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
        hideLoading();
        showAuth();
      });
  } else {
    showAuth();
  }
})();
