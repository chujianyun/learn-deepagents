/* ============================================================
   DeepSeek 风格聊天界面 —— 前端逻辑
   会话持久化(localStorage) · SSE 流式渲染 · 模型切换 · 联网搜索开关
   ============================================================ */
(() => {
  'use strict';

  // ---------- 工具 ----------
  const $ = (s) => document.querySelector(s);
  const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));

  const LS_CONVS = 'dschat.conversations.v1';
  const LS_SETTINGS = 'dschat.settings.v1';

  const FALLBACK_MODELS = [
    { id: 'deepseek-ai/DeepSeek-V4.1-Flash', name: 'DeepSeek-V4.1-Flash', vendor: 'DeepSeek', desc: '默认模型，响应快，适合日常对话与写作' },
    { id: 'MiniMaxAI/MiniMax-H3', name: 'MiniMax-H3', vendor: 'MiniMax', desc: '长上下文理解，适合长文档与复杂任务' },
    { id: 'tencent/Hy4-preview', name: 'Hy4-preview', vendor: 'Tencent', desc: '腾讯混元预览版，中文表达自然' },
    { id: 'zai-org/GLM-5.3-Flash', name: 'GLM-5.3-Flash', vendor: 'Zhipu', desc: '智谱轻量旗舰，推理与代码能力均衡' },
  ];

  const TOOL_LABELS = {
    tavily_search: '联网搜索',
    write_todos: '规划任务',
    task: '委派子智能体',
  };

  const ICONS = {
    copy: '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M5 15H4.5A1.5 1.5 0 0 1 3 13.5v-9A1.5 1.5 0 0 1 4.5 3h9A1.5 1.5 0 0 1 15 4.5V5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
    refresh: '<svg viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2.3 6.3M20 5v6h-6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    trash: '<svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m3 0l-.8 12a2 2 0 0 1-2 1.9H8.8a2 2 0 0 1-2-1.9L6 7m4 4v6m4-6v6" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    chevron: '<svg class="chevron" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    check: '<svg viewBox="0 0 24 24"><path d="M4.5 12.5l5 5 10-11" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    globe: '<svg class="globe" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M3.5 12h17M12 3.5c2.4 2.3 3.6 5.2 3.6 8.5s-1.2 6.2-3.6 8.5c-2.4-2.3-3.6-5.2-3.6-8.5s1.2-6.2 3.6-8.5z" fill="none" stroke="currentColor" stroke-width="1.7"/></svg>',
    link: '<svg viewBox="0 0 24 24"><path d="M10 14a5 5 0 0 0 7.1 0l2.4-2.4a5 5 0 0 0-7.1-7.1L11 5.9M14 10a5 5 0 0 0-7.1 0l-2.4 2.4a5 5 0 0 0 7.1 7.1L13 18.1" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
    wrench: '<svg class="globe" viewBox="0 0 24 24"><path d="M14.7 6.3a4.5 4.5 0 0 0-6 5.6L3 17.6V21h3.4l5.7-5.7a4.5 4.5 0 0 0 5.6-6L14.5 12l-2.5-2.5 2.7-3.2z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>',
    info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M12 11v5.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="12" cy="7.8" r="1.15" fill="currentColor"/></svg>',
  };

  // ---------- 状态 ----------
  let config = { models: FALLBACK_MODELS, default_model: FALLBACK_MODELS[0].id, search_available: true };
  let settings = { model: FALLBACK_MODELS[0].id, search: false };
  let conversations = [];
  let currentId = null;
  let streaming = false;
  let abortCtrl = null;
  let liveMsg = null;          // 正在流式输出的 assistant 消息
  let liveConv = null;

  const segNodes = new WeakMap();   // segment -> DOM
  const msgNodes = new WeakMap();   // message -> DOM
  const pendingRender = new Set();
  let rafId = 0;

  // ---------- DOM 引用 ----------
  const appEl = $('#app');
  const sidebarEl = $('#sidebar');
  const scrimEl = $('#scrim');
  const historyEl = $('#history');
  const chatScrollEl = $('#chatScroll');
  const welcomeEl = $('#welcome');
  const messagesEl = $('#messages');
  const inputEl = $('#input');
  const sendBtn = $('#sendBtn');
  const searchToggle = $('#searchToggle');
  const modelBtn = $('#modelBtn');
  const modelBtnName = $('#modelBtnName');
  const modelMenu = $('#modelMenu');
  const scrollBottomBtn = $('#scrollBottomBtn');
  const toastEl = $('#toast');
  const modalOverlay = $('#modalOverlay');

  // ---------- 持久化 ----------
  function loadState() {
    try {
      const s = JSON.parse(localStorage.getItem(LS_SETTINGS) || 'null');
      if (s && typeof s === 'object') settings = { ...settings, ...s };
    } catch { /* 忽略损坏数据 */ }
    try {
      const c = JSON.parse(localStorage.getItem(LS_CONVS) || 'null');
      if (Array.isArray(c)) conversations = c;
    } catch { conversations = []; }
  }

  function save() {
    try {
      settings.lastConvId = currentId;
      localStorage.setItem(LS_SETTINGS, JSON.stringify(settings));
      // 仅排除运行时临时字段，保留 fallbackFrom 等业务字段
      const replacer = (k, v) => (k === '_thinkStart' ? undefined : v);
      localStorage.setItem(LS_CONVS, JSON.stringify(conversations.slice(0, 60), replacer));
    } catch { /* 存储满时静默降级 */ }
  }

  const currentConv = () => conversations.find((c) => c.id === currentId) || null;
  const modelInfo = (id) => config.models.find((m) => m.id === id) || { name: id, vendor: '' };

  // ---------- Markdown ----------
  marked.setOptions({ gfm: true, breaks: true });

  function renderMarkdown(text) {
    try {
      return DOMPurify.sanitize(marked.parse(text), { ADD_ATTR: ['target'] });
    } catch {
      return `<p>${escapeHtml(text).replace(/\n/g, '<br>')}</p>`;
    }
  }

  function decorateMarkdown(container) {
    container.querySelectorAll('pre > code').forEach((code) => {
      const pre = code.parentElement;
      if (pre.parentElement && pre.parentElement.classList.contains('code-block')) return;
      const lang = (code.className.match(/language-([\w+-]+)/) || [])[1] || 'text';
      try { hljs.highlightElement(code); } catch { /* 未知语言时保持原样 */ }
      const wrap = document.createElement('div');
      wrap.className = 'code-block';
      const head = document.createElement('div');
      head.className = 'code-head';
      head.innerHTML = `<span>${escapeHtml(lang)}</span>`;
      const copyBtn = document.createElement('button');
      copyBtn.className = 'code-copy';
      copyBtn.textContent = '复制';
      copyBtn.addEventListener('click', () => {
        copyText(code.textContent).then(() => {
          copyBtn.textContent = '已复制';
          setTimeout(() => { copyBtn.textContent = '复制'; }, 1500);
        });
      });
      head.appendChild(copyBtn);
      pre.replaceWith(wrap);
      wrap.append(head, pre);
    });
    container.querySelectorAll('a').forEach((a) => {
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
    });
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.cssText = 'position:fixed;opacity:0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
  }

  // ---------- 提示 ----------
  let toastTimer = 0;
  function toast(text) {
    toastEl.textContent = text;
    toastEl.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastEl.hidden = true; }, 2400);
  }

  // ---------- 侧边栏 ----------
  function groupLabel(ts) {
    const day = 86400000;
    const today = new Date(); today.setHours(0, 0, 0, 0);
    const t0 = today.getTime();
    if (ts >= t0) return '今天';
    if (ts >= t0 - day) return '昨天';
    if (ts >= t0 - 7 * day) return '7 天内';
    return '更早';
  }

  function renderSidebar() {
    historyEl.innerHTML = '';
    const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
    if (!sorted.length) {
      historyEl.innerHTML = '<div class="history-empty">暂无历史对话<br>开启一段新的探索吧</div>';
      return;
    }
    let lastGroup = '';
    sorted.forEach((conv) => {
      const g = groupLabel(conv.updatedAt);
      if (g !== lastGroup) {
        const label = document.createElement('div');
        label.className = 'history-group-label';
        label.textContent = g;
        historyEl.appendChild(label);
        lastGroup = g;
      }
      const item = document.createElement('button');
      item.className = 'history-item' + (conv.id === currentId ? ' active' : '');
      item.title = conv.title;
      const title = document.createElement('span');
      title.className = 'history-item-title';
      title.textContent = conv.title;
      const del = document.createElement('span');
      del.className = 'history-item-delete';
      del.innerHTML = ICONS.trash;
      del.title = '删除对话';
      del.addEventListener('click', (e) => {
        e.stopPropagation();
        askDeleteConversation(conv.id);
      });
      item.append(title, del);
      item.addEventListener('click', () => openConversation(conv.id));
      historyEl.appendChild(item);
    });
  }

  // ---------- 消息 DOM ----------
  function segmentsToText(msg) {
    return (msg.segments || [])
      .filter((s) => s.kind === 'text' && s.text)
      .map((s) => s.text)
      .join('\n\n');
  }

  function buildSegmentNode(seg) {
    const node = document.createElement('div');
    if (seg.kind === 'thinking') {
      node.className = 'think-card';
      node.innerHTML = `
        <button class="think-header"></button>
        <div class="think-body"></div>`;
      node.querySelector('.think-header').addEventListener('click', () => {
        seg.collapsed = !seg.collapsed;
        node.classList.toggle('collapsed', seg.collapsed);
        save();
      });
    } else if (seg.kind === 'search') {
      node.className = 'search-card';
      node.innerHTML = `
        <button class="search-header"></button>
        <div class="search-sources"></div>`;
      node.querySelector('.search-header').addEventListener('click', () => {
        if (seg.status === 'running') return;
        seg.open = !seg.open;
        node.classList.toggle('open', seg.open);
        save();
      });
    } else {
      node.className = 'md-body';
    }
    return node;
  }

  function updateSegmentNode(node, seg, msg, isLive) {
    if (seg.kind === 'thinking') {
      node.classList.toggle('collapsed', !!seg.collapsed);
      const header = node.querySelector('.think-header');
      const thinkingNow = isLive && !seg.secs;
      const label = thinkingNow
        ? '<span class="think-pulse"></span><span>正在思考…</span>'
        : `<span>已深度思考${seg.secs ? `（用时 ${seg.secs} 秒）` : ''}</span>`;
      header.innerHTML = label + ICONS.chevron;
      node.querySelector('.think-body').textContent = seg.text;
    } else if (seg.kind === 'search') {
      const header = node.querySelector('.search-header');
      const sourcesWrap = node.querySelector('.search-sources');
      const isSearch = seg.name === 'tavily_search';
      const label = TOOL_LABELS[seg.name] || seg.name || '工具调用';
      node.classList.toggle('running', seg.status === 'running');
      node.classList.toggle('open', !!seg.open && seg.status !== 'running');
      if (seg.status === 'running') {
        const q = seg.query ? `「${escapeHtml(seg.query)}」` : '';
        header.innerHTML = `<span class="spinner"></span><span class="search-label">正在${label}${q}…</span>`;
        sourcesWrap.innerHTML = '';
      } else {
        const n = (seg.sources || []).length;
        const summary = isSearch
          ? (n ? `已搜索到 ${n} 个网页` : '未找到相关网页')
          : `已完成${label}`;
        header.innerHTML = `${isSearch ? ICONS.globe : ICONS.wrench}<span class="search-label">${summary}${seg.query ? ` · ${escapeHtml(seg.query)}` : ''}</span>${n ? ICONS.chevron : ''}`;
        sourcesWrap.innerHTML = (seg.sources || []).map((s) => `
          <a class="search-source" href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(s.url)}">
            ${ICONS.link}
            <span class="search-source-title">${escapeHtml(s.title)}</span>
            <span class="search-source-domain">${escapeHtml(s.domain)}</span>
          </a>`).join('');
      }
    } else {
      node.innerHTML = renderMarkdown(seg.text || '');
      decorateMarkdown(node);
      const isLastText = seg === [...msg.segments].reverse().find((s) => s.kind === 'text');
      if (isLive && isLastText) {
        const cursor = document.createElement('span');
        cursor.className = 'stream-cursor';
        (node.lastElementChild || node).appendChild(cursor);
      }
    }
  }

  function buildMessageEl(msg, conv) {
    const root = document.createElement('div');
    if (msg.role === 'user') {
      root.className = 'msg msg-user';
      const bubble = document.createElement('div');
      bubble.className = 'msg-user-bubble';
      bubble.textContent = msg.content;
      root.appendChild(bubble);
    } else {
      root.className = 'msg msg-assistant';
      root.innerHTML = '<div class="segments"></div><div class="tail"></div>';
    }
    msgNodes.set(msg, root);
    return root;
  }

  function renderAssistantTail(root, msg, conv, isLive) {
    const tail = root.querySelector('.tail');
    tail.innerHTML = '';
    if (msg.error) {
      const card = document.createElement('div');
      card.className = 'error-card';
      card.innerHTML = `<div>请求出错：${escapeHtml(msg.error)}</div>`;
      const retry = document.createElement('button');
      retry.className = 'error-retry';
      retry.innerHTML = ICONS.refresh + '<span>重试</span>';
      retry.addEventListener('click', () => regenerate(msg));
      card.appendChild(retry);
      tail.appendChild(card);
    }
    if (msg.stopped) {
      const tag = document.createElement('div');
      tag.className = 'stopped-tag';
      tag.textContent = '已停止生成';
      tail.appendChild(tag);
    }
    if (isLive) return;

    const hasContent = segmentsToText(msg) || msg.error || msg.stopped;
    if (!hasContent) return;

    const actions = document.createElement('div');
    actions.className = 'msg-actions' + (msg.error || msg.stopped ? ' always-show' : '');
    const text = segmentsToText(msg);
    if (text) {
      const copyBtn = document.createElement('button');
      copyBtn.className = 'msg-action-btn';
      copyBtn.title = '复制';
      copyBtn.innerHTML = ICONS.copy;
      copyBtn.addEventListener('click', () => { copyText(text); toast('已复制到剪贴板'); });
      actions.appendChild(copyBtn);
    }
    const isLast = conv && conv.messages[conv.messages.length - 1] === msg;
    if (isLast) {
      const regen = document.createElement('button');
      regen.className = 'msg-action-btn';
      regen.title = '重新生成';
      regen.innerHTML = ICONS.refresh;
      regen.addEventListener('click', () => regenerate(msg));
      actions.appendChild(regen);
    }
    const meta = document.createElement('span');
    meta.className = 'msg-meta';
    meta.textContent = modelInfo(msg.model).name + (msg.fallbackFrom ? '（代答）' : '') + (msg.search ? ' · 联网' : '');
    actions.appendChild(meta);
    tail.appendChild(actions);
  }

  function updateAssistantEl(root, msg, conv, isLive) {
    const segWrap = root.querySelector('.segments');
    // 模型代答提示卡：所选模型平台未开放时，说明实际由哪个模型代答
    let fb = root.querySelector('.fallback-card');
    if (msg.fallbackFrom) {
      if (!fb) {
        fb = document.createElement('div');
        fb.className = 'fallback-card';
        segWrap.before(fb);
      }
      if (fb.dataset.from !== msg.fallbackFrom || fb.dataset.to !== msg.model) {
        fb.dataset.from = msg.fallbackFrom;
        fb.dataset.to = msg.model;
        fb.innerHTML = `${ICONS.info}<div class="fallback-text"><strong>${escapeHtml(modelInfo(msg.fallbackFrom).name)}</strong> 暂未在平台开放调用，本次回答已自动改用 <strong>${escapeHtml(modelInfo(msg.model).name)}</strong> 代答。</div>`;
        const switchBtn = document.createElement('button');
        switchBtn.className = 'fallback-switch';
        switchBtn.textContent = '切换模型';
        switchBtn.addEventListener('click', (e) => { e.stopPropagation(); openModelMenu(); });
        fb.appendChild(switchBtn);
      }
    } else if (fb) {
      fb.remove();
    }
    msg.segments.forEach((seg) => {
      let node = segNodes.get(seg);
      if (!node) {
        node = buildSegmentNode(seg);
        segNodes.set(seg, node);
      }
      if (node.parentElement !== segWrap) segWrap.appendChild(node);
      updateSegmentNode(node, seg, msg, isLive);
    });
    let dots = root.querySelector('.waiting-dots');
    if (isLive && !msg.segments.length) {
      if (!dots) {
        dots = document.createElement('div');
        dots.className = 'waiting-dots';
        dots.innerHTML = '<span></span><span></span><span></span>';
        segWrap.appendChild(dots);
      }
    } else if (dots) {
      dots.remove();
    }
    renderAssistantTail(root, msg, conv, isLive);
  }

  function renderConversation() {
    const conv = currentConv();
    const hasMsgs = !!(conv && conv.messages.length);
    welcomeEl.hidden = hasMsgs;
    messagesEl.hidden = !hasMsgs;
    chatScrollEl.classList.toggle('empty', !hasMsgs);
    messagesEl.innerHTML = '';
    if (hasMsgs) {
      conv.messages.forEach((msg) => {
        const root = buildMessageEl(msg, conv);
        messagesEl.appendChild(root);
        if (msg.role === 'assistant') {
          updateAssistantEl(root, msg, conv, streaming && msg === liveMsg);
        }
      });
    }
    document.title = conv ? `${conv.title} - DeepSeek` : 'DeepSeek - 探索未至之境';
    requestAnimationFrame(() => { chatScrollEl.scrollTop = chatScrollEl.scrollHeight; });
    updateScrollBottomBtn();
  }

  // ---------- 滚动 ----------
  function nearBottom() {
    return chatScrollEl.scrollHeight - chatScrollEl.scrollTop - chatScrollEl.clientHeight < 90;
  }

  function autoScroll() {
    if (nearBottom()) chatScrollEl.scrollTop = chatScrollEl.scrollHeight;
  }

  function updateScrollBottomBtn() {
    const conv = currentConv();
    scrollBottomBtn.hidden = !conv || !conv.messages.length || nearBottom();
  }

  chatScrollEl.addEventListener('scroll', updateScrollBottomBtn);
  scrollBottomBtn.addEventListener('click', () => {
    chatScrollEl.scrollTo({ top: chatScrollEl.scrollHeight, behavior: 'smooth' });
  });

  // ---------- 流式渲染调度 ----------
  function scheduleRender(msg) {
    pendingRender.add(msg);
    if (rafId) return;
    rafId = requestAnimationFrame(() => {
      rafId = 0;
      const conv = currentConv();
      pendingRender.forEach((m) => {
        const root = msgNodes.get(m);
        if (root && root.isConnected) updateAssistantEl(root, m, conv, m === liveMsg && streaming);
      });
      pendingRender.clear();
      autoScroll();
      updateScrollBottomBtn();
    });
  }

  // ---------- SSE 事件处理 ----------
  function handleEvent(evt, msg) {
    switch (evt.type) {
      case 'thinking': {
        if (!msg._thinkStart) msg._thinkStart = Date.now();
        const last = msg.segments[msg.segments.length - 1];
        if (last && last.kind === 'thinking') last.text += evt.text;
        else msg.segments.push({ kind: 'thinking', text: evt.text, collapsed: false });
        break;
      }
      case 'delta': {
        msg.segments.forEach((s) => {
          if (s.kind === 'thinking' && !s.secs) {
            s.secs = Math.max(1, Math.round((Date.now() - (msg._thinkStart || Date.now())) / 1000));
            s.collapsed = true;
          }
        });
        const last = msg.segments[msg.segments.length - 1];
        if (last && last.kind === 'text') last.text += evt.text;
        else msg.segments.push({ kind: 'text', text: evt.text });
        break;
      }
      case 'tool_start': {
        msg.segments.push({
          kind: 'search', id: evt.id, name: evt.name,
          query: evt.query || '', status: 'running', sources: [], open: false,
        });
        break;
      }
      case 'tool_end': {
        const seg = msg.segments.find((s) => s.kind === 'search' && s.id === evt.id);
        if (seg) {
          seg.status = 'done';
          seg.sources = evt.sources || [];
        } else {
          msg.segments.push({
            kind: 'search', id: evt.id, name: evt.name, query: '',
            status: 'done', sources: evt.sources || [], open: false,
          });
        }
        break;
      }
      case 'notice':
        toast(evt.message);
        break;
      case 'model_fallback':
        // 所选模型平台未开放：后端已透明改用默认模型，前端记录代答关系
        msg.fallbackFrom = evt.from;
        msg.model = evt.to;
        markAvailability(evt.from, false);
        markAvailability(evt.to, true);
        break;
      case 'error':
        msg.error = evt.message || '未知错误';
        break;
      case 'done':
        if (evt.model) msg.model = evt.model;
        if (evt.fallback && evt.requested_model) msg.fallbackFrom = evt.requested_model;
        break;
    }
    scheduleRender(msg);
  }

  // ---------- 模型可用性 ----------
  function markAvailability(id, ok) {
    const m = config.models.find((x) => x.id === id);
    if (m && m.available !== ok) {
      m.available = ok;
      buildModelMenu();
      updateModelUI();
    }
  }

  async function refreshAvailability() {
    try {
      const resp = await fetch('/api/config');
      if (!resp.ok) return;
      const cfg = await resp.json();
      let changed = false;
      (cfg.models || []).forEach((m) => {
        const local = config.models.find((x) => x.id === m.id);
        if (local && local.available !== m.available) {
          local.available = m.available;
          changed = true;
        }
      });
      if (changed) { buildModelMenu(); updateModelUI(); }
    } catch { /* 网络异常时保持现状 */ }
  }

  function finalizeMessage(msg) {
    msg.segments.forEach((s) => {
      if (s.kind === 'thinking' && !s.secs) {
        s.secs = Math.max(1, Math.round((Date.now() - (msg._thinkStart || Date.now())) / 1000));
        s.collapsed = true;
      }
      if (s.kind === 'search' && s.status === 'running') s.status = 'done';
    });
    delete msg._thinkStart;
  }

  // ---------- 发送 / 流式请求 ----------
  function ensureConversation() {
    let conv = currentConv();
    if (!conv) {
      conv = { id: uid(), title: '新对话', createdAt: Date.now(), updatedAt: Date.now(), messages: [] };
      conversations.unshift(conv);
      currentId = conv.id;
    }
    return conv;
  }

  function buildHistory(conv, uptoMsg) {
    const out = [];
    for (const m of conv.messages) {
      if (m === uptoMsg) break;
      const content = m.role === 'user' ? m.content : segmentsToText(m);
      if (content && content.trim()) out.push({ role: m.role, content });
    }
    return out;
  }

  async function runStream(conv, msg) {
    streaming = true;
    liveMsg = msg;
    liveConv = conv;
    abortCtrl = new AbortController();
    setStreamingUI(true);
    const root = msgNodes.get(msg);
    if (root) updateAssistantEl(root, msg, conv, true);
    autoScroll();

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: buildHistory(conv, msg),
          model: msg.model,
          search: !!msg.search,
        }),
        signal: abortCtrl.signal,
      });
      if (!resp.ok || !resp.body) throw new Error(`服务返回 HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const rawEvent = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          for (const line of rawEvent.split('\n')) {
            if (line.startsWith('data: ')) {
              try { handleEvent(JSON.parse(line.slice(6)), msg); } catch { /* 忽略坏帧 */ }
            }
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') msg.stopped = true;
      else msg.error = err.message || '网络请求失败，请确认后端服务已启动';
    } finally {
      finalizeMessage(msg);
      streaming = false;
      liveMsg = null;
      liveConv = null;
      abortCtrl = null;
      setStreamingUI(false);
      conv.updatedAt = Date.now();
      save();
      renderSidebar();
      const root2 = msgNodes.get(msg);
      if (root2 && root2.isConnected) updateAssistantEl(root2, msg, currentConv(), false);
      autoScroll();
      updateScrollBottomBtn();
      inputEl.focus();
      refreshAvailability();
    }
  }

  function send() {
    const text = inputEl.value.trim();
    if (!text || streaming) return;
    const conv = ensureConversation();
    conv.messages.push({ role: 'user', id: uid(), content: text, model: settings.model, search: settings.search });
    if (conv.title === '新对话') conv.title = text.replace(/\s+/g, ' ').slice(0, 30);
    conv.updatedAt = Date.now();
    const msg = {
      role: 'assistant', id: uid(), segments: [],
      model: settings.model, search: settings.search, ts: Date.now(),
    };
    conv.messages.push(msg);
    inputEl.value = '';
    autoResize();
    updateSendBtn();
    save();
    renderSidebar();
    renderConversation();
    runStream(conv, msg);
  }

  function regenerate(msg) {
    if (streaming) return;
    const conv = currentConv();
    if (!conv) return;
    const idx = conv.messages.indexOf(msg);
    if (idx < 0) return;
    conv.messages.splice(idx);
    const fresh = {
      role: 'assistant', id: uid(), segments: [],
      model: msg.model || settings.model, search: msg.search ?? settings.search, ts: Date.now(),
    };
    conv.messages.push(fresh);
    save();
    renderConversation();
    runStream(conv, fresh);
  }

  function stopStream() {
    if (abortCtrl) abortCtrl.abort();
  }

  function setStreamingUI(on) {
    // SVG 元素没有 hidden 属性反射，必须用 toggleAttribute 操作内容属性
    sendBtn.querySelector('.icon-send').toggleAttribute('hidden', on);
    sendBtn.querySelector('.icon-stop').toggleAttribute('hidden', !on);
    sendBtn.title = on ? '停止生成' : '发送';
    sendBtn.setAttribute('aria-label', on ? '停止生成' : '发送');
    updateSendBtn();
  }

  // ---------- 会话操作 ----------
  function openConversation(id) {
    if (streaming && liveConv && liveConv.id !== id) stopStream();
    currentId = id;
    save();
    renderSidebar();
    renderConversation();
    closeMobileSidebar();
    inputEl.focus();
  }

  function newChat() {
    if (streaming) stopStream();
    currentId = null;
    save();
    renderSidebar();
    renderConversation();
    closeMobileSidebar();
    inputEl.focus();
  }

  let deleteTargetId = null;
  function askDeleteConversation(id) {
    deleteTargetId = id;
    modalOverlay.hidden = false;
    $('#modalCancel').focus();
  }

  function confirmDelete() {
    const id = deleteTargetId;
    modalOverlay.hidden = true;
    deleteTargetId = null;
    if (!id) return;
    if (streaming && liveConv && liveConv.id === id) stopStream();
    conversations = conversations.filter((c) => c.id !== id);
    if (currentId === id) currentId = null;
    save();
    renderSidebar();
    renderConversation();
    toast('对话已删除');
  }

  $('#modalCancel').addEventListener('click', () => { modalOverlay.hidden = true; deleteTargetId = null; });
  $('#modalConfirm').addEventListener('click', confirmDelete);
  modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) { modalOverlay.hidden = true; deleteTargetId = null; }
  });

  // ---------- 模型菜单 ----------
  function buildModelMenu() {
    modelMenu.innerHTML = '<div class="model-menu-title">选择模型</div>' + config.models.map((m) => `
      <button class="model-item${m.id === settings.model ? ' selected' : ''}" role="option"
        aria-selected="${m.id === settings.model}" data-id="${escapeHtml(m.id)}">
        <div class="model-item-main">
          <div class="model-item-name"><span>${escapeHtml(m.name)}</span><span class="model-vendor">${escapeHtml(m.vendor)}</span>${m.available === false ? '<span class="model-badge">未开放 · 自动代答</span>' : ''}</div>
          <div class="model-item-desc">${escapeHtml(m.desc)}</div>
        </div>
        <span class="model-item-check">${ICONS.check}</span>
      </button>`).join('');
    modelMenu.querySelectorAll('.model-item').forEach((item) => {
      item.addEventListener('click', () => {
        settings.model = item.dataset.id;
        save();
        updateModelUI();
        closeModelMenu();
        const info = modelInfo(settings.model);
        if (info.available === false) {
          toast(`${info.name} 暂未在平台开放，发送时将自动使用 ${modelInfo(config.default_model).name} 代答`);
        } else {
          toast(`已切换到 ${info.name}${streaming ? '，下一条消息生效' : ''}`);
        }
      });
    });
  }

  function updateModelUI() {
    modelBtnName.textContent = modelInfo(settings.model).name;
    modelMenu.querySelectorAll('.model-item').forEach((item) => {
      const sel = item.dataset.id === settings.model;
      item.classList.toggle('selected', sel);
      item.setAttribute('aria-selected', String(sel));
    });
  }

  function openModelMenu() {
    modelMenu.hidden = false;
    const rect = modelBtn.getBoundingClientRect();
    const width = modelMenu.offsetWidth;
    let left = Math.min(rect.left, window.innerWidth - width - 12);
    left = Math.max(12, left);
    modelMenu.style.left = `${left}px`;
    modelMenu.style.top = `${rect.bottom + 6}px`;
    modelBtn.setAttribute('aria-expanded', 'true');
  }

  function closeModelMenu() {
    modelMenu.hidden = true;
    modelBtn.setAttribute('aria-expanded', 'false');
  }

  modelBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    modelMenu.hidden ? openModelMenu() : closeModelMenu();
  });

  document.addEventListener('click', (e) => {
    if (!modelMenu.hidden && !modelMenu.contains(e.target) && !modelBtn.contains(e.target)) closeModelMenu();
  });

  // ---------- 搜索开关 ----------
  function updateSearchUI() {
    searchToggle.classList.toggle('active', settings.search);
    searchToggle.setAttribute('aria-pressed', String(settings.search));
    if (!config.search_available) {
      searchToggle.disabled = true;
      searchToggle.title = '未配置 TAVILY_API_KEY，联网搜索不可用';
    }
  }

  searchToggle.addEventListener('click', () => {
    settings.search = !settings.search;
    save();
    updateSearchUI();
    toast(settings.search ? '联网搜索已开启' : '联网搜索已关闭');
  });

  // ---------- 输入区 ----------
  function autoResize() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
  }

  function updateSendBtn() {
    sendBtn.disabled = streaming ? false : !inputEl.value.trim();
  }

  inputEl.addEventListener('input', () => { autoResize(); updateSendBtn(); });
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      if (!streaming) send();
    }
  });

  sendBtn.addEventListener('click', () => { streaming ? stopStream() : send(); });

  // ---------- 侧边栏开合 ----------
  const isMobile = () => window.matchMedia('(max-width: 768px)').matches;

  function openMobileSidebar() {
    sidebarEl.classList.add('open');
    scrimEl.hidden = false;
    requestAnimationFrame(() => scrimEl.classList.add('show'));
  }

  function closeMobileSidebar() {
    sidebarEl.classList.remove('open');
    scrimEl.classList.remove('show');
    setTimeout(() => { scrimEl.hidden = true; }, 250);
  }

  $('#sidebarCollapseBtn').addEventListener('click', () => {
    if (isMobile()) closeMobileSidebar();
    else appEl.classList.add('sidebar-collapsed');
  });
  $('#sidebarOpenBtn').addEventListener('click', () => {
    if (isMobile()) openMobileSidebar();
    else appEl.classList.remove('sidebar-collapsed');
  });
  scrimEl.addEventListener('click', closeMobileSidebar);

  $('#newChatBtn').addEventListener('click', newChat);
  $('#topNewChatBtn').addEventListener('click', newChat);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (!modalOverlay.hidden) { modalOverlay.hidden = true; deleteTargetId = null; }
      else if (!modelMenu.hidden) closeModelMenu();
      else if (sidebarEl.classList.contains('open')) closeMobileSidebar();
    }
  });

  window.addEventListener('resize', () => {
    if (!modelMenu.hidden) closeModelMenu();
    if (!isMobile()) { sidebarEl.classList.remove('open'); scrimEl.hidden = true; scrimEl.classList.remove('show'); }
  });

  // ---------- 初始化 ----------
  async function init() {
    loadState();
    // 刷新后恢复上次打开的会话
    if (settings.lastConvId && conversations.some((c) => c.id === settings.lastConvId)) {
      currentId = settings.lastConvId;
    }
    renderSidebar();
    renderConversation();
    updateSearchUI();
    updateModelUI();
    autoResize();
    updateSendBtn();

    try {
      const resp = await fetch('/api/config');
      if (resp.ok) config = await resp.json();
    } catch {
      toast('无法连接后端配置接口，使用默认模型列表');
    }
    if (!config.models.some((m) => m.id === settings.model)) {
      settings.model = config.default_model || config.models[0].id;
    }
    if (!config.search_available && settings.search) settings.search = false;
    save();
    buildModelMenu();
    updateModelUI();
    updateSearchUI();
    if (!isMobile()) inputEl.focus();
  }

  init();
})();