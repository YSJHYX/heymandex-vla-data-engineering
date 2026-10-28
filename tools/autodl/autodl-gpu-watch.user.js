// ==UserScript==
// @name         AutoDL 克隆实例余量监控
// @namespace    local.autodl.gpu-watch
// @version      1.0.0
// @description  低频监控型号余量和主机闲卡，默认只通知；自动提交前持久锁定
// @match        https://www.autodl.com/*
// @grant        none
// @sandbox      raw
// @run-at       document-start
// @noframes
// ==/UserScript==

(() => {
  'use strict';
  const SETTINGS_KEY = 'autodl.gpu-watch.settings.v1';
  const LOCK_KEY = 'autodl.gpu-watch.submit-lock.v1';
  const RELOAD_KEY = 'autodl.gpu-watch.last-reload.v1';
  const NOTIFY_KEY = 'autodl.gpu-watch.notifications.v1';
  const LIST_PATH = '/api/v1/user/machine/list';
  const STOCK_PATH = '/api/v1/machine/region/gpu_type';
  // 没有实页证据，不能杜撰地区代码。可将实页核实过的完整列表填在这里。
  const ALL_REGION_SIGNS = Object.freeze([]);
  const DEFAULTS = Object.freeze({autoSubmit: false, autoReload: false,
    notifyOn: true, mailNotify: false, targetModels: [], minIdle: 1,
    maxPrice: 0, pollMs: 4000});
  const normalize = value => String(value ?? '').replace(/\s+/g, '').toLowerCase();
  const text = node => (node?.innerText ?? node?.textContent ?? '').trim();
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const cleanModel = value => String(value).replace(/\s*[（(]\s*\d+\s*[\/／]\s*\d+\s*[)）]\s*$/, '').trim();
  const visible = node => !!node && node.offsetParent !== null;

  function modelMatches(actual, target) {
    const a = normalize(actual), t = normalize(target);
    if (!t) return false;
    let start = 0, at;
    while ((at = a.indexOf(t, start)) !== -1) {
      // 去空格后，5090 D 的 D 紧跟 5090；不能把它当作 5090。
      if (!/[a-z]/i.test(a.charAt(at + t.length))) return true;
      start = at + 1;
    }
    return false;
  }

  function parseCells(cells) {
    const values = Array.from(cells, cell => typeof cell === 'string' ? cell.trim() : text(cell));
    const index = values.findIndex((value, i) => i > 1 && /(\d+)\s*[\/／]\s*(\d+)/.test(value));
    if (index < 0) return {cells: values, reason: '没有可识别的空闲格'};
    const count = values[index].match(/(\d+)\s*[\/／]\s*(\d+)/);
    const prices = values.map(value => value.match(/¥\s*([\d.]+)\s*\/\s*时/));
    const price = prices.find(Boolean);
    const idle = Number(count[1]), total = Number(count[2]);
    return {cells: values, model: values[index - 1], idle, total,
      price: price && Number.isFinite(Number(price[1])) ? Number(price[1]) : null,
      // 主机 ID 没有可靠表头契约；整行内容作核对依据，第二格仅作去重标识。
      id: values[1], reason: idle > total ? '空闲数大于总数，放弃该行' : null};
  }

  function hostRows(doc) {
    // Element UI 的表头、表体分表；不在数据 table 内找“主机ID”。
    return Array.from(doc.querySelectorAll('tr')).filter(row =>
      visible(row) && row.cells.length >= 4 &&
      !row.closest('.el-table__fixed, .el-table__fixed-right') &&
      row.cells[0].querySelector('input[type="radio"], input[type="checkbox"]')
    ).map(row => ({...parseCells(row.cells), row,
      radio: row.cells[0].querySelector('input[type="radio"], input[type="checkbox"]')}));
  }

  function regionArray(value) {
    if (!Array.isArray(value)) return null;
    const result = [...new Set(value.filter(v => typeof v === 'string' || typeof v === 'number')
      .map(String).map(v => v.trim()).filter(Boolean))];
    return result.length ? result : null;
  }

  function regionsFromURL(href) {
    // 只认确切字段名，不猜 region、地区名称或数值的含义。兼容 hash 路由。
    const url = new URL(href);
    const sources = [url.searchParams, new URLSearchParams(url.hash.split('?')[1] || '')];
    for (const params of sources) {
      const values = [...params.getAll('region_sign_list'), ...params.getAll('region_sign_list[]')];
      for (const value of values) {
        try { const parsed = regionArray(JSON.parse(value)); if (parsed) return parsed; } catch { /* 普通字符串 */ }
      }
      const parsed = regionArray(values.flatMap(value => value.split(',')));
      if (parsed) return parsed;
    }
    return null;
  }

  function extractAvailability(raw, names) {
    const found = new Map();
    const visitNumbers = (node, out) => {
      if (!node || typeof node !== 'object') return;
      for (const [key, value] of Object.entries(node)) {
        if (/idle|free/i.test(key) && typeof value === 'number' && Number.isFinite(value) && value >= 0) out.push(value);
        else if (value && typeof value === 'object') visitNumbers(value, out);
      }
    };
    const visit = node => {
      if (!node || typeof node !== 'object') return;
      for (const [key, value] of Object.entries(node)) {
        // 必须以型号名为 key；不把列表非空、gpu_idle_num 请求参数当作闲卡。
        const name = names.find(name => normalize(name) === normalize(key));
        if (name && value && typeof value === 'object') {
          const counts = []; visitNumbers(value, counts);
          if (counts.length) found.set(normalize(name), Math.max(found.get(normalize(name)) ?? -1, ...counts));
        }
        if (value && typeof value === 'object') visit(value);
      }
    };
    visit(raw);
    return found;
  }

  function sanitizeSettings(input = {}) {
    if (!input || typeof input !== 'object') input = {};
    const out = {...DEFAULTS};
    for (const key of ['autoSubmit', 'autoReload', 'notifyOn', 'mailNotify']) out[key] = input[key] === true;
    out.targetModels = Array.isArray(input.targetModels) ? [...new Set(input.targetModels.filter(v => typeof v === 'string' && v.trim()).map(cleanModel))] : [];
    const finite = (value, fallback) => Number.isFinite(Number(value)) ? Number(value) : fallback;
    out.minIdle = Math.max(1, Math.floor(finite(input.minIdle, 1) || 1));
    out.maxPrice = Math.max(0, finite(input.maxPrice, 0));
    // 最短 4 秒；这是 DOM 检查间隔，余量接口另有 12 秒间隔。
    out.pollMs = Math.min(60000, Math.max(4000, finite(input.pollMs, 4000) || 4000));
    return out;
  }

  function lockStore(storage) {
    return {
      read() {
        const value = storage.getItem(LOCK_KEY);
        if (!value) return null;
        try { return JSON.parse(value); } catch { return {phase: '锁记录损坏，需人工核实'}; }
      },
      reserve(detail) {
        if (this.read()) return false;
        const value = JSON.stringify({...detail, time: Date.now(), phase: '提交前锁定'});
        storage.setItem(LOCK_KEY, value);
        if (storage.getItem(LOCK_KEY) !== value) throw new Error('提交锁写入失败');
        return true;
      },
      phase(phase) {
        storage.setItem(LOCK_KEY, JSON.stringify({...this.read(), phase}));
      },
      clear() { storage.removeItem(LOCK_KEY); }
    };
  }

  async function submitCandidate(ctx, candidate) {
    // 外层必须持有浏览器跨标签独占锁；先持久锁定，再做任何提交动作。
    if (!ctx.allowed() || !ctx.lock.reserve({host: candidate.id, model: candidate.model})) return false;
    ctx.report('已锁定，正在选择主机');
    try {
      if (!await ctx.select(candidate)) throw new Error('主机选中失败，放弃提交');
      await ctx.wait(800);
      if (!ctx.allowed() || !ctx.revalidate(candidate)) throw new Error('页面、型号、价格或选中状态已变，放弃提交');
      // 即使 click 抛错或网络超时，也保留锁，绝不自动重试。
      const button = ctx.createButton();
      if (!button) throw new Error('未找到唯一可用的“创建并开机”按钮');
      ctx.lock.phase('即将点击创建，结果待核实');
      button.click();
      ctx.report('已点击创建，锁定中；结果待核实');
      await ctx.wait(1500);
      if (ctx.allowed()) {
        const confirmation = ctx.confirmButton();
        if (confirmation) {
          ctx.lock.phase('即将点击确认，结果待核实');
          confirmation.click();
          ctx.report('已点击确认，锁定中；请核实实例结果');
        } else ctx.report('没有可安全识别的确认框，保留锁；请检查页面');
      }
      ctx.lock.phase('已触发创建流程，结果待核实');
      ctx.submitted(candidate);
      return true;
    } catch (error) {
      ctx.report(`${error.message}；保留锁，请人工核实后解除`);
      return false;
    }
  }

  function installHooks(win, onRequest, onStatus, onStock) {
    const nativeFetch = win.fetch;
    const proto = win.XMLHttpRequest.prototype;
    const original = {open: proto.open, setRequestHeader: proto.setRequestHeader, send: proto.send};
    const xhrMeta = new WeakMap();
    const safe = fn => { try { fn(); } catch { /* 旁路观测不能破坏页面请求 */ } };
    const sameOrigin = url => { try { return new URL(url, win.location.href).origin === win.location.origin; } catch { return false; } };
    const inspectResponse = async (meta, response) => {
      if (!sameOrigin(meta.url)) return;
      safe(() => onStatus(response.status, meta));
      if (new URL(meta.url, win.location.href).pathname === STOCK_PATH) {
        try { onStock(await response.clone().json(), meta, response.status); } catch { /* 非 JSON 由诊断识别 */ }
      }
    };
    function wrappedFetch(input, init) {
      let meta;
      safe(() => {
        const isRequest = input instanceof win.Request;
        meta = {url: new URL(isRequest ? input.url : String(input), win.location.href).href,
          method: String(init?.method ?? (isRequest ? input.method : 'GET')).toUpperCase(),
          headers: Object.fromEntries(new win.Headers(init?.headers ?? (isRequest ? input.headers : undefined))),
          body: typeof init?.body === 'string' ? init.body : null};
        if (sameOrigin(meta.url)) {
          onRequest(meta);
          if (isRequest && !Object.hasOwn(init || {}, 'body') && /machine\/list$/.test(new URL(meta.url).pathname)) {
            input.clone().text().then(body => safe(() => onRequest({...meta, body}))).catch(() => {});
          }
        }
      });
      const promise = nativeFetch.apply(this, arguments);
      if (meta) promise.then(response => inspectResponse(meta, response)).catch(() => {});
      return promise;
    }
    function open(method, url) {
      const result = original.open.apply(this, arguments);
      safe(() => xhrMeta.set(this, {url: new URL(String(url), win.location.href).href,
        method: String(method).toUpperCase(), headers: {}, body: null}));
      return result;
    }
    function setRequestHeader(name, value) {
      const result = original.setRequestHeader.apply(this, arguments);
      safe(() => {
        const meta = xhrMeta.get(this), key = String(name).toLowerCase();
        if (meta) meta.headers[key] = meta.headers[key] ? `${meta.headers[key]}, ${value}` : String(value);
      });
      return result;
    }
    function send(body) {
      const meta = xhrMeta.get(this);
      if (meta && sameOrigin(meta.url)) safe(() => {
        meta.body = typeof body === 'string' ? body : null;
        onRequest(meta);
        this.addEventListener('loadend', () => safe(() => {
          onStatus(this.status, meta);
          if (new URL(meta.url).pathname === STOCK_PATH) {
            const raw = this.responseType === 'json' ? this.response : JSON.parse(this.responseText);
            onStock(raw, meta, this.status);
          }
        }), {once: true});
      });
      return original.send.apply(this, arguments);
    }
    win.fetch = wrappedFetch; proto.open = open; proto.setRequestHeader = setRequestHeader; proto.send = send;
    return {nativeFetch: nativeFetch.bind(win), restore() {
      if (win.fetch === wrappedFetch) win.fetch = nativeFetch;
      if (proto.open === open) proto.open = original.open;
      if (proto.setRequestHeader === setRequestHeader) proto.setRequestHeader = original.setRequestHeader;
      if (proto.send === send) proto.send = original.send;
    }};
  }

  // 在 Node 中仅导出可测试的逻辑，不安装浏览器钩子。
  if (typeof module === 'object' && module.exports) {
    module.exports = {modelMatches, parseCells, hostRows, extractAvailability, regionArray,
      regionsFromURL, sanitizeSettings, lockStore, submitCandidate, installHooks, LOCK_KEY};
    return;
  }
  if (window.top !== window.self || window.__autodlGpuWatchInstalled) return;
  window.__autodlGpuWatchInstalled = true;

  let config, storageOK = true;
  try { config = sanitizeSettings(JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}')); }
  catch { config = sanitizeSettings(); storageOK = false; }
  const store = lockStore(localStorage);
  const state = {auth: null, template: null, raw: null, stocks: new Map(), stockAt: 0,
    stockRegions: '', lastQuery: 0, cooldownUntil: 0, queryBusy: false, filterAt: 0,
    positiveSince: 0, staleAttemptAt: 0, staleFingerprint: '', sequence: 0,
    notificationAt: new Map(), lastMessage: '', lastWarn: new Map(), ui: null,
    active: false, leader: false, stopping: false, refreshPending: false};
  const log = (...args) => console.log('[抢卡]', ...args);
  const warn = (key, message) => {
    if (Date.now() - (state.lastWarn.get(key) || 0) > 60000) {
      state.lastWarn.set(key, Date.now()); log(message);
    }
  };
  const report = message => {
    state.lastMessage = message;
    if (state.ui) state.ui.status.textContent = message;
    log(message);
  };
  const parseBody = meta => { try { return JSON.parse(meta?.body || 'null'); } catch { return null; } };
  function regions() {
    const fromBody = regionArray(parseBody(state.template)?.region_sign_list);
    return fromBody || regionsFromURL(location.href) || regionArray(ALL_REGION_SIGNS);
  }
  const regionKey = list => JSON.stringify(list || null);
  function onRequest(meta) {
    // Authorization 只保存在内存中；不落盘、不导出、不发给邮件桥。
    const auth = Object.entries(meta.headers).find(([key]) => key.toLowerCase() === 'authorization');
    if (auth) state.auth = auth[1];
    if (new URL(meta.url).pathname === LIST_PATH && meta.method === 'POST') {
      state.template = {...meta, headers: {...meta.headers}};
      state.sequence++;
    }
  }
  function onStatus(status, meta) {
    if (![LIST_PATH, STOCK_PATH].includes(new URL(meta.url).pathname)) return;
    if (status === 429 || status === 503) {
      state.cooldownUntil = Math.max(state.cooldownUntil, Date.now() + 90000);
      report(`接口返回 ${status}，脚本冷却 90 秒`);
    }
    if (status === 401 || status === 403) {
      state.auth = null;
      warn('登录失效', '认证不可用，等待页面自己的新请求，不执行登录');
    }
  }
  function onStock(raw, meta, status) {
    state.raw = raw;
    if (status < 200 || status >= 300) return;
    const requestRegions = regionArray(parseBody(meta)?.region_sign_list);
    if (!requestRegions || regionKey(requestRegions) !== regionKey(regions())) return;
    state.stocks = extractAvailability(raw, modelOptions().map(item => item.name));
    state.stockAt = Date.now(); state.stockRegions = regionKey(requestRegions);
    state.lastQuery = Date.now();
    if (!state.stocks.size) warn('余量结构', '余量响应未找到以页面型号为 key 的 idle/free 数值；请导出诊断');
  }
  // 必须同步安装；不能等 DOM、面板或异步代码完成后再挂钩子。
  const hooks = installHooks(window, onRequest, onStatus, onStock);

  function clonePage() {
    const breadcrumb = document.querySelector('.el-breadcrumb');
    return /克隆实例/.test(text(breadcrumb || document.body));
  }
  function modelOptions() {
    const groups = Array.from(document.querySelectorAll('.el-form-item, .el-row, section'))
      .filter(node => !node.closest('#autodl-gpu-watch') && /GPU\s*型号/i.test(text(node)) &&
        node.querySelector('.el-checkbox'));
    // 最小的确切 GPU 型号容器；若无法识别，留诊断，不扫描其它业务复选框。
    groups.sort((a, b) => text(a).length - text(b).length);
    const group = groups[0];
    if (!group) return [];
    return Array.from(group.querySelectorAll('.el-checkbox')).filter(visible).map(label => {
      const input = label.querySelector('input[type="checkbox"]');
      const raw = text(label.querySelector('.el-checkbox__label') || label);
      const count = raw.match(/[（(]\s*(\d+)\s*[\/／]\s*(\d+)\s*[)）]/);
      return {name: cleanModel(raw), input, label, displayedIdle: count ? Number(count[1]) : null};
    }).filter(item => item.input && item.name);
  }
  function idleFor(name) {
    // 过期或地区变化后必须回到未知；不能用旧地区余量触发刷新。
    if (Date.now() - state.stockAt > 30000 || state.stockRegions !== regionKey(regions())) return null;
    return state.stocks.get(normalize(name)) ?? null;
  }
  function matchingRow(row) {
    return !row.reason && row.idle >= config.minIdle && row.price !== null &&
      (config.maxPrice === 0 || row.price <= config.maxPrice) &&
      config.targetModels.some(name => modelMatches(row.model, name));
  }
  async function queryStock() {
    if (document.hidden || state.queryBusy || Date.now() < state.cooldownUntil ||
      Date.now() - state.lastQuery < 12000 || !config.targetModels.length) return;
    const list = regions();
    if (!list) { warn('地区', '地区未知：请在页面选择地区，让页面产生主机列表请求'); return; }
    if (!state.auth) { warn('认证', '尚未捕获页面 Authorization：请操作一次地区或型号筛选'); return; }
    state.lastQuery = Date.now(); state.queryBusy = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    const meta = {url: new URL(STOCK_PATH, location.origin).href, method: 'POST',
      body: JSON.stringify({region_sign_list: list})};
    try {
      const response = await hooks.nativeFetch(meta.url, {method: 'POST', credentials: 'same-origin',
        headers: {'Content-Type': 'application/json', Authorization: state.auth},
        body: meta.body, signal: controller.signal});
      onStatus(response.status, meta);
      state.raw = await response.json();
      onStock(state.raw, meta, response.status);
    } catch (error) { warn('余量查询', `余量查询未完成（${error.name}），保留未知状态`); }
    finally { clearTimeout(timeout); state.queryBusy = false; }
  }

  async function setModelFilter(wanted) {
    const options = modelOptions();
    const wantedNames = new Set(wanted.map(normalize));
    if (!wantedNames.size || !options.length) return false;
    for (const option of options) {
      if (wantedNames.has(normalize(option.name)) && option.input.disabled) {
        warn('型号禁用', `型号 ${option.name} 被页面禁用，不猜测其组件状态`); return false;
      }
    }
    let changed = false;
    // 先勾目标，避免短暂全空导致页面恢复“全部”；再取消其它及“全部”。
    const ordered = [...options.filter(o => wantedNames.has(normalize(o.name))),
      ...options.filter(o => !wantedNames.has(normalize(o.name)))];
    for (const old of ordered) {
      if (!clonePage() || document.hidden || safeLock() || Date.now() < state.cooldownUntil) return false;
      const current = modelOptions().find(o => normalize(o.name) === normalize(old.name));
      if (!current) return false;
      const checked = wantedNames.has(normalize(current.name));
      if (current.input.checked !== checked) {
        if (current.input.disabled) return false;
        current.label.click(); changed = true;
        // 每次重绘后重新定位，不操作失效节点。
        await sleep(80);
      }
    }
    if (changed) { state.filterAt = Date.now(); state.refreshPending = true; }
    return true;
  }
  function filterIsSelected() {
    const options = modelOptions();
    const selected = options.filter(o => o.input.checked).map(o => o.name);
    return selected.length > 0 && selected.every(name => config.targetModels.some(t => normalize(t) === normalize(name)));
  }
  async function gentleRefresh(wanted) {
    if (!clonePage() || document.hidden || safeLock() || !wanted.length || Date.now() - state.filterAt < 15000) return;
    const option = modelOptions().find(o => normalize(o.name) === normalize(wanted[0]));
    if (!option || option.input.disabled || !option.input.checked) return;
    state.filterAt = Date.now(); state.refreshPending = true;
    state.staleAttemptAt = Date.now();
    const name = option.name;
    log('型号余量为正，表格无闲卡：取消再勾回型号，请页面自行重拉');
    option.label.click();
    await sleep(150);
    if (!clonePage() || document.hidden || safeLock() || Date.now() < state.cooldownUntil) return;
    const current = modelOptions().find(o => normalize(o.name) === normalize(name));
    if (current && !current.input.checked && !current.input.disabled) current.label.click();
  }

  function usableButton(button) {
    return visible(button) && !button.disabled && button.getAttribute('aria-disabled') !== 'true' &&
      !button.classList.contains('is-loading') && !button.classList.contains('is-disabled');
  }
  function createButton() {
    const found = Array.from(document.querySelectorAll('button')).filter(button =>
      usableButton(button) && text(button) === '创建并开机' &&
      !button.closest('#autodl-gpu-watch, .el-dialog__wrapper, .el-message-box__wrapper'));
    return found.length === 1 ? found[0] : null;
  }
  function confirmButton() {
    const dialogs = Array.from(document.querySelectorAll('.el-dialog__wrapper, .el-message-box__wrapper'))
      .filter(node => visible(node) && /创建|开机|克隆/.test(text(node)));
    if (dialogs.length !== 1) { warn('确认框', '确认框数量或上下文无法确认，交由人工处理'); return null; }
    const buttons = Array.from(dialogs[0].querySelectorAll('button')).filter(button =>
      usableButton(button) && /^(确定|确认|立即创建|创建)$/.test(text(button)));
    if (buttons.length !== 1) { warn('确认按钮', '确认按钮不唯一或不可用，交由人工处理'); return null; }
    return buttons[0];
  }

  function notify(candidate, submitted = false) {
    const key = candidate.id;
    try {
      const saved = JSON.parse(localStorage.getItem(NOTIFY_KEY) || '{}');
      for (const [host, at] of Object.entries(saved)) {
        if (typeof at === 'number' && Date.now() - at < 300000) state.notificationAt.set(host, at);
      }
    } catch { /* 存储不可读时仍保留内存去重 */ }
    const duplicate = Date.now() - (state.notificationAt.get(key) || 0) < 300000;
    const message = submitted ? `已触发 ${candidate.model} 主机 ${candidate.id} 创建流程；结果待核实，脚本已锁定` :
      `${candidate.model} 主机 ${candidate.id}：空闲 ${candidate.idle}/${candidate.total}，¥${candidate.price}/时`;
    if (!duplicate) {
      state.notificationAt.set(key, Date.now());
      try {
        localStorage.setItem(NOTIFY_KEY, JSON.stringify(Object.fromEntries(
          Array.from(state.notificationAt).filter(([, at]) => Date.now() - at < 300000).slice(-100))));
      } catch { /* 去重写入失败不触发提交 */ }
      log(message);
      if (config.notifyOn && 'Notification' in window && Notification.permission === 'granted') {
        try { new Notification('AutoDL 抢卡', {body: message, tag: `autodl-${key}`}); }
        catch { log('浏览器通知未显示，请查看状态条'); }
      }
    }
    // 不把“点过按钮”宣称为“抢到成功”。仅自动流程触发后发送待核实通知。
    if (submitted && config.mailNotify) {
      hooks.nativeFetch('http://127.0.0.1:8765/send', {method: 'POST', mode: 'no-cors',
        credentials: 'omit', headers: {'Content-Type': 'text/plain'},
        body: JSON.stringify({event: '提交待核实', host: candidate.id, model: candidate.model,
          idle: candidate.idle, price: candidate.price, message})})
        .then(() => log('邮件桥请求已发出；no-cors 无法读取结果，请检查桥日志'))
        .catch(() => log('邮件桥访问失败，请检查本地网络权限、控制台和桥日志'));
    }
  }

  function diagnostic() {
    const rows = hostRows(document);
    console.group('[抢卡] 导出诊断（不含 Authorization 值）');
    console.log('主机行 cells 数组', rows.map(row => row.cells));
    console.log('表格结构', Array.from(document.querySelectorAll('.el-table')).map(table => ({
      table数: table.querySelectorAll('table').length,
      表头: Array.from(table.querySelectorAll('.el-table__header-wrapper th')).map(text),
      主表数据行数: Array.from(table.querySelectorAll('.el-table__body-wrapper tr'))
        .filter(row => !row.closest('.el-table__fixed, .el-table__fixed-right')).length})));
    console.log('接口与地区', {主机列表已捕获: !!state.template, Authorization有无: !!state.auth,
      地区: regions(), 页面型号: modelOptions().map(o => ({型号: o.name, 已选: o.input.checked, 页面余量: o.displayedIdle})),
      余量时间: state.stockAt ? new Date(state.stockAt).toISOString() : null,
      冷却至: state.cooldownUntil, 提交锁: safeLock(), 跨标签独占锁: state.leader});
    console.log('行解析', rows.map(({row, radio, ...value}) => value));
    console.log('余量接口原始响应', state.raw);
    console.groupEnd();
    report('诊断已打印到控制台；不含认证头，请勿公开整个 Network 请求');
  }
  function safeLock() {
    try { return store.read(); } catch { storageOK = false; return {phase: '存储不可用，禁止自动提交'}; }
  }
  function save() {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(config)); }
    catch { storageOK = false; config.autoSubmit = false; report('设置无法保存，已禁止自动提交'); }
  }
  function panel() {
    const root = document.createElement('div'); root.id = 'autodl-gpu-watch';
    root.innerHTML = `<style>
      #autodl-gpu-watch{font:13px/1.5 system-ui;color:#eee;position:relative;z-index:2147483646}
      #autodl-gpu-watch details{position:fixed;right:14px;top:14px;width:288px;background:#182331;padding:12px;border-radius:10px;box-shadow:0 3px 16px #0005}
      #autodl-gpu-watch summary{cursor:pointer;font-weight:bold}
      #autodl-gpu-watch label{display:block;margin:8px 0}
      #autodl-gpu-watch input[type=number]{width:88px;color:#111}
      #autodl-gpu-watch select{width:100%;height:110px;color:#111}
      #autodl-gpu-watch button{margin:4px;padding:4px 7px;color:#111;cursor:pointer}
      #autodl-gpu-watch .status{position:fixed;left:14px;bottom:14px;max-width:70vw;background:#182331;padding:10px;border-radius:8px;white-space:pre-wrap;pointer-events:none}
      #autodl-gpu-watch small{display:block;color:#bdcbdc}
    </style><details open><summary>AutoDL 余量监控</summary>
      <label>目标型号（Ctrl / ⌘ 多选）<select multiple data-key="targetModels"></select></label>
      <label><input type="checkbox" data-key="autoSubmit"> 自动提交（将产生租用费用）</label>
      <label><input type="checkbox" data-key="autoReload"> 余量为正但表格滞后时允许整页刷新</label>
      <label><input type="checkbox" data-key="notifyOn"> 浏览器通知</label>
      <label><input type="checkbox" data-key="mailNotify"> 提交流程后通知本机邮件桥</label>
      <label>最低空闲卡数 <input type="number" min="1" step="1" data-key="minIdle"></label>
      <label>每卡时价上限 <input type="number" min="0" step="0.01" data-key="maxPrice"> 元（0 不限）</label>
      <label>检查间隔 <input type="number" min="4000" step="1000" data-key="pollMs"> 毫秒</label>
      <small>默认只通知。数量、付费方式及源实例由页面当前设置决定。提交前即锁定；超时与失败均不自动重试。</small>
      <button data-action="permission">授权通知</button><button data-action="diagnostic">导出诊断</button>
      <button data-action="replay">重放列表（仅诊断）</button><button data-action="unlock">人工解除提交锁</button>
    </details><div class="status">等待页面型号和主机列表…</div>`;
    document.body.append(root);
    state.ui = {root, status: root.querySelector('.status'), select: root.querySelector('select')};
    for (const input of root.querySelectorAll('[data-key]')) {
      const key = input.dataset.key;
      if (input.type === 'checkbox') input.checked = config[key];
      else if (key !== 'targetModels') input.value = config[key];
      input.addEventListener('change', () => {
        const next = {...config};
        next[key] = key === 'targetModels' ? Array.from(input.selectedOptions, o => o.value) :
          input.type === 'checkbox' ? input.checked : Number(input.value);
        config = sanitizeSettings(next); save();
        state.staleAttemptAt = 0; state.positiveSince = 0;
        if (input.type !== 'checkbox' && key !== 'targetModels') input.value = config[key];
      });
    }
    root.querySelector('[data-action=permission]').onclick = async () => {
      if ('Notification' in window) report(`通知权限：${await Notification.requestPermission()}`);
      else report('此浏览器不支持系统通知；请查看状态条');
    };
    root.querySelector('[data-action=diagnostic]').onclick = diagnostic;
    root.querySelector('[data-action=unlock]').onclick = () => {
      if (!confirm('请先在实例列表和订单中核实本次创建结果。解除后再次启用自动提交可能重复扣费。确定已核实并解除锁？')) return;
      config.autoSubmit = false; save();
      root.querySelector('[data-key=autoSubmit]').checked = false;
      try { store.clear(); report('已人工解除锁；自动提交已关闭'); } catch { report('无法解除锁：存储不可用'); }
    };
    root.querySelector('[data-action=replay]').onclick = replayList;
  }
  function updateOptions() {
    const names = modelOptions().filter(o => o.name !== '全部').map(o => o.name);
    // 捕获响应可能早于 Vue 渲染型号，型号出现后再解析已捕获的同地区响应。
    if (state.raw && Date.now() - state.stockAt <= 30000 && state.stockRegions === regionKey(regions())) {
      state.stocks = extractAvailability(state.raw, names);
    }
    const select = state.ui.select;
    if (JSON.stringify(Array.from(select.options, o => o.value)) !== JSON.stringify(names)) {
      select.replaceChildren(...names.map(name => {
        const option = document.createElement('option'); option.value = name; option.textContent = name;
        option.selected = config.targetModels.includes(name); return option;
      }));
    }
    if (!names.length) warn('型号结构', '无法定位 GPU 型号复选框容器，请导出诊断');
  }
  async function replayList() {
    if (!state.leader || document.hidden || Date.now() < state.cooldownUntil || Date.now() - state.lastQuery < 12000) {
      report('重放暂缓：非运行标签、页面隐藏、请求间隔或冷却限制'); return;
    }
    const meta = state.template;
    if (!meta || !state.auth || meta.method !== 'POST' || !meta.body) { report('尚未捕获完整主机列表请求'); return; }
    state.lastQuery = Date.now();
    const headers = {...meta.headers, authorization: state.auth};
    for (const key of Object.keys(headers)) {
      if (key.toLowerCase() === 'authorization' && key !== 'authorization') delete headers[key];
    }
    try {
      const response = await hooks.nativeFetch(meta.url, {method: meta.method, headers, body: meta.body, credentials: 'same-origin'});
      onStatus(response.status, meta);
      const raw = await response.json();
      // 未知 Vue 组件契约，不把重放响应硬塞进页面。重放仅供诊断。
      console.log('[抢卡] 主机列表重放响应（仅诊断，未改页面）', raw);
      report('主机列表已重放；响应仅供诊断，不据列表非空判断有卡');
    } catch { report('主机列表重放失败，请检查控制台 Network'); }
  }
  function summary(rows, suffix = '') {
    const stock = config.targetModels.map(name => `${name}:${idleFor(name) ?? '?'}`).join(' | ') || '未选目标型号';
    const cool = Date.now() < state.cooldownUntil ? ` | 冷却 ${Math.ceil((state.cooldownUntil - Date.now()) / 1000)} 秒` : '';
    state.ui.status.textContent = `${stock}\n主机行 ${rows.length} | ${config.autoSubmit ? '自动提交开启' : '只通知'}${cool}${suffix ? `\n${suffix}` : ''}`;
  }

  async function tick() {
    if (!clonePage()) { summary([], '当前不是克隆实例页，暂停'); return; }
    updateOptions();
    const lock = safeLock();
    if (lock) { summary(hostRows(document), `已锁定：${lock.phase}。请核实实例，禁止自动重试`); return; }
    if (document.hidden) { summary([], '页面隐藏，暂停查询和操作'); return; }
    if (Date.now() < state.cooldownUntil) { summary(hostRows(document)); return; }
    await queryStock();
    if (!clonePage() || document.hidden || safeLock() || Date.now() < state.cooldownUntil) return;
    const options = modelOptions();
    const wanted = options.filter(o => config.targetModels.includes(o.name) &&
      (idleFor(o.name) === null || idleFor(o.name) >= config.minIdle)).map(o => o.name);
    // 所有目标已知为 0 时，绝不碰页面筛选；只等待下一次余量查询。
    if (wanted.length) await setModelFilter(wanted);
    if (state.refreshPending) {
      state.refreshPending = false;
      summary(hostRows(document), '筛选已变更，等待下一轮页面数据'); return;
    }
    const rows = hostRows(document);
    const candidate = rows.find(matchingRow);
    if (candidate) {
      summary(rows, `发现 ${candidate.id}：${candidate.idle}/${candidate.total}，¥${candidate.price}/时`);
      if (!config.autoSubmit) { notify(candidate); return; }
      if (!storageOK || !state.leader || !navigator.locks || !filterIsSelected()) {
        warn('提交条件', '存储、跨标签独占锁或页面型号勾选未核实，放弃自动提交'); return;
      }
      const initialURL = location.href;
      const initialConfig = JSON.stringify(config);
      const current = () => hostRows(document).find(row => row.id === candidate.id && matchingRow(row));
      await submitCandidate({lock: store, wait: sleep, report,
        allowed: () => clonePage() && !document.hidden && config.autoSubmit &&
          JSON.stringify(config) === initialConfig && location.href === initialURL && Date.now() >= state.cooldownUntil,
        select: async () => {
          const row = current(); if (!row || !row.radio?.isConnected) return false;
          // 按需求只恢复原生 disabled；Vue 若拒绝选中，不改组件内部数据。
          const wasDisabled = row.radio.disabled;
          try { if (wasDisabled) row.radio.disabled = false; row.radio.click(); }
          finally { if (wasDisabled) row.radio.disabled = true; }
          await sleep(80);
          const selected = current(); return !!selected?.radio.checked;
        },
        revalidate: () => { const row = current(); return !!row?.radio.checked && filterIsSelected(); },
        createButton, confirmButton, submitted: row => notify(row, true)
      }, candidate);
      return;
    }
    summary(rows);
    const positive = wanted.filter(name => idleFor(name) > 0);
    const fingerprint = JSON.stringify({positive, regions: regions(), targets: config.targetModels});
    if (!positive.length) { state.positiveSince = 0; state.staleAttemptAt = 0; return; }
    if (fingerprint !== state.staleFingerprint) {
      state.staleFingerprint = fingerprint; state.positiveSince = Date.now(); state.staleAttemptAt = 0;
    }
    // 有闲卡但价格不符时不刷新；只有目标表格本身无闲卡才尝试拉新。
    if (rows.some(row => !row.reason && row.idle >= config.minIdle && positive.some(name => modelMatches(row.model, name)))) return;
    if (!state.staleAttemptAt) {
      await gentleRefresh(positive); return;
    }
    if (config.autoReload && Date.now() - state.staleAttemptAt >= 10000 && !safeLock()) {
      try {
        const previous = Number(localStorage.getItem(RELOAD_KEY) || 0);
        if (Date.now() - previous >= 30000) {
          localStorage.setItem(RELOAD_KEY, String(Date.now()));
          report('已先重拉型号，仍无闲卡；按设置整页刷新'); location.reload(); return;
        }
      } catch { warn('刷新存储', '无法记录刷新时间，禁止整页刷新'); }
    }
    if (Date.now() - state.staleAttemptAt >= 15000) await gentleRefresh(positive);
  }

  async function run() {
    state.leader = true;
    try {
      while (!state.stopping) {
        try { await tick(); }
        catch (error) { report(`本轮放弃（${error.name}），请导出诊断`); }
        await sleep(config.pollMs);
      }
    } finally { state.leader = false; }
  }
  window.addEventListener('pagehide', () => { state.stopping = true; });
  (async () => {
    const deadline = Date.now() + 20000;
    while (Date.now() < deadline) {
      if (document.body && /克隆实例/.test(document.body.innerText)) {
        state.active = true; panel();
        if (!navigator.locks) { report('浏览器不支持跨标签独占锁；本页监控暂停'); return; }
        await navigator.locks.request('autodl.gpu-watch.monitor.v1', {ifAvailable: true}, async lock => {
          if (!lock) { report('其它标签正在运行，本页暂停；关闭其它监控页后再刷新本页'); return; }
          await run();
        });
        return;
      }
      await sleep(250);
    }
    // 20 秒没到克隆页：静默退出并恢复钩子。
    hooks.restore(); state.auth = null; state.template = null;
  })();
})();
