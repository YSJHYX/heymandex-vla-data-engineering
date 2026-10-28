// 使用 Node 自带测试器；不安装依赖，不连接 AutoDL。
'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const api = require('./autodl-gpu-watch.user.js');
const makeStorage = () => {
  const values = new Map();
  return {getItem: k => values.get(k) ?? null, setItem: (k, v) => values.set(k, v), removeItem: k => values.delete(k)};
};

test('型号忽略空格和大小写，5090 不吞 5090 D', () => {
  assert.equal(api.modelMatches('RTX 5090 32GB', 'rtx5090'), true);
  assert.equal(api.modelMatches('RTX 5090 D 32GB', 'RTX 5090'), false);
  assert.equal(api.modelMatches('RTX 5090D', 'RTX 5090'), false);
  assert.equal(api.modelMatches('RTX 5090 D 32GB', 'RTX 5090 D'), true);
  assert.equal(api.modelMatches('RTX 5090', ''), false);
});

test('按内容解析变动列，取第一笔现价；不把原价当现价', () => {
  const row = api.parseCells(['', '99984bf43', '756机', 'RTX 5090 32GB', '1 / 8', 'CPU: 25核', '¥2.88/时\n¥3.03/时']);
  assert.deepEqual([row.id, row.model, row.idle, row.total, row.price], ['99984bf43', 'RTX 5090 32GB', 1, 8, 2.88]);
  const shifted = api.parseCells(['', '主机一', '机房', '额外列', 'RTX 5090', '2 ／ 8', '¥ 3.1 / 时']);
  assert.deepEqual([shifted.model, shifted.idle, shifted.price], ['RTX 5090', 2, 3.1]);
  assert.equal(api.parseCells(['', '主机一', 'RTX 5090', '1 / 8', '未知价格']).price, null);
  assert.ok(api.parseCells(['', '主机一', 'RTX 5090', '9 / 8']).reason);
});

test('表头与表体分表，数据表无需包含主机ID；排除隐藏和固定列副本', () => {
  const radio = {};
  const row = (values, hidden = false, fixed = false) => ({
    offsetParent: hidden ? null : {}, closest: () => fixed ? {} : null,
    cells: values.map((v, i) => ({textContent: v, querySelector: () => i === 0 ? radio : null}))
  });
  const values = ['', '99984bf43', '756机', 'RTX 5090 32GB', '1 / 8', '¥2.88/时'];
  const good = row(values);
  const doc = {querySelectorAll: () => [row(['主机ID', 'GPU', '闲卡']), good, row(values, true), row(values, false, true)]};
  assert.equal(api.hostRows(doc).length, 1);
  assert.equal(api.hostRows(doc)[0].row, good);
});

test('余量递归按型号 key 读取，仅取数值 idle/free，同型号取最大；不以列表非空判断', () => {
  const raw = {data: [{gpu: {'RTX 5090': {idle: 0, total: 2544}}},
    {'RTX 5090': {counts: {free_gpu: 3}, total: 2544}}, {'RTX 5090 D': {idle: 8}}]};
  assert.equal(api.extractAvailability(raw, ['RTX 5090']).get('rtx5090'), 3);
  assert.equal(api.extractAvailability({data: [{gpu_idle_num: 99}]}, ['RTX 5090']).size, 0);
  assert.equal(api.extractAvailability({'RTX 5090': {idle: '3', total: 8}}, ['RTX 5090']).size, 0);
});

test('地区解析支持 query 和 hash、JSON 与逗号列表；不存在时返回未知', () => {
  assert.deepEqual(api.regionsFromURL('https://www.autodl.com/?region_sign_list=%5B%22test-a%22%5D'), ['test-a']);
  assert.deepEqual(api.regionsFromURL('https://www.autodl.com/#/clone?region_sign_list=a,b'), ['a', 'b']);
  assert.equal(api.regionsFromURL('https://www.autodl.com/?region=猜测值'), null);
  assert.equal(api.regionArray([]), null);
});

test('默认只通知，检查间隔不能降到 4 秒以下', () => {
  assert.equal(api.sanitizeSettings().autoSubmit, false);
  assert.equal(api.sanitizeSettings({pollMs: 10}).pollMs, 4000);
  assert.equal(api.sanitizeSettings({autoSubmit: 'false'}).autoSubmit, false);
  assert.equal(api.sanitizeSettings({pollMs: 'Infinity'}).pollMs, 4000);
  assert.equal(api.sanitizeSettings(null).autoSubmit, false);
});

function context(overrides = {}) {
  const clicks = [];
  const storage = makeStorage();
  const ctx = {lock: api.lockStore(storage), allowed: () => true, report: () => {},
    select: async () => true, wait: async () => {}, revalidate: () => true,
    createButton: () => ({click: () => clicks.push('创建')}),
    confirmButton: () => ({click: () => clicks.push('确认')}), submitted: () => clicks.push('通知'), ...overrides};
  return {ctx, clicks, storage};
}
const candidate = {id: '主机一', model: 'RTX 5090'};

test('选中失败时绝不点创建，锁继续保留', async () => {
  const {ctx, clicks} = context({select: async () => false});
  assert.equal(await api.submitCandidate(ctx, candidate), false);
  assert.deepEqual(clicks, []);
  assert.ok(ctx.lock.read());
});

test('创建前已经持久锁定；重复执行、刷新后新实例仍不能再次提交', async () => {
  const {ctx, clicks, storage} = context();
  ctx.createButton = () => ({click: () => {assert.ok(ctx.lock.read()); clicks.push('创建');}});
  assert.equal(await api.submitCandidate(ctx, candidate), true);
  assert.equal(await api.submitCandidate(ctx, candidate), false);
  assert.deepEqual(clicks, ['创建', '确认', '通知']);
  ctx.lock = api.lockStore(storage);
  assert.equal(await api.submitCandidate(ctx, candidate), false);
});

test('创建点击抛错、确认框未知、筛选或存储失效均不自动重试', async () => {
  for (const overrides of [
    {createButton: () => ({click() {throw new Error('模拟网络/页面错误');}})},
    {confirmButton: () => null}, {revalidate: () => false}, {createButton: () => null}
  ]) {
    const {ctx} = context(overrides);
    await api.submitCandidate(ctx, candidate);
    assert.ok(ctx.lock.read());
    assert.equal(await api.submitCandidate(ctx, candidate), false);
  }
  const {ctx, clicks} = context({lock: {reserve() {throw new Error('存储不可写');}}});
  await assert.rejects(api.submitCandidate(ctx, candidate));
  assert.deepEqual(clicks, []);
});

test('fetch 和 XHR 原语义保留，捕获 Authorization 与原始筛选体', async () => {
  const requests = [], stocks = [], statuses = [];
  class FakeXHR {
    open(method, url) {this.method = method; this.url = url;}
    setRequestHeader() {}
    addEventListener(name, cb) {this.done = cb;}
    send(body) {this.body = body; this.status = 200; this.responseType = 'json';
      this.response = {'RTX 5090': {idle: 0}}; this.done(); return '原始send返回';}
  }
  const native = async () => new Response(JSON.stringify({'RTX 5090': {idle: 2}}), {status: 200});
  const win = {fetch: native, XMLHttpRequest: FakeXHR, Request, Headers,
    location: {href: 'https://www.autodl.com/clone', origin: 'https://www.autodl.com'}};
  const hooks = api.installHooks(win, m => requests.push({...m}), (s) => statuses.push(s), raw => stocks.push(raw));
  const body = JSON.stringify({region_sign_list: ['test-region'], gpu_type_name: 'RTX 5090'});
  const response = await win.fetch('/api/v1/user/machine/list', {method: 'POST', headers: {Authorization: 'test-token-not-a-secret'}, body});
  assert.equal(response.status, 200);
  assert.equal(requests[0].body, body);
  assert.equal(requests[0].headers.authorization, 'test-token-not-a-secret');
  const xhr = new win.XMLHttpRequest(); xhr.open('POST', '/api/v1/machine/region/gpu_type');
  xhr.setRequestHeader('Authorization', 'test-token-not-a-secret');
  assert.equal(xhr.send(body), '原始send返回');
  assert.equal(stocks[0]['RTX 5090'].idle, 0);
  assert.ok(statuses.includes(200));
  hooks.restore(); assert.equal(win.fetch, native);
});
