const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, 'shutdown_control.js'), 'utf8');

function fixture(options = {}) {
  let now = 1_000_000, serial = 0;
  const actions = [], receipts = new Map(), timers = new Map(), handlers = new Map();
  class ClockDate extends Date {
    constructor(...args) { super(...(args.length ? args : [now])); }
    static now() { return now; }
  }
  const setTimer = (callback, delay) => {
    const id = ++serial;
    timers.set(id, { callback, at: now + delay });
    return id;
  };
  const app = {
    isReady: () => !!options.ready,
    once(event, callback) { handlers.set(event, callback); },
    quit() {
      actions.push('app.quit');
      for (const event of ['before-quit', 'will-quit']) {
        const callback = handlers.get(event); handlers.delete(event); callback?.();
      }
    }
  };
  let attached = true;
  const observer = { restored: true, registrationIdentifier: 'owned-registration', debuggerAttached: true, webContentsId: 7 };
  const rendererWindow = {
    __REACT_DEVTOOLS_GLOBAL_HOOK__: { __modlyCleanupOwned: !options.foreignHook },
    __modlyRoots: new Set(), __modlyCleanupStore: {},
    __modlyCleanup: { stop() { actions.push('observer.stop'); return { stopped: true }; } }
  };
  const rendererContext = vm.createContext({ window: rendererWindow });
  const debuggerApi = {
    isAttached: () => attached,
    async sendCommand(method, params) {
      actions.push(method);
      if (method === 'Page.removeScriptToEvaluateOnNewDocument') assert.equal(params.identifier, 'owned-registration');
      if (method === 'Runtime.releaseObjectGroup') assert.equal(params.objectGroup, 'modly-cleanup-qualification');
      return {};
    },
    detach() { actions.push('debugger.detach'); attached = false; }
  };
  const wc = {
    id: options.wrongWebContents ? 8 : 7, debugger: debuggerApi,
    async executeJavaScript(expression, userGesture) {
      assert.equal(userGesture, false);
      actions.push('executeJavaScript');
      if (options.hang) return new Promise(() => {});
      return vm.runInContext(expression, rendererContext);
    }
  };
  const context = vm.createContext({
    Date: ClockDate, setTimeout: setTimer, clearTimeout: id => timers.delete(id),
    __modlyObserver: observer,
    require(name) {
      if (name === 'electron') return { app, BrowserWindow: { getAllWindows: () => options.noWindow ? [] : [{ webContents: wc }] } };
      if (name === 'path') return path;
      if (name === 'fs') return { writeFileSync(destination, bytes, flags) {
        assert.equal(flags.flag, 'wx');
        const filename = path.basename(destination);
        assert.equal(receipts.has(filename), false, 'exclusive receipt');
        receipts.set(filename, JSON.parse(bytes)); actions.push('save:' + filename);
      } };
      throw Error('Unexpected module: ' + name);
    }
  });
  const config = { privateRoot: '/owned-private', startedEpoch: 1000, shutdownEpoch: 1220 };
  const install = overrides => vm.runInContext('(' + source + ')(' + JSON.stringify({ ...config, ...overrides }) + ')', context);
  const flush = async () => { for (let i = 0; i < 32; i++) await Promise.resolve(); };
  const advanceTo = async target => {
    for (;;) {
      const due = [...timers].filter(([, row]) => row.at <= target).sort((a, b) => a[1].at - b[1].at)[0];
      if (!due) break;
      now = due[1].at; timers.delete(due[0]); due[1].callback(); await flush();
    }
    now = target; await flush();
  };
  return { actions, receipts, timers, install, advanceTo, context, rendererWindow, observer };
}

test('arms the 220-second reserve before readiness and restores owned observers before normal quit', async () => {
  const f = fixture(), installed = f.install();
  assert.equal(installed.requested, false); assert.equal(installed.restored, false);
  assert.equal([...f.timers.values()][0].at, 1_220_000);
  await f.advanceTo(1_219_999); assert.equal(f.receipts.size, 0);
  await f.advanceTo(1_220_000);
  assert.equal(f.receipts.get('shutdown-request.json').elapsedSeconds, 220);
  assert.equal(f.receipts.get('shutdown-request.json').supervisorForce, false);
  assert.equal(f.receipts.get('observer-restoration.json').restored, true);
  assert.equal(f.actions.includes('app.quit'), false);
  await f.advanceTo(1_220_050);
  assert.equal(f.receipts.get('app-before-quit.json').restored, true);
  assert.equal(f.receipts.get('app-will-quit.json').requested, true);
  for (const action of ['observer.stop', 'Page.removeScriptToEvaluateOnNewDocument', 'Runtime.releaseObjectGroup', 'debugger.detach', 'save:observer-restoration.json'])
    assert.ok(f.actions.indexOf(action) < f.actions.indexOf('app.quit'), action + ' precedes app.quit');
  assert.equal(Object.hasOwn(f.rendererWindow, '__modlyCleanup'), false);
  assert.equal(Object.hasOwn(f.rendererWindow, '__REACT_DEVTOOLS_GLOBAL_HOOK__'), false);
});

test('manual finish clears reserve timer and repeated finish does not duplicate receipts or quit', async () => {
  const f = fixture(); f.install();
  const first = await f.context.__modlyCleanupShutdown.finish('manual-complete');
  assert.equal(first.restoration.restored, true);
  const again = await f.context.__modlyCleanupShutdown.finish('duplicate');
  assert.equal(again.requested, true);
  await f.advanceTo(1_221_000);
  assert.equal(f.actions.filter(action => action === 'app.quit').length, 1);
  assert.equal(f.receipts.get('shutdown-request.json').reason, 'manual-complete');
});

test('restoration failure remains explicit while normal application quit is still requested', async () => {
  const f = fixture({ foreignHook: true }); f.install();
  const result = await f.context.__modlyCleanupShutdown.finish('failed-restoration');
  assert.equal(result.restoration.restored, false);
  assert.match(result.restoration.error, /Foreign React observer/);
  assert.equal(f.receipts.has('observer-restoration.json'), false);
  assert.equal(f.actions.includes('debugger.detach'), false);
  await f.advanceTo(1_000_050);
  assert.equal(f.actions.includes('app.quit'), true);
  assert.equal(f.receipts.get('app-before-quit.json').restored, false);
});

test('hanging renderer restoration consumes at most one second before normal quit scheduling', async () => {
  const f = fixture({ hang: true }); f.install();
  await f.advanceTo(1_220_999);
  assert.equal(f.actions.includes('app.quit'), false);
  await f.advanceTo(1_221_000);
  assert.match(f.receipts.get('observer-restoration-failure.json').error, /1second restoration bound/);
  await f.advanceTo(1_221_050);
  assert.equal(f.actions.includes('app.quit'), true);
  assert.ok(f.receipts.get('app-before-quit.json').elapsedSeconds < 240);
});

test('rejects an overlong reserve or already-ready application before installing controls', () => {
  assert.throws(() => fixture().install({ shutdownEpoch: 1222 }), /Shutdown reserve/);
  assert.throws(() => fixture({ ready: true }).install(), /before readiness/);
});
