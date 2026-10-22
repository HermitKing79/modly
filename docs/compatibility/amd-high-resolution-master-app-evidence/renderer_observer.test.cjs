'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const filename = path.join(__dirname, 'renderer_observer.js')

function fixture(options = {}) {
  const events = [], calls = []
  let source, release
  const registered = new Promise((resolve) => { release = resolve })
  function BrowserWindow() {}
  const original = function (...args) { calls.push({ receiver: this, args }); return Promise.resolve('loaded') }
  BrowserWindow.prototype.loadFile = original
  const debug = {
    isAttached: () => !!options.preAttached,
    attach(version) { events.push(['attach', version]); if (options.attachFails) throw Error('attach denied') },
    async sendCommand(command, params) {
      events.push([command])
      if (command === 'Page.addScriptToEvaluateOnNewDocument') {
        source = params.source
        if (options.registerFails) throw Error('registration denied')
        await registered
        return { identifier: 'synthetic-registration' }
      }
      return {}
    },
    detach() { events.push(['detach']) },
  }
  const context = { require(name) { assert.equal(name, 'electron'); return { BrowserWindow } } }
  const installer = vm.runInNewContext('(' + fs.readFileSync(filename, 'utf8') + ')', context)
  return { BrowserWindow, original, debug, context, installer, events, calls, release, source: () => source }
}

test('one-shot wrapper awaits registration and preserves receiver, arguments and return value', async () => {
  const f = fixture(), receipt = f.installer()
  assert.equal(receipt.installed, true); assert.equal(receipt.ready, false)
  const win = new f.BrowserWindow(); win.webContents = { debugger: f.debug }
  const options = { query: { synthetic: 'only' } }
  const pending = win.loadFile('synthetic-index.html', options)
  assert.equal(f.BrowserWindow.prototype.loadFile, f.original)
  assert.equal(f.context.__modlyObserver.restored, true)
  await new Promise(setImmediate)
  assert.equal(f.calls.length, 0); assert.equal(f.context.__modlyObserver.ready, false)
  f.release(); assert.equal(await pending, 'loaded')
  assert.equal(f.calls.length, 1); assert.equal(f.calls[0].receiver, win)
  assert.equal(f.calls[0].args[0], 'synthetic-index.html'); assert.equal(f.calls[0].args[1], options)
  assert.equal(f.context.__modlyObserver.ready, true)
  assert.deepEqual(f.events.map(x => x[0]), ['attach', 'Page.enable', 'Page.addScriptToEvaluateOnNewDocument'])
  assert.throws(() => f.installer(), /already installed/)
})

test('injected production-compatible hook tracks only live roots and releases unmounted roots', async () => {
  const f = fixture(); f.installer()
  const win = new f.BrowserWindow(); win.webContents = { debugger: f.debug }
  f.release(); await win.loadFile('synthetic.html')
  const window = {}; vm.runInNewContext(f.source(), { window })
  const hook = window.__REACT_DEVTOOLS_GLOBAL_HOOK__
  assert.equal(hook.supportsFiber, true); assert.equal(hook.isDisabled, false)
  assert.equal(hook.inject({ rendererPackageName: 'react-dom' }), 1)
  assert.equal(hook.inject({ rendererPackageName: '@react-three/fiber' }), 2)
  const dom = { current: { child: {} } }, r3f = { current: { child: {} } }
  hook.onCommitFiberRoot(1, dom); hook.onCommitFiberRoot(2, r3f); hook.onCommitFiberRoot(2, r3f)
  assert.equal(window.__modlyRoots.size, 2)
  dom.current.child = null; hook.onCommitFiberRoot(1, dom)
  assert.equal(window.__modlyRoots.size, 1)
  hook.onCommitFiberUnmount(2, { tag: 3, stateNode: r3f })
  assert.equal(window.__modlyRoots.size, 0)
  const existing = {}, collision = { __REACT_DEVTOOLS_GLOBAL_HOOK__: existing }
  assert.throws(() => vm.runInNewContext(f.source(), { window: collision }), /observer collision/)
  assert.equal(collision.__REACT_DEVTOOLS_GLOBAL_HOOK__, existing)
  assert.equal(collision.__modlyRoots, undefined)
})

test('registration failure restores loadFile, detaches only owned debugger and never navigates', async () => {
  const f = fixture({ registerFails: true }); f.installer()
  const win = new f.BrowserWindow(); win.webContents = { debugger: f.debug }
  await assert.rejects(win.loadFile('synthetic.html'), /registration denied/)
  assert.equal(f.BrowserWindow.prototype.loadFile, f.original)
  assert.equal(f.calls.length, 0); assert.equal(f.context.__modlyObserver.ready, false)
  assert.equal(f.context.__modlyObserver.restored, true)
  assert.equal(f.context.__modlyObserver.errors.length, 1)
  assert.equal(f.events.filter(x => x[0] === 'detach').length, 1)
})

test('preexisting debugger is not detached and failed attachment never navigates', async () => {
  for (const options of [{ preAttached: true }, { attachFails: true }]) {
    const f = fixture(options); f.installer()
    const win = new f.BrowserWindow(); win.webContents = { debugger: f.debug }
    await assert.rejects(win.loadFile('synthetic.html'))
    assert.equal(f.calls.length, 0); assert.equal(f.context.__modlyObserver.ready, false)
    assert.equal(f.BrowserWindow.prototype.loadFile, f.original)
    assert.equal(f.events.some(x => x[0] === 'detach'), false)
  }
})
