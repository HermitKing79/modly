// CPU-only guard contracts. This never starts Electron or either Python runtime.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import vm from 'node:vm'
import { test } from 'node:test'
import { admitRuntimeProbes, buildPreview, createPlan, guardedSources, renderBootstrap, windowsLauncher } from './build_preview.mjs'

const sourceDir = path.resolve(import.meta.dirname, '../..')
const options = { repo: sourceDir, seedRuntime: path.resolve(sourceDir, '../artifacts/seed/runtime'),
  destination: path.resolve(sourceDir, '../artifacts/new-preview/preview'),
  apiPython: 'C:/existing/api/python.exe', cpuPython: 'D:/existing/cpu/python.exe' }

test('build planning refuses source/seed aliases and an existing destination', () => {
  assert.throws(() => createPlan({ ...options, destination: sourceDir }), /separate/)
  assert.throws(() => createPlan({ ...options, destination: options.seedRuntime }), /separate/)
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'modly-preview-build-'))
  try { assert.throws(() => createPlan({ ...options, destination: root }), /already exists/) }
  finally { fs.rmdirSync(root) }
})

test('main guards pin API, isolate settings, refuse a busy port and bind nonce health', () => {
  const sources = guardedSources(options)
  const bridge = sources.get('python-bridge')
  assert.match(bridge, /preview_api:app/)
  assert.match(bridge, /preview_nonce/)
  assert.match(bridge, /generation_enabled !== false/)
  assert.match(bridge, /createServer/)
  assert.doesNotMatch(bridge, /taskkill|netstat|lsof|execSync/)
  assert.match(bridge, /C:\/existing\/api\/python\.exe/)
  const settings = sources.get('settings-store')
  assert.doesNotMatch(settings, /readFileSync/)
  assert.match(settings, /settings writes are disabled/)
  assert.match(sources.get('python-setup'), /setup\/install is disabled/)
  assert.match(sources.get('index'), /Noncanonical AMD Image Test Preview/)
})

test('build refuses runtime substitution before touching a destination', async () => {
  await assert.rejects(buildPreview(options), /previously qualified/)
  assert.equal(fs.existsSync(options.destination), false)
})

test('Windows launcher has one untimed route and a fixed bounded smoke route through the same launcher', () => {
  const source = windowsLauncher(options.apiPython)
  assert.doesNotMatch(source, /%\*/)
  assert.match(source, /if not "%~2"=="" goto bad_args/)
  const commands = source.split('\r\n').filter(line => line.includes('-B -I -S'))
  assert.equal(commands.length, 2)
  assert.equal(commands[1], commands[0]+' --test-timeout 180')
  assert.doesNotMatch(commands[0], /timeout|inspect/)
})

test('runtime admission rejects API-local Pillow or a changed CPU decoder before building a new preview', () => {
  const api = { components: { Pillow_available: false } }
  const cpu = { version: '3.12.10', components: { Pillow: '12.3.0', webp: true } }
  admitRuntimeProbes(api,cpu)
  assert.throws(() => admitRuntimeProbes({ components: { Pillow_available: true } }, cpu), /API-local Pillow/)
  assert.throws(() => admitRuntimeProbes(api, { ...cpu, version: '3.13.0' }), /capability\/version/)
  assert.throws(() => admitRuntimeProbes(api, { ...cpu, components: { Pillow: '12.2.0', webp: true } }), /capability\/version/)
})

function harness({ attached = true, nonce = '0123456789abcdef0123456789abcdef', attachPath, recordNonce } = {}) {
  const profile = path.resolve('/private-preview/profile')
  const file = attachPath ?? path.join(profile, 'launches/session-1/attached.json')
  const handled = new Map(), events = new Map(), paths = new Map(), calls = [], created = new Set()
  const env = { MODLY_AMD_PREVIEW_NONCE: nonce, MODLY_AMD_PREVIEW_ATTACH_FILE: file,
    PYTHONPATH: 'foreign', ELECTRON_RENDERER_URL: 'https://foreign.invalid', CHROME_LOG_FILE: '/foreign/log', ELECTRON_LOG_FILE: '/foreign/log' }
  const app = { setPath: (key, value) => paths.set(key, value), setName: value => calls.push(['name', value]),
    getPath: key => paths.get(key), disableHardwareAcceleration: () => calls.push(['gpu-disabled']),
    commandLine: { appendSwitch: (...args) => calls.push(['switch', ...args]) },
    whenReady: () => Promise.resolve(), on: (name, fn) => events.set(name, fn),
    exit: code => calls.push(['exit', code]) }
  let beforeRequest
  const electron = { app, ipcMain: { handle: (name, listener) => handled.set(name, listener),
    on: (name, listener) => events.set(name, listener) },
    session: { defaultSession: { webRequest: { onBeforeRequest: (_filter, fn) => { beforeRequest = fn } },
      setPermissionRequestHandler: () => {}, setPermissionCheckHandler: () => {}, on: () => {} } } }
  const fakeFs = { mkdirSync: value => created.add(value), realpathSync: value => value, existsSync: value => value === file ? attached : true,
    readFileSync: () => JSON.stringify({ nonce: recordNonce ?? nonce, attached }), writeFileSync: () => {} }
  let fakeTime = 0
  const context = { require: name => {
    if (name === 'electron') return electron
    if (name === 'node:fs') return fakeFs
    if (name === 'node:path') return path
    if (name === './index.js') { calls.push(['main']); return {} }
    throw Error(`Unexpected require ${name}`)
  }, process: { env, stderr: { write: () => {} }, platform: 'win32' }, console,
    URL, setTimeout: attached ? setTimeout : fn => { fn(); return {} }, clearTimeout,
    Date: attached ? Date : class extends Date { static now() { return fakeTime += 10001 } } }
  vm.runInNewContext(renderBootstrap({ ...options, profile }), context)
  return { handled, events, paths, calls, env, created,
    registerHandle: (...args) => electron.ipcMain.handle(...args),
    request: d => new Promise(resolve => beforeRequest(d, resolve)) }
}

test('bootstrap requires launcher attachment before requiring main', async () => {
  const h = harness({ attachPath: path.resolve('/foreign/attached.json') })
  await new Promise(resolve => setImmediate(resolve))
  assert.ok(h.calls.some(([name, value]) => name === 'exit' && value === 1))
  assert.ok(!h.calls.some(([name]) => name === 'main'))
})

test('storage and hardware flags are applied before waiting for Job attachment; timeout refuses main', async () => {
  const h = harness({ attached: false })
  assert.ok(h.paths.has('userData'))
  assert.ok(h.calls.some(([name]) => name === 'gpu-disabled'))
  await new Promise(resolve => setImmediate(resolve))
  assert.ok(h.calls.some(([name, value]) => name === 'exit' && value === 1))
  assert.ok(!h.calls.some(([name]) => name === 'main'))
})

test('Windows native picker Desktop resolves inside the private USERPROFILE before main or attachment waiting', async () => {
  for (const attached of [true, false]) {
    const h = harness({ attached })
    assert.equal(h.paths.get('desktop'), path.join(h.env.USERPROFILE, 'Desktop'))
    assert.ok(h.created.has(h.paths.get('desktop')))
    await new Promise(resolve => setImmediate(resolve))
  }
})

test('attachment with another launch nonce refuses main', async () => {
  const h = harness({ recordNonce: 'f'.repeat(32) })
  await new Promise(resolve => setImmediate(resolve))
  assert.ok(h.calls.some(([name, value]) => name === 'exit' && value === 1))
  assert.ok(!h.calls.some(([name]) => name === 'main'))
})

test('attached bootstrap isolates storage, disables GPU and applies default-deny IPC and HTTP', async () => {
  const h = harness()
  await new Promise(resolve => setImmediate(resolve))
  assert.ok(h.calls.some(([name]) => name === 'main'))
  for (const key of ['userData', 'sessionData', 'temp', 'logs', 'crashDumps']) {
    assert.ok(h.paths.get(key).includes('private-preview'), key)
  }
  assert.equal(h.env.PYTHONPATH, undefined)
  assert.equal(h.env.ELECTRON_RENDERER_URL, undefined)
  assert.equal(h.env.CUDA_VISIBLE_DEVICES, '')
  assert.equal(h.env.HF_HUB_OFFLINE, '1')
  assert.ok(h.env.CHROME_LOG_FILE.includes('private-preview'))
  assert.ok(h.env.ELECTRON_LOG_FILE.includes('private-preview'))
  for (const channel of ['settings:set', 'setup:run', 'model:download', 'shell:openExternal', 'unknown:new']) {
    // The original listeners must never execute even when a new channel is added.
    h.handled.clear()
    h.registerHandle(channel, () => { throw Error('unsafe listener ran') })
    await assert.rejects(h.handled.get(channel)(), /Preview blocks IPC/)
  }
  h.registerHandle('fs:readFileBase64', () => 'original bounded reader')
  assert.equal(h.handled.get('fs:readFileBase64')(), 'original bounded reader')
  h.registerHandle('extensions:list', () => { throw Error('remote trusted-repo fetch attempted') })
  assert.equal((await h.handled.get('extensions:list')()).length, 0)
  assert.equal((await h.request({ method: 'GET', url: 'http://127.0.0.1:8765/health' })).cancel, false)
  assert.equal((await h.request({ method: 'POST', url: 'http://127.0.0.1:8765/amd/image-test' })).cancel, false)
  for (const [method, url] of [['GET', 'https://example.com/'], ['POST', 'http://127.0.0.1:8765/generate'],
    ['GET', 'http://127.0.0.1:8765/amd/image-test'], ['GET', 'http://localhost:8765/health']]) {
    assert.equal((await h.request({ method, url })).cancel, true)
  }
})
