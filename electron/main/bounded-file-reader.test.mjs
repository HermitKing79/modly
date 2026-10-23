import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { mkdtemp, open, rmdir, unlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { resolve } from 'node:path'
import test from 'node:test'

const {
  MAX_SCENE_MANIFEST_BYTES,
  readLocalFileBase64,
} = await import(new URL('./bounded-file-reader.ts', import.meta.url).href)
const { createElectronApi } = await import(new URL('../preload/electron-api.ts', import.meta.url).href)

test('the authoritative readFileBase64 IPC delegates to the bounded reader', () => {
  const handlers = readFileSync(resolve('electron/main/ipc-handlers.ts'), 'utf8')
  assert.match(
    handlers.split('\n').find((line) => line.includes("ipcMain.handle('fs:readFileBase64'")),
    /readLocalFileBase64\(filePath, undefined, maxBytes\)/,
  )
})

async function realFile(t, bytes, name = 'image.png') {
  const directory = await mkdtemp(join(tmpdir(), 'modly-bounded-reader-'))
  const filePath = join(directory, name)
  await writeFile(filePath, bytes)
  t.after(async () => { await unlink(filePath); await rmdir(directory) })
  return filePath
}

test('explicit byte cap reads a real file at the bound and preserves one-argument use', async (t) => {
  const path = await realFile(t, Buffer.from([1, 2, 3, 4]))
  assert.equal(await readLocalFileBase64(path, undefined, 4), 'AQIDBA==')
  assert.equal(await readLocalFileBase64(path), 'AQIDBA==')
})

test('explicit cap rejects a real oversized image with its actual byte count', async (t) => {
  const path = await realFile(t, Buffer.alloc(2097153))
  await assert.rejects(readLocalFileBase64(path, undefined, 2097152), /2097153 bytes/)
})

test('invalid byte caps cannot open or allocate a file', async () => {
  for (const cap of [0, -1, 0.5, NaN, Infinity, 33554433, '2097152', null]) {
    await assert.rejects(readLocalFileBase64('/not-opened/image.png', undefined, cap), /maxBytes/)
  }
})

test('capped reading stays on one descriptor and refuses growth after stat', async (t) => {
  const path = await realFile(t, Buffer.from('a'))
  let opens = 0
  let reads = 0
  let closes = 0
  const dependencies = {
    statFile: async () => { throw new Error('Path stat must not be used') },
    readBytes: async () => { throw new Error('Unbounded read must not be used') },
    openFile: async (filePath, mode) => {
      opens++
      const descriptor = await open(filePath, mode)
      return {
        stat: async () => {
          const before = await descriptor.stat()
          if (reads === 0) await writeFile(path, Buffer.from('abcdefgh'))
          return before
        },
        read: (...args) => { reads++; return descriptor.read(...args) },
        close: async () => { closes++; await descriptor.close() },
      }
    },
  }
  await assert.rejects(readLocalFileBase64(path, dependencies, 4), /8 bytes/)
  assert.equal(opens, 1)
  assert.ok(reads > 0)
  assert.equal(closes, 1)
})

test('capped reading handles short reads and closes after descriptor errors', async () => {
  let position = 0
  let closed = false
  const handle = {
    stat: async () => ({ size: 3, isFile: () => true }),
    read: async (buffer, offset) => {
      if (position === 3) return { bytesRead: 0 }
      buffer[offset] = ++position
      return { bytesRead: 1 }
    },
    close: async () => { closed = true },
  }
  const dependencies = { openFile: async () => handle }
  assert.equal(await readLocalFileBase64('image.png', dependencies, 3), 'AQID')
  assert.equal(closed, true)
  closed = false
  handle.read = async () => { throw new Error('descriptor read failed') }
  await assert.rejects(readLocalFileBase64('image.png', dependencies, 3), /descriptor read failed/)
  assert.equal(closed, true)
})

test('an explicit cap cannot raise the existing scene-manifest limit', async (t) => {
  const path = await realFile(t, Buffer.alloc(1048577), 'scene-manifest.json')
  await assert.rejects(readLocalFileBase64(path, undefined, 33554432), /exceeds the 1 MiB limit/)
})

test('preload forwards the optional cap through IPC and keeps omitted-cap callers working', async (t) => {
  const path = await realFile(t, Buffer.from([1, 2, 3, 4]))
  const api = createElectronApi({
    invoke: async (channel, filePath, maxBytes) => {
      assert.equal(channel, 'fs:readFileBase64')
      return readLocalFileBase64(filePath, undefined, maxBytes)
    },
    send: () => {}, on: () => {}, removeAllListeners: () => {},
  }, { setZoomFactor: () => {} }, { getPathForFile: () => '' })
  await assert.rejects(api.fs.readFileBase64(path, 3), /4 bytes/)
  assert.equal(await api.fs.readFileBase64(path), 'AQIDBA==')
})

test('capped reader refuses directories instead of allocating or reading their contents', async (t) => {
  const path = await realFile(t, Buffer.from('x'))
  await assert.rejects(readLocalFileBase64(resolve(path, '..'), undefined, 4), /not a file/)
})

test('oversized scene manifests are rejected before their bytes are read', async () => {
  let reads = 0
  await assert.rejects(
    readLocalFileBase64('/workspace/room/scene-manifest.json', {
      statFile: async () => ({ size: MAX_SCENE_MANIFEST_BYTES + 1, isFile: () => true }),
      readBytes: async () => { reads++; return new Uint8Array() },
    }),
    /exceeds the 1 MiB limit/,
  )
  assert.equal(reads, 0)
})

test('scene manifests at the bound are read and encoded', async () => {
  const bytes = new TextEncoder().encode('{"schema":"modly.scene-manifest.v1"}')
  const encoded = await readLocalFileBase64('/workspace/room/scene-manifest.json', {
    statFile: async () => ({ size: MAX_SCENE_MANIFEST_BYTES, isFile: () => true }),
    readBytes: async () => bytes,
  })
  assert.equal(encoded, Buffer.from(bytes).toString('base64'))
})

test('a manifest that grows after stat is rejected before base64 transfer', async () => {
  await assert.rejects(
    readLocalFileBase64('/workspace/room/scene-manifest.json', {
      statFile: async () => ({ size: 1, isFile: () => true }),
      readBytes: async () => new Uint8Array(MAX_SCENE_MANIFEST_BYTES + 1),
    }),
    /exceeds the 1 MiB limit/,
  )
})
