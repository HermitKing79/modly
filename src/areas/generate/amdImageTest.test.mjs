import assert from 'node:assert/strict'
import test from 'node:test'
import { analyzeAlpha, imageByteLength, createImageTestSession, validateCpuResult, localValidationUrl, safePreviewDimensions } from './amdImageTest.ts'
import { readLocalImage } from '../workflows/nodes/imageUtils.ts'

test('preliminary alpha uses strict >204 and a three pixel foreground extent', () => {
  const p = new Uint8ClampedArray(4 * 5 * 5)
  for (let y = 1; y < 4; y++) for (let x = 1; x < 4; x++) p[(y * 5 + x) * 4 + 3] = 205
  assert.deepEqual(analyzeAlpha(p, 5, 5), { transparent: true, foregroundExtent: [3, 3], eligible: true })
  p.fill(204)
  assert.equal(analyzeAlpha(p, 5, 5).eligible, false)
  assert.throws(() => analyzeAlpha(p, 6, 5), /pixel/)
})
test('opaque JPEG pixels cannot pass the preliminary cutout check', () => {
  assert.equal(analyzeAlpha(new Uint8ClampedArray(4 * 9).fill(255), 3, 3).eligible, false)
})
test('input byte sizes are exact without converting or re-encoding bytes', () => {
  assert.equal(imageByteLength('AQIDBA=='), 4)
  assert.equal(imageByteLength('AQID'), 3)
  assert.throws(() => imageByteLength('corrupt'), /base64/)
})
test('unrelated or falsely complete CPU responses are rejected', () => {
  assert.throws(() => validateCpuResult({ status: 'valid', eligible: true }), /validation/)
  assert.throws(() => validateCpuResult({ schema: 'modly.amd.image-test.v1', status: 'valid', eligible: true, preprocessing_complete: true }), /validation/)
})
test('original CPU admission stays distinct from preprocessing and generation', () => {
  const r = { schema: 'modly.amd.image-test.v1', status: 'valid', eligible: true, validation: 'original_cpu_input_contract', preprocessing_complete: false, generation_qualified: false, metadata: {}, reasons: [] }
  assert.equal(validateCpuResult(r), r)
  assert.throws(() => validateCpuResult({ ...r, validation: 'not_run' }), /validation/)
})
test('private image validation never uploads to a remote or credentialed URL', () => {
  assert.equal(localValidationUrl('http://127.0.0.1:8765'), 'http://127.0.0.1:8765/amd/image-test')
  for (const url of ['https://example.com', 'http://127.0.0.1.example.com', 'file:///C:/data', 'http://user:pass@localhost:8765', 'http://localhost:8765/other']) {
    assert.throws(() => localValidationUrl(url), /local/)
  }
})
test('browser decoding requires original bounded single-frame CPU metadata', () => {
  assert.deepEqual(safePreviewDimensions({ metadata: { width: 16, height: 16, frames: 1 } }), [16, 16])
  for (const metadata of [{}, { width: 100000, height: 100000, frames: 1 }, { width: 16, height: 16, frames: 2 }, { width: -1, height: 16, frames: 1 }]) {
    assert.equal(safePreviewDimensions({ metadata }), null)
  }
})
test('shared workflow reader retains original bytes and supports an explicit preview cap', async () => {
  const fs = { readFileBase64: async () => 'AQIDBA==' }
  const result = await readLocalImage('C:\\local\\photo.jpg', fs)
  assert.equal(result.name, 'photo.jpg'); assert.equal(result.bytes, 4)
  assert.equal(result.preview, 'data:image/jpeg;base64,AQIDBA==')
  await assert.rejects(readLocalImage('photo.jpg', fs, 3), /preview/)
})
test('changing images cancels pending validation and ignores its late success', async () => {
  const pending = []; const states = []; const signals = []
  const session = createImageTestSession(async (path) => ({ path, name: path, preview: path, bytes: 4, base64: 'AQIDBA==' }), async (_, signal) => {
    signals.push(signal)
    return new Promise((resolve) => pending.push(resolve))
  }, (state) => states.push(state))
  const first = session.select('first.png'); await new Promise(r => setImmediate(r))
  const second = session.select('second.jpg'); await new Promise(r => setImmediate(r))
  assert.equal(signals[0].aborted, true)
  pending[1]({ status: 'invalid' }); await second
  pending[0]({ status: 'valid' }); await first
  assert.equal(states.at(-1).image.name, 'second.jpg')
  assert.equal(states.at(-1).result.status, 'invalid')
  session.dispose()
})
test('changing images ignores a late read failure and disposal prevents state updates', async () => {
  let rejectRead; const states = []
  const session = createImageTestSession((path) => path === 'old' ? new Promise((_, reject) => { rejectRead = reject }) : Promise.resolve({ name: path }), async () => ({ status: 'invalid' }), s => states.push(s))
  const old = session.select('old'); await session.select('new')
  rejectRead(new Error('old failure')); await old
  assert.equal(states.at(-1).image.name, 'new')
  session.dispose(); const count = states.length; await session.select('other'); assert.equal(states.length, count)
})
