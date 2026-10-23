import { useEffect, useRef, useState } from 'react'
import { useAppStore } from '@shared/stores/appStore'
import { readLocalImage, type LocalImage } from '@areas/workflows/nodes/imageUtils'
import { AMD_IMAGE_MAX_BYTES, analyzeAlpha, createImageTestSession, validateCpuResult, localValidationUrl, safePreviewDimensions, type BrowserAlpha, type CpuImageResult, type ImageTestState } from '../amdImageTest'

interface Result { browser?: BrowserAlpha; width?: number; height?: number; cpu?: CpuImageResult; unavailable?: string }
const checkerboard = { backgroundColor: '#27272a', backgroundImage: 'conic-gradient(#3f3f46 25%, transparent 0 50%, #3f3f46 0 75%, transparent 0)', backgroundSize: '20px 20px' }

async function preliminary(image: LocalImage): Promise<Pick<Result, 'browser' | 'width' | 'height'>> {
  const decoded = new Image()
  decoded.src = image.preview
  await decoded.decode()
  const width = decoded.naturalWidth, height = decoded.naturalHeight
  if (width > 4096 || height > 4096 || width * height > 16777216) throw new Error('Image exceeds 4096 pixels per side / 16M pixels. Choose a smaller original cutout.')
  const canvas = document.createElement('canvas')
  canvas.width = width; canvas.height = height
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('Browser pixel inspection is unavailable')
  context.drawImage(decoded, 0, 0)
  const browser = analyzeAlpha(context.getImageData(0, 0, width, height).data, width, height)
  canvas.width = 0; canvas.height = 0
  return { browser, width, height }
}

async function validate(image: LocalImage, signal: AbortSignal, apiUrl: string): Promise<Result> {
  // Never upload or allocate a pixel table for an adapter-oversize input.
  if (image.bytes > AMD_IMAGE_MAX_BYTES) return { unavailable: 'CPU admission not run: image exceeds the 2 MiB generation-adapter limit.' }
  signal.throwIfAborted()
  if (!apiUrl) return { unavailable: 'Local CPU validator unavailable. Start the Modly backend to validate the original input contract. Preview waits for bounded original dimensions.' }
  try {
    const url = localValidationUrl(apiUrl)
    const bytes = Uint8Array.from(atob(image.base64), c => c.charCodeAt(0))
    const form = new FormData()
    form.append('image', new Blob([bytes]), image.name)
    const response = await fetch(url, { method: 'POST', body: form, redirect: 'error', signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]) })
    if (!response.ok) throw new Error(`Local validator returned HTTP ${response.status}`)
    const cpu = validateCpuResult(await response.json())
    const dimensions = safePreviewDimensions(cpu)
    // Pillow must establish bounded original dimensions before any Chromium
    // image decoder or preview receives src. Browser conversion is not admission.
    if (dimensions) {
      const [width, height] = dimensions
      try {
        const browser = await preliminary(image)
        signal.throwIfAborted()
        return { ...browser, cpu }
      } catch (error) {
        if (signal.aborted) throw error
        return { cpu, width, height, unavailable: 'Browser preview could not decode this file. The original CPU input result is retained below.' }
      }
    }
    return { cpu }
  } catch (error) {
    if (signal.aborted) throw error
    return { unavailable: `${error instanceof Error ? error.message : 'Local CPU validation failed'}. Original-pipeline eligibility and safe preview are unverified.` }
  }
}

export default function AmdImageTestPanel({ initialPath }: { initialPath?: string }) {
  const apiUrl = useAppStore(state => state.apiUrl)
  const [state, setState] = useState<ImageTestState<Result>>({ phase: 'empty' })
  const [pickerError, setPickerError] = useState('')
  const session = useRef<ReturnType<typeof createImageTestSession<Result>>>()
  const chosenPath = useRef(initialPath)

  useEffect(() => {
    const current = createImageTestSession(path => readLocalImage(path, window.electron.fs, AMD_IMAGE_MAX_BYTES), (image, signal) => validate(image, signal, apiUrl), setState)
    session.current = current
    if (chosenPath.current) void current.select(chosenPath.current)
    return () => current.dispose()
  }, [apiUrl])
  useEffect(() => {
    if (initialPath && initialPath !== chosenPath.current) { chosenPath.current = initialPath; void session.current?.select(initialPath) }
  }, [initialPath])

  const browse = async () => {
    try {
      const path = await window.electron.fs.selectImage()
      if (!path) return
      setPickerError(''); chosenPath.current = path
      void session.current?.select(path)
    } catch { setPickerError('The existing image picker could not open. Try again when the desktop bridge is available.') }
  }
  const image = state.image, result = state.result, cpu = result?.cpu
  const filename = image?.name ?? state.path?.split(/[\\/]/).pop()
  const dimensions = cpu?.metadata.width && cpu?.metadata.height ? `${cpu.metadata.width} × ${cpu.metadata.height}` : result?.width ? `${result.width} × ${result.height}` : 'Not decoded'
  const busy = state.phase === 'loading' || state.phase === 'validating'
  return (
    <section aria-label="Experimental AMD Image Test" className="flex-1 min-h-0 overflow-y-auto px-4 py-3 text-xs text-zinc-300">
      <h2 className="text-sm font-semibold text-zinc-100">Experimental AMD Image Test</h2>
      <p className="mt-2 text-[11px] text-zinc-400 leading-relaxed">Check your own local cutout before a future AMD run. Images stay on this computer; no background removal or image conversion is performed.</p>
      <div className="mt-3 rounded-lg border border-amber-800/50 bg-amber-950/40 px-3 py-2 text-[11px] text-amber-200">
        AMD image-to-3D generation is not yet enabled. Input readiness does not qualify generation.
      </div>
      <button type="button" onClick={browse} className="mt-3 w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-xs font-medium text-zinc-100 hover:bg-zinc-700">
        {filename ? 'Change test image…' : 'Choose test image…'}
      </button>
      {pickerError && <p role="alert" className="mt-2 text-amber-300">{pickerError}</p>}
      {filename && <p className="mt-3 break-all text-[11px] text-zinc-300">{filename}</p>}
      {image && result?.browser && <div className="mt-2 rounded-lg border border-zinc-700 overflow-hidden" style={checkerboard}>
        <img src={image.preview} alt={`Selected test image: ${image.name}`} className="w-full object-contain" style={{ height: 190 }} />
      </div>}
      {image && <dl className="mt-3 grid grid-cols-2 gap-x-2 gap-y-2 text-[11px]">
        <dt className="text-zinc-500">Decoded dimensions</dt><dd>{dimensions}</dd>
        <dt className="text-zinc-500">Original file size</dt><dd>{image.bytes.toLocaleString()} bytes ({(image.bytes / 1048576).toFixed(3)} MiB)</dd>
        <dt className="text-zinc-500">2 MiB adapter limit</dt><dd className={image.bytes <= AMD_IMAGE_MAX_BYTES ? 'text-emerald-300' : 'text-amber-300'}>{image.bytes <= AMD_IMAGE_MAX_BYTES ? 'Within limit' : 'Too large — use a smaller lossless cutout'}</dd>
      </dl>}
      <div role="status" aria-live="polite" className="mt-4 space-y-3">
        {busy && <p className="text-sky-300">{state.phase === 'loading' ? 'Reading selected image…' : 'Checking pixels and original CPU input contract…'}</p>}
        {state.error && <p role="alert" className="rounded-md bg-red-950/40 px-3 py-2 text-red-300">Unable to validate this image: {state.error}. Choose an uncorrupted PNG/RGBA cutout within the input limits.</p>}
        {result?.browser && <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-3">
          <h3 className="font-medium text-zinc-200">Preliminary browser check</h3>
          <p className="mt-2 text-[11px]">Transparent pixels: {result.browser.transparent ? 'Detected' : 'Not detected'}</p>
          <p className="mt-1 text-[11px]">Foreground alpha &gt;204: {result.browser.foregroundExtent.join(' × ')} pixels {result.browser.eligible ? '— sufficient extent' : '— needs a transparent background and at least 3 × 3 extent'}</p>
          <p className="mt-2 text-[10px] text-zinc-500">A browser may convert pixels for display. This does not prove original RGBA mode, a single frame or the original resize check.</p>
        </div>}
        {cpu && <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-3">
          <h3 className={`font-medium ${cpu.status === 'valid' ? 'text-emerald-300' : 'text-amber-300'}`}>{cpu.status === 'valid' ? 'Original CPU input contract: eligible' : cpu.status === 'invalid' ? 'Original CPU input contract: rejected' : 'Original CPU input contract: unavailable'}</h3>
          {typeof cpu.metadata.mode === 'string' && <p className="mt-2 text-[11px]">Original mode: {cpu.metadata.mode} · {String(cpu.metadata.format ?? 'unknown format')}</p>}
          {typeof cpu.metadata.transparent_pixels_present === 'boolean' && <p className="mt-1 text-[11px]">Original transparent background: {cpu.metadata.transparent_pixels_present ? 'Detected' : 'Missing'}</p>}
          {Array.isArray(cpu.metadata.foreground_extent) && <p className="mt-1 text-[11px]">Original foreground &gt;204: {cpu.metadata.foreground_extent.join(' × ')} pixels</p>}
          {Array.isArray(cpu.metadata.resized_foreground_extent) && <p className="mt-1 text-[11px]">After original resize: {cpu.metadata.resized_foreground_extent.join(' × ')} pixels</p>}
          {cpu.reasons.map(reason => <div key={reason.code} className="mt-2 text-[11px] leading-relaxed"><p>{reason.message}</p><p className="mt-1 text-zinc-400">{reason.correction}</p></div>)}
          <p className="mt-2 text-[10px] text-zinc-500">Original image preprocessing: not executed. DINO, neural models and generation: not run or qualified.</p>
        </div>}
        {result?.unavailable && <p className="rounded-lg border border-amber-800/50 p-3 text-[11px] text-amber-200">{result.unavailable}</p>}
        {(result?.unavailable || cpu?.status === 'unavailable') && <button type="button" onClick={() => { if (chosenPath.current) void session.current?.select(chosenPath.current) }} className="rounded-md border border-zinc-700 px-3 py-2 text-[11px] text-zinc-200 hover:bg-zinc-800">Check input again</button>}
        {!result && !busy && !state.error && <p className="text-[11px] text-zinc-500">Use a genuinely pre-segmented RGBA image with fully transparent background pixels. Ordinary JPEG photographs have no transparency and are unsupported by this alpha-only path.</p>}
      </div>
    </section>
  )
}
