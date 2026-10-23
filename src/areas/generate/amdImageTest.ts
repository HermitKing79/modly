import type { LocalImage } from '../workflows/nodes/imageUtils'
export { imageByteLength } from '../workflows/nodes/imageUtils'

export const AMD_IMAGE_MAX_BYTES = 2 * 1024 * 1024
export interface BrowserAlpha { transparent: boolean; foregroundExtent: [number, number]; eligible: boolean }
export interface CpuImageResult {
  schema: 'modly.amd.image-test.v1'
  status: 'valid' | 'invalid' | 'unavailable'
  eligible: boolean | null
  preprocessing_complete: false
  generation_qualified: false
  validation: string
  metadata: Record<string, unknown>
  reasons: { code: string; message: string; correction: string }[]
}

export function analyzeAlpha(pixels: Uint8ClampedArray, width: number, height: number): BrowserAlpha {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width > 4096 || height > 4096 || pixels.length !== width * height * 4) {
    throw new Error('Browser pixel dimensions exceed the input envelope')
  }
  let transparent = false, left = width, right = -1, top = height, bottom = -1
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const alpha = pixels[(y * width + x) * 4 + 3]
    if (alpha === 0) transparent = true
    if (alpha > 204) { left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y) }
  }
  const foregroundExtent: [number, number] = [Math.max(0, right - left + 1), Math.max(0, bottom - top + 1)]
  return { transparent, foregroundExtent, eligible: transparent && foregroundExtent.every(n => n >= 3) }
}

export function validateCpuResult(value: unknown): CpuImageResult {
  const r = value as CpuImageResult | null
  if (!r || r.schema !== 'modly.amd.image-test.v1' || !['valid', 'invalid', 'unavailable'].includes(r.status) ||
      r.preprocessing_complete !== false || r.generation_qualified !== false ||
      (r.status === 'valid' && r.validation !== 'original_cpu_input_contract') ||
      r.eligible !== (r.status === 'valid' ? true : r.status === 'invalid' ? false : null) ||
      !r.metadata || typeof r.metadata !== 'object' || !Array.isArray(r.reasons) ||
      r.reasons.some(reason => !reason || typeof reason.message !== 'string' || typeof reason.correction !== 'string')) {
    throw new Error('Unexpected local CPU validation response')
  }
  return r
}

export function localValidationUrl(apiUrl: string): string {
  const url = new URL(apiUrl)
  if (url.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new Error('Image validation requires the local Modly backend')
  }
  return `${url.origin}/amd/image-test`
}

export function safePreviewDimensions(cpu: Pick<CpuImageResult, 'metadata'>): [number, number] | null {
  const { width, height, frames } = cpu.metadata
  return frames === 1 && typeof width === 'number' && typeof height === 'number' && Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0 && width <= 4096 && height <= 4096 && width * height <= 16777216 ? [width, height] : null
}

export interface ImageTestState<R> { phase: 'empty' | 'loading' | 'validating' | 'done' | 'error'; path?: string; image?: LocalImage; result?: R; error?: string }

export function createImageTestSession<R>(load: (path: string) => Promise<LocalImage>, validate: (image: LocalImage, signal: AbortSignal) => Promise<R>, publish: (state: ImageTestState<R>) => void) {
  let revision = 0, disposed = false, controller: AbortController | undefined
  return {
    async select(path: string) {
      if (disposed) return
      const current = ++revision
      controller?.abort()
      controller = new AbortController()
      publish({ phase: 'loading', path })
      try {
        const image = await load(path)
        if (disposed || current !== revision) return
        publish({ phase: 'validating', path, image })
        const result = await validate(image, controller.signal)
        if (!disposed && current === revision) publish({ phase: 'done', path, image, result })
      } catch (error) {
        if (!disposed && current === revision) publish({ phase: 'error', path, error: error instanceof Error ? error.message : 'Image validation failed' })
      }
    },
    dispose() { disposed = true; revision++; controller?.abort() },
  }
}
