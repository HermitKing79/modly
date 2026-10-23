// Shared helpers for node components that read image files from disk.

export function mimeFromPath(p: string): string {
  const ext = p.split('.').pop()?.toLowerCase() ?? ''
  if (ext === 'jpg' || ext === 'jpeg') return 'image/jpeg'
  if (ext === 'webp') return 'image/webp'
  return 'image/png'
}

export interface LocalImage {
  path: string
  name: string
  preview: string
  base64: string
  bytes: number
}

export function imageByteLength(base64: string): number {
  const padding = base64.endsWith('==') ? 2 : base64.endsWith('=') ? 1 : 0
  if (!base64 || base64.length % 4 || /[^A-Za-z0-9+/]/.test(base64.slice(0, base64.length - padding))) {
    throw new Error('Invalid image base64')
  }
  return base64.length / 4 * 3 - padding
}

// The same Electron picker/reader supplies workflow nodes and the input test.
export async function readLocalImage(path: string, fs: { readFileBase64: (path: string, maxBytes?: number) => Promise<string> }, maxBytes = Infinity): Promise<LocalImage> {
  const base64 = await (Number.isFinite(maxBytes) ? fs.readFileBase64(path, maxBytes) : fs.readFileBase64(path))
  const bytes = imageByteLength(base64)
  if (bytes > maxBytes) throw new Error(`Image contains ${bytes.toLocaleString()} bytes; exceeds the ${maxBytes.toLocaleString()} byte preview limit`)
  return { path, name: path.split(/[\\/]/).pop() ?? path, base64, bytes, preview: `data:${mimeFromPath(path)};base64,${base64}` }
}
