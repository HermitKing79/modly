import { basename } from 'node:path'
import { open, readFile, stat } from 'node:fs/promises'

export const SCENE_MANIFEST_FILE_NAME = 'scene-manifest.json'
export const MAX_SCENE_MANIFEST_BYTES = 1024 * 1024

type FileReadDependencies = {
  statFile: (filePath: string) => Promise<{ size: number; isFile: () => boolean }>
  readBytes: (filePath: string) => Promise<Uint8Array>
  openFile?: typeof open
}

const DEFAULT_DEPENDENCIES: FileReadDependencies = {
  statFile: stat,
  readBytes: readFile,
  openFile: open,
}

export async function readLocalFileBase64(
  filePath: string,
  dependencies: FileReadDependencies = DEFAULT_DEPENDENCIES,
  maxBytes?: number,
): Promise<string> {
  if (typeof filePath !== 'string' || filePath.trim().length === 0) {
    throw new Error('fs:readFileBase64 requires a non-empty file path')
  }

  const isSceneManifest = basename(filePath) === SCENE_MANIFEST_FILE_NAME
  if (maxBytes !== undefined) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0 || maxBytes > 32 * 1024 * 1024) {
      throw new Error('fs:readFileBase64 maxBytes must be a positive integer of at most 32 MiB')
    }
    const limit = isSceneManifest ? Math.min(maxBytes, MAX_SCENE_MANIFEST_BYTES) : maxBytes
    const oversized = (size: number) => new Error(
      isSceneManifest && limit === MAX_SCENE_MANIFEST_BYTES
        ? `Scene manifest exceeds the 1 MiB limit; actual size ${size} bytes`
        : `File size ${size} bytes exceeds the ${limit} byte limit`,
    )
    const handle = await (dependencies.openFile ?? open)(filePath, 'r')
    try {
      const fileInfo = await handle.stat()
      if (!fileInfo.isFile()) throw new Error('Selected path is not a file')
      if (!Number.isSafeInteger(fileInfo.size) || fileInfo.size < 0) throw new Error('Invalid file size')
      if (fileInfo.size > limit) throw oversized(fileInfo.size)
      // One extra byte detects growth without an unbounded read or a new path lookup.
      const buffer = Buffer.alloc(limit + 1)
      let size = 0
      while (size <= limit) {
        const { bytesRead } = await handle.read(buffer, size, buffer.length - size, null)
        if (bytesRead === 0) break
        size += bytesRead
        if (size > limit) {
          const current = await handle.stat()
          throw oversized(Math.max(size, current.size))
        }
      }
      return buffer.subarray(0, size).toString('base64')
    } finally {
      await handle.close()
    }
  }
  if (isSceneManifest) {
    const fileInfo = await dependencies.statFile(filePath)
    if (!fileInfo.isFile()) throw new Error('Scene manifest is not a file')
    if (fileInfo.size > MAX_SCENE_MANIFEST_BYTES) {
      throw new Error('Scene manifest exceeds the 1 MiB limit')
    }
  }

  const bytes = await dependencies.readBytes(filePath)
  // Recheck the transferred bytes in case the file grew between stat and read.
  if (isSceneManifest && bytes.byteLength > MAX_SCENE_MANIFEST_BYTES) {
    throw new Error('Scene manifest exceeds the 1 MiB limit')
  }
  return Buffer.from(bytes).toString('base64')
}
