// Read-only check of the actual replay workspace through existing library/viewer projection code.
import { readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'
import { listWorkspaceAssetLibrary, openWorkspaceAssetLibraryEntry, readWorkspaceAssetLibraryEntry } from '../../../electron/main/artifact-registry-service.ts'
import { projectAssetLibraryEntry, resolveAssetLibraryOpenTarget } from '../../../src/areas/generate/assetLibraryProjection.ts'

const root = fileURLToPath(new URL('../../../../artifacts/amd-worker-product-integration-279f8c0/replay/', import.meta.url))
const result = JSON.parse(await readFile(join(root, 'result.json'), 'utf8'))
const workspaceDir = join(root, 'workspace')
const library = await listWorkspaceAssetLibrary({ workspaceDir })
if (!library.success) throw new Error('Existing workspace library rejected replay workspace')
const cases = []
for (const item of result.cases) {
  const workspacePath = item.job.output_url.replace(/^\/workspace\//, '')
  const entry = library.entries.find((value) => value.workspacePath === workspacePath)
  if (!entry || !entry.openable || entry.previewKind !== '3d-model') throw new Error('Promoted GLB not discoverable/openable')
  const opened = await openWorkspaceAssetLibraryEntry({ workspaceDir, workspacePath })
  const target = resolveAssetLibraryOpenTarget(projectAssetLibraryEntry(entry))
  if (!opened.success || target.kind !== 'self' || target.url !== item.job.output_url) throw new Error('Existing viewer URL differs')
  // Ordinary JSON sidecars are intentionally filtered from list(). Their read
  // interface still exposes metadata; the job API also returns the full receipt.
  const metadata = await readWorkspaceAssetLibraryEntry({ workspaceDir, workspacePath: workspacePath.replace(/\.glb$/, '.json') })
  if (!metadata.success || metadata.entry.artifactId !== item.job.job_id || metadata.entry.source?.workspacePath !== workspacePath) throw new Error('Read-back provenance/source link differs')
  cases.push({ case: item.case, openable: true, viewer_url_exact: true, sidecar_readback_exact: true,
               sidecar_separately_listed: false })
}
const evidence = { classification: 'PASS', actual_replay_workspace: true, desktop_renderer_run: false, cases }
await writeFile(new URL('./workspace-result.json', import.meta.url), JSON.stringify(evidence, null, 2) + '\n', { flag: 'wx' })
console.log(JSON.stringify(evidence))
