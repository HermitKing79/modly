"""Read saved app observations only; never launch or import an asset."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = ROOT.parent / 'artifacts/amd-high-resolution-master-app-64d07f5'
SESSION = PRIVATE / 'desktop-session'
EXPECTED = '1970bbcc5740dc56c8c9e6e20d97e8afd3176126c57b87d886a034f876eb38ae'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(name):
    return json.loads((SESSION / name).read_text(encoding='utf-8'))


def main():
    host = read('app-host.json')
    full = read('master-buffer-material.json')
    tiny = read('tiny-buffer-material.json')
    mesh, = full['meshes']
    assert host['classification'] == 'PASS' and host['returncode'] == 0
    assert host['input_sources_unchanged']
    assert host['job_cleanup']['active_after_cleanup'] == 0
    assert not host['job_cleanup']['forced_job_cleanup']
    assert len(tiny['meshes']) == 1 and tiny['meshes'][0]['indices'] == 9
    assert mesh['vertices'] == 1957891 and mesh['triangles'] == 4173584
    assert mesh['positionSHA'] == '688de41c46daeaac404f44b5c7304ba71c9adf52fd5846cd2edbc4bb6b096c72'
    assert mesh['colourSHA'] == 'b8134eac3f81d14529bfd2f308a5fef352923152df0c15b64ffbf2880e98cfa0'
    assert mesh['colourType'] == 'Float32Array'
    assert mesh['material'] == dict(type='MeshStandardMaterial', vertexColors=True,
        color=[1, 1, 1], metalness=0.00011082693761409814,
        roughness=0.999995986777544, flatShading=True, side=2)
    exported = SESSION / 'exports/modly-original-master.glb'
    assert exported.stat().st_size == 97073940 and digest(exported) == EXPECTED
    resources = host['resource_samples']
    assert resources and all(x['admitted'] for x in resources)
    receipts = {name: dict(bytes=(SESSION/name).stat().st_size, sha256=digest(SESSION/name))
        for name in ('app-host.json', 'app-session-attempt.json', 'master-buffer-material.json',
                     'tiny-buffer-material.json', 'master-app-metrics.json', 'post-blank-state.json',
                     'original-app-root.json', 'app.stdout.txt', 'app.stderr.txt')}
    result = dict(
        classification='BLOCKED', reason='Five acceptance areas pass; camera and in-session cleanup remain unverified.',
        installed_app_version='0.4.3', exact_checkout_build=False,
        verdicts=dict(import_glb='PASS', full_resolution_viewport='PASS', vertex_colour_fidelity='PASS',
            camera_interaction_responsiveness='BLOCKED', triangle_count_reporting='PASS',
            original_byte_glb_export='PASS', model_unload_resource_cleanup='BLOCKED'),
        actual_ui_imports=dict(tiny=1, full_master=1), production_source_changes=False,
        vertices=mesh['vertices'], triangles=mesh['triangles'], live_mesh=mesh,
        export=dict(path=str(exported), bytes=exported.stat().st_size, sha256=digest(exported)),
        resource_summary=dict(samples=len(resources), all_admitted=True,
            minimum_available_ram_bytes=min(x['available_ram_bytes'] for x in resources),
            minimum_destination_free_bytes=min(x['disk_free_bytes'] for x in resources),
            peak_process_committed_bytes=host['peak_process_committed_bytes'],
            peak_job_committed_bytes=host['peak_job_committed_bytes'],
            elapsed_seconds=host['elapsed_seconds'], deadline_seconds=900,
            memory_cap_bytes=6442450944, dedicated_gpu_memory_measured=False),
        cleanup=dict(app_exit_code=host['returncode'], job=host['job_cleanup'],
            caveat='Stock PythonBridge shutdown uses taskkill /T /F; backend exit 1 is logged. This is not proof of graceful backend exit or in-session model disposal.'),
        limitations=[
            'Installed app, not an exact build of source checkout; lineage audit is separate.',
            'Restricted-context startup exited 0x80000003 before any asset import; cause unproven.',
            'Diagnostic observer awaited a renderer before original loadFile. One reviewed about:blank bootstrap unblocked it; no unassisted startup claim.',
            'Mock observer tests did not establish cold-renderer startup behavior.',
            'Saved master-orbit.jpg depicts the export dialog, not an orbit; excluded as camera evidence.',
            'Renderer query selected a helper/postpass root; its triangle counter is not the master draw count.',
            'No per-import precise timing, FPS, dedicated VRAM, pan/zoom, model replacement or WebGL resource-release proof.',
            'Final export independently hashed after exit; last download ledger was still progressing.'
        ], receipts=receipts,
        screenshots=[dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))
            for p in sorted((SESSION/'screenshots').glob('*.jpg'))])
    destination = Path(__file__).with_name('execution-summary.json')
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(classification=result['classification'], verdicts=result['verdicts'],
        summary_sha256=digest(destination), resources=result['resource_summary'])))


if __name__ == '__main__':
    main()
