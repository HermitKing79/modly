"""One owned CPU-only deterministic gate; no Blender/runtime import or render."""
import sys
sys.dont_write_bytecode = True
import json, time, traceback
from pathlib import Path
E = Path(__file__).resolve().parent
sys.path.insert(0, str(E))
from inputs import ROOT, P, OLD, GLB, COLOURS, MESH, ADDON, authenticate, load_auth, plain, digest, require, save, pinned
from candidate import referenced_vertices

def main():
    require(sys.stdin.buffer.readline(64) == b'GO\n', 'Owned exact binary GO')
    auth = load_auth(); auth.P = P; sys.addaudithook(auth.runtime_guard)
    began = time.monotonic(); out = P/'preflight'; stage = 'authentication'
    try:
        import numpy as np
        _, identities, watched, _ = authenticate()
        audit = json.loads(plain(E/'importer-audit.json').read_text())
        for relative, sha in audit['installed_source_SHA256'].items():
            path = plain(ADDON/relative)
            require(digest(path) == sha, 'Actual installed importer source identity')
            watched.append(path)
        before = {str(path): digest(path) for path in watched}
        vertices = np.load(plain(MESH/'vertices.npy'), mmap_mode='r', allow_pickle=False)
        faces = np.load(plain(MESH/'faces.npy'), mmap_mode='r', allow_pickle=False)
        colours = np.load(plain(COLOURS), mmap_mode='r', allow_pickle=False)
        require(vertices.shape == colours.shape == (1957891,3) and faces.shape == (4173584,3), 'Original complete tables')
        reader = pinned('Blender_independent_GLB_reader', OLD/'verify_stream.py', '72cbf55f1792f6abebe240e4a69627a9e780b2e3e12a0eab7c9770ccbb106cdd')
        previous = json.loads((GLB.parent/'result.json').read_text())
        stage = 'fresh_entire_GLB_readback'
        parsed = reader.verify(GLB, vertices, faces, colours, previous['source_hashes'])
        stage = 'complete_index_membership'
        used, counts = referenced_vertices(faces, len(vertices))
        require(not counts['exact_import_vertex_count_gate'], 'Freeze the demonstrated blocker; unexpected result requires review')
        require(counts['referenced_vertices'] == 1952142 and counts['unreferenced_vertices'] == 5749, 'Independent source/GLB count agreement')
        require(before == {str(path): digest(path) for path in watched}, 'All sources unchanged')
        require(not any(name == 'bpy' or name.startswith(('torch','triton','flex_gemm')) for name in sys.modules), 'No viewer/GPU/model runtime imported')
        save(out/'result.json', dict(classification='PASS', import_verdict='BLOCKED', visual_verdict='BLOCKED',
            scope='CPU diagnosis only: no Blender process, no GLB import, no render, no production import marker',
            counts=counts, input_GLB_SHA256=digest(GLB), input_GLB_bytes=GLB.stat().st_size,
            original_geometry_and_F32_COLOUR0_entire_file_exact=True, container=parsed['report'],
            referenced_membership_SHA256=__import__('hashlib').sha256(used.tobytes()).hexdigest(),
            sources_unchanged=True, installed_importer_modified=False, acceptance_relaxed=False,
            Blender_imports=0, Blender_renders=0, elapsed_seconds=time.monotonic()-began))
        print('PRE_IMPORT_EXACT_VERTEX_COUNT_BLOCKED', flush=True)
    except BaseException as error:
        if not (out/'result.json').exists():
            save(out/'result.json', dict(classification='BLOCKED', stage=stage, error=str(error), traceback=traceback.format_exc(), retry=False))
        raise

if __name__ == '__main__': main()
