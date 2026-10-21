"""Pinned read-only source and previous complete GLB verification closure."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import hashlib, importlib.util
E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-brick-full-glb-blender-df195c4'
OLD = ROOT / 'docs/compatibility/amd-brick-full-sampled-glb-evidence'
PREVIOUS = ROOT.parent / 'artifacts/amd-brick-full-sampled-glb-3db265c'

def pinned(name, path, sha):
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError('Pinned source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

original = pinned('Blender_original_inputs', OLD/'inputs.py', 'f20e771b322788a1c4fbde2031906dd4e81dcd2c55c772f232a9b33106c3b07c')
plain, digest, require, save, load_auth = original.plain, original.digest, original.require, original.save, original.load_auth
GLB = PREVIOUS/'export/brick-source-sampled-colour.glb'
COLOURS = PREVIOUS/'export/linear-COLOR0.npy'
MESH = ROOT.parent/'artifacts/amd-brick-cpu-mesh-8473579/mesh'
BLENDER = Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')
ADDON = BLENDER.parent/'5.2/scripts/addons_core/io_scene_gltf2'
PINS = {
    GLB: '1970bbcc5740dc56c8c9e6e20d97e8afd3176126c57b87d886a034f876eb38ae',
    COLOURS: '1240160cefcf961d463d9c06416e0907469c8d80a590fb44f483bb1fc220ea44',
    BLENDER: '284f4041f98e113f3dc10654a7193ffaaa9bfdfec8b87fa116620a48b5f6d4cb',
    PREVIOUS/'export/result.json': 'c7d440405c1354e9a9abef45a6892189e9a9671d54976ec0ce9a7b5adec91022',
    PREVIOUS/'verify/result.json': 'a36d1368e3910b9865416cd9e3b9653fd7d5f06634addd031657874a40c02b46',
    OLD/'verify_stream.py': '72cbf55f1792f6abebe240e4a69627a9e780b2e3e12a0eab7c9770ccbb106cdd',
    OLD/'independent-outcome.json': 'f2c149a3b1396b5f0e1aeaa21d7f93fd7680057be9a6196f2e4ee10fb3928822',
}

def authenticate():
    auth, identities, watched, receipts = original.authenticate()
    for path, sha in PINS.items():
        require(digest(plain(path)) == sha, 'Exact original GLB/source identity: ' + str(path))
    require(GLB.stat().st_size == 97073940 and COLOURS.stat().st_size == 23494820, 'Original GLB/colour sizes')
    for path in (PREVIOUS/'export/result.json', PREVIOUS/'verify/result.json'):
        require(json.loads(path.read_text())['classification'] == 'PASS', 'Previous independent complete export PASS')
    return auth, identities, list(dict.fromkeys(watched + list(PINS))), receipts
