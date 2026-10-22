"""Pinned original-patch and SDK admission for corrected constructor readback."""
import sys
sys.dont_write_bytecode = True
import hashlib, importlib.util, json, os
from pathlib import Path
E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-meshlib-patch-constructor-bbdb6ac'
OLD = ROOT / 'docs/compatibility/amd-clean-target-feasibility-evidence'
OLD_PRIVATE = ROOT.parent / 'artifacts/amd-clean-target-feasibility-d10c18e'
SDK = OLD / 'constructor-library-pins.json'
SDK_SHA = '1d9a31b199aae661f5333a6cf482c20986a3578e7d6d73185646e8e87a70267b'
GUARD = ROOT / 'docs/compatibility/amd-brick-defect-diagnosis-evidence/diagnose.py'
GUARD_SHA = '8fa7c617324ba636856b48f47a995ee64cd2a85f3c26d86d8760ec638b163529'
HISTORICAL = {
    OLD / 'import_probe.py': '14a88253ff5764df843467a415375562fbc010785bbb74451d64ee4fcb1656e7',
    OLD_PRIVATE / 'import/result.json': '06c6277e175e534996b738c552d69783960d105a676330214c6b867b5f9b966e',
    OLD_PRIVATE / 'import-host.json': '305cf4de32c08ceab8b907fac4ac49ce4899886973d3191a166fc523050ea038',
}

def require(ok, message):
    if not ok: raise ValueError(message)

def plain(path):
    path = Path(path).absolute()
    for q in (path, *path.parents):
        s = q.lstat()
        require(not q.is_symlink() and not getattr(s, 'st_file_attributes', 0) & 0x400, 'Plain non-reparse path required')
    require(not path.is_file() or path.stat().st_nlink == 1, 'Single-link file required')
    return path

def digest(path):
    with plain(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())

def read_json(path):
    path = plain(path); require(path.stat().st_size <= 2 * 1024**2, 'Bounded JSON')
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Unique JSON keys'); result[key] = value
        return result
    def nonfinite(value): raise ValueError('Finite JSON required')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=nonfinite)

def load_sdk_auth():
    require(digest(SDK) == SDK_SHA, 'Original independently reviewed SDK manifest')
    row = read_json(SDK)
    require(row['package'] == 'meshlib' and row['version'] == '3.1.4.297' and row['CPU_import_admitted'] is True, 'Exact CPU library admission')
    require(row['resolved_PE_images'] == 72 and len(row['files']) == 78, 'Exact installed CPU/source closure')
    require(row['static_PE_closure_SHA256'] == '8687f30a413b99c8b4e27d789f3af4240f70684cbe7d63d71ee4ea09cc7beef5', 'Pinned static dependency audit')
    for item in row['files'].values():
        path = plain(item['path'])
        require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], 'Audited native/source library identity: ' + str(path))
    return row

def authenticate():
    plan=load_plan();lineage=load_lineage()
    sdk = load_sdk_auth()
    require(digest(GUARD) == GUARD_SHA, 'Unchanged original CPU audit guard')
    for path, sha in HISTORICAL.items():
        require(digest(path) == sha, 'Preserved historical failure: ' + str(path))
    return list(dict.fromkeys([SDK,GUARD,*HISTORICAL,INPUT_MANIFEST,LINEAGE,*(Path(item['path']) for item in sdk['files'].values()),*(Path(item['path']) for patch in plan['patches'] for item in patch['inputfiles'].values()),*lineage_paths(lineage)]))

def begin(stage='import'):
    require(sys.stdin.buffer.readline(64) == b'GO\n', 'Owned binary GO before numerical/native imports')
    require(digest(GUARD) == GUARD_SHA, 'Pinned original CPU guard')
    spec = importlib.util.spec_from_file_location('counter_cpu_guard', GUARD)
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    require(stage in {'import','verify'},'Only approved constructor and saved-data phases')
    guard.P = P / stage
    sys.addaudithook(guard.runtime_guard)
    def extra(event, args):
        if event == 'import' and (str(args[0]).startswith(('meshlib.mrcudapy', 'meshlib.mrviewerpy')) or (stage=='verify' and str(args[0]).split('.')[0]=='meshlib')):
            raise RuntimeError('No GPU or viewer imports')
        if event in {'os.chmod', 'os.chown'}: raise RuntimeError('No ownership or permission changes')
    sys.addaudithook(extra)

INPUT_MANIFEST=OLD_PRIVATE/'input-manifest.json'
LINEAGE=OLD_PRIVATE/'patch-authentication-private.json'
MICRO=ROOT/'docs/compatibility/amd-meshlib-counter-proof-evidence'
MICRO_PRIVATE=ROOT.parent/'artifacts/amd-meshlib-counter-proof-8ff0549'
HISTORICAL.update({
    MICRO/'counter_probe.py':'c58d5ea0df31a909ec47f577e0c0698b2920f638b2eb29d458dfb16a6911d25a',
    MICRO/'independent-outcome.json':'24b6f77333e05c17092343d24b6899d8a83d4a58d2342f13601396454a0dfb8f',
    MICRO_PRIVATE/'counter/result.json':'81755459b9f13e8dbc586ef3d0b3908d4acdccb2854ad67e6dde9928c4966c9e',
    MICRO_PRIVATE/'counter-host.json':'74ddae60f738fab8fe04b46febcf42ee676e6517b376ccc41473cb3055095889',
})

def load_plan():
    require(digest(INPUT_MANIFEST)=='aeb98fd9411ab718d7fd439f6345ad2e10927f55371a4d3c6bce4f22b39204bb','Exact original independently authenticated patch manifest')
    plan=read_json(INPUT_MANIFEST)
    require([x['name'] for x in plan['patches']]==['brick','foliage'],'Only frozen brick then foliage')
    for patch,(vertices,faces) in zip(plan['patches'],[(457,781),(3177,4531)]):
        require((patch['vertex_count'],patch['triangle_count'])==(vertices,faces),'Exact accepted patch counts')
        require(set(patch['inputfiles'])=={'vertices','faces','source_vertex_ids','source_face_ids'},'Four exact original correspondence arrays')
        for item in patch['inputfiles'].values():
            path=plain(item['path']);require(path.is_relative_to(ROOT.parent/'artifacts') and path.stat().st_size==item['bytes'] and item['bytes']<=1024**2 and digest(path)==item['sha256'],'Exact bounded original patch file')
    return plan

def lineage_paths(lineage):
    result=[]
    for patch in lineage['patches'].values():
        result.append(Path(patch['receipt']))
        result.extend(Path(item['path']) for item in patch['files'].values())
    result.extend(Path(item['path']) for item in lineage['accepted_patch_assets'].values())
    result.append(Path(lineage['foliage_decision_receipt']))
    result.append(Path(lineage['foliage_linear_RGB']['path']))
    result.extend(ROOT/path for path in lineage['source_evidence_SHA256'])
    return result

def load_lineage():
    require(digest(LINEAGE)=='333d0028af652fced2049f5a9c8dce78ed4d7bc1f0d40f910c6cdf7bdaf82c52','Exact prior private patch/material authentication')
    row=read_json(LINEAGE)
    for patch in row['patches'].values():
        require(digest(patch['receipt'])==patch['receipt_sha256'],'Original patch receipt')
        for item in patch['files'].values():
            path=plain(item['path']);require(path.stat().st_size==item['bytes'] and item['bytes']<=2*1024**2 and digest(path)==item['sha256'],'Original patch geometry/colours/selection identity')
    for item in [*row['accepted_patch_assets'].values(),row['foliage_linear_RGB']]:
        require(plain(item['path']).stat().st_size==item['bytes'] and digest(item['path'])==item['sha256'],'Accepted colour/portable asset identity')
    require(digest(row['foliage_decision_receipt'])==row['foliage_decision_sha256'],'Existing numerical colour decision')
    for path,sha in row['source_evidence_SHA256'].items():
        require(digest(ROOT/path)==sha,'Existing original geometry/material correspondence source')
    return row
