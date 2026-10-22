"""Exact SDK-only admission for the one synthetic scalar-binding probe."""
import sys
sys.dont_write_bytecode = True
import hashlib, importlib.util, json, os
from pathlib import Path
E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-meshlib-counter-proof-8ff0549'
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
    sdk = load_sdk_auth()
    require(digest(GUARD) == GUARD_SHA, 'Unchanged original CPU audit guard')
    for path, sha in HISTORICAL.items():
        require(digest(path) == sha, 'Preserved historical failure: ' + str(path))
    return [SDK, GUARD, *HISTORICAL, *(Path(item['path']) for item in sdk['files'].values())]

def begin():
    require(sys.stdin.buffer.readline(64) == b'GO\n', 'Owned binary GO before numerical/native imports')
    require(digest(GUARD) == GUARD_SHA, 'Pinned original CPU guard')
    spec = importlib.util.spec_from_file_location('counter_cpu_guard', GUARD)
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    guard.P = P / 'counter'
    sys.addaudithook(guard.runtime_guard)
    def extra(event, args):
        if event == 'import' and str(args[0]).startswith(('meshlib.mrcudapy', 'meshlib.mrviewerpy')):
            raise RuntimeError('No GPU or viewer imports')
        if event in {'os.chmod', 'os.chown'}: raise RuntimeError('No ownership or permission changes')
    sys.addaudithook(extra)
