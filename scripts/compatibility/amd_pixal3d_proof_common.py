"""Stdlib-only policy shared by the standalone host and model-free guest."""
from datetime import datetime,timezone
import hashlib,json,os,re,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CONTRACT=ROOT/'docs/compatibility/amd-pixal3d-standalone-hip-contract.json'
GIB=1024**3
OPERATORS=('gfx1201','HIP_extension','HIP_INT8','HIP_ConvRot','PyTorch_SDPA','NAF','native_sparse','native_conversion')

def now():return datetime.now(timezone.utc).isoformat()

def lock():
    value=json.loads(CONTRACT.read_text(encoding='utf-8'))
    if value['schema']!='modly.pixal3d.standalone.model-free-hip.v1':raise ValueError('Unknown contract')
    if value['guest_root']!='/home/modly/pixal3d-native-proof-v1' or value['windows_root']!='D:/AI/Modly-Pixal3D-Proof-v1':raise ValueError('Isolation contract differs')
    if value['model_downloads_allowed'] or value['generation_authorized']:raise ValueError('Model-free scope required')
    return value

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def plain(path):
    path=Path(path).absolute()
    if str(path).startswith('\\\\') or path!=path.resolve():raise ValueError('Canonical local path required')
    for parent in (path,*path.parents):
        if parent.exists() and (parent.is_symlink() or getattr(parent,'is_junction',lambda:False)()):raise ValueError('Redirected path denied')
    if path.is_file() and path.stat().st_nlink!=1:raise ValueError('Hardlinked file denied')
    return path

def inside(root,name):
    if not isinstance(name,str) or not name or '\\' in name or ':' in name or Path(name).is_absolute() or '..' in name.split('/'):
        raise ValueError('Only relative unredirected paths allowed')
    path=plain(Path(root)/name);path.relative_to(plain(root));return path

def save(path,value):
    with plain(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def journal(path,phase,**values):
    with plain(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(utc=now(),phase=phase,**values),allow_nan=False)+'\n');stream.flush()

def require_ram(row,floor):
    if type(row.get('available_bytes')) is not int or row['available_bytes']<floor:raise RuntimeError('Available Windows RAM below '+str(floor)+' bytes')

def require_resources(available,D_free,C_free):
    if available<12*GIB or D_free<50*GIB or C_free<2*GIB:raise RuntimeError('Original live RAM/D/output disk floor failed')

def admit_once(read,sleep,record,execute):
    # Reuse the original qualified three-reading implementation unchanged.
    from amd_zombie_d_handoff import fresh_admission
    fresh_admission(read,sleep,record)
    final=dict(read(),timestamp_utc=now(),stage='final-before-provision')
    record(final);require_ram(final,22*GIB)
    return execute('admitted')

def verify(path,row):
    path=plain(path)
    if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise RuntimeError('Payload identity differs: '+path.name)

def hip_payload(path):
    """Inspect the prebuilt wheel before environment creation; never build it."""
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        matches=[n for n in names if re.fullmatch(r'comfy_kitchen/backends/hip/_C[^/]*\.so',n)]
        if len(matches)!=1:raise RuntimeError('Missing or ambiguous prebuilt HIP extension; no compile fallback')
        with archive.open(matches[0]) as stream:header=stream.read(64)
        if header[:6]!=b'\x7fELF\x02\x01' or int.from_bytes(header[18:20],'little')!=62:raise RuntimeError('Linux x86-64 ELF HIP payload required')
        architectures=archive.read('comfy_kitchen/backends/hip/architectures.json')
        if b'"gfx1201"' not in architectures:raise RuntimeError('Wheel does not advertise gfx1201')
        return dict(extension=matches[0],extension_sha256=hashlib.sha256(archive.read(matches[0])).hexdigest(),
                    expansion_bytes=sum(i.file_size for i in archive.infolist()),architectures_sha256=hashlib.sha256(architectures).hexdigest())

def bounded_expansion(paths):
    total=0
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                name=info.filename
                if name.startswith('/') or '..' in name.split('/') or '\\' in name or info.external_attr>>16&0xf000==0xa000:raise ValueError('Unsafe wheel path')
                total+=info.file_size
    if total>20*GIB:raise RuntimeError('Original isolated growth budget exceeded')
    return total
