"""One verified pure-Python packaging wheel addition; no dependency resolver."""
import email
import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
import stat
import sys
import zipfile

E = Path(__file__).resolve().parent
TARGET = Path('/home/modly/native-decoder-preparation-6dd94a85/site-packages')
def sha(data): return hashlib.sha256(data).hexdigest()
def deny(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm','numpy','safetensors','packaging'}:
        raise RuntimeError('Staging is standard-library-only')
sys.addaudithook(deny)
admission = json.loads((E/'packaging-admission.json').read_text())
wheel = E/admission['filename']
if wheel.stat().st_size != 66469 or sha(wheel.read_bytes()) != '29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484':
    raise RuntimeError('Approved wheel identity differs')
if TARGET.resolve(strict=True) != TARGET or (TARGET/'packaging').exists() or (TARGET/'packaging-25.0.dist-info').exists():
    raise RuntimeError('Target alias or existing packaging; no overwrite allowed')
sys.argv = [str(E/'passive_linux.py'), 'before']
runpy.run_path(str(E/'passive_linux.py'), run_name='__main__')
payloads = {}
with zipfile.ZipFile(wheel) as archive:
    metadata = email.message_from_bytes(archive.read('packaging-25.0.dist-info/METADATA'))
    wheel_meta = email.message_from_bytes(archive.read('packaging-25.0.dist-info/WHEEL'))
    if metadata['Name'] != 'packaging' or metadata['Version'] != '25.0' or metadata.get_all('Requires-Dist') or metadata['Requires-Python'] != '>=3.8':
        raise RuntimeError('Unexpected dependency/metadata contract')
    if wheel_meta['Root-Is-Purelib'] != 'true' or wheel_meta.get_all('Tag') != ['py3-none-any']:
        raise RuntimeError('Only a genuine pure-Python universal wheel is allowed')
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir(): continue
        if path.is_absolute() or '..' in path.parts or '\\' in info.filename or path.parts[0] not in {'packaging','packaging-25.0.dist-info'} or stat.S_ISLNK(info.external_attr >> 16):
            raise RuntimeError('Unsafe wheel member: '+info.filename)
        if (TARGET/info.filename).exists(): raise RuntimeError('Collision: '+info.filename)
        payloads[info.filename] = archive.read(info)
if not payloads or sum(map(len,payloads.values())) > 4*1024**2: raise RuntimeError('Bounded wheel size gate')
with (E/'packaging-install.marker').open('x') as marker:
    json.dump({'wheel_sha256':admission['sha256'],'target':str(TARGET),'installations':1},marker)
rows = {}
for name, data in payloads.items():
    destination = TARGET/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.parent.resolve().is_relative_to(TARGET): raise RuntimeError('Foreign destination')
    with destination.open('xb') as stream: stream.write(data)
    if destination.read_bytes() != data: raise RuntimeError('Staging readback differs')
    rows[name] = {'bytes':len(data),'sha256':sha(data)}
# Read the exact installed safetensors sources without importing the package.
source_rows = {}
for name in ('__init__.py','torch.py'):
    source = TARGET/'safetensors'/name
    data = source.read_bytes()
    with (E/('installed-safetensors-'+name+'.txt')).open('xb') as stream: stream.write(data)
    source_rows[name] = {'path':str(source),'sha256':sha(data)}
result = {'status':'PASS','version':'25.0','target':str(TARGET),'wheel_sha256':admission['sha256'],
    'installed_files':rows,'installed_bytes':sum(map(len,payloads.values())),
    'original_staged_payloads_verified_before':922,'baseline_venv_modified':False,
    'safetensors_source_readback':source_rows,'AI_native_imports':False,'installations':1}
with (E/'packaging-staging.json').open('x') as stream: json.dump(result,stream,indent=2)
print(json.dumps({'status':'PASS','installed_files':len(rows),'installed_bytes':result['installed_bytes']}))
