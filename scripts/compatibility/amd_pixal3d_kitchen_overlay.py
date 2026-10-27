"""One authenticated pure-Python ZIP overlay; no extraction or installation."""
import ast,email,hashlib,importlib,io,os,sys,zipfile
from pathlib import Path
from amd_pixal3d_proof_common import ROOT,plain
from amd_pixal3d_wheel_payload import wheel_manifest,readonly

PIN=dict(name='packaging',version='25.0',filename='packaging-25.0-py3-none-any.whl',bytes=66469,
    sha256='29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484',
    path='docs/compatibility/amd-wsl-packaging-decoder-loading-evidence/packaging-25.0-py3-none-any.whl',
    source='https://files.pythonhosted.org/packages/20/12/38679034af332785aac8774540895e234f4d07f7545804097de4b666afd8/packaging-25.0-py3-none-any.whl')

def archive_path():return plain(ROOT/PIN['path'])

def authenticate(progress=None):
    path=archive_path();manifest=wheel_manifest(path,PIN,path.parent,path.parent,progress)
    if manifest['RECORD_discrepancies']:raise RuntimeError('Overlay RECORD/payload disagreement')
    if any(not(n.startswith('packaging/') or n.startswith('packaging-25.0.dist-info/')) or n.endswith(('.so','.pyd','.dll','.pth','.pyc')) for n in manifest['files']):
        raise RuntimeError('Only the exact pure-Python packaging overlay is allowed')
    # Metadata parsing uses an independently authenticated bounded byte snapshot,
    # never a second unverified ZIP handle after the payload pass.
    with readonly(path) as stream:data=stream.read(PIN['bytes']+1)
    if len(data)!=PIN['bytes'] or hashlib.sha256(data).hexdigest()!=PIN['sha256']:raise RuntimeError('Overlay metadata snapshot differs')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        meta=email.message_from_bytes(archive.read('packaging-25.0.dist-info/METADATA'))
        wheel=email.message_from_bytes(archive.read('packaging-25.0.dist-info/WHEEL'))
    if meta['Name']!='packaging' or meta['Version']!='25.0' or meta.get_all('Requires-Dist',[]) or wheel['Root-Is-Purelib']!='true' or wheel.get_all('Tag')!=['py3-none-any']:
        raise RuntimeError('Unexpected overlay dependency/platform metadata')
    return dict(status='PASS',name='packaging',version='25.0',archive=str(path),archive_sha256=PIN['sha256'],
        bytes=PIN['bytes'],payload_files=len(manifest['files']),expanded_bytes=manifest['expanded_bytes'],installations=0,extraction=False,downloads=False)

def verify_loaded():
    path=archive_path();origins={}
    for name,module in sys.modules.copy().items():
        if name=='packaging' or name.startswith('packaging.'):
            origin=getattr(module,'__file__',None);stem=str(path)+os.sep+name.replace('.',os.sep)
            if origin not in (stem+'.py',stem+os.sep+'__init__.py'):raise RuntimeError('Loaded overlay origin differs: '+name)
            origins[name]=origin
    if 'packaging.version' not in origins:raise RuntimeError('Required real packaging.version was not loaded')
    return origins

def activate():
    if any(n=='packaging' or n.startswith('packaging.') for n in sys.modules):raise RuntimeError('Preloaded packaging cannot bypass overlay provenance')
    report=authenticate();path=archive_path()
    if str(path) in sys.path:raise RuntimeError('Overlay already activated; no retry')
    sys.path.insert(0,str(path))
    from packaging import version
    import packaging
    import importlib.metadata
    if packaging.__version__!='25.0' or importlib.metadata.version('packaging')!='25.0':raise RuntimeError('Loaded overlay version differs')
    origins=verify_loaded()
    if not version.parse('2.9.1')<version.parse('2.10.0'):raise RuntimeError('PEP 440 version comparison failed')
    return dict(report,module_origins=origins)

def hold_read_only(lease):
    """Extend the existing Windows deny-write/delete lease to this cached wheel."""
    import ctypes
    path=archive_path();handle=lease.api.CreateFileW(str(path),0x80000000,1,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError()
    lease.handles.append(handle)
    return authenticate()

def audit_imports(kitchen_archive,kitchen_pin):
    """Static audit of all Kitchen/overlay Python files; never import Torch."""
    kitchen_archive=plain(kitchen_archive)
    # Authentication precedes ZIP interpretation, using the same verifier as live.
    manifest=wheel_manifest(kitchen_archive,kitchen_pin,kitchen_archive.parent,kitchen_archive.parent)
    authenticate();rows=[];third_party=set();optional=[];dynamic=[];python_files=0
    for path,internal in ((kitchen_archive,'comfy_kitchen'),(archive_path(),'packaging')):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith('.py'):continue
                python_files+=1;tree=ast.parse(archive.read(name),filename=name)
                for node in ast.walk(tree):
                    roots=[]
                    if isinstance(node,ast.Import):roots=[a.name.split('.')[0] for a in node.names]
                    if isinstance(node,ast.ImportFrom) and not node.level and node.module:roots=[node.module.split('.')[0]]
                    for root in roots:
                        if root in sys.stdlib_module_names or root==internal:continue
                        row=dict(file=name,line=node.lineno,module=root)
                        if (root=='triton' and name.startswith('comfy_kitchen/backends/triton/')) or (root=='torch_npu' and name.startswith('comfy_kitchen/backends/ascend/')) or (root=='nvidia' and name.startswith('comfy_kitchen/backends/cuda/')) or (root=='_manylinux' and name=='packaging/_manylinux.py'):
                            optional.append(row)
                        else:third_party.add(root);rows.append(row)
                    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='__import__':dynamic.append(dict(file=name,line=node.lineno,expression=ast.unparse(node)))
    if third_party!={'torch','packaging'}:raise RuntimeError('Unexpected required third-party runtime roots: '+str(sorted(third_party)))
    return dict(status='STATIC_AUDIT_PASS',python_files=python_files,required_third_party=sorted(third_party),imports=rows,
        guarded_optional_backend_imports=optional,dynamic_import_sites=dynamic,
        limitation='Static Kitchen/packaging audit; not proof of full Linux Torch/native import execution')
