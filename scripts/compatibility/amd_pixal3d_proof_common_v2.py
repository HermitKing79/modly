"""V2 policy: original pins/guards, immutable v1 cache, exclusive new roots."""
import hashlib,json,os
from pathlib import Path
import amd_pixal3d_proof_common as original
from amd_pixal3d_proof_common import ROOT,GIB,OPERATORS,now,digest,plain,inside,save,journal,require_ram,require_resources,admit_once,verify,hip_payload,bounded_expansion
CONTRACT=ROOT/'docs/compatibility/amd-pixal3d-standalone-hip-v2-contract.json'

def lock():
    value=json.loads(CONTRACT.read_text(encoding='utf-8'));old=original.lock()
    if value['schema']!='modly.pixal3d.standalone.model-free-hip.v2':raise ValueError('Unknown v2 contract')
    for key in ('runtime','packages','sources','guards','kitchen_HIP_source_sha256','provision_seconds','operator_seconds','cleanup_seconds',
                'download_limit_bytes','guest_growth_limit_bytes','guest_disk_floor_bytes','D_disk_floor_bytes','windows_python',
                'windows_python_sha256','distro','guest_user','guest_python','approved_contract','approved_contract_canonical_sha256'):
        if value[key]!=old[key]:raise ValueError('Original pin/guard changed: '+key)
    if (value['windows_root']!='D:/AI/Modly-Pixal3D-Proof-v2' or value['guest_root']!='/home/modly/pixal3d-native-proof-v2'
        or value['cache_root']!='D:/AI/Modly-Pixal3D-Proof-v1'):raise ValueError('Exact v2 isolation/cache paths required')
    if value['downloads_allowed'] or value['model_downloads_allowed'] or value['generation_authorized']:raise ValueError('Cache-only model-free scope required')
    return value

def cache_files(cfg):
    root=plain(cfg['cache_root'])
    return [(inside(root,'downloads/'+row['filename']),row) for row in cfg['packages']]+[(inside(root,'sources/'+row['path']),row) for row in cfg['sources']]

def identity(path):
    s=plain(path).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,file_id=s.st_ino)

def verify_cache(cfg,progress=None):
    rows=[]
    for path,row in cache_files(cfg):
        before=identity(path)
        if before['bytes']!=row['bytes']:raise RuntimeError('Cached size differs: '+path.name)
        sha=hashlib.sha256()
        with path.open('rb') as stream:
            while block:=stream.read(1024**2):
                sha.update(block)
                if progress:progress()
        if sha.hexdigest()!=row['sha256'] or before!=identity(path):raise RuntimeError('Cached hash/identity differs: '+path.name)
        rows.append(dict(path=str(path),**before,sha256=sha.hexdigest()))
    return dict(files_verified=len(rows),bytes_verified=sum(r['bytes'] for r in rows),files=rows,archives_copied=0,downloads=0)

def cache_policy(root):
    root=Path(root).absolute()
    def protected(value):
        if not isinstance(value,(str,bytes,os.PathLike)):return False
        path=Path(os.fsdecode(value)).resolve()
        return path.is_relative_to(root)
    def audit(event,args):
        if event=='open' and protected(args[0]):
            if any(v in str(args[1] or '') for v in 'wax+') or args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):raise RuntimeError('Immutable cache write denied')
        if event in ('os.mkdir','os.remove','os.rmdir','os.truncate','os.utime','os.chmod','os.chown') and protected(args[0]):raise RuntimeError('Immutable cache mutation denied')
        if event in ('os.rename','os.link','os.symlink') and any(protected(v) for v in args[:2]):raise RuntimeError('Immutable cache rename/link denied')
    return audit

class CacheReadLease:
    """Hold Windows read-only, deny-write/delete handles across guest installation."""
    def __init__(self,cfg):self.cfg=cfg;self.handles=[]
    def __enter__(self):
        import ctypes
        from ctypes import wintypes as w
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        self.api.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,ctypes.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
        self.api.CreateFileW.restype=w.HANDLE;self.api.CloseHandle.argtypes=[w.HANDLE]
        try:
            for path,_ in cache_files(self.cfg):
                handle=self.api.CreateFileW(str(path),0x80000000,1,None,3,0x80,None)
                if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError()
                self.handles.append(handle)
        except BaseException:self.__exit__(None,None,None);raise
        return self
    def __exit__(self,*args):
        for handle in self.handles:self.api.CloseHandle(handle)
        self.handles=[]
