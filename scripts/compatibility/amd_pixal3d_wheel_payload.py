"""Read-only wheel authentication: pinned archive bytes, then exact ZIP payloads.

RECORD is retained as metadata, never used as an alternative file identity.
Only the top-level RECORD is pip-rewritten; vendored RECORDs are payload files.
"""
import base64,csv,hashlib,io,os,stat,zipfile
from contextlib import contextmanager
from pathlib import Path
from amd_pixal3d_proof_common import plain

BLOCK=1024**2

def identity(s):
    # Python 3.12 Windows lstat/fstat expose different legacy ctime semantics.
    return (s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns)+( (s.st_ctime_ns,) if os.name=='posix' else () )

@contextmanager
def readonly(path):
    path=plain(path);before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:raise RuntimeError('Regular unaliased file required: '+str(path))
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_BINARY',0))
    with os.fdopen(fd,'rb') as stream:
        if identity(before)!=identity(os.fstat(stream.fileno())):raise RuntimeError('File changed before read: '+str(path))
        yield stream
        if identity(before)!=identity(os.fstat(stream.fileno())) or identity(before)!=identity(plain(path).lstat()):raise RuntimeError('File changed during read: '+str(path))

def stream_identity(stream,progress=None):
    sha=hashlib.sha256();size=0
    while block:=stream.read(BLOCK):
        sha.update(block);size+=len(block)
        if progress:progress()
    return dict(bytes=size,sha256=sha.hexdigest())

def member_name(name):
    if not isinstance(name,str) or not name or any(c in name for c in ('\\',':','\x00')) or name.startswith('/') or any(p in ('','.','..') for p in name.split('/')):
        raise ValueError('Unsafe wheel payload path: '+str(name))
    return name

def installed_path(name,env,site,*,check_filesystem=True):
    member_name(name);parts=name.split('/')
    if parts[0].endswith('.data'):
        if len(parts)<3:raise ValueError('Incomplete wheel spread path')
        if parts[1] in ('purelib','platlib'):path=site.joinpath(*parts[2:])
        elif parts[1]=='data':path=env.joinpath(*parts[2:])
        else:raise ValueError('Unsupported wheel spread category; no guessing')
    else:path=site/name
    # Manifest construction is lexical; roots are checked once by its caller.
    # Actual installed-file reads always recheck every parent with plain().
    if check_filesystem:path=plain(path);env=plain(env)
    path.relative_to(env);return path

def claim_path(claims,path,owner):
    # normcase also rejects aliases when fixtures are checked on Windows.
    key=os.path.normcase(str(path))
    if key in claims or ('directory',key) in claims or any(os.path.normcase(str(p)) in claims for p in path.parents):raise RuntimeError('Wheel installed-path collision: '+str(path))
    claims[key]=owner
    for parent in path.parents:claims[('directory',os.path.normcase(str(parent)))]=True

def claim_directory(claims,path):
    if any(os.path.normcase(str(p)) in claims for p in (path,*path.parents)):raise RuntimeError('Wheel directory/file collision: '+str(path))
    for parent in (path,*path.parents):claims[('directory',os.path.normcase(str(parent)))]=True

def record_rows(stream):
    rows={}
    for row in csv.reader(io.TextIOWrapper(stream,encoding='utf-8',newline='')):
        if len(row)!=3:raise RuntimeError('Malformed distribution RECORD')
        name,encoded,size=row;member_name(name)
        if name in rows:raise RuntimeError('Duplicate RECORD path: '+name)
        rows[name]=dict(encoded_sha256=encoded,bytes=size)
    return rows

def discrepancies(files,rows,record):
    result=[]
    for name,row in rows.items():
        if name==record and not row['encoded_sha256'] and not row['bytes']:continue
        payload=files.get(name)
        if payload is None:result.append(dict(kind='RECORD_path_absent_from_payload',path=name,RECORD=row));continue
        encoded='sha256='+base64.urlsafe_b64encode(bytes.fromhex(payload['sha256'])).rstrip(b'=').decode()
        if row!=dict(encoded_sha256=encoded,bytes=str(payload['bytes'])):
            result.append(dict(kind='metadata_differs',path=name,RECORD=row,payload=payload))
    result.extend(dict(kind='payload_absent_from_RECORD',path=name,payload=row) for name,row in files.items() if name not in rows)
    return result

def wheel_manifest(archive,pin,env,site,progress=None,claims=None):
    """Authenticate on the same open handle before interpreting any ZIP member."""
    files={};destinations={};claims={} if claims is None else claims
    env=plain(env);site=plain(site);site.relative_to(env)
    with readonly(archive) as raw:
        whole=stream_identity(raw,progress)
        if whole!=dict(bytes=pin['bytes'],sha256=pin['sha256']):raise RuntimeError('Pinned wheel archive identity differs: '+Path(archive).name)
        raw.seek(0)
        with zipfile.ZipFile(raw) as z:
            seen=set();expanded=0
            for item in z.infolist():
                if item.orig_filename!=item.filename:raise ValueError('Truncated ZIP payload name')
                name=member_name(item.filename[:-1] if item.is_dir() else item.filename)
                if name in seen:raise RuntimeError('Duplicate ZIP path: '+name)
                seen.add(name);mode=item.external_attr>>16;kind=stat.S_IFMT(mode)
                if kind not in (0,stat.S_IFDIR if item.is_dir() else stat.S_IFREG):raise RuntimeError('Non-regular ZIP payload: '+name)
                destination=installed_path(name,env,site,check_filesystem=False)
                if item.is_dir():claim_directory(claims,destination);continue
                claim_path(claims,destination,dict(archive=Path(archive).name,member=name));destinations[name]=destination
                expanded+=item.file_size
                if expanded>20*1024**3:raise RuntimeError('Original isolated growth budget exceeded')
                with z.open(item) as stream:row=stream_identity(stream,progress)
                if row['bytes']!=item.file_size:raise RuntimeError('ZIP payload size differs: '+name)
                files[name]=row
            records=[n for n in files if len(n.split('/'))==2 and n.endswith('.dist-info/RECORD')]
            expected='-'.join(Path(archive).name.split('-')[:2])+'.dist-info/RECORD'
            if records!=[expected]:raise RuntimeError('Exact top-level distribution RECORD required')
            record=records[0]
            with z.open(record) as stream:rows=record_rows(stream)
    return dict(archive=Path(archive).name,archive_identity=whole,files=files,destinations=destinations,record=record,
        RECORD_discrepancies=discrepancies(files,rows,record),expanded_bytes=expanded)

def verify_manifest(manifest,env,site,progress=None,trusted_paths=None):
    files=0;total=0;hint_errors=[];installed_record=None
    for name,expected in manifest['files'].items():
        path=installed_path(name,env,site)
        if trusted_paths is not None and path.is_relative_to(site):trusted_paths.add(path.relative_to(site).as_posix())
        with readonly(path) as stream:
            actual=stream_identity(stream,progress)
            if name==manifest['record']:
                installed_record=dict(path=name,**actual,pip_rewritten_metadata=True)
            elif actual!=expected:raise RuntimeError('Installed payload identity differs: '+name)
            else:files+=1;total+=actual['bytes']
            # Same read-only cache hint as V3; neither a write nor a recovery guarantee.
            if hasattr(os,'posix_fadvise'):
                try:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                except OSError as exc:hint_errors.append(dict(file=name,error=str(exc)))
        if progress:progress()
    return dict(files_verified=files,bytes_verified=total,archive=manifest['archive'],archive_identity=manifest['archive_identity'],
        RECORD_discrepancies=manifest['RECORD_discrepancies'],installed_RECORD=installed_record,
        read_only_cache_hint_errors=hint_errors,physical_recovery_guaranteed=False)

def verify_installed_wheel(archive,env,site,progress=None,trusted_paths=None,*,expected,claims=None):
    return verify_manifest(wheel_manifest(archive,expected,env,site,progress,claims),env,site,progress,trusted_paths)
