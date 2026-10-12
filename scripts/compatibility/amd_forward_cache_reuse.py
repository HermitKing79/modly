"""Conservative inert cache inspection and private copies. Never loads binaries."""
import argparse
import base64
import bz2
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import struct

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'docs/compatibility/amd-instrumented-forward-evidence'
HISTORY=ROOT/'docs/compatibility/amd-sparse-decoder-forward-correction-evidence'
SITE=Path('D:/AI/Trellis2AMD/venv/Lib/site-packages')
DATABASE='miopen_custom_cache_dir/3.6.0.8d1ae90e/gfx1201_32.ukdb'


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def validate_elf(data):
    if len(data)<64 or data[:7]!=b'\x7fELF\x02\x01\x01':raise ValueError('Incomplete/unknown ELF header')
    kind,machine,version,entry,phoff,shoff,flags,ehsize,phsize,phnum,shsize,shnum,strings=struct.unpack_from('<HHIQQQIHHHHHH',data,16)
    if kind not in (1,3) or machine!=224 or version!=1 or flags!=0x4e or ehsize!=64:
        raise ValueError('ELF is not the inspected AMD gfx1201 object')
    if not 0<shnum<=8192 or shsize!=64 or shoff<64 or shoff+shsize*shnum>len(data):
        raise ValueError('Incomplete/extended section table')
    if strings>=shnum or phnum>128 or (phnum and (phsize!=56 or phoff<64 or phoff+phsize*phnum>len(data))):
        raise ValueError('Incomplete/extended ELF tables')
    for index in range(phnum):
        row=struct.unpack_from('<IIQQQQQQ',data,phoff+index*phsize)
        if row[2]+row[5]>len(data):raise ValueError('Truncated program data')
    for index in range(shnum):
        row=struct.unpack_from('<IIQQQQIIQQ',data,shoff+index*shsize)
        if row[1]!=8 and row[4]+row[5]>len(data):raise ValueError('Truncated section data')
    return {'architecture':'gfx1201','machine':machine,'flags':hex(flags),'section_count':shnum,'program_count':phnum}


def inspect_comgr(data):
    try:
        if len(data)<16:raise ValueError('Incomplete COMGR framing')
        size=struct.unpack_from('<Q',data)[0]
        if size>len(data)-16:raise ValueError('Incomplete COMGR payload')
        log=struct.unpack_from('<Q',data,8+size)[0]
        if 16+size+log!=len(data) or log:raise ValueError('Incomplete/trailing/nonempty-log COMGR entry')
        payload=data[8:8+size]
        if not payload.startswith(b'\x7fELF'):raise ValueError('Uncertain inner framing/bitcode/empty payload excluded')
        return {'eligible':True,**validate_elf(payload),'reason':'complete direct ELF in verified outer container'}
    except (ValueError,struct.error) as exc:return {'eligible':False,'reason':str(exc)}


def cache_path(name):
    if not isinstance(name,str) or not (re.fullmatch(r'amd_comgr_cache_dir/llvmcache-[0-9a-f]{64}',name) or name==DATABASE):
        raise ValueError('Unapproved cache path')
    return Path(name)


def verify_copies(root,rows):
    expected={row['path'] for row in rows}
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=expected:raise RuntimeError('Private cache file set changed')
    for row in rows:
        path=root/cache_path(row['path']);path.resolve().relative_to(root.resolve())
        if not row['eligible'] or path.is_symlink() or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise RuntimeError('Private cache identity changed: '+row['path'])


def copy_entries(source,target,rows):
    for row in rows:
        if not row.get('eligible'):raise ValueError('Cannot copy an uncertain cache entry')
        path=source/cache_path(row['path']);path.resolve().relative_to(source.resolve())
        if path.is_symlink() or digest(path)!=row['sha256']:raise RuntimeError('Source cache hash mismatch')
    if target.resolve().is_relative_to(source.resolve()):raise ValueError('Do not write into cache source')
    target.mkdir(parents=True,exist_ok=False)
    for row in rows:
        path=target/cache_path(row['path']);path.parent.mkdir(parents=True,exist_ok=True)
        with (source/row['path']).open('rb') as incoming,path.open('xb') as outgoing:
            while chunk:=incoming.read(1024*1024):outgoing.write(chunk)
    verify_copies(target,rows)


def attest_runtime(site,historical):
    distributions={row['name'].lower().replace('_','-'):row for row in historical['amd']['all_distributions']}
    files={'rocm-sdk-core':('_rocm_sdk_core/bin/amd_comgr.dll','_rocm_sdk_core/bin/amdhip64_7.dll','_rocm_sdk_core/bin/hiprtc0715.dll'),
           'rocm-sdk-libraries':('_rocm_sdk_libraries/bin/MIOpen.dll',)}
    result={}
    for package,names in files.items():
        old=distributions.get(package)
        if not old or old['version']!='10.0.0':raise RuntimeError('Historical ROCm package identity unavailable')
        record=site/(package.replace('-','_')+'-10.0.0.dist-info/RECORD')
        text=record.read_text()
        if hashlib.sha256(text.encode()).hexdigest()!=old['record_sha256']:raise RuntimeError('Installed ROCm RECORD changed')
        entries={row[0]:row[1:] for row in csv.reader(io.StringIO(text))}
        for name in names:
            signature,size=entries[name]
            if not signature.startswith('sha256='):raise RuntimeError('Native library lacks SHA256 package attestation')
            path=site/name;actual=digest(path)
            encoded=base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip('=')
            if signature!='sha256='+encoded or path.stat().st_size!=int(size):raise RuntimeError('Native library differs from unchanged package RECORD')
            result[name]={'sha256':actual,'bytes':int(size),'package':package,'version':old['version'],'record_sha256':old['record_sha256']}
    return result


def validate_database(path,expected):
    connection=sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        connection.execute('PRAGMA query_only=ON')
        if connection.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise RuntimeError('MIOpen database invalid')
        rows=connection.execute('SELECT id,kernel_name,kernel_args,kernel_hash,uncompressed_size,kernel_blob FROM kern_db ORDER BY id').fetchall()
        if len(rows)!=2 or len(expected)!=2:raise RuntimeError('Unexpected MIOpen kernel set')
        for row,old in zip(rows,expected):
            ident,name,args,key,size,blob=row
            if (ident,name,args,key,size)!=(old['id'],old['name'],old['arguments'],old['recorded_hash'],old['uncompressed_bytes']) or not 0<size<=2*1024**2:
                raise RuntimeError('MIOpen record identity changed')
            decoder=bz2.BZ2Decompressor();data=decoder.decompress(blob,max_length=size+1)
            if not decoder.eof or decoder.unused_data or len(data)!=size or hashlib.sha256(data).hexdigest()!=old['uncompressed_sha256']:
                raise RuntimeError('MIOpen kernel compression/content invalid')
            validate_elf(data)
            if '-mcpu=gfx1201' not in args:raise RuntimeError('MIOpen target mismatch')
    finally:connection.close()


def prepare():
    baseline=ROOT/'docs/compatibility/amd-forward-timeout-investigation-evidence/inventory-before.json'
    current=EVIDENCE/'inventory-before.json'
    if current.read_bytes()!=baseline.read_bytes():raise RuntimeError('Runtime/source metadata changed; cache reuse blocked')
    historical=json.loads(current.read_text());runtime=attest_runtime(SITE,historical)
    audit=json.loads((ROOT/'docs/compatibility/amd-forward-timeout-investigation-evidence/cache-audit.json').read_text())
    manifest=json.loads((HISTORY/'scratch-side-effects.json').read_text());source=HISTORY/'run/scratch'
    actual={path.relative_to(source).as_posix() for path in source.rglob('*') if path.is_file()}
    if actual!={row['path'] for row in manifest['files']}:raise RuntimeError('Historical file set changed')
    selected=[];excluded=[]
    for row in manifest['files']:
        path=source/row['path'];path.resolve().relative_to(source.resolve())
        if path.is_symlink() or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise RuntimeError('Historical cache changed')
        decision=inspect_comgr(path.read_bytes()) if row['path'].startswith('amd_comgr_cache_dir/llvmcache-') else {'eligible':False,'reason':'temporary, lock, timestamp or unrelated output'}
        if row['path']==DATABASE:
            validate_database(path,audit['miopen']['kernels']);decision={'eligible':True,'reason':'two unchanged complete gfx1201 MIOpen records'}
        target=selected if decision['eligible'] else excluded;target.append({**row,**decision})
    if not selected or not any(row['path']==DATABASE for row in selected):raise RuntimeError('No defensible cache reuse set')
    copy_entries(source,EVIDENCE/'prepared-cache',selected)
    for row in selected:
        if digest(source/row['path'])!=row['sha256']:raise RuntimeError('Source changed during copying')
    plan={'selected':selected,'excluded':excluded,'runtime_libraries':runtime,
          'historical_manifest_sha256':digest(HISTORY/'scratch-side-effects.json'),
          'inventory_sha256':digest(current),'historical_files_verified':len(actual),'executed_artifacts':0}
    with (EVIDENCE/'cache-plan.json').open('x',encoding='utf-8') as stream:json.dump(plan,stream,indent=2)
    print(json.dumps({'selected':len(selected),'excluded':len(excluded),'runtime_libraries_attested':len(runtime),'executed_artifacts':0}))


if __name__=='__main__':prepare()
