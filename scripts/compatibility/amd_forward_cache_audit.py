"""Read historical compiler artifacts as inert data; stdlib only, no AI imports."""
import argparse
import bz2
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import struct

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'docs/compatibility/amd-sparse-decoder-forward-correction-evidence'
DESTINATION=ROOT/'docs/compatibility/amd-forward-timeout-investigation-evidence'


def elf_header(data):
    if len(data)<64 or data[:6]!=b'\x7fELF\x02\x01': return None
    return {'class':64,'little_endian':True,'machine':struct.unpack_from('<H',data,18)[0],
            'flags':hex(struct.unpack_from('<I',data,48)[0])}


def audit(output):
    output=output.resolve(); output.relative_to(DESTINATION.resolve())
    saved=json.loads((OLD/'scratch-side-effects.json').read_text())
    scratch=OLD/'run/scratch'
    rows=[];formats=Counter();flags=Counter();times=[]
    actual={p.relative_to(scratch).as_posix() for p in scratch.rglob('*') if p.is_file()}
    if actual!={row['path'] for row in saved['files']}: raise RuntimeError('Historical cache file set changed')
    for original in saved['files']:
        path=scratch/original['path']; before=path.stat(); data=path.read_bytes()
        if len(data)!=original['bytes'] or hashlib.sha256(data).hexdigest()!=original['sha256']:
            raise RuntimeError('Historical artifact identity changed: '+original['path'])
        row={**original,'mtime_ns':before.st_mtime_ns}
        if path.name.startswith('llvmcache-'):
            n=struct.unpack_from('<Q',data)[0]
            if n>len(data)-16: raise RuntimeError('Truncated COMGR payload')
            log_length=struct.unpack_from('<Q',data,8+n)[0]
            if 16+n+log_length!=len(data): raise RuntimeError('COMGR length fields do not cover file')
            payload=data[8:8+n]
            offset=0 if payload[:4]==b'\x7fELF' else 8 if payload[8:12]==b'\x7fELF' else None
            header=elf_header(payload[offset:]) if offset is not None else None
            kind='ELF' if header else 'LLVM-bitcode' if payload[:4]==b'BC\xc0\xde' else 'unclassified'
            row.update(serialized_payload_bytes=n,compiler_log_bytes=log_length,kind=kind,
                       elf_offset_in_payload=offset,elf=header)
            formats[kind]+=1
            if header: flags[header['flags']]+=1
            times.append(before.st_mtime_ns)
        elif original['path'].startswith('tmp/'):
            row['header_hex']=data[:40].hex();row['elf']=elf_header(data)
            row['container_magic']='CCOB' if data[:4]==b'CCOB' else None
        if path.stat().st_mtime_ns!=before.st_mtime_ns: raise RuntimeError('Artifact changed while reading')
        rows.append(row)
    db=next((scratch/'miopen_custom_cache_dir').rglob('*.ukdb'))
    # Immutable/read-only avoids journal, lock, recovery or metadata writes.
    connection=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        connection.execute('PRAGMA query_only=ON')
        check=connection.execute('PRAGMA quick_check').fetchall()
        schema=connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table'").fetchall()
        kernels=[]
        for ident,name,args,key,size,blob in connection.execute('SELECT id,kernel_name,kernel_args,kernel_hash,uncompressed_size,kernel_blob FROM kern_db ORDER BY id'):
            if size>2*1024**2: raise RuntimeError('Kernel metadata inspection exceeds scope')
            decoder=bz2.BZ2Decompressor();unpacked=decoder.decompress(blob,max_length=size+1)
            if not decoder.eof or decoder.unused_data or len(unpacked)!=size: raise RuntimeError('Kernel blob size/compression invalid')
            kernels.append({'id':ident,'name':name,'arguments':args,'recorded_hash':key,'uncompressed_bytes':size,
                            'compressed_bytes':len(blob),'uncompressed_sha256':hashlib.sha256(unpacked).hexdigest(),
                            'elf':elf_header(unpacked)})
    finally: connection.close()
    after={p.relative_to(scratch).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in scratch.rglob('*') if p.is_file()}
    if after!={row['path']:row['sha256'] for row in saved['files']}: raise RuntimeError('Read-only cache audit changed historical artifacts')
    result={'historical_manifest_verified':True,'read_only':True,'executed_artifacts':0,'file_count':len(rows),
            'total_bytes':sum(row['bytes'] for row in rows),'comgr_directory_files':len(list((scratch/'amd_comgr_cache_dir').iterdir())),
            'comgr_entry_count':len(times),'comgr_formats':dict(formats),'comgr_elf_flags':dict(flags),
            'comgr_mtime_span_seconds':(max(times)-min(times))/1e9,
            'comgr_first_mtime_utc':datetime.datetime.fromtimestamp(min(times)/1e9,datetime.timezone.utc).isoformat(),
            'comgr_last_mtime_utc':datetime.datetime.fromtimestamp(max(times)/1e9,datetime.timezone.utc).isoformat(),
            'miopen':{'path':str(db),'open_mode':'ro,immutable,query_only','quick_check':check,'schema':schema,'kernels':kernels},
            'process_evidence':json.loads((OLD/'run/results.json').read_text())['process'],'files':rows}
    with output.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({key:value for key,value in result.items() if key not in ('files','process_evidence','miopen')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output)
