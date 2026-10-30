"""Read-only ext4 metadata interpretation; no children, privilege or repair path."""
import hashlib,os,re,stat,struct
GIB=1024**3

def require(ok,message):
    if not ok:raise ValueError(message)

def crc32c(data):
    value=0xffffffff
    for byte in data:
        value^=byte
        for _ in range(8):value=(value>>1)^(0x82f63b78 if value&1 else 0)
    return value

def parse_superblock(data,device_bytes):
    require(len(data)==1024,'Incomplete primary superblock')
    u32=lambda offset:struct.unpack_from('<I',data,offset)[0]
    require(struct.unpack_from('<H',data,56)[0]==0xef53,'ext4 magic differs')
    log=u32(24);require(log<=6,'Unsupported ext4 block size')
    high=u32(0x150);wide=bool(u32(0x60)&0x80)
    require(wide or high==0,'High block count without 64bit feature')
    count=u32(4)+(high<<32 if wide else 0);size=1024<<log
    require(count>0 and count*size<=device_bytes,'Superblock capacity exceeds device')
    checked=bool(u32(0x64)&0x400)
    if checked:
        require(data[0x175]==1,'Unsupported metadata checksum type')
        require(crc32c(data[:1020])==u32(1020),'Primary superblock checksum differs')
    return dict(status='PRESENT',block_count=count,block_size_bytes=size,filesystem_bytes=count*size,
                checksum_verified=checked,sha256=hashlib.sha256(data).hexdigest())

def read_superblock(path,device,device_bytes):
    require(re.fullmatch(r'/dev/[A-Za-z0-9._-]+',path) is not None,'Noncanonical root device path')
    try:fd=os.open(path,os.O_RDONLY|getattr(os,'O_CLOEXEC',0)|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    except (PermissionError,FileNotFoundError) as exc:
        return dict(status='UNAVAILABLE',reason=type(exc).__name__,errno=exc.errno,path=path)
    try:
        before=os.fstat(fd)
        require(stat.S_ISBLK(before.st_mode) and before.st_rdev==device,'Root device handle identity differs')
        first=parse_superblock(os.pread(fd,1024,1024),device_bytes)
        second=parse_superblock(os.pread(fd,1024,1024),device_bytes)
        after=os.fstat(fd)
        require((before.st_dev,before.st_ino,before.st_rdev)==(after.st_dev,after.st_ino,after.st_rdev),'Device identity changed')
        require(all(first[k]==second[k] for k in ('block_count','block_size_bytes','filesystem_bytes','checksum_verified')),'Filesystem geometry changed during query')
        return dict(first,path=path,second_sha256=second['sha256'])
    finally:os.close(fd)

def assess(row):
    for key in ('root_device_bytes','usable_bytes','available_bytes','fragment_bytes','block_size_bytes','total_blocks','free_blocks','available_blocks'):
        require(type(row.get(key)) is int and row[key]>=0,'Missing/invalid capacity: '+key)
    require(row.get('ext4_rw') is True,'Root ext4 read/write state unverified')
    require(row['root_device_bytes']==64*GIB,'Root device not exactly 64 GiB')
    unit=row['fragment_bytes'];require(unit>0 and row['block_size_bytes']>0,'Invalid statvfs block units')
    require(0<=row['available_blocks']<=row['free_blocks']<=row['total_blocks'],'Inconsistent statvfs counts')
    require(row['usable_bytes']==row['total_blocks']*unit and row['available_bytes']==row['available_blocks']*unit,'statvfs byte/count mismatch')
    require(0<row['usable_bytes']<=row['root_device_bytes'],'Filesystem usable capacity invalid')
    raw=row.get('superblock',{});exact=None
    if raw.get('status')=='PRESENT':
        exact=raw.get('filesystem_bytes')
        require(type(exact) is int and 0<exact<=64*GIB and row['usable_bytes']<=exact,'Superblock/statvfs disagreement')
        require(raw.get('block_size_bytes')==row['block_size_bytes'],'Superblock/statvfs block size differs')
        expansion=('FULL_64_GIB_CONFIRMED' if exact==64*GIB else 'PARTIAL_EXPANSION_CONFIRMED' if exact>32*GIB else 'UNCHANGED_32_GIB' if exact==32*GIB else 'BELOW_ORIGINAL_CAPACITY')
    else:
        require(raw.get('status')=='UNAVAILABLE','Missing exact-geometry result')
        expansion='EXPANDED_BEYOND_32_GIB_EXACT_UNAVAILABLE' if row['usable_bytes']>32*GIB else 'EXACT_GEOMETRY_UNAVAILABLE'
    disk='PASS' if row['available_bytes']>=5*GIB else 'FAIL'
    go=disk=='PASS' and (exact==64*GIB or exact is None)
    return dict(expansion=expansion,exact_filesystem_bytes=exact,exact_64GiB_confirmed=exact==64*GIB,
                decoder_disk_gate=disk,recommendation='GO_FOR_DECODE_PREPARATION_ONLY' if go else 'NO_GO_REVIEW',
                no_decode_authorised=True)
