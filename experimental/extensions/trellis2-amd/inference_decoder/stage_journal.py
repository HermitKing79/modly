"""Mirror only the existing durable stage journal for the Windows deadline monitor."""
import os
from pathlib import Path
import stat
import sys

MAX_JOURNAL_BYTES=16*1024**2
MAX_EVENT_BYTES=128*1024


def attest_directory(path):
    path=Path(path)
    if not path.is_absolute():raise ValueError('Absolute pre-existing journal directory required')
    for parent in (*reversed(path.parents),path):
        info=parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info,'st_file_attributes',0)&0x400):
            raise ValueError('Journal parents must be plain directories without links or reparse points')
    if path.resolve(strict=True)!=path:
        raise ValueError('Journal directory resolves to a different location')
    return path


class StageJournal:
    def __init__(self,base,output,host_directory=None):
        self.base=base;self.output=Path(output).resolve(strict=True)
        self.host=None if host_directory is None else attest_directory(host_directory)
        self.offset=0
        if (self.output/'phases.jsonl').exists():
            raise FileExistsError('Fresh local stage journal required')
        if self.host is not None:
            if any(self.host.iterdir()):raise FileExistsError('Fresh empty host journal directory required')
            self.path=self.host/'phases.jsonl'
            with self.path.open('xb') as stream:stream.flush();os.fsync(stream.fileno())
            info=self.path.lstat();self.identity=(info.st_dev,info.st_ino)

    @property
    def writable_roots(self):
        return () if self.host is None else (self.host,)

    def audit(self,event,args):
        """The broader import policy grants this directory only for this file."""
        if self.host is None:return
        if event=='open':
            path,mode,flags=args
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(
                flags and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if not writing:return
            paths=(path,)
        elif event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.chown','os.utime'):
            paths=args[:1]
        elif event in ('os.rename','os.link','os.symlink'):
            paths=args[:2]
        else:return
        for value in paths:
            if isinstance(value,int):
                if sys.platform!='linux':continue
                try:value=os.readlink('/proc/self/fd/'+str(value))
                except OSError:continue
            path=Path(os.fsdecode(value)).resolve()
            if path.is_relative_to(self.host):
                append=(isinstance(mode,str) and mode.startswith('a')) or bool(flags and flags&os.O_APPEND) if event=='open' else False
                if event!='open' or path!=self.path or not append or flags and flags&os.O_TRUNC:
                    raise RuntimeError('Host evidence permits only phases.jsonl journal appends')

    def install_guard(self):
        if self.host is not None:sys.addaudithook(self.audit)

    def __call__(self,output,phase,**detail):
        if Path(output).resolve(strict=True)!=self.output:
            raise ValueError('Stage journal output identity changed')
        self.base.journal(self.output,phase,**detail)
        if self.host is None:return
        attest_directory(self.host)
        info=self.path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink!=1
                or (info.st_dev,info.st_ino)!=self.identity or info.st_size!=self.offset):
            raise RuntimeError('Host stage journal file identity or length changed')
        with (self.output/'phases.jsonl').open('rb') as stream:
            stream.seek(self.offset);block=stream.read(MAX_EVENT_BYTES+1)
        if (not block or len(block)>MAX_EVENT_BYTES or not block.endswith(b'\n')
                or self.offset+len(block)>MAX_JOURNAL_BYTES):
            raise RuntimeError('Stage event exceeds bounded journal bridge')
        with self.path.open('ab') as stream:
            stream.write(block);stream.flush();os.fsync(stream.fileno())
        self.offset+=len(block)
