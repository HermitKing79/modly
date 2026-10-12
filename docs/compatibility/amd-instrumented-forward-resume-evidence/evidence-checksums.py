"""Hash saved evidence, excluding preserved local binary cache directories."""
import hashlib
from pathlib import Path

p=Path(__file__).parent;rows=[]
for path in sorted(p.rglob('*')):
    relative=path.relative_to(p)
    if path.is_file() and path.suffix in ('.json','.gz','.txt') and relative.parts[0]!='prepared-cache' and relative.parts[:2]!=('run','scratch'):
        rows.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+relative.as_posix())
with (p/'SHA256SUMS.sha256').open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(rows)+'\n')
print('Saved',len(rows),'evidence checksums')
