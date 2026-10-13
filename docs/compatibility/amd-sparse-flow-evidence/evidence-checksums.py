"""Hash published evidence; scratch products are retained locally, never executed."""
import hashlib
from pathlib import Path

root=Path(__file__).parent
target=root/'SHA256SUMS.sha256'
rows=[]
for path in sorted(root.rglob('*')):
    if path.is_file() and path!=target and 'scratch' not in path.relative_to(root).parts:
        rows.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.relative_to(root).as_posix())
with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(rows)+'\n')
print('Hashed',len(rows),'evidence files')
