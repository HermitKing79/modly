"""Hash saved diagnostic evidence only; no external artifacts are executed."""
import hashlib
from pathlib import Path

p=Path(__file__).parent
rows=[hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.name
      for path in sorted(p.iterdir()) if path.is_file() and path.suffix in ('.json','.gz','.txt')]
with (p/'SHA256SUMS.sha256').open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(rows)+'\n')
print('Saved checksums for',len(rows),'evidence files')
