import hashlib,json,importlib.metadata as md,os,subprocess,sys
from pathlib import Path
E=Path(__file__).resolve().parent
def deny(event,args):
 if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm'}:raise RuntimeError('Source read forbids AI imports')
sys.addaudithook(deny)
T=Path('/opt/modly-wsl/venv/lib/python3.12/site-packages/torch');rows={}
for rel in ('utils/_device.py','overrides.py','__init__.py'):
 p=T/rel;data=p.read_bytes();copy=E/('installed-torch-'+rel.replace('/','-')+'.txt')
 with copy.open('xb') as f:f.write(data)
 rows[rel]={'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'copy':copy.name}
root=Path('/opt/modly-wsl/venv');tree=[]
for d,dirs,files in os.walk(root,followlinks=False):
 for name in dirs+files:
  p=Path(d)/name;s=p.lstat();tree.append([p.relative_to(root).as_posix(),s.st_size,s.st_mtime_ns,s.st_mode])
record={'sources':rows,'venv_metadata_sha256':hashlib.sha256(json.dumps(sorted(tree),separators=(',',':')).encode()).hexdigest(),'python_packages':sorted([[d.metadata['Name'],d.version] for d in md.distributions(path=[str(T.parent)])]),'dpkg_packages':subprocess.check_output(['dpkg-query','-W','-f=${Package}\t${Version}\n'],text=True,timeout=15).splitlines(),'torch_imported':False,'GPU_actions':False}
with (E/'installed-source-before.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps({'sources':rows,'venv_sha256':record['venv_metadata_sha256'],'GPU_actions':False}))
