import hashlib,importlib.metadata as md,json,os,subprocess,sys
from pathlib import Path
E=Path(__file__).resolve().parent
def deny(event,args):
 if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm'}:raise RuntimeError('Preservation prohibits AI imports')
sys.addaudithook(deny)
before=json.loads((E/'installed-source-before.json').read_text());root=Path('/opt/modly-wsl/venv');tree=[]
for d,dirs,files in os.walk(root,followlinks=False):
 for name in dirs+files:
  p=Path(d)/name;s=p.lstat();tree.append([p.relative_to(root).as_posix(),s.st_size,s.st_mtime_ns,s.st_mode])
h=hashlib.sha256(json.dumps(sorted(tree),separators=(',',':')).encode()).hexdigest()
pkgs=sorted([[d.metadata['Name'],d.version] for d in md.distributions(path=[str(root/'lib/python3.12/site-packages')])])
dpkg=subprocess.check_output(['dpkg-query','-W','-f=${Package}\t${Version}\n'],text=True,timeout=15).splitlines()
source_same=all(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'] for row in before['sources'].values())
record={'status':'PASS','baseline_metadata_sha256':h,'baseline_venv_unchanged':h==before['venv_metadata_sha256'],'python_packages_unchanged':pkgs==before['python_packages'],'dpkg_unchanged':dpkg==before['dpkg_packages'],'inspected_sources_unchanged':source_same,'AI_imports':False,'GPU_actions':False}
if not all(record[k] for k in ('baseline_venv_unchanged','python_packages_unchanged','dpkg_unchanged','inspected_sources_unchanged')):raise RuntimeError('Preservation mismatch')
with (E/'linux-preservation-after.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
