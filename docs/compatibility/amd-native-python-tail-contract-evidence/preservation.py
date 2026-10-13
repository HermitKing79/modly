"""Passive inventories and historical hashes; never import AI runtimes."""
import gzip, hashlib, json, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3]
EV=Path(__file__).parent
OLD=ROOT/'docs/compatibility/amd-native-postprocessor-admission-evidence'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
def refs(remote=False):
    text=git('ls-remote','--heads','origin') if remote else git('for-each-ref','--format=%(objectname) %(refname:short)','refs/heads')
    return {line.split()[1].removeprefix('refs/heads/'):line.split()[0] for line in text.splitlines()}
def audit(e,a):
    if e=='import' and str(a[0]).split('.')[0] in ('torch','triton','trellis2','cumesh','flex_gemm','o_voxel'):raise RuntimeError('Passive inventory forbids AI imports')
sys.addaudithook(audit)
sys.path.insert(0,str(ROOT/'scripts/compatibility'))
import amd_bounded_gpu_proof as old
phase=sys.argv[1];assert phase in ('before','after')
args=SimpleNamespace(amd_venv=Path('D:/AI/Trellis2AMD/venv'),modly_venv=Path('C:/Users/wattz/Documents/Modly/extensions/trellis2/venv'),amd_source=Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD'),extension_source=ROOT.parent/'gguf-source')
inventory=old.snapshot(args);inventory['model_cache_metadata']={}
roots=json.loads(gzip.decompress((OLD/'model-cache-metadata-after.json.gz').read_bytes()))
raw={}
for name in roots:
    base=Path(name);rows=[]
    for directory,ds,fs in os.walk(base,followlinks=False):
        for child in ds+fs:
            p=Path(directory)/child;s=p.lstat();rows.append([p.relative_to(base).as_posix(),s.st_size,s.st_mtime_ns,s.st_mode])
    raw[name]=sorted(rows)
    data=json.dumps(old.normalize_metadata_rows(raw[name]),separators=(',',':')).encode()
    inventory['model_cache_metadata'][name]=dict(entries=len(rows),normalized_sha256=hashlib.sha256(data).hexdigest())
assert inventory==json.loads((OLD/'inventory-after.json').read_text()),'Installed environment/source metadata changed'
old_git=json.loads((OLD/'git-before.json').read_text())
for name,h in old_git['historical_files'].items():assert sha(ROOT/name)==h,name
intentional_changes={}
reviewed=json.loads((EV/'independent-review.json').read_text())['reviewed_sha256'] if phase=='after' else {}
allowed_harness=set()  # No production permission changes: every prior hash stays strict.
for name,row in json.loads((OLD/'checksums.json').read_text()).items():
    actual=sha(ROOT/name)
    if phase=='after' and name in allowed_harness:
        assert actual==reviewed[name],name
        intentional_changes[name]=dict(previous_sha256=row['sha256'],reviewed_sha256=actual)
    else:assert actual==row['sha256'],name
remote=refs(True);local=refs()
source=args.amd_source
row=dict(base_commit='c9e1a35ce9941753e64536885e59d947cd0a7f1e',branch=git('branch','--show-current'),head=git('rev-parse','HEAD'),origin=git('remote','get-url','origin'),remote_refs=remote,local_refs=local,
    amd_source_head=git('rev-parse','HEAD',cwd=source),amd_source_status=git('status','--short',cwd=source),
    amd_modified_files={name:sha(source/name) for name in ('nodes.py','nodes.py.before-profiler-fix')},
    installed_inventory_unchanged=True,torch_imported='torch' in sys.modules)
row['intentional_reviewed_harness_changes']=intentional_changes
for key in ('amd_source_head','amd_source_status','amd_modified_files'):assert row[key]==old_git[key],key
for name,h in old_git['remote_refs'].items():
    if name!='codex/amd-native-postprocessor-admission':assert remote[name]==h,(name,'remote')
for name,h in old_git['local_refs'].items():
    if name!='codex/amd-native-postprocessor-admission':assert local[name]==h,(name,'local')
assert remote['codex/amd-native-postprocessor-admission']==row['base_commit']
assert local['codex/amd-native-postprocessor-admission']==row['base_commit']
if phase=='before':
    assert row['head']==row['base_commit']
    row['historical_files']={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'docs/compatibility').rglob('*') if p.is_file() and EV not in p.parents and p.name!='amd-native-python-tail-contract.md'}
else:
    baseline=json.loads((EV/'git-before.json').read_text())
    for key in ('remote_refs','local_refs'):
        for name,h in baseline[key].items():
            if name!=row['branch']:assert row[key][name]==h,(name,key)
    for name,h in baseline['historical_files'].items():assert sha(ROOT/name)==h,name
    row['historical_files_verified']=len(baseline['historical_files'])
for name,value in [('inventory-'+phase+'.json',inventory),('git-'+phase+'.json',row)]:
    with (EV/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
with (EV/('model-cache-metadata-'+phase+'.json.gz')).open('xb') as f:f.write(gzip.compress(json.dumps(raw).encode(),mtime=0))
print(json.dumps({'phase':phase,'historical_files':len(row.get('historical_files',{})) or row.get('historical_files_verified'),'preservation_verified':True,'torch_imported':False}))
