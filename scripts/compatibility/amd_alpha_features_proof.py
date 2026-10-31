"""Phase19 read-only saved-output census. No reconstruction entry point."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent))
import amd_alpha_wrap_proof as old
import amd_cgal_proof as owner
from amd_closure_proof import check
PRIOR=old.BASE;BASE=PRIOR.parent/'amd-alpha-features-20261031-v1';BUILD=BASE/'build'
STAGES=('authenticate','deps-cgal','deps-boost','deps-tools','deps-numpy','configure','build','fixtures','census','analyse','render','visual-finalise','preserve')

def command(stage):
 if stage=='configure':
  c=old.command('configure');c[c.index('-S')+1]=str(BASE/'project');c[c.index('-B')+1]=str(BUILD);return c
 if stage=='build':return [str(owner.CMAKE),'--build',str(BUILD),'--config','Release','--target','official_sample','--parallel','1']
 if stage=='fixtures':return [str(BUILD/'Release/official_sample.exe'),'fixtures',str(BASE/'fixtures.json')]
 if stage=='census':return [str(BUILD/'Release/official_sample.exe'),str(old.ROOT14),str(PRIOR),str(BASE/'census')]
 raise ValueError('No native route')

def spawn_allowed(stage,args):
 if stage not in ('configure','build','fixtures','census'):return False
 exe,argv,cwd,_=args;c=command(stage)
 return exe==c[0] and argv in (c,tuple(c),subprocess.list2cmdline(c)) and cwd==str(BASE)

def reviewed(stage):
 r=json.loads((BASE/('visual-review.json' if stage=='visual-finalise' else 'analysis-review.json' if stage in ('analyse','render') else 'execution-review.json')).read_text())
 if r['classification']!='APPROVED' or stage not in r['allowed_stages']:raise ValueError('Independent stage approval required')
 for name,h in r['files'].items():
  if owner.sha(Path(name))!=h:raise ValueError('Reviewed source changed')

def successful(stage):
 r=json.loads((BASE/(stage+'-run')/'host-result.json').read_text())
 if r['classification'] not in ('PASS','PASS_WITH_OWNED_COMPILER_HELPER_CLEANUP'):raise ValueError('Prerequisite failed')

def worker(stage):
 if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO/EOF required')
 from amd_zombie_d_compare import audit_policy
 ordinary=audit_policy(BASE)
 def guard(event,args):
  if event=='subprocess.Popen':
   if not spawn_allowed(stage,args):raise RuntimeError('Exact CPU command/cwd required')
   return
  ordinary(event,args)
 sys.addaudithook(guard)
 if owner.sha(owner.ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json')!='397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0':raise ValueError('Frozen criteria differs')
 if stage=='authenticate':
  files={}
  for name,row in json.loads((PRIOR/'evidence-manifest.json').read_text()).items():files[name]=check(PRIOR/name,row)
  a=json.loads((PRIOR/'authentication.json').read_text())
  for name,row in a['original_snapshot'].items():check(Path(name),row)
  for name,row in a['Phase14_files'].items():check(old.ROOT14/name,row)
  for name,row in a['protected_records'].items():
   s=Path(name).stat()
   if (s.st_size,s.st_mtime_ns,s.st_ino)!=(row['bytes'],row['mtime_ns'],row['file_id']):raise ValueError('Protected identity')
  owner.save(BASE/'authentication.json',dict(classification='PASS',Phase18_files=files,original_snapshot=a['original_snapshot'],Phase14_files=a['Phase14_files'],protected_records=a['protected_records']))
  p=json.loads((PRIOR/'compiler-helper-pin.json').read_text());check(Path(p['path']),p);owner.save(BASE/'compiler-helper-pin.json',p);return
 if stage.startswith('deps-'):old.BASE=BASE;old.dependency(stage[5:]);return
 if stage=='preserve':
  a=json.loads((BASE/'authentication.json').read_text())
  for name,row in a['Phase18_files'].items():check(PRIOR/name,row)
  for name,row in a['Phase14_files'].items():check(old.ROOT14/name,row)
  for name,row in a['original_snapshot'].items():check(Path(name),row)
  for name,row in a['protected_records'].items():
   s=Path(name).stat()
   if (s.st_size,s.st_mtime_ns,s.st_ino)!=(row['bytes'],row['mtime_ns'],row['file_id']):raise ValueError('Protected identity changed')
  owner.save(BASE/'preservation-after.json',dict(classification='PASS',Phase18_files=len(a['Phase18_files']),Phase14_files=len(a['Phase14_files']),original_sources=len(a['original_snapshot']),protected_records=len(a['protected_records'])));return
 reviewed(stage);successful('authenticate')
 if stage in ('analyse','render','visual-finalise'):
  successful('census');sys.path.insert(0,str(owner.CPU_SITE))
  if stage in ('render','visual-finalise'):successful('analyse')
  import amd_alpha_features_result as r
  getattr(r,stage.replace('-','_'))();return
 for dep in ('cgal','boost','tools','numpy'):successful('deps-'+dep)
 if stage=='build':successful('configure')
 if stage in ('fixtures','census'):
  successful('build');p=json.loads((BASE/'binary.json').read_text());check(BUILD/'Release/official_sample.exe',p)
 if stage=='census':
  successful('fixtures')
  if json.loads((BASE/'fixtures.json').read_text())['classification']!='PASS':raise ValueError('Native fixtures failed')
  a=json.loads((BASE/'authentication.json').read_text())
  for name in ('raw-wrap.off','simplified.off','feature-input.bin','result.json'):check(PRIOR/name,old.prior.previous.old.manifest_identity(a['Phase18_files'],name))
  for name in ('input/mesh.bin','input/receipt.json'):check(old.ROOT14/name,old.prior.previous.old.manifest_identity(a['Phase14_files'],name))
 r=subprocess.run(command(stage),executable=command(stage)[0],cwd=str(BASE),shell=False)
 if r.returncode:raise RuntimeError('Native read-only stage failed')
 if stage=='build':
  p=BUILD/'Release/official_sample.exe';owner.save(BASE/'binary.json',dict(bytes=p.stat().st_size,sha256=owner.sha(p)))

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES);p.add_argument('--worker',choices=STAGES);a=p.parse_args(argv)
 if a.worker:worker(a.worker);return 0
 if not a.stage:print('INERT: read-only census; no reconstruction route');return 0
 owner.BASE=BASE;owner.BUILD=BUILD;owner.__file__=__file__;owner.command=command
 owner.compiler_cleanup_stage=lambda s:s in ('configure','build')
 owner.compiler_outputs_complete=lambda s:(BUILD/'official_sample.vcxproj').is_file() if s=='configure' else (BUILD/'Release/official_sample.exe').is_file()
 return owner.supervise(a.stage)
if __name__=='__main__':raise SystemExit(main())
