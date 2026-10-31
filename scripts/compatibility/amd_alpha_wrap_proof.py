"""One-shot isolated CPU Alpha Wrapping proof; inert without an explicit stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_cgal_proof as owner
import amd_patch_subdivision_proof as prior
from amd_closure_proof import check
ROOT13=owner.BASE
ROOT14=prior.previous.PRIOR
ROOT17=prior.BASE
BASE=ROOT13.parent/'amd-alpha-wrap-20261031-v1'
BUILD=BASE/'build'
SAMPLE=ROOT13/'isolated/CGAL-6.2.1/data/meshes/blobby-shuffled.off'
ALPHA_DENOMINATOR=20
OFFSET_DENOMINATOR=2000
TARGETS={'build-official':'official_sample','build-check':'alpha_check','build-wrap':'alpha_wrap','build-simplify':'alpha_simplify','build-validate':'alpha_validate'}
DEPS=('cgal','examples','boost','tools','numpy')
OFFICIAL_STAGES=('configure','build-official','build-check','official','check-official')
STAGES=('authenticate',*('deps-'+x for x in DEPS),'configure',*TARGETS,'official','check-official','wrap','check-raw','simplify','check-final','prepare-validation','validate','assess','preserve')

def dependency_spec(name):
    files={'cgal':'CGAL-6.2.1-library.zip','examples':'CGAL-6.2.1-examples.zip','boost':'boost_1_87_0.zip','tools':'cmake-3.31.10-windows-x86_64.zip'}
    if name=='numpy':return json.loads((ROOT13/'numpy-provenance.json').read_text())
    if name not in files:raise ValueError('Unapproved dependency')
    return next(r for r in json.loads((ROOT13/'downloads-provenance.json').read_text()) if r['name']==files[name])

def dependency(name):
    row=dependency_spec(name);archive=ROOT13/'downloads'/row['name'];check(archive,row)
    selected={};begin=time.monotonic()
    examples={'CGAL-6.2.1/examples/Alpha_wrap_3/triangle_soup_wrap.cpp','CGAL-6.2.1/examples/Alpha_wrap_3/output_helper.h','CGAL-6.2.1/examples/Surface_mesh_simplification/edge_collapse_envelope.cpp','CGAL-6.2.1/data/meshes/blobby-shuffled.off'}
    with zipfile.ZipFile(archive) as z:
        seen=set()
        for member in z.infolist():
            if member.is_dir():continue
            rel=owner.safe_member(member.filename)
            if name=='boost' and not owner.selected_boost(rel):continue
            if name=='examples' and member.filename not in examples:continue
            key=member.filename.casefold()
            if key in seen:raise ValueError('Archive path collision')
            seen.add(key);data=z.read(member)
            dest=(owner.CPU_SITE/rel) if name=='numpy' else (owner.ENV/rel)
            if name=='examples' and member.filename.endswith('blobby-shuffled.off'):dest=SAMPLE
            selected[member.filename]=dict(path=str(dest),**check(dest,dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())))
    if not selected or (name=='examples' and len(selected)!=len(examples)):raise ValueError('Incomplete dependency scope')
    owner.save(BASE/('dependency-'+name+'.json'),dict(classification='PASS',archive=row,files=selected,elapsed_seconds=time.monotonic()-begin,complete_selected_payloads=True))

def command(stage):
    if stage=='configure':return [str(owner.CMAKE),'-S',str(BASE/'project'),'-B',str(BUILD),'-G','Visual Studio 17 2022','-A','x64','-T','v143','-DCGAL_DIR='+owner.CGAL.as_posix(),'-DBoost_DIR='+(ROOT13/'boost-header-config').as_posix(),'-DCGAL_DISABLE_GMP=ON','-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF','-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON','-DCMAKE_EXPORT_PACKAGE_REGISTRY=OFF']
    if stage in TARGETS:return [str(owner.CMAKE),'--build',str(BUILD),'--config','Release','--target',TARGETS[stage],'--parallel','1']
    exe=lambda name:str(BUILD/'Release'/(name+'.exe'))
    if stage=='official':return [exe('official_sample'),SAMPLE.as_posix(),'20','600']
    if stage=='check-official':return [exe('alpha_check'),str(BASE/'official/blobby-shuffled_20_600.off'),str(BASE/'official-check.json')]
    if stage=='wrap':return [exe('alpha_wrap'),str(ROOT14/'input/mesh.bin'),str(BASE/'raw-wrap.off'),str(BASE/'wrap.json')]
    if stage=='check-raw':return [exe('alpha_check'),str(BASE/'raw-wrap.off'),str(BASE/'raw-check.json')]
    if stage=='simplify':return [exe('alpha_simplify'),str(BASE/'raw-wrap.off'),str(BASE/'simplified.off'),str(BASE/'simplification.json')]
    if stage=='check-final':return [exe('alpha_check'),str(final_candidate()),str(BASE/'final-check.json')]
    if stage=='validate':return [exe('alpha_validate'),str(ROOT14),str(final_candidate()),str(BASE/'validation')]
    raise ValueError('No native route')

def final_candidate():
    # An interrupted/failed simplification never silently falls back to the wrap.
    if (BASE/'simplify-run').exists():return BASE/'simplified.off'
    return BASE/'raw-wrap.off'

def spawn_allowed(stage,args):
    if stage not in ('configure',*TARGETS,'official','check-official','wrap','check-raw','simplify','check-final','validate'):return False
    exe,argv,cwd,_=args;c=command(stage);directory=BASE/'official' if stage=='official' else BASE
    return exe==c[0] and argv in (c,tuple(c),subprocess.list2cmdline(c)) and cwd==str(directory)

def successful(stage):
    row=json.loads((BASE/(stage+'-run')/'host-result.json').read_text())
    if row['classification'] not in ('PASS','PASS_WITH_OWNED_COMPILER_HELPER_CLEANUP'):raise ValueError('Prerequisite stage failed: '+stage)

def reviewed(stage):
    row=json.loads((BASE/('execution-review-official.json' if stage in OFFICIAL_STAGES else 'execution-review-ai.json')).read_text())
    if row['classification']!='APPROVED' or stage not in row['allowed_stages']:raise ValueError('Independent stage approval required')
    for name,h in row['files'].items():
        if owner.sha(Path(name))!=h:raise ValueError('Reviewed source changed: '+name)

def require_mesh(path):
    r=json.loads(path.read_text())
    if r.get('classification')!='PASS_CLOSED_EMBEDDED_MESH':raise ValueError('Native geometry baseline failed')

def artifact(stage,path):
    pin=json.loads((BASE/(stage+'-artifact.json')).read_text())
    if pin['path']!=str(path):raise ValueError('Artifact path differs')
    check(path,pin)

def worker(stage):
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO and EOF required')
    from amd_zombie_d_compare import audit_policy
    ordinary=audit_policy(BASE)
    def guard(event,args):
        if event=='subprocess.Popen':
            if not spawn_allowed(stage,args):raise RuntimeError('Exact CPU command and cwd required')
            return
        ordinary(event,args)
    sys.addaudithook(guard)
    if owner.sha(owner.ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json')!='397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0':raise ValueError('Frozen criteria differs')
    if stage=='authenticate':
        prior.BASE=BASE;prior.authenticate()
        for name,row in json.loads((ROOT17/'evidence-manifest.json').read_text()).items():check(ROOT17/name,row)
        for name,row in json.loads((ROOT13/'final-build-identities.json').read_text()).items():check(ROOT13/name,row)
        pin=json.loads((ROOT13/'compiler-helper-pin.json').read_text());check(Path(pin['path']),pin)
        owner.save(BASE/'compiler-helper-pin.json',pin);return
    if stage.startswith('deps-'):dependency(stage[5:]);return
    if stage=='preserve':
        a=json.loads((BASE/'authentication.json').read_text())
        for name,row in a['Phase16_files'].items():check(prior.PRIOR/name,row)
        for name,row in a['Phase14_files'].items():check(ROOT14/name,row)
        for name,row in a['original_snapshot'].items():check(Path(name),row)
        for name,row in a['protected_records'].items():
            s=Path(name).stat()
            if (s.st_size,s.st_mtime_ns,s.st_ino)!=(row['bytes'],row['mtime_ns'],row['file_id']):raise ValueError('Protected identity changed')
        owner.save(BASE/'preservation-after.json',dict(classification='PASS',Phase16_files=len(a['Phase16_files']),Phase14_files=len(a['Phase14_files']),original_sources=len(a['original_snapshot']),protected_records=len(a['protected_records'])));return
    reviewed(stage);successful('authenticate')
    if stage in ('prepare-validation','assess'):
        successful('check-final' if stage=='prepare-validation' else 'validate');sys.path.insert(0,str(owner.CPU_SITE))
        import amd_alpha_wrap_result
        (amd_alpha_wrap_result.prepare if stage=='prepare-validation' else amd_alpha_wrap_result.assess)();return
    for dep in DEPS:
        if stage=='configure' or stage in TARGETS:
            if json.loads((BASE/('dependency-'+dep+'.json')).read_text())['classification']!='PASS':raise ValueError('Dependency not admitted')
    if stage in TARGETS:successful('configure')
    if stage=='official':
        successful('build-official');(BASE/'official').mkdir(exist_ok=False)
    if stage=='check-official':successful('official');successful('build-check');artifact('official',BASE/'official/blobby-shuffled_20_600.off')
    if stage=='wrap':
        successful('check-official');require_mesh(BASE/'official-check.json');successful('build-wrap')
        a=json.loads((BASE/'authentication.json').read_text())
        check(ROOT14/'input/mesh.bin',prior.previous.old.manifest_identity(a['Phase14_files'],'input/mesh.bin'))
        owner.save(BASE/'AI-wrap-started.json',dict(one_shot=True,alpha_over_D=.05,offset_over_D=.0005,input_sha256=owner.sha(ROOT14/'input/mesh.bin')))
    if stage=='check-raw':successful('wrap');artifact('wrap',BASE/'raw-wrap.off')
    if stage=='simplify':
        successful('check-raw');require_mesh(BASE/'raw-check.json');successful('build-simplify')
        artifact('wrap',BASE/'raw-wrap.off')
        if not 1000<json.loads((BASE/'raw-check.json').read_text())['triangles']<=100000:raise ValueError('Simplification not admitted')
    if stage=='check-final':
        successful('simplify' if (BASE/'simplify-run').exists() else 'check-raw')
        artifact('simplify' if (BASE/'simplify-run').exists() else 'wrap',final_candidate())
    if stage=='validate':
        successful('check-final');require_mesh(BASE/'final-check.json');successful('build-validate')
        if json.loads((BASE/'final-check.json').read_text())['triangles']>1000:raise ValueError('Frozen output budget')
        successful('prepare-validation')
        pin=json.loads((BASE/'validation-inputs.json').read_text())
        for name,row in pin.items():check(Path(name),row)
    if stage not in ('configure',*TARGETS):
        exe=Path(command(stage)[0]);pin=json.loads((BASE/(exe.stem+'-binary.json')).read_text())
        if owner.sha(exe)!=pin['sha256']:raise ValueError('Native executable changed')
    result=subprocess.run(command(stage),executable=command(stage)[0],cwd=str(BASE/'official' if stage=='official' else BASE),shell=False)
    if result.returncode:raise RuntimeError('Native stage failed: '+str(result.returncode))
    if stage in TARGETS:
        exe=BUILD/'Release'/(TARGETS[stage]+'.exe')
        owner.save(BASE/(TARGETS[stage]+'-binary.json'),dict(sha256=owner.sha(exe),bytes=exe.stat().st_size,CPU_only=True))
    if stage in ('official','wrap','simplify'):
        path=BASE/({'official':'official/blobby-shuffled_20_600.off','wrap':'raw-wrap.off','simplify':'simplified.off'}[stage])
        owner.save(BASE/(stage+'-artifact.json'),dict(path=str(path),bytes=path.stat().st_size,sha256=owner.sha(path)))

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES);p.add_argument('--worker',choices=STAGES);a=p.parse_args(argv)
    if a.worker:worker(a.worker);return 0
    if not a.stage:print('INERT: explicit bounded CPU proof stage required');return 0
    owner.BASE=BASE;owner.BUILD=BUILD;owner.__file__=__file__;owner.command=command
    owner.compiler_cleanup_stage=lambda s:s=='configure' or s in TARGETS
    owner.compiler_outputs_complete=lambda s: (BUILD/'official_sample.vcxproj').is_file() if s=='configure' else (BUILD/'Release'/(TARGETS.get(s,'INVALID')+'.exe')).is_file()
    return owner.supervise(a.stage)

if __name__=='__main__':raise SystemExit(main())
