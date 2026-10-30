"""Phase 14: isolated, source-only, bounded CPU experiment; inert by default."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import amd_cgal_proof as owner

PRIOR = owner.BASE
BASE = PRIOR.parent / 'amd-mesh-aware-reconstruction-20261030-v1'
SITE = owner.CPU_SITE
BUILD = BASE / 'build'
PROTOCOL_SHA='9278cbb6554ed278494ca5c77b437c410cd89004d2c1a3805e0230e3dd2b7e23'
STAGES = ('authenticate','prepare-mesh','configure-shared','build-runtime','fixture','segment','screen','residual-assess','feature-check','verify-backend','augment','configure-reconstruction','build-reconstruction','prepare-candidates','screen-refined','configure-owned','build-owned','owned-reconstruct','prepare-validation','configure-validation','build-validation','validate','prepare-domains','configure-domains','build-domains','certify-supported-domains','assess-candidate','inspect-domain-witnesses')
DATA_ACTIONS={'prepare-mesh':'prepare','screen':'screen','residual-assess':'residual_assess','feature-check':'feature_check','augment':'augment','screen-refined':'screen_refined'}

def source_read_allowed(path):
    from amd_zombie_d_compare import plain
    p=plain(path)
    if p.name in ('amd_closed_solid_benchmark.py','amd-closed-solid-reference.json'):return False
    allowed={PRIOR/'ai-input/samples.npz',PRIOR/'ai-input/receipt.json'}
    raw=BASE.parent/'amd-closed-solid-raw-recovery-v1/mesh'
    return not p.is_relative_to(BASE.parent) or p.is_relative_to(BASE) or p.is_relative_to(raw) or p.is_relative_to(SITE) or p in allowed

def command(stage):
    if stage=='configure-domains':
        c=command('configure-shared');c[c.index('-S')+1]=str(BASE/'domains-project');c[c.index('-B')+1]=str(BASE/'domains-build');return c
    if stage=='build-domains':return [str(owner.CMAKE),'--build',str(BASE/'domains-build'),'--config','Release','--target','official_sample','--parallel','1']
    if stage=='certify-supported-domains':return [str(BASE/'domains-build/Release/official_sample.exe'),str(BASE)]
    if stage=='configure-validation':
        c=command('configure-shared');c[c.index('-S')+1]=str(BASE/'validation-project');c[c.index('-B')+1]=str(BASE/'validation-build');return c
    if stage=='build-validation':return [str(owner.CMAKE),'--build',str(BASE/'validation-build'),'--config','Release','--target','official_sample','--parallel','1']
    if stage=='validate':return [str(BASE/'validation-build/Release/official_sample.exe'),str(BASE)]
    if stage=='configure-owned':
        c=command('configure-shared');c[c.index('-S')+1]=str(BASE/'owned-project');c[c.index('-B')+1]=str(BASE/'owned-build');return c
    if stage=='build-owned':return [str(owner.CMAKE),'--build',str(BASE/'owned-build'),'--config','Release','--target','official_sample','--parallel','1']
    if stage=='owned-reconstruct':return [str(BASE/'owned-build/Release/official_sample.exe'),'owned-session',str(BASE/'owned-session')]
    if stage=='configure-reconstruction':
        c=command('configure-shared');c[c.index('-S')+1]=str(BASE/'reconstruction-project');c[c.index('-B')+1]=str(BASE/'reconstruction-build');return c
    if stage=='build-reconstruction':return [str(owner.CMAKE),'--build',str(BASE/'reconstruction-build'),'--config','Release','--target','official_sample','--parallel','1']
    if stage in ('prepare-candidates','reconstruct'):
        return [str(BASE/'reconstruction-build/Release/official_sample.exe'),stage,str(BASE)]
    if stage == 'configure-shared':
        return [str(owner.CMAKE), '-S',str(BASE/'project'),'-B',str(BUILD),'-G','Visual Studio 17 2022','-A','x64','-T','v143',
                '-DCGAL_DIR='+owner.CGAL.as_posix(),'-DBoost_DIR='+(PRIOR/'boost-header-config').as_posix(),
                '-DEigen3_DIR='+(PRIOR/'eigen-build-final').as_posix(),'-DCGAL_DISABLE_GMP=ON',
                '-DMODLY_SCIP_ROOT='+owner.SCIP.as_posix(),'-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF',
                '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON','-DCMAKE_EXPORT_PACKAGE_REGISTRY=OFF']
    if stage == 'build-runtime':
        return [str(owner.CMAKE),'--build',str(BUILD),'--config','Release','--target','official_sample','--parallel','1']
    if stage=='fixture':return [str(BUILD/'Release/official_sample.exe'),'segment',str(BASE/'fixture')]
    if stage=='segment':
        return [str(BUILD/'Release/official_sample.exe'),stage,str(BASE)]
    raise ValueError('No native command')

def spawn_allowed(stage, args):
    if stage not in ('configure-shared','build-runtime','fixture','segment','configure-reconstruction','build-reconstruction','prepare-candidates','configure-owned','build-owned','owned-reconstruct','configure-validation','build-validation','validate','configure-domains','build-domains','certify-supported-domains'):return False
    exe,argv,cwd,_env=args;expected=command(stage)
    return exe==expected[0] and argv in (expected,tuple(expected),subprocess.list2cmdline(expected)) and cwd==str(BASE)

def worker(stage):
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO and EOF required')
    from amd_zombie_d_compare import audit_policy
    ordinary=audit_policy(BASE)
    def guard(event,args):
        if event=='subprocess.Popen':
            if not spawn_allowed(stage,args):raise RuntimeError('Exact CPU command/cwd required')
            return
        ordinary(event,args)
    sys.addaudithook(guard)
    if owner.sha(BASE/'protocol.json')!=PROTOCOL_SHA:raise RuntimeError('Locked protocol differs')
    if owner.sha(owner.ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json')!='397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0':raise RuntimeError('Frozen criteria differs')
    if stage in ('prepare-mesh','screen','residual-assess','feature-check','augment','screen-refined','owned-reconstruct','prepare-validation','prepare-domains','assess-candidate','inspect-domain-witnesses'):
        def source_only(event,args):
            if event=='import' and str(args[0]).startswith('amd_closed_solid_'):
                raise RuntimeError('Procedural reference imports forbidden')
            if event=='open' and isinstance(args[0],(str,bytes,Path)):
                if not source_read_allowed(args[0]):raise RuntimeError('Non-AI evidence forbidden to fitting: '+str(args[0]))
        sys.addaudithook(source_only)
    if stage=='authenticate':
        import amd_mesh_aware_input as data
        data.authenticate();return
    if stage=='verify-backend':
        import amd_mesh_aware_input as data
        data.verify_backend();return
    if stage=='owned-reconstruct':
        sys.path.insert(0,str(SITE))
        import amd_mesh_aware_input as data
        pin=json.loads((BASE/'owned-build-pin.json').read_text())
        if owner.sha(BASE/'owned-build/Release/official_sample.exe')!=pin['exe_sha256']:raise RuntimeError('Owned binary differs')
        qualified=json.loads((BASE/'augmented-plane-screen.json').read_text())
        if qualified['classification']!='PASS' or owner.sha(BASE/'augmented/train.ply')!=qualified['PLY_sha256']:raise RuntimeError('Source-only training admission differs')
        session=BASE/'owned-session';session.mkdir(exist_ok=False)
        for name in ('augmented','arrangement','candidate'):(session/name).mkdir(exist_ok=False)
        with (session/'augmented/train.ply').open('xb') as output:output.write((BASE/'augmented/train.ply').read_bytes())
        env=dict(os.environ,PATH=str(owner.SCIP/'bin')+os.pathsep+os.environ.get('PATH',''))
        child=subprocess.Popen(command(stage),executable=command(stage)[0],cwd=str(BASE),env=env,shell=False,stdin=subprocess.PIPE)
        try:
            while not (session/'READY.txt').exists():
                if child.poll() is not None:raise RuntimeError('Owned constructor failed before READY')
                time.sleep(.05)
            if (session/'READY.txt').read_bytes()!=b'READY\n':raise RuntimeError('Exact live object READY required')
            data.screen_refined(session)
            final=json.loads((session/'refined-screen.json').read_text())
            if final['classification']!='PASS':raise RuntimeError('Live object source-support gate refused')
            owner.save(BASE/'actual-MIP-authorisation.json',dict(one_shot=True,same_native_object_screened_and_solved=True,session=str(session),screen_sha256=owner.sha(session/'refined-screen.json')))
            child.stdin.write(b'GO\n');child.stdin.close()
            if child.wait()!=0:raise RuntimeError('Owned native reconstruction failed')
        finally:
            if not child.stdin.closed:child.stdin.close()
            if child.poll() is None:child.wait(timeout=5)
        return
    if stage in DATA_ACTIONS or stage in ('prepare-validation','prepare-domains','assess-candidate','inspect-domain-witnesses'):
        sys.path.insert(0,str(SITE))
        import amd_mesh_aware_input as data
        if stage in ('prepare-validation','prepare-domains','assess-candidate','inspect-domain-witnesses'):
            import amd_mesh_aware_result as result
            {'prepare-validation':result.prepare,'prepare-domains':result.prepare_domains,'assess-candidate':result.assess,'inspect-domain-witnesses':result.witnesses}[stage]();return
        getattr(data,DATA_ACTIONS[stage])();return
    if stage=='certify-supported-domains':
        pin=json.loads((BASE/'domains-build-pin.json').read_text())
        if owner.sha(BASE/'domains-build/Release/official_sample.exe')!=pin['exe_sha256']:raise RuntimeError('Domain binary differs')
        if owner.sha(BASE/'finite-domain-input.bin')!=json.loads((BASE/'finite-domain-input.json').read_text())['sha256']:raise RuntimeError('Finite domains differ')
    if stage=='validate':
        pin=json.loads((BASE/'validation-build-pin.json').read_text())
        if owner.sha(BASE/'validation-build/Release/official_sample.exe')!=pin['exe_sha256']:raise RuntimeError('Validation binary differs')
        if owner.sha(BASE/'owned-session/candidate/raw-reconstruction.off')!=json.loads((BASE/'candidate-input-pin.json').read_text())['raw_candidate_sha256']:raise RuntimeError('Candidate differs')
        if owner.sha(BASE/'validation-points.bin')!=json.loads((BASE/'candidate-input-pin.json').read_text())['validation_points_sha256']:raise RuntimeError('Frozen source samples differ')
        if owner.sha(BASE/'input/mesh.bin')!=json.loads((BASE/'input/receipt.json').read_text())['input_sha256']:raise RuntimeError('Reference triangle geometry differs')
    if stage in ('prepare-candidates','reconstruct'):
        screen=BASE/('augmented-plane-screen.json' if stage=='prepare-candidates' else 'refined-screen.json')
        if json.loads(screen.read_text())['classification']!='PASS':raise RuntimeError('Unchanged support gate refused')
        pin=json.loads((BASE/'reconstruction-build-pin.json').read_text())
        if owner.sha(BASE/'reconstruction-build/Release/official_sample.exe')!=pin['exe_sha256']:raise RuntimeError('Qualified reconstruction binary differs')
        if owner.sha(BASE/'augmented/train.ply')!=json.loads((BASE/'augmented-plane-screen.json').read_text())['PLY_sha256']:raise RuntimeError('Training PLY differs')
        if stage=='reconstruct':
            final=json.loads(screen.read_text())
            if not all(p['strict_triangles']==p['source_triangles'] for p in final['original_patch_survival']):raise RuntimeError('Incomplete finite source-patch survival')
            for name,key in (('prepared-planes.csv','planes_SHA256'),('prepared-members.csv','members_SHA256'),('candidates.off','candidate_arrangement_SHA256')):
                if owner.sha(BASE/'arrangement'/name)!=final[key]:raise RuntimeError('Actual final CGAL admission input differs')
            if owner.sha(BASE/'finite-source-domains.npz')!=final['domains_SHA256']:raise RuntimeError('Finite domains differ')
        (BASE/('arrangement' if stage=='prepare-candidates' else 'candidate')).mkdir(exist_ok=False)
    if stage in ('fixture','segment'):
        pin=json.loads((BASE/'binary-qualification.json').read_text())
        if owner.sha(BUILD/'Release/official_sample.exe')!=pin['exe_sha256']:raise RuntimeError('Qualified binary differs')
    if stage=='segment':
        receipt=json.loads((BASE/'input/receipt.json').read_text())
        if owner.sha(BASE/'input/mesh.bin')!=receipt['input_sha256']:raise RuntimeError('Source-only binary input differs')
    env=dict(os.environ,PATH=str(owner.SCIP/'bin')+os.pathsep+os.environ.get('PATH',''))
    result=subprocess.run(command(stage),executable=command(stage)[0],cwd=str(BASE),env=env,shell=False)
    if result.returncode:raise RuntimeError('Native CPU stage failed: '+str(result.returncode))
    if stage=='fixture':
        sys.path.insert(0,str(SITE));import numpy as np
        from amd_mesh_aware_input import read_planes
        rows=read_planes(np,BASE/'fixture/regions.csv')
        if len(rows)!=2 or sorted(p['faces'] for p in rows)!=[128,128]:raise ValueError('Native region fixture failed')
        owner.save(BASE/'fixture-result.json',dict(classification='PASS',regions=len(rows),faces=[p['faces'] for p in rows],mixed_winding=True,junction_blocking_scope='Python adjacency unit tests; native fixture contains two disjoint planar sheets'))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES);p.add_argument('--worker',choices=STAGES);a=p.parse_args(argv)
    if a.worker:worker(a.worker);return 0
    if a.stage:
        # Reuse the reviewed Job controller verbatim, with a fresh entry and paths.
        owner.BASE=BASE;owner.BUILD=BUILD;owner.__file__=__file__
        owner.command=command
        if a.stage in ('configure-reconstruction','build-reconstruction','configure-owned','build-owned','configure-validation','build-validation','configure-domains','build-domains'):
            owner.BUILD=BASE/('owned-build' if a.stage.endswith('owned') else 'domains-build' if a.stage.endswith('domains') else 'validation-build' if a.stage.endswith('validation') else 'reconstruction-build')
            owner.compiler_cleanup_stage=lambda s:s in ('configure-reconstruction','build-reconstruction','configure-owned','build-owned','configure-validation','build-validation','configure-domains','build-domains')
            owner.compiler_outputs_complete=lambda s: ((owner.BUILD/'official_sample.vcxproj').is_file() if s.startswith('configure-') else (owner.BUILD/'Release/official_sample.exe').is_file() and (owner.BUILD/'Release/official_sample.exe').stat().st_size>100000)
        return owner.supervise(a.stage)
    print('INERT: explicit CPU stage required');return 0

if __name__=='__main__':raise SystemExit(main())
