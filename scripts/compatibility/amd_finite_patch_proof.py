"""Phase 15 isolated CPU proof. Inert without an explicit, one-shot stage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_cgal_proof as owner
import amd_mesh_aware_proof as previous
PRIOR=previous.BASE
BASE=PRIOR.parent/'amd-finite-patch-reconstruction-20261030-v1'
SITE=previous.SITE
BUILD=BASE/'build'
PLAN_SHA='758bd8f15f29da6c89603a489960c4c0c8ff5189323b46d9ac72530b94c6296c'
STAGES=('authenticate','stage-inputs','configure','build','build-reviewed','fixtures','reconstruct','execute-constrained','prepare-validation','validate','prepare-domains','certify-domains','assess-candidate','assess-geometry','witnesses')

def manifest_identity(mapping,name):
    key=name.replace('\\','/')
    matches=[row for candidate,row in mapping.items() if candidate.replace('\\','/')==key]
    if len(matches)!=1:raise ValueError('Missing or ambiguous manifest identity: '+name)
    return matches[0]

def eligibility_admission(summary,records):
    """Postrun offline tightening; does not change the consumed Phase15 result."""
    count=summary.get('candidate_faces',0)
    if not 4<=count<=20000 or len(records)!=count or summary.get('eligibility_records')!=count:return 'INCOMPLETE_FACE_RECORDS'
    if [r.get('variable') for r in records]!=list(range(count)) or len({r.get('key') for r in records})!=count:return 'AMBIGUOUS_FACE_MAPPING'
    states={'CERTIFIED_ELIGIBLE','CONFIRMED_UNSUPPORTED','UNRESOLVED'}
    if any(r.get('state') not in states for r in records):return 'UNKNOWN_CERTIFICATE_STATE'
    eligible=sum(r['state']=='CERTIFIED_ELIGIBLE' for r in records)
    if eligible!=summary.get('eligible_faces') or eligible<4:return 'INSUFFICIENT_CERTIFIED_FACES'
    if any(r['state']=='UNRESOLVED' and r.get('reason')=='QUERY_LIMIT' for r in records):return 'GLOBAL_QUERY_WORK_INCOMPLETE'
    return 'ADMITTED'

def command(stage):
    if stage=='configure':
        c=previous.command('configure-shared');c[c.index('-S')+1]=str(BASE/'project');c[c.index('-B')+1]=str(BUILD);return c
    if stage in ('build','build-reviewed'):return [str(owner.CMAKE),'--build',str(BUILD),'--config','Release','--target','official_sample','--parallel','1']
    if stage in ('fixtures','reconstruct','execute-constrained'):return [str(BUILD/'Release/official_sample.exe'),'fixtures' if stage=='fixtures' else 'reconstruct',str(BASE),str(PRIOR),'finite-patch-v1']
    if stage=='validate':return [str(PRIOR/'validation-build/Release/official_sample.exe'),str(BASE)]
    if stage=='certify-domains':return [str(PRIOR/'domains-build/Release/official_sample.exe'),str(BASE)]
    raise ValueError('No native CPU route for '+stage)

def spawn_allowed(stage,args):
    if stage not in ('configure','build','build-reviewed','fixtures','reconstruct','execute-constrained','validate','certify-domains'):return False
    exe,argv,cwd,_env=args;c=command(stage)
    return exe==c[0] and argv in (c,tuple(c),subprocess.list2cmdline(c)) and cwd==str(BASE)

def source_allowed(path):
    from amd_zombie_d_compare import plain
    p=plain(path)
    if p.name in ('amd_closed_solid_benchmark.py','amd-closed-solid-reference.json'):return False
    old=owner.BASE # Controller does not modify this in the worker process.
    allowed={old/'ai-input/samples.npz',old/'ai-input/receipt.json'}
    raw=BASE.parent/'amd-closed-solid-raw-recovery-v1/mesh'
    return not p.is_relative_to(BASE.parent) or p.is_relative_to(BASE) or p.is_relative_to(PRIOR) or p.is_relative_to(raw) or p.is_relative_to(SITE) or p in allowed

def authenticate():
    manifest=json.loads((PRIOR/'evidence-manifest-v2.json').read_text());checked={}
    for name,e in manifest.items():
        p=PRIOR/name;s=p.stat()
        if s.st_size!=e['bytes'] or owner.sha(p)!=e['sha256']:raise ValueError('Phase14 evidence differs: '+name)
        checked[name]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=e['sha256'])
    old=json.loads((PRIOR/'authentication.json').read_text());source={}
    for name,e in old['original_snapshot'].items():
        p=Path(name);s=p.stat()
        if s.st_size!=e['bytes'] or owner.sha(p)!=e['sha256']:raise ValueError('Original AI source differs')
        source[name]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=e['sha256'])
    for pin,relative in (('owned-build-pin.json','owned-build/Release/official_sample.exe'),('validation-build-pin.json','validation-build/Release/official_sample.exe'),('domains-build-pin.json','domains-build/Release/official_sample.exe')):
        if owner.sha(PRIOR/relative)!=json.loads((PRIOR/pin).read_text())['exe_sha256']:raise ValueError('Reused binary differs')
    if owner.sha(owner.ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json')!='397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0':raise ValueError('Frozen criteria differs')
    owner.save(BASE/'authentication.json',dict(classification='PASS',Phase14_manifest_sha256=owner.sha(PRIOR/'evidence-manifest-v2.json'),Phase14_files=checked,original_snapshot=source,original_binding=old['binding'],reuse_domains_from_heldout_areas=True,validation_sets_are_not_blind_to_eligibility_constraints=True))

def use_old_result(stage):
    sys.path.insert(0,str(SITE))
    import amd_mesh_aware_result as result
    # Only module globals in this disposable worker change; old source stays intact.
    result.run.BASE=BASE
    actions={'prepare-validation':result.prepare,'prepare-domains':result.prepare_domains,'assess-candidate':result.assess,'witnesses':result.witnesses}
    actions[stage]()

def stage_inputs():
    names=['input/mesh.bin','input/source-analysis.npz','input/receipt.json','augmented/samples.npz','augmented-plane-screen.json','region-labels.bin','owned-session/refined-screen.json']
    identities={}
    for name in names:
        source=PRIOR/name;dest=BASE/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise RuntimeError('Staging path consumed')
        shutil.copyfile(source,dest)
        if owner.sha(source)!=owner.sha(dest):raise RuntimeError('Source-copy identity differs')
        identities[name]=dict(bytes=dest.stat().st_size,sha256=owner.sha(dest))
    owner.save(BASE/'staged-inputs.json',identities)
    import amd_mesh_aware_input as data
    data.run.BASE=BASE
    data.verify_backend()

def worker(stage):
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Exact owned GO and EOF required')
    from amd_zombie_d_compare import audit_policy
    ordinary=audit_policy(BASE)
    def audit(event,args):
        if event=='subprocess.Popen':
            if not spawn_allowed(stage,args):raise RuntimeError('Exact native CPU command and cwd required')
            return
        ordinary(event,args)
        if stage not in ('authenticate','stage-inputs','configure','build','build-reviewed'):
            if event=='import' and str(args[0]).startswith('amd_closed_solid_'):raise RuntimeError('Ground truth import forbidden')
            if event=='open' and isinstance(args[0],(str,bytes,Path)) and not source_allowed(args[0]):raise RuntimeError('Ground truth or unrelated artifact read forbidden')
    sys.addaudithook(audit)
    if owner.sha(BASE/'plan.json')!=PLAN_SHA:raise RuntimeError('Reviewed execution plan changed')
    if owner.sha(owner.ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json')!='397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0':raise RuntimeError('Frozen criteria changed')
    if stage=='authenticate':authenticate();return
    if stage=='stage-inputs':stage_inputs();return
    if stage=='assess-geometry':
        sys.path.insert(0,str(SITE));import amd_finite_patch_result as result
        result.assess_geometry();return
    if stage=='prepare-domains':
        shutil.copyfile(PRIOR/'finite-domain-input.bin',BASE/'finite-domain-input.bin')
        original=json.loads((PRIOR/'finite-domain-input.json').read_text())
        if owner.sha(BASE/'finite-domain-input.bin')!=original['sha256']:raise RuntimeError('Reused finite-domain identity differs')
        original.update(heldout_source_geometry_used_for_evaluation_only=False,full_source_domains_used_in_solver_eligibility=True,neither_validation_set_blind_to_constraints=True)
        owner.save(BASE/'finite-domain-input.json',original);return
    if stage in ('prepare-validation','prepare-domains','assess-candidate','witnesses'):use_old_result(stage);return
    if stage in ('fixtures','reconstruct','execute-constrained'):
        pin=json.loads((BASE/'build-pin.json').read_text())
        if owner.sha(BUILD/'Release/official_sample.exe')!=pin['exe_sha256']:raise RuntimeError('Pinned classifier/MIP adapter differs')
    if stage=='validate':
        if owner.sha(PRIOR/'validation-build/Release/official_sample.exe')!=json.loads((PRIOR/'validation-build-pin.json').read_text())['exe_sha256']:raise RuntimeError('Reused validation binary differs')
        pin=json.loads((BASE/'candidate-input-pin.json').read_text())
        if owner.sha(BASE/'owned-session/candidate/raw-reconstruction.off')!=pin['raw_candidate_sha256'] or owner.sha(BASE/'validation-points.bin')!=pin['validation_points_sha256']:raise RuntimeError('Validation identities differ')
    if stage=='certify-domains':
        if owner.sha(PRIOR/'domains-build/Release/official_sample.exe')!=json.loads((PRIOR/'domains-build-pin.json').read_text())['exe_sha256']:raise RuntimeError('Reused domain binary differs')
        if owner.sha(BASE/'finite-domain-input.bin')!=json.loads((BASE/'finite-domain-input.json').read_text())['sha256']:raise RuntimeError('Domain input differs')
    env=dict(os.environ,PATH=str(owner.SCIP/'bin')+os.pathsep+os.environ.get('PATH',''))
    if stage in ('reconstruct','execute-constrained'):
        if json.loads((BASE/'fixtures/result.json').read_text())['classification']!='PASS':raise RuntimeError('Synthetic fixtures not qualified')
        reviewed=json.loads((BASE/('pre-execution-review-v2.json' if stage=='execute-constrained' else 'pre-execution-review.json')).read_text())
        if reviewed['classification']!='APPROVED':raise RuntimeError('Independent pre-execution review required')
        bindings={'entry_sha256':Path(__file__),'adapter_source_sha256':Path(__file__).with_name('amd_finite_patch_reconstruct.cpp'),'private_header_sha256':BASE/'project/overlay/CGAL/Polygonal_surface_reconstruction.h','solver_overlay_sha256':BASE/'project/overlay/CGAL/SCIP_mixed_integer_program_traits.h','exe_sha256':BUILD/'Release/official_sample.exe','plan_sha256':BASE/'plan.json','feature_contract_sha256':BASE/'feature-validation-contract.json'}
        if any(owner.sha(path)!=reviewed[key] for key,path in bindings.items()):raise RuntimeError('Reviewed source, executable or contract identity differs')
        auth=json.loads((BASE/'authentication.json').read_text())['Phase14_files']
        for name in ('input/mesh.bin','finite-domain-input.bin','augmented/train.ply','owned-session/arrangement/prepared-planes.csv','owned-session/arrangement/prepared-members.csv'):
            if owner.sha(PRIOR/name)!=manifest_identity(auth,name)['sha256']:raise RuntimeError('Authenticated reconstruction source differs: '+name)
        child=subprocess.Popen(command(stage),executable=command(stage)[0],cwd=str(BASE),env=env,shell=False,stdin=subprocess.PIPE)
        try:
            ready=BASE/'ELIGIBILITY-READY.json'
            while not ready.exists():
                if child.poll() is not None:
                    result=BASE/'native-result.json'
                    if child.returncode==0 and result.exists() and json.loads(result.read_text())['classification']=='NO_GO_INSUFFICIENT_ELIGIBLE':return
                    raise RuntimeError('Eligibility stage ended before GO')
                time.sleep(.05)
            status=json.loads(ready.read_text())
            records=[json.loads(line) for line in (BASE/'owned-session/eligibility.jsonl').read_text().splitlines()]
            decision=eligibility_admission(status,records)
            if decision!='ADMITTED':
                owner.save(BASE/'pre-solve-refusal.json',dict(classification=decision,AI_MIP_started=False,eligibility_summary_sha256=owner.sha(ready),prospective_postrun_guard=True))
                raise RuntimeError('Pre-solve work admission refused: '+decision)
            owner.save(BASE/'one-shot-MIP-authorisation.json',dict(one_AI_solve=True,ready_sha256=owner.sha(ready),same_live_object=True,eligibility_constraints_consume_historical_and_locked_geometry=True))
            child.stdin.write(b'GO\n');child.stdin.close()
            if child.wait()!=0:raise RuntimeError('Constrained native reconstruction failed')
        finally:
            if not child.stdin.closed:child.stdin.close()
            if child.poll() is None:child.wait(timeout=5)
        return
    result=subprocess.run(command(stage),executable=command(stage)[0],cwd=str(BASE),env=env,shell=False)
    if result.returncode:raise RuntimeError('Native stage returned '+str(result.returncode))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=STAGES);p.add_argument('--worker',choices=STAGES);a=p.parse_args(argv)
    if a.worker:worker(a.worker);return 0
    if a.stage:
        owner.BASE=BASE;owner.BUILD=BUILD;owner.__file__=__file__;owner.command=command
        owner.compiler_cleanup_stage=lambda s:s in ('configure','build','build-reviewed')
        owner.compiler_outputs_complete=lambda s:((BUILD/'official_sample.vcxproj').is_file() if s=='configure' else (BUILD/'Release/official_sample.exe').is_file() and (BUILD/'Release/official_sample.exe').stat().st_size>100000)
        return owner.supervise(a.stage)
    print('INERT: Phase15 requires an explicit bounded CPU stage');return 0

if __name__=='__main__':raise SystemExit(main())
