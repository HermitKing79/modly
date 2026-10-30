"""Phase16 bounded closure/certification proof; no production execution route."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_cgal_proof as owner
import amd_finite_patch_proof as old
import amd_closure_graph as graph
PRIOR=old.PRIOR
PHASE15=old.BASE
BASE=PRIOR.parent/'amd-closure-feasibility-20261030-v1'
BUILD=BASE/'build'
STAGES=('authenticate','backend','configure','build','build-mapping','build-diagnostic','configure-certify','build-certify','build-refine','certify','refine','capture','capture-mapping','analyse-mapping','diagnostic','merge-certificates','fidelity-screen','analyse','preserve')

def check(path,entry):
    s=path.stat()
    if path.is_symlink() or s.st_nlink!=1 or s.st_size!=entry['bytes'] or owner.sha(path)!=entry['sha256']:
        raise ValueError('Authenticated file differs: '+str(path))
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,file_id=s.st_ino,sha256=entry['sha256'])

def authenticate():
    old.BASE=BASE
    old.authenticate()
    checked={}
    for name,row in json.loads((PHASE15/'evidence-manifest.json').read_text()).items():
        checked[name]=check(PHASE15/name,row)
    pin=json.loads((PHASE15/'build-pin.json').read_text())
    for name,key in (('build/Release/official_sample.exe','exe_sha256'),('project/overlay/CGAL/Polygonal_surface_reconstruction.h','private_header_sha256'),('project/overlay/CGAL/SCIP_mixed_integer_program_traits.h','solver_overlay_sha256')):
        if owner.sha(PHASE15/name)!=pin[key]:raise ValueError('Phase15 binary/header differs')
    previous=json.loads((PHASE15/'preservation-after.json').read_text())['protected_after']
    for name,row in previous.items():
        s=Path(name).stat()
        if any(getattr(s,attr)!=row[key] for key,attr in [('bytes','st_size'),('mtime_ns','st_mtime_ns'),('file_id','st_ino')]):raise ValueError('Protected identity differs: '+name)
    owner.save(BASE/'phase15-authentication.json',dict(classification='PASS',files=checked,protected_records=previous))
    rows=[json.loads(x) for x in (PHASE15/'owned-session/eligibility.jsonl').read_text().splitlines()]
    states={'CERTIFIED_ELIGIBLE':0,'CONFIRMED_UNSUPPORTED':1,'UNRESOLVED':2}
    if len(rows)!=1105 or [r['variable'] for r in rows]!=list(range(1105)) or len({r['key'] for r in rows})!=1105:raise ValueError('Prior mapping')
    counts={s:sum(r['state']==s for r in rows) for s in states}
    if list(counts.values())!=[79,719,307]:raise ValueError('Prior states')
    with (BASE/'prior-faces.txt').open('x',encoding='ascii') as f:
        f.write('1105\n')
        for r in rows:
            if r['state']=='CONFIRMED_UNSUPPORTED' and r['reason']!='OUTSIDE_COMPATIBLE_DOMAIN':raise ValueError('Unsupported provenance')
            values=[r['variable'],r['group'],states[r['state']],*r['witness'],r['witness_distance_native'],r['key']]
            f.write(' '.join(map(str,values))+'\n')
    owner.save(BASE/'prior-input-pin.json',dict(sha256=owner.sha(BASE/'prior-faces.txt'),counts=counts,original_records_sha256=owner.sha(PHASE15/'owned-session/eligibility.jsonl')))

def command(stage):
    if stage=='configure-certify':
        c=command('configure');c[c.index('-S')+1]=str(BASE/'certify-project');c[c.index('-B')+1]=str(BASE/'certify-build');return c
    if stage in ('build-certify','build-refine'):return [str(owner.CMAKE),'--build',str(BASE/'certify-build'),'--config','Release','--target','official_sample','--parallel','1']
    if stage=='certify':return [str(BASE/'certify-build/Release/official_sample.exe'),str(BASE/'certification'),str(PRIOR),'closure-certify-v1']
    if stage=='refine':return [str(BASE/'certify-build/Release/official_sample.exe'),str(BASE/'refinement'),str(PRIOR),'closure-refine-v1']
    if stage=='configure':
        c=old.previous.command('configure-shared');c[c.index('-S')+1]=str(BASE/'project');c[c.index('-B')+1]=str(BUILD);return c
    if stage in ('build','build-mapping','build-diagnostic'):return [str(owner.CMAKE),'--build',str(BUILD),'--config','Release','--target','official_sample','--parallel','1']
    if stage=='capture':return [str(BUILD/'Release/official_sample.exe'),str(BASE),str(PRIOR),'closure-capture-v1']
    if stage=='capture-mapping':return [str(BUILD/'Release/official_sample.exe'),str(BASE/'mapping-capture'),str(PRIOR),'closure-mapping-v1']
    if stage=='diagnostic':return [str(BUILD/'Release/official_sample.exe'),str(BASE/'diagnostic'),str(PRIOR),'closure-diagnostic-v1']
    raise ValueError('No native route')

def spawn_allowed(stage,args):
    if stage not in ('configure','build','build-mapping','build-diagnostic','configure-certify','build-certify','build-refine','certify','refine','capture','capture-mapping','diagnostic'):return False
    exe,argv,cwd,_=args;c=command(stage)
    return exe==c[0] and argv in (c,tuple(c),subprocess.list2cmdline(c)) and cwd==str(BASE)

def bindings():
    return {'entry':Path(__file__),'capture_source':Path(__file__).with_name('amd_closure_capture.cpp'),
            'graph_source':Path(graph.__file__),'header':BASE/'project/overlay/CGAL/Polygonal_surface_reconstruction.h',
            'common':BASE/'project/phase15-common.inc','solver':BASE/'project/overlay/CGAL/SCIP_mixed_integer_program_traits.h',
            'cmake_project':BASE/'project/CMakeLists.txt','validation_include':Path(__file__).with_name('amd_mesh_aware_validate.cpp'),
            'compiler_helper_pin':BASE/'compiler-helper-pin.json',
            'executable':BUILD/'Release/official_sample.exe','plan':BASE/'plan.json',
            'prior_faces':BASE/'prior-faces.txt'}

def certificate_bindings(refine=False):
    result={'entry':Path(__file__),'source':Path(__file__).with_name('amd_closure_certify.cpp'),
            'core':BASE/'certify-project/certificate-core.inc','project':BASE/'certify-project/CMakeLists.txt',
            'executable':BASE/'certify-build/Release/official_sample.exe','input':BASE/'certification/frozen-polygons.txt',
            'plan':BASE/'certificate-plan.json','native_polygons':BASE/'diagnostic/actual-polygons.jsonl',
            'native_fans':BASE/'diagnostic/actual-fans.jsonl','validation_include':Path(__file__).with_name('amd_mesh_aware_validate.cpp')}
    if refine:result.update(input=BASE/'refinement/frozen-polygons.txt',plan=BASE/'refinement-plan.json',targets=BASE/'refinement/refine-ids.txt',original_certificates=BASE/'certification/certificates.jsonl')
    return result

def diagnostic_admission(root):
    receipt=json.loads((root/'capture-result.json').read_text())
    faces=[json.loads(x) for x in (root/'live-faces.jsonl').read_text().splitlines()]
    fans=[json.loads(x) for x in (root/'live-fans.jsonl').read_text().splitlines()]
    witnesses=[json.loads(x) for x in (root/'witness-verification.jsonl').read_text().splitlines()]
    if receipt.get('classification')!='AUTHENTICATED_ADJACENCY_CAPTURE' or receipt.get('unsupported_witnesses_verified')!=719 or not receipt.get('all_face_edges_accounted'):raise ValueError('Incomplete witness capture')
    if len(faces)!=1105 or [r['variable'] for r in faces]!=list(range(1105)) or len({r['key'] for r in faces})!=1105:raise ValueError('Live mapping')
    if [r['fan'] for r in fans]!=list(range(len(fans))):raise ValueError('Live fan numbering')
    if any(r['state'] not in ('CONFIRMED_UNSUPPORTED','POTENTIALLY_ELIGIBLE') for r in faces):raise ValueError('Unknown diagnostic state')
    unsupported=[r['variable'] for r in faces if r['state']=='CONFIRMED_UNSUPPORTED']
    if len(unsupported)!=719 or len(witnesses)!=719 or {r['variable'] for r in witnesses}!=set(unsupported):raise ValueError('Missing witness')
    if any(not r['PASS'] or not r['unsupported_lower_bound_native']>0.20000001 for r in witnesses):raise ValueError('Unverified exclusion')
    proof=graph.prune(1105,[r['faces'] for r in fans],unsupported)
    proof['evidence_sha256']={name:owner.sha(root/name) for name in ('capture-result.json','live-faces.jsonl','live-fans.jsonl','witness-verification.jsonl','actual-polygons.jsonl','actual-fans.jsonl','live-correspondence.jsonl')}
    return proof

def merge_certificates(first,updates):
    rows={r['variable']:r for r in first}
    if len(rows)!=1105 or set(rows)!=set(range(1105)):raise ValueError('Incomplete original certificates')
    unresolved={i for i,r in rows.items() if r['state']=='UNRESOLVED'}
    if len(updates)!=2 or len(unresolved)!=2 or {r['variable'] for r in updates}!=unresolved:raise ValueError('Target scope differs')
    for r in updates:
        original=rows[r['variable']]
        if r['key']!=original['key'] or r['group']!=original['group'] or original['reason']!='DEPTH_LIMIT_OR_NUMERICAL_BAND':raise ValueError('Refined identity differs')
        rows[r['variable']]=r
    if any('RESOURCE_EXHAUSTION' in r['coverage'] for r in rows.values()):raise ValueError('Certification resource incomplete')
    return [rows[i] for i in range(1105)]

def worker(stage):
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO and EOF required')
    from amd_zombie_d_compare import audit_policy
    ordinary=audit_policy(BASE)
    def audit(event,args):
        if event=='subprocess.Popen':
            if not spawn_allowed(stage,args):raise RuntimeError('Exact CPU command required')
            return
        ordinary(event,args)
    sys.addaudithook(audit)
    if owner.sha(owner.ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json')!='397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0':raise ValueError('Frozen criteria changed')
    if stage=='authenticate':authenticate();return
    if stage=='backend':
        import amd_mesh_aware_input as data
        data.run.BASE=BASE;data.verify_backend();return
    if stage=='fidelity-screen':
        review=json.loads((BASE/'fidelity-review.json').read_text())
        for name,p in {'entry':Path(__file__),'fidelity':Path(__file__).with_name('amd_closure_fidelity.py'),'graph':Path(graph.__file__),'certificates':BASE/'complete-certificates.jsonl','screen':BASE/'certified-closure-screen.json','native_polygons':BASE/'diagnostic/actual-polygons.jsonl','native_fans':BASE/'diagnostic/actual-fans.jsonl'}.items():
            if review['classification']!='APPROVED' or owner.sha(p)!=review['bindings'][name]:raise ValueError('Fidelity review differs')
        previous=json.loads((BASE/'certificate-review.json').read_text())['bindings']
        if any(review['bindings'][name]!=previous[name] for name in ('native_polygons','native_fans')):raise ValueError('Original certified arrangement differs')
        import amd_closure_fidelity
        amd_closure_fidelity.run();return
    if stage=='analyse-mapping':
        from amd_closure_mapping import map_faces
        oldrows=[json.loads(x) for x in (PHASE15/'owned-session/eligibility.jsonl').read_text().splitlines()]
        live=[json.loads(x) for x in (BASE/'mapping-capture/actual-polygons.jsonl').read_text().splitlines()]
        result=map_faces(oldrows,live)
        result.update(original_sha256=owner.sha(PHASE15/'owned-session/eligibility.jsonl'),actual_sha256=owner.sha(BASE/'mapping-capture/actual-polygons.jsonl'))
        owner.save(BASE/'identity-correspondence.json',result);return
    if stage=='merge-certificates':
        first=[json.loads(x) for x in (BASE/'certification/certificates.jsonl').read_text().splitlines()]
        updates=[json.loads(x) for x in (BASE/'refinement/certificates.jsonl').read_text().splitlines()]
        rows=merge_certificates(first,updates)
        live=[json.loads(x) for x in (BASE/'diagnostic/actual-polygons.jsonl').read_text().splitlines()]
        if any(r['key']!=live[r['variable']]['key'] for r in rows):raise ValueError('Frozen certified face differs')
        with (BASE/'complete-certificates.jsonl').open('x') as f:
            for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
        fans=[json.loads(x)['faces'] for x in (BASE/'diagnostic/actual-fans.jsonl').read_text().splitlines()]
        proof=graph.prune(1105,fans,[r['variable'] for r in rows if r['state']!='CERTIFIED_ELIGIBLE'])
        proof.update(certificates_sha256=owner.sha(BASE/'complete-certificates.jsonl'),fan_sha256=owner.sha(BASE/'diagnostic/actual-fans.jsonl'),counts={s:sum(r['state']==s for r in rows) for s in ('CERTIFIED_ELIGIBLE','CONFIRMED_UNSUPPORTED','UNRESOLVED')},credible_route_only_not_closure_proof=bool(proof['remaining']))
        owner.save(BASE/'certified-closure-screen.json',proof);return
    if stage in ('certify','refine'):
        review=json.loads((BASE/('refinement-review.json' if stage=='refine' else 'certificate-review.json')).read_text())
        if review['classification']!='APPROVED' or any(owner.sha(p)!=review['bindings'][k] for k,p in certificate_bindings(stage=='refine').items()):raise ValueError('Reviewed certification differs')
        if json.loads((BASE/'diagnostic/native-result.json').read_text())['classification']!='OPTIMISTIC_NONEMPTY_SELECTION_EXISTS':raise ValueError('Stage A did not qualify')
        a=json.loads((BASE/'authentication.json').read_text())['Phase14_files']
        for name in ('input/mesh.bin','finite-domain-input.bin'):check(PRIOR/name,old.manifest_identity(a,name))
    if stage=='analyse':
        native=json.loads((BASE/'native-result.json').read_text())
        if native['classification']!='AUTHENTICATED_ADJACENCY_CAPTURE' or native['solver_calls']!=0:raise ValueError('Capture incomplete')
        faces=[json.loads(x) for x in (BASE/'live-faces.jsonl').read_text().splitlines()]
        rows=[json.loads(x) for x in (BASE/'live-fans.jsonl').read_text().splitlines()]
        if [r['fan'] for r in rows]!=list(range(len(rows))) or [r['variable'] for r in faces]!=list(range(len(faces))):raise ValueError('Incomplete native records')
        result=graph.prune(len(faces),[r['faces'] for r in rows],[r['variable'] for r in faces if r['state']=='CONFIRMED_UNSUPPORTED'])
        result.update(live_faces_sha256=owner.sha(BASE/'live-faces.jsonl'),live_fans_sha256=owner.sha(BASE/'live-fans.jsonl'),not_a_claim_about_original_AI_geometry=True,solver_calls=0)
        owner.save(BASE/'closure-proof.json',result);return
    if stage=='preserve':
        a=json.loads((BASE/'authentication.json').read_text());b=json.loads((BASE/'phase15-authentication.json').read_text())
        for name,row in a['Phase14_files'].items():check(PRIOR/name,row)
        for name,row in a['original_snapshot'].items():check(Path(name),row)
        for name,row in b['files'].items():check(PHASE15/name,row)
        for name,row in b['protected_records'].items():
            s=Path(name).stat()
            if (s.st_size,s.st_mtime_ns,s.st_ino)!=(row['bytes'],row['mtime_ns'],row['file_id']):raise ValueError('Protected record changed')
        owner.save(BASE/'preservation-after.json',dict(classification='PASS',Phase14_files=len(a['Phase14_files']),Phase15_files=len(b['files']),original_sources=len(a['original_snapshot']),protected_records=len(b['protected_records'])));return
    if stage in ('capture','capture-mapping','diagnostic'):
        reviews={'capture':'pre-execution-review.json','capture-mapping':'mapping-review.json','diagnostic':'diagnostic-review.json'}
        review=json.loads((BASE/reviews[stage]).read_text())
        if review['classification']!='APPROVED' or any(owner.sha(p)!=review['bindings'][k] for k,p in bindings().items()):raise ValueError('Reviewed capture differs')
        a=json.loads((BASE/'authentication.json').read_text())['Phase14_files']
        for name in ('input/mesh.bin','finite-domain-input.bin','augmented/train.ply','owned-session/arrangement/prepared-planes.csv','owned-session/arrangement/prepared-members.csv'):
            check(PRIOR/name,old.manifest_identity(a,name))
        if json.loads((BASE/'backend-reauthentication.json').read_text())['classification']!='PASS':raise ValueError('Backend not qualified')
        if stage=='diagnostic':
            d=BASE/'diagnostic'
            if owner.sha(BASE/'diagnostic-plan.json')!=review['diagnostic_plan_sha256']:raise ValueError('Diagnostic scope differs')
            if owner.sha(d/'prior-polygons.txt')!=review['prior_polygons_sha256']:raise ValueError('Diagnostic input differs')
            c=command(stage);env=dict(os.environ,PATH=str(owner.SCIP/'bin')+os.pathsep+os.environ.get('PATH',''))
            child=subprocess.Popen(c,executable=c[0],cwd=str(BASE),env=env,shell=False,stdin=subprocess.PIPE)
            try:
                while not (d/'DIAGNOSTIC-READY.json').exists():
                    if child.poll() is not None:raise RuntimeError('Native diagnostic admission failed')
                    time.sleep(.05)
                proof=diagnostic_admission(d);owner.save(d/'necessary-condition-proof.json',proof)
                go=bool(proof['remaining'])
                owner.save(d/'host-decision.json',dict(classification='DIAGNOSTIC_GO' if go else 'STOP_GRAPH_PROVEN_EMPTY',remaining=len(proof['remaining']),proof_sha256=owner.sha(d/'necessary-condition-proof.json'),supported_asset_authorised=False,production_solve_authorised=False))
                child.stdin.write(b'DIAGNOSTIC_GO\n' if go else b'STOP\n');child.stdin.close();code=child.wait()
                if go and code!=0:raise RuntimeError('Diagnostic native failure')
                if not go and ((d/'DIAGNOSTIC-MIP-started.txt').exists() or code!=10):raise RuntimeError('Native graph stop mismatch')
            finally:
                if not child.stdin.closed:child.stdin.close()
                if child.poll() is None:child.wait(timeout=5)
            return
    c=command(stage);env=dict(os.environ,PATH=str(owner.SCIP/'bin')+os.pathsep+os.environ.get('PATH',''))
    if subprocess.run(c,executable=c[0],cwd=str(BASE),env=env,shell=False).returncode:raise RuntimeError('Native CPU stage failed')

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES);p.add_argument('--worker',choices=STAGES);a=p.parse_args()
    if a.worker:worker(a.worker);return 0
    if not a.stage:print('INERT: explicit bounded stages on exclusive paths; no production route');return 0
    chosen=BASE/'certify-build' if a.stage in ('configure-certify','build-certify','build-refine') else BUILD
    owner.BASE=BASE;owner.BUILD=chosen;owner.__file__=__file__;owner.command=command
    owner.compiler_cleanup_stage=lambda s:s in ('configure','build','build-mapping','build-diagnostic','configure-certify','build-certify','build-refine')
    owner.compiler_outputs_complete=lambda s:(chosen/'official_sample.vcxproj').is_file() if s in ('configure','configure-certify') else (chosen/'Release/official_sample.exe').is_file() and (chosen/'Release/official_sample.exe').stat().st_size>100000
    return owner.supervise(a.stage)

if __name__=='__main__':raise SystemExit(main())
