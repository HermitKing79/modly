"""One closed-solid run only after separate approval. Default is no-work dry run."""
import argparse,json,os,re,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_closed_solid_benchmark as b
import amd_zombie_d_handoff as legacy
ROOT=b.ROOT;CASE=b.CASE;PRIVATE=b.PRIVATE;COMPARISON=b.RAW
REFERENCE=b.REFERENCE.relative_to(ROOT).as_posix()
FLOOR=legacy.FLOOR;PREFERRED=legacy.PREFERRED;LIVE_FLOOR=legacy.LIVE_FLOOR
PYTHON=legacy.PYTHON;PYTHON_SHA=legacy.PYTHON_SHA;FORWARDS=dict(legacy.FORWARDS)
STAGES=('occupancy','shape','decode','comparison')
digest=legacy.digest;plain=legacy.plain;save=legacy.save;windows_memory=legacy.windows_memory
fresh_admission=legacy.fresh_admission;validate_admission=legacy.validate_admission;guard_reservation=legacy.guard_reservation
verify_private_inputs=legacy.verify_private_inputs;publication_files=legacy.publication_files
QUALIFIED_COMMIT='bc7160c7439a5d0ea9653e166620b6d535c498ce'
QUALIFIED_FILES=['scripts/compatibility/'+n for n in (
    'amd_zombie_d_handoff.py','amd_zombie_d_stages.py','amd_zombie_d_compare.py','amd_zombie_cpu_mesh.py',
    'amd_generation_occupancy.py','amd_generation_shape.py','amd_generation_decode.py','amd_generation_mesh.py','amd_flexible_dual_grid_cpu.py')]
CRITICAL=QUALIFIED_FILES+['scripts/compatibility/'+n for n in (
    'amd_closed_solid_benchmark.py','amd_closed_solid_cpu.py','amd_closed_solid_render.py',
    'amd_closed_solid_handoff.py','amd_closed_solid_stages.py','Invoke-ClosedSolidBenchmark.ps1','test_amd_closed_solid.py',
    'amd_brick_four_edge_forensics.py','amd_zombie_ram_diagnostic.py','amd_forward_owned_job.py')]+[
    REFERENCE,'docs/compatibility/amd-brick-defect-diagnosis-evidence/topology.py']

def authority():
    path=plain(ROOT/REFERENCE)
    if path.stat().st_size>65536:raise ValueError('Bounded published reference contract required')
    row=json.loads(path.read_bytes())
    if (row.get('schema')!='modly.amd.closed-solid-reference.v1' or row.get('dimensions')!=list(b.DIMENSIONS)
        or row.get('chamfer')!=b.CHAMFER or row.get('seed')!=b.SEED or row.get('resolution')!=512
        or row.get('candidate_count')!=1 or row.get('reference_views')!=1 or row.get('neural_execution_hold') is not True
        or row.get('automatic_retries')!=0 or row.get('ground_truth_may_replace_neural_output') is not False
        or row.get('source_path')!=(CASE/'occupancy/source-image.png').relative_to(ROOT).as_posix()
        or row.get('input_contract_path')!=(CASE/'occupancy/input-contract.json').relative_to(ROOT).as_posix()):raise ValueError('Exact closed-solid reference contract required')
    verify_private_inputs(ROOT,row)
    expected={str((b.PREP/name).relative_to(ROOT.parent).as_posix()) for name in ('source/mesh.ply','source/vertices.npy','source/faces.npy','reference.png','conditioning-512.png','silhouette-512.png','render.json','qualification.json')}
    if set(row['files'])!=expected:raise ValueError('Exact ground truth/reference artifact closure required')
    legacy.verify_qualification(ROOT.parent,row['files'])
    expected_evidence={(b.PREP/name).relative_to(ROOT.parent).as_posix() for name in b.PREPARATION_EVIDENCE}
    if set(row['preparation_evidence'])!=expected_evidence:raise ValueError('Exact owned preparation evidence required')
    legacy.verify_qualification(ROOT.parent,row['preparation_evidence'])
    for stage in ('render','qualify'):
        receipt=json.loads((b.PREP/(stage+'-run/host-result.json')).read_bytes())
        if (receipt['classification']!='PASS' or receipt['worker_returncode']!=0 or receipt['process_cap_bytes']!=6*1024**3
            or receipt['deadline_seconds']!=180 or receipt['elapsed_seconds']>=180 or receipt['peak_process_committed_bytes']>6*1024**3
            or not receipt['worker_exit_verified'] or receipt['cleanup_errors'] or not receipt['job_cleanup']['owned_tree_exit_verified']
            or receipt['job_cleanup']['forced_job_cleanup']):raise ValueError('Bounded natural CPU preparation required')
        for resource in [receipt['initial_resource'],*receipt['samples']]:
            if (resource['required_RAM_bytes']!=12*1024**3 or resource['required_disk_bytes']!=2*1024**3 or not resource['admitted']
                or resource['available_RAM_bytes']<12*1024**3 or resource['free_disk_bytes']<2*1024**3):raise ValueError('CPU preparation resource evidence differs')
    sources=json.loads((b.PREP/'sources-before.json').read_bytes());after=json.loads((b.PREP/'sources-after.json').read_bytes())
    if after!=dict(unchanged=True,files=len(sources)):raise ValueError('Preparation source settlement differs')
    for name in ('amd_closed_solid_benchmark.py','amd_closed_solid_cpu.py','amd_closed_solid_render.py'):
        if digest(b.PREP/'executed-source'/name)!=sources[str((ROOT/'scripts/compatibility'/name).resolve())]['sha256']:raise ValueError('Executed preparation source differs')
    q=json.loads((b.PREP/'qualification.json').read_bytes())
    if q['classification']!='PASS_REFERENCE' or q['nontrivial_self_intersections']!=0 or q['conditioning_512_pixel_sha256']!=row['conditioning_512_pixel_sha256']:raise ValueError('Reference qualification differs')
    topology=q['topology']
    if (any(topology[k]!=value for k,value in dict(vertices=24,triangles=44,components=1,boundary=0,nonmanifold=0,winding=0,euler=2).items())
        or not topology['convex'] or topology['signed_volume']<=0 or q['all_triangle_pairs_checked']!=946
        or q['source_image_sha256']!=row['source_sha256'] or q['input_contract_sha256']!=row['input_contract_sha256']
        or q['crop_rectangle']!=row['crop_rectangle'] or q['neural_attempt_reserved'] is not False):raise ValueError('Exact closed-solid qualification required')
    return row

def verify_publication(commit):
    if not re.fullmatch('[0-9a-f]{40}',commit or '') or legacy.git(['rev-parse','HEAD']).strip()!=commit:raise ValueError('Exact published current checkout required')
    for name in CRITICAL:
        actual=(ROOT/name).read_text(encoding='utf-8').replace('\r\n','\n')
        if legacy.git(['show',commit+':'+name]).replace('\r\n','\n')!=actual:raise ValueError('Published source differs: '+name)
        if name in QUALIFIED_FILES and legacy.git(['show',QUALIFIED_COMMIT+':'+name]).replace('\r\n','\n')!=actual:raise ValueError('Previously working stage changed: '+name)

def require_fresh_outputs():
    for path in (PRIVATE,COMPARISON,CASE/'shape',CASE/'decode'):
        plain(path)
        if path.exists():raise FileExistsError('One-shot destination consumed; no retry: '+str(path))
    if {p.name for p in plain(CASE/'occupancy').iterdir()}!={'source-image.png','input-contract.json'}:raise FileExistsError('Prepared occupancy already consumed')

def execute_once(stage):
    for name in STAGES:
        print('CLOSED_SOLID_STAGE='+name,flush=True)
        if stage(name)!=0:raise RuntimeError('Stage failed; no retry or later stage: '+name)

def run(commit):
    import ctypes
    if os.name!='nt' or ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('Normal non-administrator Windows required')
    if Path(sys.executable).resolve()!=PYTHON.resolve() or digest(PYTHON)!=PYTHON_SHA:raise ValueError('Qualified Windows interpreter required')
    if {'amd_generation_occupancy','amd_generation_shape','amd_generation_decode','amd_sparse_decoder_forward'}.intersection(sys.modules):raise RuntimeError('Fresh one-shot process required')
    verify_publication(commit);authority();require_fresh_outputs();PRIVATE.mkdir(exist_ok=False)
    result=dict(classification='FAIL',harness_commit=commit,maximum_forwards=FORWARDS,neural_attempt_reserved=False,postprocessing=False)
    try:
        if shutil.disk_usage(PRIVATE).free<2*1024**3:raise RuntimeError('Original 2 GiB output disk floor failed')
        readings=[]
        def record(row):
            readings.append(row);save(PRIVATE/f'admission-{len(readings)}.json',row)
            print('AVAILABLE_RAM_GIB='+str(row['available_bytes']/1024**3),flush=True)
        rows=fresh_admission(windows_memory,time.sleep,record);validate_admission(rows,datetime.now(timezone.utc))
        save(PRIVATE/'admission.json',dict(readings=rows,required_bytes=FLOOR,preferred_only_bytes=PREFERRED))
        sys.path[:0]=[str(ROOT/'scripts/compatibility'),'D:/AI/Trellis2AMD/venv/Lib/site-packages']
        os.environ['WORKSPACE_DIR']=str(ROOT);os.environ['MODLY_ZOMBIE_D_COMMIT']=commit;os.environ['MODLY_CLOSED_SOLID_COMMIT']=commit
        import amd_closed_solid_stages as stages
        bridge=stages.Bridge(commit,read=windows_memory)
        guard_reservation(windows_memory,lambda row:save(PRIVATE/'pipeline-final-admission.json',row),
            lambda:save(PRIVATE/'pipeline-attempt-started.json',dict(harness_commit=commit,maximum_pipelines=1,maximum_forwards=FORWARDS,source_sha256=authority()['source_sha256'])))
        result['neural_attempt_reserved']=True;execute_once(bridge.dispatch)
        result['classification']='BENCHMARK_EVIDENCE_COMPLETE_RAW_QUALITY_REQUIRES_REVIEW';result['raw_evidence']=str(COMPARISON)
    except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc);raise
    finally:save(PRIVATE/'pipeline-result.json',result);print('CLOSED_SOLID_RESULT='+result['classification'],flush=True)
    return 0

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--approved-one-attempt',action='store_true');p.add_argument('--harness-commit');a=p.parse_args(argv)
    if not a.run:
        print(json.dumps(dict(mode='DRY_RUN',execution_hold=True,separate_approval_required=True,neural_execution=False,WSL_launched=False,marker_reserved=False,
            stages=list(STAGES),seed=b.SEED,resolution=512,maximum_forwards=FORWARDS,admission_bytes=FLOOR,preferred_only_bytes=PREFERRED,
            readings=3,spacing_seconds=4,final_pre_marker_bytes=FLOOR,live_host_floor_bytes=LIVE_FLOOR,case=str(CASE),raw=str(COMPARISON),postprocessing=False)));return 0
    if not a.approved_one_attempt or not re.fullmatch('[0-9a-f]{40}',a.harness_commit or ''):p.error('Execution HOLD: separate approval, --approved-one-attempt and exact --harness-commit required')
    try:return run(a.harness_commit)
    except Exception as exc:print(type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True);return 2
if __name__=='__main__':raise SystemExit(main())
