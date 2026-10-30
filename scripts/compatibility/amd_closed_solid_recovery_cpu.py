"""Scoped unchanged CPU extraction/measurement/rendering after recovery PASS."""
import argparse,contextlib,json,os,re,subprocess,sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_closed_solid_recovery as h

def lineage_for(execution,original):
    def checked(contract,shape,admission,input_contract,input_sha,feature,commit):
        if commit!=execution:raise ValueError('Foreign CPU recovery execution commit')
        return original(contract,shape,admission,input_contract,input_sha,feature,h.UPSTREAM)
    return checked

def original_render_command():
    return [str(h.original.b.BLENDER),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1','--python',str(Path(__file__).with_name('amd_closed_solid_render.py')),'--','--generated']

def render_command(command):
    if command!=original_render_command():raise ValueError('Only unchanged generated reference-camera render permitted')
    return command[:-3]+[str(Path(__file__).resolve()),'--','--render','--generated']

def closure(row):
    clean=(row['edges']['boundary_incidence_1']==0 and row['edges']['nonmanifold_incidence_gt_2']==0
        and row['two_face_winding_conflicts']==0 and row['components']['face_components']==1
        and row['duplicate_index_triangles_ignoring_winding']==0
        and row['degeneracy']['zero_area_faces_float64']==0 and row['degeneracy']['repeated_index_faces']==0
        and row['degeneracy']['nonfinite_vertex_rows']==0 and row['degeneracy']['faces_with_nonfinite_coordinates']==0)
    return dict(closed_oriented_edge_manifold_single_component=clean,
        component_definition='Face-bearing vertex-index connectivity; not edge-connected sheets',
        watertightness_scope='Combinatorial edge closure/orientation only; no geometric watertightness certification',
        self_intersections_verified=False,vertex_manifold_verified=False,texture_ready=False,geometry_repaired=False,
        recommendation='CANDIDATE_REQUIRES_VISUAL_AND_INTERSECTION_REVIEW' if clean else 'NO_GO_RAW_CLOSED_SOLID_TOPOLOGY')

@contextlib.contextmanager
def configured(commit):
    if not re.fullmatch('[a-f0-9]{40}',commit or ''):raise ValueError('Exact CPU recovery commit required')
    b=h.original.b
    cpu=b.load('_closed_solid_recovery_cpu_algorithms',Path(__file__).with_name('amd_closed_solid_cpu.py'))
    routed=b.load('_closed_solid_recovery_cpu_spec',Path(__file__).with_name('amd_closed_solid_benchmark.py'));routed.RAW=h.RAW
    cpu.b=routed;cpu.__file__=str(Path(__file__).resolve())
    factory=cpu.decode_reader
    def reader():
        value=factory();value.DECODE=h.DECODE;value.SOURCE=h.DECODE/'run/guest/shape-decoded.safetensors';value.BASE=h.RAW
        value.validate_lineage=lineage_for(commit,value.validate_lineage)
        return value
    cpu.decode_reader=reader
    measure=cpu.topology_and_shape
    def analyzed():
        measure();source=h.RAW/'raw-quality.json'
        h.save(h.RAW/'closed-solid-closure.json',dict(closure(json.loads(source.read_text())),raw_quality_sha256=h.digest(source)))
    cpu.topology_and_shape=analyzed
    launch=subprocess.Popen
    def route(command,**kwargs):
        if command and command[0]==str(b.BLENDER):command=render_command(command)
        return launch(command,**kwargs)
    cpu.subprocess=SimpleNamespace(Popen=route,PIPE=subprocess.PIPE)
    proxy=ModuleType('amd_closed_solid_handoff');proxy.__dict__.update(vars(h.original))
    proxy.verify_publication=h.verify_publication;proxy.CRITICAL=h.CRITICAL
    old=sys.modules.get('amd_closed_solid_handoff');sys.modules['amd_closed_solid_handoff']=proxy
    try:yield cpu
    finally:
        if old is None:sys.modules.pop('amd_closed_solid_handoff',None)
        else:sys.modules['amd_closed_solid_handoff']=old

def future(commit):
    previous=os.environ.get('MODLY_CLOSED_SOLID_RECOVERY_COMMIT');os.environ['MODLY_CLOSED_SOLID_RECOVERY_COMMIT']=commit
    try:
        with configured(commit) as cpu:return cpu.future(commit)
    finally:
        if previous is None:os.environ.pop('MODLY_CLOSED_SOLID_RECOVERY_COMMIT',None)
        else:os.environ['MODLY_CLOSED_SOLID_RECOVERY_COMMIT']=previous

def main(argv=None):
    args=sys.argv[sys.argv.index('--')+1:] if argv is None and '--' in sys.argv else argv
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=('extract','analyze','compare'));p.add_argument('--render',action='store_true');p.add_argument('--generated',action='store_true');a=p.parse_args(args)
    if not a.worker and not a.render:print('DRY_RUN: CPU routing only, no work');return 0
    commit=os.environ.get('MODLY_CLOSED_SOLID_RECOVERY_COMMIT','')
    with configured(commit) as cpu:
        if a.render:
            if not a.generated or a.worker:raise ValueError('Only generated render mode')
            script=Path(__file__).with_name('amd_closed_solid_render.py');old_argv=sys.argv
            old_b=sys.modules.get('amd_closed_solid_benchmark');sys.modules['amd_closed_solid_benchmark']=cpu.b
            try:
                sys.argv=[str(script),'--','--generated']
                exec(compile(script.read_bytes(),str(script),'exec'),dict(__name__='__main__',__file__=str(script)))
            finally:
                sys.argv=old_argv
                if old_b is None:sys.modules.pop('amd_closed_solid_benchmark',None)
                else:sys.modules['amd_closed_solid_benchmark']=old_b
        else:return cpu.worker(a.worker)
    return 0

if __name__=='__main__':raise SystemExit(main())
