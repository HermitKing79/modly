"""Source-safe textual outcome only; no private image/geometry publication."""
from pathlib import Path
import hashlib,json
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-uv-bake-18c0c92'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    stages={}
    for name,base in [('initial_nomination',P),('read_only_nine_rays',P),('corrected_UV_preparation',P/'qualified')]:
        stage='nomination' if name=='read_only_nine_rays' else 'bake'
        result=base/(stage+'-host-result.json');r=json.loads(result.read_text())
        samples=[r['initial_resources']]+r['resource_samples']
        assert r['watched_sources_unchanged'] and r['job_cleanup']['owned_tree_exit_verified'] and not r['job_cleanup']['forced_job_cleanup']
        stages[name]={k:r[k] for k in ('classification','returncode','elapsed_seconds','peak_process_committed_bytes','job_cleanup','watched_sources_unchanged')}
        stages[name].update(measurement='Windows Job peak owned-process committed bytes, not working-set or whole-host peak',
            observed_Windows_available_RAM_min_bytes=min(x['available_ram_bytes'] for x in samples),
            observed_destination_free_disk_min_bytes=min(x['disk_free_bytes'] for x in samples),
            samples=len(samples),host_result_sha256=sha(result),attempt_marker_sha256=sha(base/(stage+'-attempt.json')),
            stdout_sha256=sha(base/stage/'stdout.txt'),stderr_sha256=sha(base/stage/'stderr.txt'),
            exact_stderr=(base/stage/'stderr.txt').read_text())
    assert stages['corrected_UV_preparation']['returncode']==1 and 'Analytic positive-area UV overlap' in stages['corrected_UV_preparation']['exact_stderr']
    assert not any((P/'qualified'/x).exists() for x in ('export','compare','finish'))
    assert {x.name for x in (P/'qualified/bake').iterdir()}=={'scratch','stdout.txt','stderr.txt'}
    record=dict(classification='BLOCKED',base='18c0c92fe348e8a7fc04e9b5c9443423484ee5b6',branch='codex/amd-brick-uv-bake-proof',
        blocker='Analytic positive-area overlap in the selected unchanged patch planar UV projection',
        interpretation='Preparation/UV parameterization blockage. No completed texture bake or measured visual-detail result; not evidence of bad learned materials or native/GPU incompatibility.',
        real_asset_texture_sampling_calls=0,real_texture_images_written=0,Blender_launches=0,model_or_GPU_execution=False,
        stages=stages,focused_CPU_tests=dict(passed=29,failed=0,seconds=.512,log_sha256=sha(E/'focused-tests-final.txt')),
        source_semantics='Feasible normalized sparse trilinear CPU port; synthetic tests only; native GPU bitwise parity unqualified',
        original_mesh_sha256='ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd',
        genuine_material_sha256='8cd735630149e28829efe686d30f1c7c255dde76748dc3d525236fb5c2c81394',
        preservation_after_sha256=sha(E/'preservation-after.json'),
        no_actual_texture_or_render_quality_claim=True,
        missing_observations=['Exact overlap pair/area and selected face count were not journalled before rejection','No baked map/GLB/render exists; texture coverage, seams and detail-loss metrics are unmeasured'],
        next_experiment='One separately approved seam-aware UV unwrap of this same small unchanged patch, preserving every original oriented triangle, then repeat the source-spatial bake and independent comparison. Do not repair geometry or scale to the full model.')
    with (E/'execution-results.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:record[k] for k in ('classification','stages','focused_CPU_tests')}))

if __name__=='__main__':main()
