"""Fixed focused CPU suite; never launches a worker, WSL, model or OS Job."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
PYTHON=r'C:\Users\wattz\Documents\Modly\dependencies\venv\Scripts\python.exe'
BRANCH='codex/amd-upstream-image-shape-integration'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!=BRANCH:raise ValueError('Exact isolated upstream integration branch required')
    isolation=sys.argv[1:]==['--fixture-isolation']
    if sys.argv[1:] and not isolation:raise ValueError('Only the focused fixture-isolation followup is supported')
    out=Path(__file__).parent/('cpu-isolated' if isolation else 'cpu-final');out.mkdir(exist_ok=False)
    cases=[('host-authority',['tests.test_amd_upstream_job'],18),
           ('geometry-handoff',['test_amd_upstream_geometry'],7),
           ('affected-current-bridge-regression',['test_amd_current_geometry'],10),
           ('capability-and-non-amd-registry',[
               'tests.test_amd_stage_orchestrator.OrchestratorTests.test_generated_mode_cannot_launch_without_neural_qualification',
               'tests.test_amd_scaffold.AMDScaffoldTests.test_unopted_extension_is_not_discovered_or_imported_in_host',
               'tests.test_generator_registry.GeneratorRegistryDiscoveryTests.test_legacy_generator_supports_eager_and_lazy_sibling_imports'],3)]
    if isolation:cases=cases[1:3]
    results=[]
    for name,modules,count in cases:
        relocate=("import tempfile\n_original_temp=tempfile.TemporaryDirectory\n"
                  "def _current_temp(*a,**kw):\n"
                  "    if str(kw.get('dir','')).replace('\\\\','/').endswith('docs/compatibility/amd-current-image-geometry-evidence'):\n"
                  "        kw['dir']='docs/compatibility/amd-upstream-image-shape-evidence'\n"
                  "    return _original_temp(*a,**kw)\n"
                  "tempfile.TemporaryDirectory=_current_temp\n")
        code=relocate+("import sys,unittest;sys.path[:0]=['api','scripts/compatibility'];"
              "r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames("+
              repr(modules)+"));raise SystemExit(not r.wasSuccessful())")
        start=time.perf_counter()
        result=subprocess.run([PYTHON,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=90)
        elapsed=time.perf_counter()-start
        path=out/(name+'.log');path.write_text(result.stdout+result.stderr,encoding='utf-8')
        passed=result.returncode==0 and ('Ran '+str(count)+' tests in') in result.stderr and '\nOK\n' in result.stderr
        results.append(dict(name=name,modules=modules,tests=count,passed=passed,exit_code=result.returncode,
                            elapsed_seconds=elapsed,log=path.name,log_sha256=sha(path)))
        if not passed:break
    sources=['api/services/amd_upstream_job.py','api/tests/test_amd_upstream_job.py',
        'scripts/compatibility/amd_generation_decode.py','scripts/compatibility/test_amd_upstream_geometry.py',
        'api/services/amd_stage_orchestrator.py','api/services/amd_stage_contract.py',
        'api/services/amd_completed_result.py','api/services/generator_registry.py','api/routers/generation.py',
        'scripts/compatibility/amd_generation_occupancy.py','scripts/compatibility/amd_generation_shape.py']
    sources += ['experimental/extensions/trellis2-amd/inference_decoder/'+name+'.py' for name in
                ('current_input','run_contract','admission','runtime','chunked_large','chunked_final','neural','decode','artifact_io')]
    receipt=dict(base='5501dfd43263a9a155bed50413264b692c171905',branch=branch,
        passed=len(results)==len(cases) and all(row['passed'] for row in results),
        distinct_tests=sum(row['tests'] for row in results),runs=results,python=PYTHON,python_sha256=sha(Path(PYTHON)),
        source_sha256={name:sha(ROOT/name) for name in sources},
        scope='Synthetic CPU contracts and real byte/NPY readers; inherited neural predecessor verifiers substituted only in wiring fixture',
        no_gpu=True,no_wsl=True,no_neural_worker=True,no_model_loading=True,no_os_job=True,bytecode_disabled=True,
        measured_memory_peak=None,genuine_upstream_execution='NOT RUN',
        excluded_router_attempt=dict(result='INTERRUPTED_NO_VERDICT',module='tests.test_generation_router',
            note='An exploratory compatibility selection stopped making progress; its owned command was interrupted. No router retry or repair, and no router PASS is claimed.'))
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=receipt['passed'],tests=receipt['distinct_tests'],
                          elapsed_seconds=sum(row['elapsed_seconds'] for row in results))))
    return 0 if receipt['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
