"""Run only the declared CPU contract/arithmetic regressions; no inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
API=Path(r'C:\Users\wattz\Documents\Modly\dependencies\venv\Scripts\python.exe')
AMD=Path(r'D:\AI\Trellis2AMD\venv\Scripts\python.exe')
CODE='experimental/extensions/trellis2-amd/inference_decoder'
BASE='c6345043b7aa15425161fa53d99ac0f922faf12d'
BRANCH='codex/amd-current-image-geometry-admission'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def command(python,names):
    script=("import sys,unittest;sys.path[:0]="+repr([CODE,'scripts/compatibility','api'])+
            ";r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames("+
            repr(names)+"));raise SystemExit(not r.wasSuccessful())")
    return [str(python),'-B','-c',script]


def main():
    if subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD'],cwd=ROOT,text=True).strip()!=BRANCH:
        raise ValueError('Exact isolated qualification branch required')
    out=Path(__file__).parent/'cpu-final';out.mkdir(exist_ok=False)
    cases=[('current-contract',API,['test_current_input'],11),
           ('host-bridge-and-legacy',AMD,['test_amd_current_geometry','test_amd_generation_decode'],27),
           ('arithmetic-and-historical-profile',AMD,['test_decoder','test_large_tiling','test_final_support',
                                                  'test_tiling_contract','test_admission_reclaim'],48),
           ('writer-readers-and-execution-denial',AMD,['test_current_artifact'],9)]
    rows=[]
    for name,python,modules,count in cases:
        started=time.perf_counter()
        result=subprocess.run(command(python,modules),cwd=ROOT,capture_output=True,text=True,timeout=90)
        elapsed=time.perf_counter()-started
        log=out/(name+'.log');log.write_text(result.stdout+result.stderr,encoding='utf-8')
        passed=(result.returncode==0 and ('Ran '+str(count)+' tests in') in result.stderr and '\nOK\n' in result.stderr)
        rows.append(dict(name=name,modules=modules,expected_tests=count,passed=passed,
            exit_code=result.returncode,elapsed_seconds=elapsed,python=str(python),python_sha256=sha(python),
            log=log.name,log_sha256=sha(log)))
        if not passed:break
    sources=[p for p in (ROOT/CODE).glob('*.py')]
    sources += [ROOT/p for p in ('scripts/compatibility/amd_generation_decode.py',
        'scripts/compatibility/amd_generation_mesh.py','scripts/compatibility/test_amd_current_geometry.py',
        'api/services/amd_stage_contract.py','api/services/amd_stage_orchestrator.py')]
    receipt=dict(schema='modly.amd.current-image-cpu-tests.v1',base=BASE,branch=BRANCH,
        passed=len(rows)==len(cases) and all(row['passed'] for row in rows),
        distinct_tests=sum(row['expected_tests'] for row in rows),runs=rows,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        bytecode_disabled=True,torch_device='CPU only; _lazy_init denied in arithmetic/serializer tests',
        GPU=False,WSL=False,model_loading=False,native_FlexGEMM=False,
        cpu_bridge_predecessors='labelled synthetic receipts; inherited genuine lineage verifier remains mandatory',
        process_memory_peak='not measured; no GPU memory measurement claimed')
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':receipt['passed'],'tests':receipt['distinct_tests'],
                      'elapsed_seconds':sum(row['elapsed_seconds'] for row in rows)}))
    return 0 if receipt['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
