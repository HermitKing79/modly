"""Record preserved CPU diagnostics and checksum new bounded evidence only."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
EV=Path(__file__).parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


notes=[]
for name in ('cpu-probe','cpu-probe-2','cpu-probe-3','cpu-probe-4','cpu-probe-5'):
    path=EV/name/'results.json';r=json.loads(path.read_text(encoding='utf-8'));w=r['worker']
    notes.append(dict(directory=name,results_sha256=sha(path),returncode=r['returncode'],elapsed_seconds=r['elapsed_seconds'],
        harness_sha256=w.get('harness_sha256'),computed_key=w.get('first_kernel',{}).get('computed_hash'),
        interpretation={'cpu-probe':'CPU probe exited cleanly, but reconstructed attrs omitted empty entries and callback pointers were unsigned; miss was not native incompatibility.',
                        'cpu-probe-2':'Installed production binder caught omitted empty attrs. CPU test contract failed; no native execution.',
                        'cpu-probe-3':'All attrs corrected, but callbacks still unsigned; miss was not native incompatibility.',
                        'cpu-probe-4':'Actual int32 native callbacks and all attrs produce exact canonical first-dispatch/helper hits.',
                        'cpu-probe-5':'Final candidate, restricted to verified first SPLITK2 dispatch, independently reproduces actual binder/compiler/helper keys.'}[name]))
record=dict(classification='UNKNOWN',notes=notes,all_intermediate_evidence_preserved=True,
    prohibited_activity=dict(gpu_initialization=False,weights_loaded=False,kernel_or_helper_execution=False,compilation=False,autotuning=False),
    correction_test_logs=['review-regression-red.log','allocation-review-red.log','final-candidate-tests.log'],
    code_scope='Read-only identity probe and prospective synthetic admission planners only; no production native interception or GPU worker implemented.')
with (EV/'cpu-preparation-notes.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
inventory_same=(EV/'inventory-before.json').read_bytes()==(EV/'inventory-after.json').read_bytes()
assert inventory_same
assert not (EV/'attempt-started.json').exists() and not (EV/'run').exists()
probe=json.loads((EV/'cpu-probe-5/results.json').read_text(encoding='utf-8'))
assert probe['worker']['harness_sha256']==sha(ROOT/'scripts/compatibility/amd_shape_cache_identity.py')
paths=[p for p in EV.rglob('*') if p.is_file() and 'scratch' not in p.relative_to(EV).parts and p.name not in ('checksums.json','publication-verification.json')]
paths.extend(ROOT/'scripts/compatibility'/n for n in ('amd_shape_cache_identity.py','amd_shape_decoder_admission.py','test_amd_shape_cache_identity.py','test_amd_shape_decoder_admission.py'))
paths.append(ROOT/'docs/compatibility/amd-shape-decoder-cache-admission.md')
rows={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)}
with (EV/'checksums.json').open('x',encoding='utf-8') as f:json.dump(rows,f,indent=2)
print(json.dumps(dict(files_hashed=len(rows),inventory_byte_identical=inventory_same,final_cpu_probe_harness_matches=True,no_gpu_attempt=True)))
