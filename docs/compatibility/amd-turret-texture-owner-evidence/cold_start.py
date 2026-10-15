"""One real WSL CPU bootstrap; shared decoder startup, no AI/native imports."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

E = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('texture_candidate', E/'decode.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
base = candidate.base


def guest(output):
    output.mkdir(exist_ok=False)
    record = {'classification': 'CPU_UNKNOWN', 'forbidden_import_attempts': [],
              'real_weights_loaded': False, 'forward_calls': 0}
    policy = None
    try:
        base.await_go(sys.stdin.buffer)
        record['authorization_accepted'] = True
        record['candidate_identities'] = candidate.candidate_files()
        def deny_ai(event, args):
            if event == 'import' and str(args[0]).split('.')[0] in {
                    'torch', 'triton', 'flex_gemm', 'trellis2', 'cumesh', 'o_voxel', 'hip_utils'}:
                record['forbidden_import_attempts'].append(str(args[0]))
                raise RuntimeError('CPU bootstrap forbids AI/GPU/native imports')
        sys.addaudithook(deny_ai)
        from amd_wsl_native_import_policy import ImportPolicy
        scratch = candidate.SCRATCH.with_name(candidate.SCRATCH.name + '-cpu')
        from artifact_io import ARTIFACT_ROOT
        probe = ARTIFACT_ROOT/'probe'
        policy = ImportPolicy(scratch, writable_roots=(output, probe))
        policy.install_audit_hook()
        # This is the actual production function, not a copied startup sequence.
        candidate.prepare_bootstrap(record, scratch)
        import numpy
        import amd_texture_field_artifact as writer
        import artifact_io
        stage = (candidate.STAGE/'site-packages').resolve(strict=True)
        if (numpy.__version__ != '2.2.6' or not Path(numpy.__file__).resolve().is_relative_to(stage)
                or Path(writer.__file__).resolve() != candidate.ROOT/'scripts/compatibility/amd_texture_field_artifact.py'
                or Path(artifact_io.__file__).resolve() != E/'artifact_io.py'
                or sys.prefix != '/opt/modly-wsl/venv' or sys.version.split()[0] != '3.12.3'):
            raise RuntimeError('Cold-start runtime/module origin differs')
        excluded = ('torch', 'triton', 'flex_gemm', 'trellis2', 'cumesh', 'o_voxel', 'hip_utils')
        absent = not any(n.split('.')[0] in excluded for n in sys.modules)
        if not absent or record['forbidden_import_attempts'] or policy.snapshot()['rejected_calls']:
            raise RuntimeError('CPU bootstrap crossed an excluded boundary')
        record.update(numpy_version=numpy.__version__,
            numpy_origin=numpy.__file__, writer_origin=writer.__file__, bridge_origin=artifact_io.__file__,
            python=sys.version.split()[0], prefix=sys.prefix, staged_path=str(stage),
            staged_path_index=sys.path.index(str(stage)), excluded_modules_absent=absent,
            policy=policy.snapshot())
        from output_owner import ownership
        artifact_io.check_directory(probe)
        authority = ownership(probe)
        raw = numpy.array([[-5,-4,-3,-2,-1,0],[1,2,3,4,5,6]], dtype=numpy.float32)
        features = raw*numpy.float32(.5)+numpy.float32(.5)
        coords = numpy.array([[0,511,2,3],[0,0,1,0]], dtype=numpy.int32)
        record['fixture'] = writer.write(probe, raw, features, coords,
            {'mode':'fixture','purpose':'actual-WSL-owned-DrvFS-writer-acceptance'}, authority=authority)
        assert writer.verify(probe/writer.ARTIFACT_NAME, probe, record['fixture']['metadata'], authority=authority)['verified']
        negatives = []
        for label, call in (
                ('default-UID-remains-closed', lambda: writer.verify(probe/writer.ARTIFACT_NAME, probe)),
                ('default-file-UID-remains-closed', lambda: writer._bounded_read(probe/writer.ARTIFACT_NAME, writer.MAX_FILE_BYTES)),
                ('foreign-root-context', lambda: writer.approved_root(probe, authority=ownership(artifact_io.FIELD))),
                ('foreign-file-name', lambda: writer.verify(probe/'foreign.safetensors', probe, authority=authority))):
            try:
                call()
            except ValueError:
                negatives.append(label)
            else:
                raise AssertionError('Ownership negative accepted: '+label)
        record['ownership_negatives'] = negatives
        record['ownership_context'] = {'root':str(authority.root),'root_identity':list(authority.root_identity),
            'worker_identity':list(authority.worker_identity),'mount':authority.mount,
            'windows_owner_sid':authority.windows_owner_sid,'attestation_sha256':authority.attestation_sha256}
        record['policy'] = policy.snapshot()
        if record['policy']['rejected_calls'] or any(n.split('.')[0] in excluded for n in sys.modules):
            raise RuntimeError('CPU writer crossed an excluded boundary')
        record['classification'] = 'CPU_PASS'
        print(json.dumps(record), flush=True)
    except BaseException as exc:
        record.update(classification='CPU_UNKNOWN', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        record['finished_utc'] = base.utc_now()
        base.save_json(output/'results.json', record)
    return 0 if record['classification'] == 'CPU_PASS' else 1


def host(output):
    if output.resolve() != (E/'cold-run').resolve():
        raise RuntimeError('Exclusive cold-start evidence path required')
    if (E/'neural-attempt.marker').exists() or (E/'run').exists():
        raise RuntimeError('CPU cold start must precede any inference reservation')
    candidate.candidate_files()
    from output_owner import ROOT as ARTIFACT_ROOT, host_verify
    host_verify(ARTIFACT_ROOT/'field')
    host_verify(ARTIFACT_ROOT/'probe')
    # Resolve real helpers before SCRIPT selects this CPU-only guest.
    helpers = {name: base.load_sibling(name) for name in ('amd_bounded_gpu_proof', 'amd_forward_owned_job')}
    for name, module in helpers.items():
        if Path(module.__file__).resolve() != candidate.ROOT/'scripts/compatibility'/(name+'.py'):
            raise RuntimeError('Helper lookup redirected')
    command = base.worker_command(base.guest_path(Path(__file__)), base.guest_path(output/'guest'))
    if command[command.index('--worker')-1] != base.guest_path(Path(__file__)):
        raise RuntimeError('Wrong CPU guest selected')
    base.save_json(E/'cold-helper-preflight.json', {'worker_command': command,
        'helper_origins': {n: m.__file__ for n, m in helpers.items()}, 'GPU_authorized': False})
    with patch.object(base, 'SCRIPT', Path(__file__).resolve()), \
            patch.object(base, 'ATTEMPT_MARKER', E/'cold-attempt.marker'), \
            patch.object(base, 'load_sibling', side_effect=lambda name: helpers[name]):
        raw_exit = base.supervise(output)
    raw = json.loads((output/'results.json').read_text())
    worker = raw.get('worker', {})
    passed = (raw_exit == 1 and raw.get('classification') == 'UNKNOWN'
        and raw.get('worker_returncode') == 0 and worker.get('classification') == 'CPU_PASS'
        and worker.get('candidate_identities') == candidate.candidate_files()
        and worker.get('forbidden_import_attempts') == [] and worker.get('excluded_modules_absent') is True
        and worker.get('numpy_version') == '2.2.6' and raw.get('observer_healthy') is True
        and 'error' not in worker
        and raw.get('windows_job_cleanup', {}).get('owned_tree_exit_verified') is True
        and raw['windows_job_cleanup'].get('forced_job_cleanup') is False
        and raw.get('windows_job_grace', {}).get('empty') is True
        and raw.get('distro_termination', {}).get('returncode') == 0
        and not any('error' in k for k in raw))
    persistence = None
    try:
        if passed:
            from output_owner import ROOT as ARTIFACT_ROOT, host_verify
            from amd_texture_field_artifact import verify
            import hashlib
            import struct
            probe = ARTIFACT_ROOT/'probe'
            before = host_verify(probe, files=True)
            persistence = verify(probe/'texture-field.safetensors', probe, worker['fixture']['metadata'])
            # Independent host arithmetic/LE encodings, not the guest's outputs.
            expected = {'raw':struct.pack('<12f',*range(-5,7)),
                'features':struct.pack('<12f',*(-2+i*.5 for i in range(12))),
                'coordinates':struct.pack('<8i',0,511,2,3,0,0,1,0)}
            for name, payload in expected.items():
                if persistence['metadata']['tensors'][name]['sha256'] != hashlib.sha256(payload).hexdigest():
                    raise ValueError('Independent Windows fixture bytes differ: '+name)
            after = host_verify(probe, files=True)
            if 'torch' in sys.modules or len(worker.get('ownership_negatives',[])) != 4:
                raise RuntimeError('CPU-only ownership acceptance incomplete')
            base.save_json(E/'persistence-preflight.json', dict(persistence,
                Windows_ownership_before=before, Windows_ownership_after=after,
                independent_host_payload_comparison=True, torch_imported=False, GPU_initialized=False))
    except BaseException as exc:
        passed = False
        base.save_json(output/'host-verification-error.json', {'error':str(exc),'traceback':traceback.format_exc()})
    verdict = {'classification': 'PASS' if passed else 'UNKNOWN', 'GPU_authorized': False,
        'production_supervisor_exit': raw_exit, 'acceptance_attempts': 1,
        'independent_Windows_fixture_verified':bool(passed and persistence and persistence['verified']),
        'note': 'Unchanged vector-add supervisor remains UNKNOWN; this separate CPU verdict requires genuine cold startup and clean monitored containment.'}
    base.save_json(output/'CPU-verdict.json', verdict)
    print(json.dumps(verdict), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        raise SystemExit(guest(args.worker))
    if args.output is None:
        parser.error('--output required')
    raise SystemExit(host(args.output))
