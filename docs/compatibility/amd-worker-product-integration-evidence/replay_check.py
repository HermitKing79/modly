"""Bounded historical-result replay through the real IPC receiver and image-job API.

This is a test adapter, never registered in Modly. It does not run inference.
The accepted master and its original host/export/verification receipts stay read-only.
"""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-worker-product-integration-279f8c0'
OUT = P / 'replay'
OLD = ROOT.parent / 'artifacts/amd-brick-full-sampled-glb-3db265c'
MASTER = OLD / 'export/brick-source-sampled-colour.glb'
MASTER_SHA = '1970bbcc5740dc56c8c9e6e20d97e8afd3176126c57b87d886a034f876eb38ae'
TINY = ROOT.parent / 'artifacts/amd-turret-compact-glb-d3eec48/fixture.glb'
TINY_SHA = 'a4fe7dd08d3fd6c9f14d53c1be09f1b2a3b4ab134c4100aee381602f7a6826f1'
IMAGE = ROOT / 'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/source-image.webp'
IMAGE_SHA = 'a31ba3b084757faa66e4bb91a85bd956fcc1eaaaf298a091fd08044351cd0293'
CAP, DEADLINE = 6 * 1024**3, 180


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def worker():
    if sys.stdin.buffer.readline(32) != b'GO\n':
        raise ValueError('CPU replay authorisation required')
    os.environ.update(MODELS_DIR=str(OUT / 'models'), WORKSPACE_DIR=str(OUT / 'workspace'), EXTENSIONS_DIR='')
    sys.path.insert(0, str(ROOT / 'api'))
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import routers.generation as generation
    import services.generator_registry as registry
    from services.extension_process import ExtensionProcess
    from services.amd_completed_result import CONTRACT, CompletionAuthority, CompletionContext, FilePin

    image = IMAGE.read_bytes()
    if hashlib.sha256(image).hexdigest() != IMAGE_SHA:
        raise ValueError('Original image identity')
    # generator_registry creates this exact test workspace at import time.
    workspace = OUT / 'workspace'; workspace.mkdir(exist_ok=True)
    registry.WORKSPACE_DIR = workspace
    records = []
    for name, source, expected in [('tiny-control', TINY, TINY_SHA), ('historical-master', MASTER, MASTER_SHA)]:
        started = time.monotonic()
        if sha(source) != expected:
            raise ValueError('Historical GLB identity')
        leaf = OUT / name; leaf.mkdir()
        staged = leaf / 'completed.glb'
        with source.open('rb') as incoming, staged.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024**2)
            outgoing.flush(); os.fsync(outgoing.fileno())
        if name == 'historical-master':
            paths = {'export': OLD / 'export/result.json', 'export_host': OLD / 'export-host.json',
                     'verification': OLD / 'verify/result.json', 'verification_host': OLD / 'verify-host.json'}
            hashes = {'export': 'c7d440405c1354e9a9abef45a6892189e9a9671d54976ec0ce9a7b5adec91022',
                      'export_host': 'e884c6264b4b71e1bea6d1bc5746c0c8ba1c5f5a72e02e73f144d278e63383cb',
                      'verification': 'a36d1368e3910b9865416cd9e3b9653fd7d5f06634addd031657874a40c02b46',
                      'verification_host': '3ba191131390b6017732822783c06cdd2f9027824be0bc36316211c2449a496d'}
            producer = ROOT / 'docs/compatibility/amd-brick-full-sampled-glb-evidence/export_worker.py'
        else:
            # Synthetic PASS receipts belong only to the explicitly labelled tiny transport control.
            paths = {}; hashes = {}
            for key in ('export', 'export_host', 'verification', 'verification_host'):
                paths[key] = leaf / (key + '.json')
                save(paths[key], {'classification': 'PASS', 'test_control_only': True})
                hashes[key] = sha(paths[key])
            producer = Path(__file__)
        for key, path in paths.items():
            if sha(path) != hashes[key] or json.loads(path.read_text())['classification'] != 'PASS':
                raise ValueError('Pinned prior proof differs')
        manifest = {'id': 'test-only/completed-amd-replay', 'experimental': True,
                    'completed_output_contract': CONTRACT, 'completed_result_mode': 'replay_fixture'}
        adapter = ExtensionProcess(leaf, manifest)
        adapter._loaded = True
        source_run = 'accepted-source-sampled-glb-3db265c' if name == 'historical-master' else 'tiny-integration-control'
        producer_sha = sha(producer)
        authority = None
        def send(message):
            nonlocal authority
            if message['action'] != 'generate':
                raise AssertionError('No real worker process or inference is permitted')
            context = CompletionContext.from_dict(message['completed_result_context'])
            receipt = leaf / 'completion.json'
            save(receipt, {'contract': CONTRACT, 'product_context': context.to_dict(), 'status': 'PASS',
                           'source_run_id': source_run, 'worker_source_sha256': producer_sha,
                           'output': {'path': staged.name, 'bytes': staged.stat().st_size, 'sha256': expected},
                           'proof_sha256': hashes})
            authority = CompletionAuthority(leaf, receipt.name, sha(receipt), IMAGE_SHA, source_run,
                                           FilePin(producer, producer_sha),
                                           {key: FilePin(path, hashes[key]) for key, path in paths.items()}, 'replay_fixture')
            adapter._queue.put({'type': 'done', 'id': context.request_id,
                                'completion': {'receipt_path': str(receipt), 'receipt_sha256': sha(receipt)}})
        adapter._send = send  # Test-only saved worker reply; actual production receiver/authenticator below.
        adapter.configure_completion_verifier(lambda context, candidate: authority, mode='replay_fixture')
        class Registry:
            def get_generator(self, *args): return adapter
            def get_manifest(self, *args): return {'output': 'mesh'}
            def activate_ready_generator(self, *args): return adapter
            def assert_weight_variant_installed(self, *args): pass
            def model_status(self, *args): return {'loaded': True, 'downloaded': True, 'name': 'Replay only'}
        generation.generator_registry = Registry()
        app = FastAPI(); app.include_router(generation.router, prefix='/generate')
        with TestClient(app) as client:
            response = client.post('/generate/from-image', files={'image': ('source.webp', image, 'image/webp')},
                                   data={'model_id': manifest['id'], 'collection': 'Replay', 'remesh': 'none'})
            response.raise_for_status()
            status = client.get('/generate/status/' + response.json()['job_id']); status.raise_for_status()
            result = status.json()
        if result['status'] != 'done' or result['result_origin'] != 'amd-replay' or 'replay' not in result['step'].lower():
            raise ValueError('Real API did not return explicit completed replay: ' + str(result))
        output = workspace / result['output_url'].removeprefix('/workspace/')
        if not output.is_relative_to(workspace) or sha(output) != expected or sha(staged) != expected or sha(source) != expected:
            raise ValueError('Output/source exact byte correspondence failed')
        meta = json.loads(output.with_suffix('.json').read_text())
        if meta != result['result_metadata'] or meta['mode'] != 'replay_fixture':
            raise ValueError('API/asset provenance differs')
        records.append({'case': name, 'classification': 'PASS', 'inference_executed': False,
                        'source': str(source), 'output': str(output), 'bytes': output.stat().st_size,
                        'sha256': expected, 'job': result, 'elapsed_seconds': time.monotonic() - started})
    if any(key == 'torch' or key.startswith(('triton', 'flex_gemm')) for key in sys.modules):
        raise AssertionError('Forbidden model/native import during replay')
    save(OUT / 'result.json', {'classification': 'PASS', 'test_kind': 'completed-artifact replay/integration test',
                             'GPU_inference': False, 'model_loaded': False, 'cases': records})


def host():
    sys.path.insert(0, str(ROOT / 'scripts/compatibility'))
    from amd_forward_owned_job import OwnedJob, ExtendedLimits
    spec = importlib.util.spec_from_file_location('existing_cpu_monitor', ROOT / 'docs/compatibility/amd-brick-pbr-exposure-evidence/render_host.py')
    monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor)
    monitor.PRIVATE = P
    before = monitor.resources()
    if not before['admitted']:
        raise RuntimeError('Replay resource admission: ' + json.dumps(before))
    OUT.mkdir(exist_ok=False)
    save(OUT / 'attempt.json', {'CPU_only': True, 'replay_only': True, 'initial_resources': before,
                              'worker_sha256': sha(__file__), 'memory_limit_bytes': CAP, 'deadline_seconds': DEADLINE})
    job = OwnedJob(); limits = ExtendedLimits(); limits.Basic.Flags = 0x2300
    limits.ProcessMemory = limits.JobMemory = CAP
    if not job.api.SetInformationJobObject(job.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        job.close(); raise ctypes.WinError()
    child = None; attached = False; began = time.monotonic()
    result = {'initial_resources': before, 'resource_samples': []}
    try:
        with (OUT / 'stdout.txt').open('xb') as stdout, (OUT / 'stderr.txt').open('xb') as stderr:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'],
                                     stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                                              CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES=''))
            job.attach(child); attached = True
            child.stdin.write(b'GO\n'); child.stdin.close()
            while child.poll() is None:
                if time.monotonic() - began >= DEADLINE: raise TimeoutError('180-second CPU replay deadline')
                row = monitor.resources(); row['elapsed_seconds'] = time.monotonic() - began
                result['resource_samples'].append(row)
                if not row['admitted']: raise RuntimeError('Live replay resource admission failed')
                time.sleep(.2)
            result['returncode'] = child.returncode
            result['process_memory'] = monitor.process_memory(child)
            until = time.monotonic() + 5
            while job.active() and time.monotonic() < until: time.sleep(.025)
    except BaseException:
        result['error'] = traceback.format_exc()
        if child is not None and not attached and child.poll() is None: child.kill(); child.wait(timeout=5)
    finally:
        peak = ExtendedLimits()
        if not job.api.QueryInformationJobObject(job.handle, 9, ctypes.byref(peak), ctypes.sizeof(peak), None):
            result['measurement_error'] = 'Job memory query failed'
        result.update(peak_process_committed_bytes=peak.PeakProcessMemory, peak_job_committed_bytes=peak.PeakJobMemory)
        result['cleanup'] = job.finish()
        if child is not None: child.wait(timeout=5)
        result['elapsed_seconds'] = time.monotonic() - began
        result['classification'] = 'BLOCKED'
        if result.get('returncode') == 0 and not result.get('error') and not result.get('measurement_error') and not result['cleanup']['forced_job_cleanup']:
            try:
                receipt = json.loads((OUT / 'result.json').read_text())
                for case in receipt['cases']:
                    if sha(case['output']) != case['sha256'] or sha(case['source']) != case['sha256']:
                        raise ValueError('Independent post-exit full-file readback failed')
                result['independent_post_exit_readback'] = True
                result['classification'] = 'PASS'
            except BaseException:
                result['post_exit_verification_error'] = traceback.format_exc()
        save(OUT / 'host-result.json', result)
    print(json.dumps({'classification': result['classification'], 'elapsed_seconds': result['elapsed_seconds'],
                      'peak_job_committed_bytes': result['peak_job_committed_bytes'], 'cleanup': result['cleanup']}))
    return 0 if result['classification'] == 'PASS' else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']: worker()
    elif not sys.argv[1:]: raise SystemExit(host())
    else: raise SystemExit('Unsupported CPU replay arguments')
