"""Focused application IPC preparation test; no real loading or generated assets."""
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]


def child(scratch):
    scratch = scratch.resolve()
    scratch.relative_to(ROOT)
    outputs = scratch / 'outputs'
    outputs.mkdir()
    os.environ.update(WORKSPACE_DIR=str(outputs), MODELS_DIR=str(scratch / 'private-models'),
                      MODLY_ENABLE_EXPERIMENTAL_AMD='1', MODLY_AMD_RUNTIME_CONFIG=str(scratch / 'runtime.json'))
    config = {'python': 'D:/AI/Trellis2AMD/venv/Scripts/python.exe',
              'source_root': 'D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD',
              'mode': 'geometry-preparation-alpha-only', 'geometry': {
                  'models_root': 'D:/AI/Trellis2AMD/ComfyUI/models',
                  'pipeline_dir': 'D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B',
                  'dino_dir': 'D:/AI/Trellis2AMD/ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m'}}
    path = scratch / 'runtime.json'
    path.write_text(json.dumps(config))
    sys.path.insert(0, str(ROOT / 'api'))
    from PIL import Image
    from services.extension_process import ExtensionProcess
    from services.amd_runtime import CONTRACT
    from services.generators.base import GenerationCancelled
    image = Image.new('RGBA', (12, 10), (40, 80, 120, 0))
    for x in range(2, 10):
        for y in range(2, 8):
            image.putpixel((x, y), (40, 80, 120, 255))
    transparent, opaque = io.BytesIO(), io.BytesIO()
    image.save(transparent, format='PNG')
    Image.new('RGBA', (12, 10), (40, 80, 120, 255)).save(opaque, format='PNG')
    manifest = {'id': 'trellis2-amd/generate', 'node_id': 'generate', 'ext_id': 'trellis2-amd',
                'runtime_contract': CONTRACT, 'experimental': True}
    worker = ExtensionProcess(ROOT / 'experimental/extensions/trellis2-amd', manifest)
    worker.outputs_dir = outputs
    worker.model_dir = scratch / 'private-models/generate'
    result = {'host_python': sys.version, 'host_executable': sys.executable, 'runtime_config': config}
    try:
        worker.load()
        owned = worker._proc
        result['worker_identity'] = worker.runtime_identity
        assert worker.runtime_identity['preparation_only'] is True and worker.runtime_identity['weights_loaded'] is False
        try:
            worker.prepare_geometry(opaque.getvalue())
        except RuntimeError as exc:
            result['opaque_rejection'] = str(exc)
            assert 'alpha-only' in str(exc)
        else:
            raise AssertionError('Opaque alpha-only image accepted')
        prepared = worker.prepare_geometry(transparent.getvalue())
        result['prepared'] = prepared
        result['monitor'] = worker.last_preparation_monitor
        assert prepared['pipeline']['class'] == 'Trellis2ImageTo3DPipeline'
        assert prepared['pipeline']['factory_reused'] is True
        assert len(prepared['pipeline']['model_slots']) == 9 and all(value is None for value in prepared['pipeline']['model_slots'].values())
        assert len(prepared['components']) == 6 and all(value['state'] == 'unloaded' for value in prepared['components'].values())
        assert prepared['weights_loaded'] is False and prepared['glb_generated'] is False
        assert prepared['tensor_payload_bytes_read'] == 0 and prepared['gpu']['allocated_bytes'] == prepared['gpu']['reserved_bytes'] == 0
        assert worker._proc.pid == owned.pid
        result['same_process_recovery'] = True
        try:
            worker.generate(transparent.getvalue(), {'mock': False})
        except RuntimeError as exc:
            result['generation_rejection'] = str(exc)
            assert 'generation is disabled' in str(exc)
        else:
            raise AssertionError('Preparation worker accepted generation')
        worker.unload()
        assert not worker.is_loaded()
        worker.load()
        second = worker.prepare_geometry(transparent.getvalue())
        assert worker._proc.pid == owned.pid and second['weights_loaded'] is False
        result['unload_reload_same_worker'] = True
        worker.shutdown()
        assert owned.poll() == 0 and worker._proc is None
        result['cooperative_shutdown'] = {'pid': owned.pid, 'exit': owned.returncode, 'reaped': True}
        # A fresh owned worker tests cancellation BEFORE preparation/import.
        worker.load()
        cancelled = worker._proc
        cancel = threading.Event()
        cancel.set()
        try:
            worker.prepare_geometry(transparent.getvalue(), cancel_event=cancel)
        except GenerationCancelled:
            pass
        else:
            raise AssertionError('Cancellation was ignored')
        assert worker._proc is None and cancelled.poll() is not None
        result['cancelled_worker'] = {'pid': cancelled.pid, 'exit': cancelled.returncode, 'reaped': True,
                                    'preparation_not_started': True}
        files = [str(item.relative_to(outputs)) for item in outputs.rglob('*') if item.is_file()]
        assert not any(item.lower().endswith('.glb') for item in files)
        result.update(output_files=files, real_checkpoint_loads=0, glb_files_generated=0,
                      host_torch_imported='torch' in sys.modules, classification='PASS')
    finally:
        worker.stop()
    (scratch / 'application-ipc.json').write_text(json.dumps(result, indent=2))


def supervise(output):
    output = output.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / 'scratch'
    scratch.mkdir()
    spec = importlib.util.spec_from_file_location('scaffold_capture', Path(__file__).with_name('amd_worker_scaffold_proof.py'))
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    command = ['C:/Users/wattz/Documents/Modly/extensions/trellis2/venv/Scripts/python.exe',
               '-B', '-I', str(Path(__file__).resolve()), '--child', '--output', str(scratch)]
    env = {**os.environ, 'TMP': str(scratch), 'TEMP': str(scratch), 'PYTHONDONTWRITEBYTECODE': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    run = capture.run_host(command, env, 90)
    for stream in ('stdout', 'stderr'):
        (output / (stream + '.txt')).write_bytes(run.pop(stream))
    path = scratch / 'application-ipc.json'
    application = json.loads(path.read_text()) if path.is_file() else None
    passed = (run['exit'] == 0 and run['timeout'] is False and run['tree_cleanup_verified']
              and application is not None and application['classification'] == 'PASS')
    run.update(command=command, timeout_seconds=90, classification='PASS' if passed else 'PARTIAL', application=application)
    (output / 'results.json').write_text(json.dumps(run, indent=2))
    print(json.dumps({key: value for key, value in run.items() if key not in ('application', 'command')}))
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(child(args.output) if args.child else supervise(args.output))
