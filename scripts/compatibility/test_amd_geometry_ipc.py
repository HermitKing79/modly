"""Preparation IPC contract/lifecycle tests, with no worker or GPU launches."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api"))
from services import amd_runtime
from services.extension_process import ExtensionProcess

MANIFEST = {"id": "trellis2-amd/generate", "node_id": "generate", "experimental": True,
            "runtime_contract": amd_runtime.CONTRACT}


class GeometryIPCContractTests(unittest.TestCase):
    def config(self):
        return {"python": str(Path(sys.executable).resolve()), "source_root": str(ROOT),
                "mode": "geometry-preparation-alpha-only", "geometry": {
                    "models_root": str(ROOT), "pipeline_dir": str(ROOT), "dino_dir": str(ROOT)}}

    def test_explicit_preparation_config_is_separate_from_mock_lists(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            path = Path(tmp) / 'runtime.json'
            path.write_text(json.dumps(self.config()))
            with patch.dict(os.environ, {"MODLY_ENABLE_EXPERIMENTAL_AMD": "1", "MODLY_AMD_RUNTIME_CONFIG": str(path)}):
                self.assertEqual(amd_runtime.runtime_config(MANIFEST)['mode'], 'geometry-preparation-alpha-only')
                for change in ({'mode': 'unknown'}, {'models': {}}, {'geometry': {}}):
                    path.write_text(json.dumps({**self.config(), **change}))
                    with self.assertRaises(ValueError):
                        amd_runtime.runtime_config(MANIFEST)

    def test_preparation_identity_cannot_be_used_as_mock_identity(self):
        config = self.config()
        identity = {'contract': amd_runtime.CONTRACT, 'model_id': MANIFEST['id'], 'python': config['python'],
                    'python_version': '3.12.10', 'torch': amd_runtime.VERSIONS['torch'], 'hip': '7.15.26333',
                    'cuda': None, 'format': 'safetensors', 'mock_only': False, 'preparation_only': True,
                    'runtime_mode': config['mode'], 'gpu_initialized': False, 'weights_loaded': False}
        amd_runtime.verify_identity(identity, MANIFEST, config)
        with self.assertRaises(ValueError):
            amd_runtime.verify_identity(identity, MANIFEST, {'python': config['python']})
        with self.assertRaises(ValueError):
            amd_runtime.verify_identity({**identity, 'weights_loaded': True}, MANIFEST, config)

    def worker(self):
        worker = ExtensionProcess(ROOT / 'experimental/extensions/trellis2-amd', MANIFEST)
        worker._amd_config = self.config()
        return worker

    def test_prepared_diagnostics_are_returned_without_an_output_asset(self):
        worker = self.worker()
        diagnostics = {'schema': 'modly.amd.geometry-preparation.v1', 'mode': 'alpha-only',
                       'weights_loaded': False, 'asset_generated': False, 'glb_generated': False,
                       'ready_for_inference': False, 'pipeline_prepared': True}
        with patch.object(worker, 'is_loaded', return_value=True), patch.object(worker, '_send') as send, \
             patch.object(worker, '_recv', return_value={'type': 'prepared', 'id': 'id', 'diagnostics': diagnostics}), \
             patch('services.extension_process.uuid.uuid4', return_value='id'), \
             patch('services.amd_observer.windows_observation', return_value={}), \
             patch('services.amd_observer.healthy_observation', return_value=True):
            self.assertEqual(worker.prepare_geometry(b'RGBA bytes', {'image_mode': 'alpha-only'}), diagnostics)
        self.assertEqual(send.call_args.args[0]['action'], 'prepare_geometry')
        self.assertNotIn('outputs_dir', send.call_args.args[0])

    def test_unhealthy_observer_prevents_preparation_message(self):
        worker = self.worker()
        with patch.object(worker, 'is_loaded', return_value=True), patch.object(worker, '_send') as send, \
             patch.object(worker, 'stop') as stop, patch('services.amd_observer.windows_observation', return_value={}), \
             patch('services.amd_observer.healthy_observation', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'observer'):
                worker.prepare_geometry(b'RGBA', {'image_mode': 'alpha-only'})
        send.assert_not_called()
        stop.assert_called_once()

    def test_cancellation_stops_and_reaps_without_preparation(self):
        from services.generators.base import GenerationCancelled
        worker, cancel = self.worker(), threading.Event()
        cancel.set()
        with patch.object(worker, '_send') as send, patch.object(worker, 'stop') as stop:
            with self.assertRaises(GenerationCancelled):
                worker.prepare_geometry(b'RGBA', {'image_mode': 'alpha-only'}, cancel_event=cancel)
        send.assert_not_called()
        stop.assert_called_once()

    def test_wrong_mode_or_inference_claim_fails(self):
        worker = self.worker()
        worker._amd_config = {'python': self.config()['python']}
        with self.assertRaises(ValueError):
            worker.prepare_geometry(b'RGBA', {'image_mode': 'alpha-only'})
        worker._amd_config = self.config()
        with patch.object(worker, 'is_loaded', return_value=True), patch.object(worker, '_send'), \
             patch.object(worker, '_recv', return_value={'type': 'prepared', 'id': 'id', 'diagnostics': {'glb_generated': True}}), \
             patch.object(worker, 'stop'), patch('services.extension_process.uuid.uuid4', return_value='id'), \
             patch('services.amd_observer.windows_observation', return_value={}), \
             patch('services.amd_observer.healthy_observation', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'diagnostics'):
                worker.prepare_geometry(b'RGBA', {'image_mode': 'alpha-only'})

    def test_only_explicit_validation_errors_keep_worker_alive(self):
        for recoverable in (True, False):
            worker = self.worker()
            reply = {'type': 'error', 'id': 'id', 'message': 'rejected', 'loaded': True, 'recoverable': recoverable}
            with patch.object(worker, 'is_loaded', return_value=True), patch.object(worker, '_send'), \
                 patch.object(worker, '_recv', return_value=reply), patch.object(worker, 'stop') as stop, \
                 patch('services.extension_process.uuid.uuid4', return_value='id'), \
                 patch('services.amd_observer.windows_observation', return_value={}), \
                 patch('services.amd_observer.healthy_observation', return_value=True):
                with self.assertRaisesRegex(RuntimeError, 'rejected'):
                    worker.prepare_geometry(b'RGBA')
                self.assertEqual(stop.call_count, 0 if recoverable else 1)

    def test_observer_exception_stops_worker(self):
        worker = self.worker()
        with patch.object(worker, 'is_loaded', return_value=True), patch.object(worker, '_send') as send, \
             patch.object(worker, 'stop') as stop, \
             patch('services.amd_observer.windows_observation', side_effect=OSError('observer unavailable')):
            with self.assertRaisesRegex(OSError, 'observer unavailable'):
                worker.prepare_geometry(b'RGBA')
            send.assert_not_called()
            stop.assert_called_once()

    def test_inflight_cancel_and_timeout_stop_worker_and_observe_afterward(self):
        from services.generators.base import GenerationCancelled
        for cancelled in (True, False):
            worker, cancel = self.worker(), threading.Event()
            def sent(*args):
                if cancelled:
                    cancel.set()
            with patch.object(worker, 'is_loaded', return_value=True), patch.object(worker, '_send', side_effect=sent), \
                 patch.object(worker, 'stop') as stop, patch('services.amd_observer.windows_observation', return_value={}) as observe, \
                 patch('services.amd_observer.healthy_observation', return_value=True), \
                 patch('services.extension_process.AMD_PREPARATION_TIMEOUT', 0):
                with self.assertRaises(GenerationCancelled if cancelled else TimeoutError):
                    worker.prepare_geometry(b'RGBA', cancel_event=cancel)
                stop.assert_called_once()
                self.assertEqual(observe.call_count, 2)


if __name__ == '__main__':
    unittest.main()
