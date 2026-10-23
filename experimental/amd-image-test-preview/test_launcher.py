"""CPU contracts only: no Electron, Job, model or installed environment writes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('preview_launcher', HERE / 'launch_preview.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class PreviewLauncherTests(unittest.TestCase):
    def fixture(self, root):
        files = {}
        for name in launcher.REQUIRED_FILES:
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'fixture ' + name.encode())
            files[name] = launcher.sha(p)
        for name in ('api-python.exe', 'cpu-python.exe'):
            (root / name).write_bytes(b'synthetic-not-executable')
        c = dict(schema=launcher.SCHEMA, source_commit=launcher.BASE, root=str(root),
                 executable=str(root/'runtime/Modly.exe'), profile=str(root/'profile'),
                 api_dir=str(root/'api'), api_python=str(root/'api-python.exe'),
                 cpu_python=str(root/'cpu-python.exe'), files=files,
                 api_probe={'python_sha256':launcher.sha(root/'api-python.exe')},
                 cpu_probe={'python_sha256':launcher.sha(root/'cpu-python.exe')},
                 runtime_tree_sha256=launcher.runtime_tree_sha256(root/'runtime'))
        return c

    def save(self, root, c):
        p = root/'preview.json';p.write_text(json.dumps(c), encoding='utf8');return p

    def test_exact_fixture_and_immutable_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.fixture(root)
            self.assertEqual(launcher.validate_config(self.save(root,c))['root'], str(root))
            (root/'api/services/amd_image_test.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed preview file'):
                launcher.validate_config(root/'preview.json')

    def test_foreign_profile_and_runtime_hash_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.fixture(root);c['profile']=str(root.parent)
            with self.assertRaisesRegex(ValueError,'Confined'):
                launcher.validate_config(self.save(root,c))
            c['profile']=str(root/'profile');c['cpu_probe']['python_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'runtime'):
                launcher.validate_config(self.save(root,c))

    def test_traversal_absolute_and_missing_guard_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.fixture(root)
            for bad in ('../foreign.py','/foreign.py','C:/foreign.py'):
                c['files'][bad]='0'*64
                with self.assertRaises(ValueError):launcher.validate_config(self.save(root,c))
                del c['files'][bad]
            del c['files']['runtime/resources/app/out/main/preview-bootstrap.cjs']
            with self.assertRaisesRegex(ValueError,'closure'):
                launcher.validate_config(self.save(root,c))

    def test_busy_port_refused_without_process_or_termination(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));listener.listen()
            with mock.patch.object(launcher.subprocess,'Popen') as spawn:
                with self.assertRaisesRegex(ValueError,'already in use'):
                    launcher.check_port(listener.getsockname()[1])
                spawn.assert_not_called()
                self.assertNotEqual(listener.fileno(),-1)

    def test_resource_floors_do_not_change(self):
        for ram,disk in ((12*1024**3-1,3*1024**3),(13*1024**3,2*1024**3-1)):
            with self.assertRaises(ValueError):launcher.admit(dict(available_ram_bytes=ram,disk_free_bytes=disk))
        launcher.admit(dict(available_ram_bytes=12*1024**3,disk_free_bytes=2*1024**3))

    def test_runtime_result_must_match_exact_record(self):
        with mock.patch.object(launcher.subprocess,'run',return_value=mock.Mock(returncode=0,stdout=b'{"changed":true}',stderr=b'')):
            with self.assertRaisesRegex(ValueError,'Runtime probe changed'):
                launcher.verify_runtime('not-executed',HERE/'runtime_probe.py','cpu',{'python_sha256':'fixture'}, {})

    def test_unlisted_runtime_dependency_change_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.fixture(root);self.save(root,c)
            (root/'runtime/foreign-module.js').write_text('changed')
            with self.assertRaisesRegex(ValueError,'runtime tree'):
                launcher.validate_config(root/'preview.json')

    def test_offline_environment_isolated_and_no_python_injection(self):
        env=launcher.environment(Path('C:/preview/profile').absolute(),'nonce',{'PYTHONPATH':'foreign','HF_HOME':'foreign','TEMP':'foreign','CHROME_LOG_FILE':'foreign','ELECTRON_LOG_FILE':'foreign'})
        self.assertNotIn('PYTHONPATH',env)
        self.assertEqual(env['MODLY_AMD_PREVIEW_NONCE'],'nonce')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(env['HF_HUB_OFFLINE'],'1')
        self.assertIn('profile',env['TEMP']);self.assertNotEqual(env['HF_HOME'],'foreign')
        self.assertNotIn('CHROME_LOG_FILE',env);self.assertNotIn('ELECTRON_LOG_FILE',env)

    def test_launch_entry_has_no_download_or_retry_route(self):
        source=(HERE/'launch_preview.py').read_text()
        self.assertNotIn('taskkill',source)
        self.assertNotIn('pip install',source)
        self.assertNotIn('--inspect',source)
        self.assertIn('check_port()',source)


if __name__=='__main__':unittest.main()
