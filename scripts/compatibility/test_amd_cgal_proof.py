import sys
sys.dont_write_bytecode = True
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import amd_cgal_proof as p


class Isolation(unittest.TestCase):
    def test_archive_paths(self):
        self.assertEqual(p.safe_member('root/include/test.h'), Path('root/include/test.h'))
        for name in ('../escape', '/escape', 'C:/escape', 'a\\escape', 'a/../x', 'a./b', 'a /b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                p.safe_member(name)

    def test_boost_header_selection_does_not_require_unselected_library_docs(self):
        self.assertTrue(p.selected_boost(Path('boost_1_87_0/boost/version.hpp')))
        self.assertTrue(p.selected_boost(Path('boost_1_87_0/LICENSE_1_0.txt')))
        self.assertFalse(p.selected_boost(Path('boost_1_87_0/libs/json/LICENSE_1_0.txt')))

    def test_resource_floor_and_frozen_caps(self):
        self.assertFalse(p.admission(12*p.GIB-1, 20*p.GIB))
        self.assertFalse(p.admission(20*p.GIB, 2*p.GIB-1))
        self.assertTrue(p.admission(12*p.GIB, 2*p.GIB))
        self.assertEqual(p.CAP, 6*p.GIB)
        self.assertEqual(p.DEADLINE, 180)

    def test_windows_exact_spawn_event(self):
        import subprocess
        expected = p.command('configure-shared')
        args = (expected[0], subprocess.list2cmdline(expected), str(p.BASE), {})
        self.assertTrue(p.spawn_allowed('configure-shared', args))
        for bad in ((expected[0], args[1]+' & wsl.exe', str(p.BASE), {}),
                    ('wsl.exe', args[1], str(p.BASE), {}),
                    (expected[0], args[1], str(p.ROOT), {})):
            self.assertFalse(p.spawn_allowed('configure-shared', bad))

    def test_compiler_cleanup_cannot_apply_to_solver_or_unknown_stage(self):
        self.assertTrue(p.compiler_cleanup_stage('build-runtime'))
        self.assertTrue(p.compiler_cleanup_stage('configure-shared'))
        for stage in ('sample-official', 'ai-solver', 'authenticate', 'configure-unknown'):
            self.assertFalse(p.compiler_cleanup_stage(stage))

    def test_cmake_definitions_use_forward_paths(self):
        self.assertTrue(all('\\' not in x for x in p.command('configure-shared') if x.startswith('-D')))

    def test_default_inert_and_worker_requires_GO(self):
        import io
        from types import SimpleNamespace
        from unittest.mock import patch
        self.assertEqual(p.main([]), 0)
        with patch.object(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(b'NO\n'))):
            with self.assertRaisesRegex(RuntimeError, 'GO'):
                p.main(['--worker', 'prepare-ai'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
