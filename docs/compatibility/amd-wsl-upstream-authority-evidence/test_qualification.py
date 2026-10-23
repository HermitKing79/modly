"""CPU-only harness contracts. No WSL, Job or fixture process is launched."""
import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

E=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('wsl_authority_q',E/'qualification.py')
q=importlib.util.module_from_spec(s);s.loader.exec_module(q)


class QualificationTests(unittest.TestCase):
    def test_preflight_never_reserves_or_starts_execution(self):
        with patch.object(q,'read',side_effect=ValueError('missing admission')), \
             patch.object(q.subprocess,'Popen') as launch,patch.object(Path,'mkdir') as reserve:
            with self.assertRaises(ValueError):q.preflight()
            launch.assert_not_called();reserve.assert_not_called()

    def test_fixture_has_no_child_process_or_model_calls(self):
        tree=ast.parse((E/'fixture.py').read_text())
        imports={a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(imports & {'subprocess','multiprocessing','torch','triton','flex_gemm'})
        self.assertFalse(calls & {'fork','Popen','execv','execve','execvpe','system','spawn'})


if __name__=='__main__':unittest.main(verbosity=2)
