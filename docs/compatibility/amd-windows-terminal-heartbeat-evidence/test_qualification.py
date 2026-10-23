"""CPU-only control provenance tests; never launches a process or Job."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

E=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('authority_qualification',E/'qualification.py')
q=importlib.util.module_from_spec(s);s.loader.exec_module(q)


class QualificationTests(unittest.TestCase):
    def test_no_success_from_status_without_live_identity(self):
        self.assertIsNone(q.live_proof({'pid':10},None,{},1))

    def test_foreign_nonce_rejected_before_identity(self):
        with self.assertRaises(ValueError):
            q.live_proof({'pid':10,'nonce':'foreign'},{}, {'nonce':'owned'}, 1)

    def test_preflight_never_reserves_or_launches(self):
        with patch.object(q,'read',side_effect=ValueError('missing reviewed admission')), \
             patch.object(q.subprocess,'Popen') as launch, patch.object(Path,'mkdir') as reserve:
            with self.assertRaises(ValueError):q.preflight()
            launch.assert_not_called();reserve.assert_not_called()

    def test_fixture_has_no_process_or_model_import(self):
        import ast
        tree=ast.parse((E/'fixture.py').read_text())
        imports={a.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for a in node.names}
        self.assertTrue(imports <= {'json','os','sys','time'})
        self.assertNotIn('subprocess',(E/'fixture.py').read_text())


if __name__=='__main__':unittest.main()
