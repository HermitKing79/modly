"""Focused standard-library checks; never imports Torch or enters WSL."""
import ast
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import json
import uuid

E=Path(__file__).resolve().parent
OLD=E.parent/'amd-wsl-turret-chunked-decode-evidence'

class WriterPreparation(unittest.TestCase):
    def test_nonrecursive_leaf_creation(self):
        source=E/'artifact_io.py' if (E/'artifact_io.py').exists() else E/'neural.py'
        calls=[n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr=='mkdir']
        self.assertTrue(calls)
        for call in calls:
            kw={k.arg:ast.literal_eval(k.value) for k in call.keywords}
            self.assertFalse(kw.get('parents',False),'writer must never create ancestors')
            self.assertIs(kw.get('exist_ok'),False)

    def test_computation_and_validator_unchanged(self):
        for name in ('admission.py','runtime.py','chunked_final.py'):
            self.assertEqual((E/name).read_bytes(),(OLD/name).read_bytes())
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n,ast.FunctionDef)}
        old,new=functions(OLD/'neural.py'),functions(E/'neural.py')
        for name in ('run_neural','validate_output','detach_failure'):self.assertEqual(new[name],old[name])

    def test_host_preparation_rejects_traversal_before_mutation(self):
        script=E/'artifact_parent.ps1'
        self.assertTrue(script.exists(),'trusted-host directory preflight missing')
        import artifact_paths
        r=subprocess.run(artifact_paths.host_command('../foreign'),capture_output=True,text=True,timeout=10)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('Unapproved artifact leaf',r.stderr)

    def test_receipt_published_only_after_complete_write(self):
        import artifact_io
        target=E/('atomic-publication-fixture-'+uuid.uuid4().hex+'.json')
        self.assertFalse(target.exists())
        original=artifact_io.os.rename;observed=[]
        def rename(source,destination):
            self.assertFalse(target.exists())
            observed.append(json.loads(Path(source).read_text()))
            return original(source,destination)
        with patch.object(artifact_io.os,'rename',rename):artifact_io.durable_json(target,{'complete':True})
        self.assertEqual(observed,[{'complete':True}])
        with self.assertRaises(FileExistsError):artifact_io.durable_json(target,{'complete':False})
        self.assertEqual(json.loads(target.read_text()),{'complete':True})

if __name__=='__main__':unittest.main(verbosity=2)
