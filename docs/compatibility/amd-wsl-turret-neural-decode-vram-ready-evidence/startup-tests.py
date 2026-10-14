"""Only the changed production startup configuration; no Torch/runtime imports."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name('decode.py')
TREE=ast.parse(SOURCE.read_text())
WORKER=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
KEYS=('PYTORCH_TUNABLEOP_ENABLED','PYTORCH_TUNABLEOP_TUNING','PYTORCH_TUNABLEOP_RECORD_UNTUNED')

def production_check():
    node=next((n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='verify_tunable'),None)
    if node is None:raise AssertionError('Production independent flag check missing')
    ns={'os':os}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns['verify_tunable']

def fake_torch(values,calls):
    def getter(name):
        def run():calls.append(name);return values[name]
        return run
    return SimpleNamespace(cuda=SimpleNamespace(tunable=SimpleNamespace(
        is_enabled=getter('enabled'),tuning_is_enabled=getter('tuning'),record_untuned_is_enabled=getter('record_untuned'))))

class StartupTests(unittest.TestCase):
    def test_independent_genuine_call_syntax_and_evidence(self):
        calls=[];record={}
        with patch.dict(os.environ,{k:'0' for k in KEYS}):
            production_check()(fake_torch(dict(enabled=False,tuning=False,record_untuned=False),calls),record,'test')
        self.assertEqual(calls,['enabled','tuning','record_untuned'])
        self.assertEqual(record['tunable_admission']['test']['environment'],{k:'0' for k in KEYS})

    def test_every_true_or_nonboolean_flag_fails_and_is_recorded(self):
        with patch.dict(os.environ,{k:'0' for k in KEYS}):
            for field in ('enabled','tuning','record_untuned'):
                for value in (True,0,None):
                    values=dict(enabled=False,tuning=False,record_untuned=False);values[field]=value
                    record={};calls=[]
                    with self.subTest(field=field,value=value),self.assertRaisesRegex(RuntimeError,'TunableOp configuration admission failed'):
                        production_check()(fake_torch(values,calls),record,'test')
                    self.assertEqual(len(calls),3)
                    self.assertIs(record['tunable_admission']['test']['flags'][field],value)

    def test_every_missing_or_wrong_environment_value_fails(self):
        for key in KEYS:
            for value in (None,'1','false',''):
                with self.subTest(key=key,value=value),patch.dict(os.environ,{k:'0' for k in KEYS}):
                    if value is None:del os.environ[key]
                    else:os.environ[key]=value
                    with self.assertRaisesRegex(RuntimeError,'TunableOp configuration admission failed'):
                        production_check()(fake_torch(dict(enabled=False,tuning=False,record_untuned=False),[]),{},'test')

    def test_actual_assignment_overrides_inheritance_before_dependencies(self):
        updates=[n for n in ast.walk(WORKER) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
            and ast.unparse(n.value.func)=='os.environ.update' and set(KEYS).issubset({k.arg for k in n.value.keywords})]
        self.assertEqual(len(updates),1)
        assignment=updates[0]
        with patch.dict(os.environ,{k:'1' for k in KEYS}):
            exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(SOURCE),'exec'),{'os':os})
            self.assertEqual({k:os.environ[k] for k in KEYS},{k:'0' for k in KEYS})
        imports=[n for n in ast.walk(WORKER) if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertLess(assignment.lineno,min(n.lineno for n in imports))

    def test_exact_two_admission_locations(self):
        calls=[n for n in ast.walk(WORKER) if isinstance(n,ast.Call) and ast.unparse(n.func)=='verify_tunable']
        self.assertEqual([ast.literal_eval(n.args[2]) for n in sorted(calls,key=lambda n:n.lineno)],['after_torch_import','pre_gpu'])
        torch_import=next(n for n in ast.walk(WORKER) if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
        self.assertEqual(calls[0].lineno,torch_import.lineno+1)
        load=next(n for n in ast.walk(WORKER) if isinstance(n,ast.Call) and ast.unparse(n.func)=='component.load_cpu')
        self.assertLess(calls[0].lineno,load.lineno)
        run=next(n for n in ast.walk(WORKER) if isinstance(n,ast.Call) and ast.unparse(n.func)=='run_neural')
        self.assertLess(calls[1].lineno,run.lineno)

if __name__=='__main__':unittest.main()
