"""Focused host-only flag recording and acceptance tests; no Torch import."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('probe',Path(__file__).with_name('probe.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class Tests(unittest.TestCase):
    def test_independent_queries_never_short_circuit(self):
        calls=[]
        class Flags:
            def is_enabled(self): calls.append('enabled');return True
            def tuning_is_enabled(self): calls.append('tuning');return True
            def record_untuned_is_enabled(self): calls.append('record');return False
        self.assertEqual(p.read_flags(Flags()),dict(enabled=True,tuning=True,record_untuned=False))
        self.assertEqual(calls,['enabled','tuning','record'])

    def test_nonboolean_flag_rejected(self):
        class Flags:
            def is_enabled(self):return 0
        with self.assertRaisesRegex(RuntimeError,'boolean'):p.read_flags(Flags())

    def test_environment_absence_and_extras_preserved(self):
        env={'PYTORCH_TUNABLEOP_TUNING':'1','PYTORCH_TUNABLEOP_VERBOSE':'0','UNRELATED':'value'}
        rows=p.tunable_environment(env)
        self.assertEqual(rows['PYTORCH_TUNABLEOP_ENABLED'],dict(set=False,value=None))
        self.assertEqual(rows['PYTORCH_TUNABLEOP_TUNING'],dict(set=True,value='1'))
        self.assertEqual(rows['PYTORCH_TUNABLEOP_VERBOSE'],dict(set=True,value='0'))
        self.assertNotIn('UNRELATED',rows)
        self.assertEqual(env['PYTORCH_TUNABLEOP_TUNING'],'1')

    def test_qualified_record_and_negative_gates(self):
        env=p.tunable_environment({})
        record={'classification':'QUALIFIED','gpu_entry_functions_instrumented':['torch.cuda._lazy_init','torch._C._cuda_init'],
            'gpu_entry_calls':[],'gpu_initialized_before':False,'gpu_initialized_after':False,
            'controls_called':list(p.CONTROLS),'flags_before':dict(enabled=False,tuning=True,record_untuned=False),
            'flags_after':dict(enabled=False,tuning=False,record_untuned=False),
            'environment_before':env,'environment_after':copy.deepcopy(env),'forbidden_imports':[],
            'restoration_verified':True,'policy':dict(rejected_calls=0,native_calls=0,tensor_operations_allowed=0),
            'runtime':dict(torch=p.base.TORCH_RUNTIME_VERSION,distribution_version=p.base.TORCH_DISTRIBUTION_VERSION,
                git_version=p.base.TORCH_GIT_VERSION,hip=p.base.HIP_VERSION,cuda=None,prefix='/opt/modly-wsl/venv',
                python='3.12.3',gpu_initialized_before=False)}
        p.qualify(record)
        for key,value in [('gpu_entry_calls',['torch.cuda._lazy_init']),('gpu_initialized_after',True),
                          ('flags_after',dict(enabled=False,tuning=True,record_untuned=False)),
                          ('controls_called',['enable']),('restoration_verified',False)]:
            changed=copy.deepcopy(record);changed[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):p.qualify(changed)
        for field,value in [('tensor_operations_allowed',1),('native_calls',1),('rejected_calls',1)]:
            changed=copy.deepcopy(record);changed['policy'][field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):p.qualify(changed)
        changed=copy.deepcopy(record);changed['environment_after']['PYTORCH_TUNABLEOP_TUNING']['value']='1'
        with self.assertRaises(RuntimeError):p.qualify(changed)

    def test_paths_and_helper_anchor(self):
        self.assertEqual(p.base.SCRIPT.parent,p.ROOT/'scripts/compatibility')
        command=p.base.worker_command(p.base.guest_path(p.SCRIPT),p.base.guest_path(p.E/'run/guest'))
        self.assertEqual(command[command.index('--worker')-1],p.base.guest_path(p.SCRIPT))
        self.assertIn('90s',command)
        self.assertNotIn('loading.py',command)

if __name__=='__main__':unittest.main()
