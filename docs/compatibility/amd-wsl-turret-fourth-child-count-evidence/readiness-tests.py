"""Metadata verdicts and real paths only; no Torch/GPU import."""
import copy
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('ready',Path(__file__).with_name('readiness.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class Tests(unittest.TestCase):
    def test_complete_acceptance_and_each_negative_boundary(self):
        row={'classification':'QUALIFIED','runtime':dict(torch=r.base.TORCH_RUNTIME_VERSION,
             distribution_version=r.base.TORCH_DISTRIBUTION_VERSION,git_version=r.base.TORCH_GIT_VERSION,
             hip=r.base.HIP_VERSION,cuda=None,python='3.12.3',prefix='/opt/modly-wsl/venv',gpu_initialized_before=False),
             'gpu':dict(count=1,name='AMD Radeon RX 9070 XT',architecture='gfx1201',free_bytes=14*1024**3),
             'free_bytes_after_release':14*1024**3,'memory_before':dict(allocated=0,reserved=0),
             'memory_release':dict(allocated=0,reserved=0),'policy':dict(tensor_operations_allowed=0,rejected_calls=0),
             'forbidden_imports':[],'tunable_admission':{'after_torch_import':dict(
             flags=dict(enabled=False,tuning=False,record_untuned=False),
             environment={k:'0' for k in ('PYTORCH_TUNABLEOP_ENABLED','PYTORCH_TUNABLEOP_TUNING','PYTORCH_TUNABLEOP_RECORD_UNTUNED')})}}
        r.qualify(row)
        for section,key,value in [('gpu','free_bytes',13*1024**3-1),('gpu','count',2),('gpu','architecture','gfx1100'),
                                  ('memory_release','allocated',1),('memory_before','reserved',1),
                                  ('policy','tensor_operations_allowed',1),('policy','rejected_calls',1)]:
            bad=copy.deepcopy(row);bad[section][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):r.qualify(bad)
        for key,value in [('forbidden_imports',['flex_gemm']),('free_bytes_after_release',13*1024**3-1),('tunable_admission',{})]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):r.qualify(bad)

    def test_original_helper_anchor_and_guest(self):
        self.assertEqual(r.base.SCRIPT.parent,r.ROOT/'scripts/compatibility')
        command=r.base.worker_command(r.base.guest_path(r.SCRIPT),r.base.guest_path(r.E/'readiness-run/guest'))
        self.assertEqual(command[command.index('--worker')-1],r.base.guest_path(r.SCRIPT))
        self.assertIn('90s',command)
        self.assertNotIn('decode.py',command)

if __name__=='__main__':unittest.main()
