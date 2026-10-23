"""Counterexamples pin actual boundaries; none qualifies neural execution."""
import copy
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'api'),str(ROOT/'scripts/compatibility')]
class NoAI(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'torch','triton','trellis2','flex_gemm','transformers'}:
            raise AssertionError('Model/native import forbidden in this CPU evidence check: '+name)
sys.meta_path.insert(0,NoAI())
import amd_shape_sampling_contract as policy
from test_amd_shape_sampling_result import candidate
from services.amd_completed_result import CompletionContext
from services.amd_stage_process import OwnedStageRunner,StageCommand
from services.amd_stage_orchestrator import StageOrchestrator,StagePlan,StageBlocked
from tests.test_amd_upstream_job import UpstreamJobTests

class EvidenceBoundaryTests(unittest.TestCase):
    def test_test_created_observation_only_produces_CPU_sidecar(self):
        fixture=UpstreamJobTests();fixture.setUp()
        try:
            _,handoff=fixture.complete()
            document=json.loads(handoff.receipt_path.read_bytes())
            self.assertEqual(document['qualification'],'CPU_CONTRACT_ONLY')
            self.assertIs(document['genuine_execution_qualified'],False)
        finally:fixture.doCleanups()

    def test_actual_runner_precancelled_result_is_not_execution_authority(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);event=threading.Event();event.set()
            context=CompletionContext.new('synthetic-no-launch',b'CPU fixture',{},model_id='trellis2-amd')
            command=StageCommand('conditioning','windows',('must-not-execute',),root,context.to_dict(),25)
            with patch('services.amd_stage_process.subprocess.Popen',side_effect=AssertionError('No process allowed')) as spawn, \
                 patch('services.amd_stage_process._OwnedJob',side_effect=AssertionError('No Job allowed')) as job:
                result=OwnedStageRunner(root).run(command,event)
            spawn.assert_not_called();job.assert_not_called()
            self.assertEqual(result['terminal'],'cancelled')
            self.assertIs(result['no_stage_launched'],True)
            self.assertIs(result['natural'],True)
            self.assertIsNone(result['returncode'])
            self.assertIs(type(result),dict)
            self.assertNotIn('supervisor_authority',result)

    def test_generated_orchestrator_still_blocks_before_dispatch(self):
        with tempfile.TemporaryDirectory() as name:
            image=b'labelled CPU fixture';params={}
            context=CompletionContext.new('synthetic-neural-block',image,params,model_id='trellis2-amd')
            plan=StagePlan(context,'generated',{'image_mode':'alpha-only','resolution':512},())
            with self.assertRaises(StageBlocked):
                StageOrchestrator(Path(name),None).run(plan,image,params,threading.Event())

    def test_manifest_classifier_is_not_actual_five_file_readback(self):
        record=candidate()
        with tempfile.TemporaryDirectory() as name:
            for leaf,row in record['artifacts'].items():row['path']=str(Path(name)/leaf)
            self.assertFalse(any(Path(row['path']).exists() for row in record['artifacts'].values()))
            self.assertEqual(policy.classify(record,True,True,True),'PASS')
        # This PASS is the original receipt classifier's scope, NOT adapter success.

    def test_manifest_classifier_does_not_authenticate_model_or_provenance_fields(self):
        for changed in ({'checkpoint_sha256':'0'*64},{'config_sha256':'0'*64},
                        {'seed':-1},{'parameters':{'steps':99}}):
            with self.subTest(changed=changed):
                record=candidate();record['provenance']=copy.deepcopy(changed)
                self.assertEqual(policy.classify(record,True,True,True),'PASS')

    def test_original_manifest_negative_controls_remain_effective(self):
        for name in ('shape-initial-noise.npy','shape-latent-provenance.json'):
            record=candidate();record['artifacts'].pop(name)
            self.assertEqual(policy.classify(record,True,True,True),'PARTIAL')
        record=candidate();record['artifacts']['shape-initial-noise.npy']['payload_sha256']='b'*64
        self.assertEqual(policy.classify(record,True,True,True),'PARTIAL')
        record=candidate();record['sampling']['steps'][0]['t']=0.1
        self.assertEqual(policy.classify(record,True,True,True),'PARTIAL')

if __name__=='__main__':
    # Load this class only: imported historical test cases are fixture helpers.
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceBoundaryTests))
    raise SystemExit(not result.wasSuccessful())
