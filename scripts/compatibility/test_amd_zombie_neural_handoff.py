"""Synthetic handoff policy tests. Never construct a real runtime or launch WSL."""
import copy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).with_name('amd_zombie_neural_handoff.py')
spec=importlib.util.spec_from_file_location('handoff',SCRIPT) if SCRIPT.exists() else None
subject=None
if spec:
    subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)
FLOOR=23622320128


@contextmanager
def scratch_directory():
    parent=SCRIPT.parents[3]/'artifacts/amd-zombie-neural-handoff-tests'
    parent.mkdir(exist_ok=True)
    # Retain only newly created test-owned scratch, never historical evidence.
    yield tempfile.mkdtemp(prefix='policy-',dir=parent)


class Exists(unittest.TestCase):
    def test_standalone_handoff_exists(self):
        self.assertIsNotNone(subject,'Standalone handoff is missing')


@unittest.skipIf(subject is None,'Handoff not implemented')
class Policy(unittest.TestCase):
    def test_exact_floor_passes_with_three_reads_and_two_four_second_sleeps(self):
        sleeps=[];records=[]
        rows=subject.fresh_admission(lambda:dict(available_bytes=FLOOR),sleeps.append,records.append)
        self.assertEqual(len(rows),3)
        self.assertEqual(sleeps,[4,4]);self.assertEqual(len(records),3)

    def test_any_low_read_stops_without_retry(self):
        for position in range(3):
            values=iter([FLOOR]*position+[FLOOR-1]);recorded=[];sleeps=[]
            with self.assertRaises(RuntimeError):
                subject.fresh_admission(lambda:dict(available_bytes=next(values)),sleeps.append,recorded.append)
            self.assertEqual(len(recorded),position+1);self.assertEqual(len(sleeps),position)

    def test_final_gate_precedes_exclusive_reservation(self):
        with scratch_directory() as scratch:
            root=Path(scratch);calls=[]
            runtime=SimpleNamespace(evidence=root/'decode',commit='1'*40,contract=SimpleNamespace(sha256='a'*64),
                host_admission=lambda:dict(available_ram_bytes=FLOOR-1))
            with self.assertRaises(RuntimeError):
                subject.reserve_guarded(runtime,lambda output:calls.append(output),root/'decode/run',root/'low.json')
            self.assertEqual(calls,[]);self.assertFalse(runtime.evidence.exists())
            runtime.host_admission=lambda:dict(available_ram_bytes=FLOOR)
            subject.reserve_guarded(runtime,lambda output:calls.append(output),root/'decode/run',root/'pass.json')
            self.assertEqual(calls,[root/'decode/run'])
            with self.assertRaises(FileExistsError):
                subject.reserve_guarded(runtime,lambda output:calls.append(output),root/'decode/run',root/'pass.json')
            self.assertEqual(len(calls),1)

    def test_wrong_run_and_low_disk_do_not_reserve(self):
        with scratch_directory() as scratch:
            root=Path(scratch);calls=[]
            runtime=SimpleNamespace(evidence=root/'decode',commit='1'*40,contract=SimpleNamespace(sha256='a'*64),
                host_admission=lambda:dict(available_ram_bytes=FLOOR))
            with self.assertRaises(ValueError):
                subject.reserve_guarded(runtime,calls.append,root/'other',root/'wrong.json')
            with self.assertRaises(RuntimeError):
                subject.reserve_guarded(runtime,calls.append,root/'decode/run',root/'disk.json',disk_free=lambda p:2147483647)
            self.assertEqual(calls,[])

    def test_fresh_receipt_rejects_stale_close_spacing_or_forged_pass(self):
        now=datetime.now(timezone.utc)
        rows=[dict(timestamp_utc=(now-timedelta(seconds=8-i*4)).isoformat(),available_bytes=FLOOR) for i in range(3)]
        self.assertIsNone(subject.validate_admission(rows,now))
        for mutation in ('stale','spacing','low','count'):
            altered=copy.deepcopy(rows)
            if mutation=='stale':
                altered=[dict(r,timestamp_utc=(now-timedelta(minutes=2,seconds=8-i*4)).isoformat()) for i,r in enumerate(altered)]
            if mutation=='spacing':altered[1]['timestamp_utc']=altered[0]['timestamp_utc']
            if mutation=='low':altered[1]['available_bytes']=FLOOR-1
            if mutation=='count':altered.pop()
            with self.assertRaises(ValueError):subject.validate_admission(altered,now)

    def test_unpublished_source_is_rejected_before_execution(self):
        with scratch_directory() as scratch:
            root=Path(scratch);(root/'entry.py').write_text('working\n')
            def git(args):return '1'*40 if args==['rev-parse','HEAD'] else 'different\n'
            with self.assertRaises(ValueError):subject.verify_publication('1'*40,root,['entry.py'],git)

    def test_dry_run_has_no_runtime_or_wsl_import(self):
        before=set(sys.modules)
        result=subject.main([])
        self.assertEqual(result,0)
        added=set(sys.modules)-before
        self.assertFalse(any(n.startswith(('amd_generation','amd_zombie_decode','torch','trellis2')) for n in added))

    def test_routing_preserves_original_module_and_wraps_only_reservation(self):
        # Simulate the external runtime constructor; exercising the real one
        # would repeat full lineage work and import the guest decoder module.
        namespace={'STATIC_FILES':['inherited.py']}
        exec('class Runtime:\n def __init__(self): pass\n',namespace)
        runtime=namespace['Runtime']();runtime.reserve_attempt=lambda output:None
        route=SimpleNamespace(EVIDENCE='old',RUN_ID='old',source_inputs=None)
        def construct(commit):
            self.assertEqual(route.EVIDENCE,subject.EVIDENCE)
            self.assertEqual(route.RUN_ID,'run-zombie-standalone-d77c7ef-v1')
            self.assertIs(route.source_inputs,zombie.source_inputs)
            return runtime
        route.make_runtime=construct
        historical=SimpleNamespace(load_module=lambda name,path:route)
        zombie=SimpleNamespace(historical=historical,__file__=str(SCRIPT),EVIDENCE='historical',RUN_ID='historical',source_inputs=object())
        original=zombie.__dict__.copy()
        actual=subject.create_runtime(zombie,'1'*40)
        self.assertEqual(zombie.__dict__,original)
        self.assertIs(actual,runtime)
        self.assertEqual(namespace['STATIC_FILES'][0],'inherited.py')
        with patch.object(subject,'reserve_guarded',return_value='guarded'):
            self.assertEqual(actual.reserve_attempt(Path('owned/run')),'guarded')

    def test_missing_execution_commit_stops_before_any_runtime(self):
        with self.assertRaises(SystemExit) as caught:subject.main(['--run'])
        self.assertEqual(caught.exception.code,2)


if __name__=='__main__':unittest.main()
