"""Mocked pipeline safety tests; never import model runtimes or dispatch neural work."""
import contextlib,importlib.util,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT=Path(__file__).with_name('amd_zombie_c_handoff.py')
subject=None
if SCRIPT.exists():
    spec=importlib.util.spec_from_file_location('c_handoff',SCRIPT);subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)

class Exists(unittest.TestCase):
    def test_handoff_exists(self):self.assertIsNotNone(subject,'New input-bound standalone handoff missing')

@unittest.skipIf(subject is None,'not implemented')
class Policy(unittest.TestCase):
    def test_dry_run_cannot_dispatch_or_query_ram(self):
        with patch.object(subject,'run',side_effect=AssertionError('dispatch')),patch.object(subject,'windows_memory',side_effect=AssertionError('RAM')):
            out=io.StringIO()
            with contextlib.redirect_stdout(out):self.assertEqual(subject.main([]),0)
        plan=json.loads(out.getvalue());self.assertEqual(plan['mode'],'DRY_RUN');self.assertFalse(plan['neural_execution'])
        self.assertEqual(plan['stages'],['occupancy','shape','decode','comparison'])
        self.assertEqual(plan['maximum_forwards'],dict(dino=1,occupancy_sampling=22,occupancy_decode=1,shape=21,geometry_decode=1))

    def test_admission_exact22_passes_and24_not_required(self):
        sleeps=[];records=[]
        rows=subject.fresh_admission(lambda:dict(available_bytes=22*1024**3),sleeps.append,records.append)
        self.assertEqual(len(rows),3);self.assertEqual(sleeps,[4,4]);self.assertEqual(len(records),3)

    def test_low_read_stops_without_later_read(self):
        for position in range(3):
            values=iter([subject.FLOOR]*position+[subject.FLOOR-1]);record=[]
            with self.assertRaises(RuntimeError):subject.fresh_admission(lambda:dict(available_bytes=next(values)),lambda _:None,record.append)
            self.assertEqual(len(record),position+1)

    def test_gate_never_reserves_below_floor(self):
        calls=[]
        with self.assertRaises(RuntimeError):subject.guard_reservation(lambda:dict(available_bytes=subject.FLOOR-1),calls.append,lambda:calls.append('reserved'))
        self.assertEqual(len(calls),1);self.assertEqual(calls[0]['classification'],'NO_GO_FINAL_RAM')
        calls=[];subject.guard_reservation(lambda:dict(available_bytes=subject.FLOOR),calls.append,lambda:calls.append('reserved'))
        self.assertEqual(calls[-1],'reserved')

    def test_each_failed_stage_prevents_later_stages_and_never_retries(self):
        for index,failed in enumerate(subject.STAGES):
            calls=[]
            def stage(name):calls.append(name);return 1 if name==failed else 0
            with self.assertRaises(RuntimeError):subject.execute_once(stage)
            self.assertEqual(calls,list(subject.STAGES[:index+1]))

    def test_success_runs_each_stage_exactly_once(self):
        calls=[];subject.execute_once(lambda name:calls.append(name) or 0);self.assertEqual(calls,list(subject.STAGES))

    def test_only_two_exact_private_inputs_can_use_hash_authority(self):
        paths={'private/source.png','private/input-contract.json'}
        result=subject.publication_files(['old.py',*paths],paths,['manifest.json','new.py'])
        self.assertEqual(set(result),{'old.py','manifest.json','new.py'})
        result=subject.publication_files(['old.py','foreign/image.png'],paths,['manifest.json'])
        self.assertIn('foreign/image.png',result)

    def test_private_input_authentication_rejects_modified_source_contract_and_redirect(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parents[3]/'artifacts') as tmp:
            root=Path(tmp);(root/'source.png').write_bytes(b'new');(root/'contract.json').write_bytes(b'{}')
            authority=dict(source_path='source.png',input_contract_path='contract.json',source_sha256=subject.digest(root/'source.png'),input_contract_sha256=subject.digest(root/'contract.json'),source_bytes=3)
            subject.verify_private_inputs(root,authority)
            for path,blob in [('source.png',b'old'),('contract.json',b'[]')]:
                before=(root/path).read_bytes();(root/path).write_bytes(blob)
                with self.assertRaises(ValueError):subject.verify_private_inputs(root,authority)
                (root/path).write_bytes(before)
            authority['source_path']='../source.png'
            with self.assertRaises(ValueError):subject.verify_private_inputs(root,authority)

    def test_stage_bridge_has_no_imports_at_dry_run_or_module_import(self):
        self.assertFalse({'torch','trellis2','transformers','amd_zombie_c_stages'}.intersection(sys.modules))

    def test_run_requires_exact_commit(self):
        for commit in ('','HEAD','a'*39,'Z'*40):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):subject.main(['--run','--harness-commit',commit])

    def test_wrong_head_or_unpublished_source_fails_publication(self):
        with patch.object(subject,'git',return_value='b'*40):
            with self.assertRaises(ValueError):subject.verify_publication('a'*40)
        with tempfile.TemporaryDirectory(dir=SCRIPT.parents[3]/'artifacts') as tmp:
            root=Path(tmp);(root/'code.py').write_text('changed')
            def git(args):return 'a'*40 if args[0]=='rev-parse' else 'published'
            with patch.object(subject,'ROOT',root),patch.object(subject,'CRITICAL',['code.py']),patch.object(subject,'git',side_effect=git):
                with self.assertRaises(ValueError):subject.verify_publication('a'*40)

if __name__=='__main__':unittest.main()
