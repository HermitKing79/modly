"""CPU-only tests for the one-run host composition; no WSL or GPU imports."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT/'docs/compatibility/amd-brick-pbr-postrestart-evidence/host.py'

def cpu_only(event,args):
    if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','trellis2','flex_gemm','o_voxel','cumesh'}:
        raise RuntimeError('CPU host acceptance forbids AI/native/model imports')
    if event in ('subprocess.Popen','os.system','socket.connect','socket.bind','socket.getaddrinfo'):
        raise RuntimeError('CPU host acceptance forbids actual processes and network')

sys.addaudithook(cpu_only)


def load():
    spec = importlib.util.spec_from_file_location('brick_pbr_host_cpu_tests', PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HostCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = load()

    def test_original_limits_and_no_exception(self):
        h = self.host
        self.assertEqual(h.PRELAUNCH_RAM, 37*1024**3//2)
        self.assertEqual(h.LIVE_RAM, 12*1024**3)
        self.assertEqual(h.IO_SECONDS, 30)
        self.assertEqual(h.DEADLINES, {'native':60,'cpu':120,'gpu':240,'host':480})

    def test_committed_identity_only_accepts_proven_git_crlf_checkout_conversion(self):
        h=self.host;name='api/services/amd_geometry_assets.py'
        attrs=b''.join((name.encode()+b'\0'+key+b'\0unspecified\0')
                       for key in (b'text',b'filter',b'working-tree-encoding',b'ident'))
        with patch.object(h.subprocess,'check_output',side_effect=['i/lf w/crlf attr/\t'+name,attrs,'true']) as git:
            self.assertTrue(h.committed_blob_matches(name,b'a\nb\n',b'a\r\nb\r\n'))
            self.assertEqual(git.call_count,3)
        for blob,working in ((b'a\nb\n',b'a\r\nc\r\n'),(b'a\n',b'a\r\n\r'),(b'a\n',b'a\n\n')):
            self.assertFalse(h.committed_blob_matches(name,blob,working))
        for eol,metadata,autocrlf in (
                ('i/lf w/lf',attrs,'true'),('i/lf w/crlf',attrs.replace(b'filter\0unspecified',b'filter\0foreign'),'true'),
                ('i/lf w/crlf',attrs.replace(b'text\0unspecified',b'text\0unset'),'true'),
                ('i/lf w/crlf',attrs.replace(b'ident\0unspecified',b'ident\0set'),'true'),
                ('i/lf w/crlf',attrs,'false')):
            with self.subTest(eol=eol,metadata=metadata,autocrlf=autocrlf),patch.object(h.subprocess,'check_output',side_effect=[eol,metadata,autocrlf]):
                self.assertFalse(h.committed_blob_matches(name,b'a\n',b'a\r\n'))
        with patch.object(h.subprocess,'check_output') as git:
            self.assertTrue(h.committed_blob_matches(name,b'a\r\n',b'a\r\n'));git.assert_not_called()

    def test_remaining_host_reference_uses_actual_reading_and_refuses_shortfall(self):
        h = self.host
        for stage, remaining in [('native',3067854848),('cpu',1143431168)]:
            good = h.assess_remaining(stage,h.LIVE_RAM+remaining)
            self.assertTrue(good['admitted'])
            self.assertEqual(good['reference_remaining_bytes'],remaining)
            self.assertFalse(h.assess_remaining(stage,h.LIVE_RAM+remaining-1)['admitted'])
        with self.assertRaises(ValueError): h.assess_remaining('foreign',2**40)

    def test_adverse_recent_ram_step_cannot_spend_the_remaining_margin(self):
        h=self.host;base=h.LIVE_RAM+1143431168
        history=[{'available_ram_bytes':base+1100},{'available_ram_bytes':base+600},
                 {'available_ram_bytes':base+100}]
        self.assertFalse(h.assess_remaining('cpu',base+100,recent=history)['admitted'])
        self.assertTrue(h.assess_remaining('cpu',base+1000,recent=history)['admitted'])

    def test_source_cache_is_exact_empty_nonexistent_and_writes_disabled(self):
        h=self.host;h.require_source_cache()
        with patch.object(h,'EMPTY_BYTECODE',ROOT),patch.object(sys,'pycache_prefix',str(ROOT)):
            with self.assertRaises(RuntimeError):h.require_source_cache()

    def test_frozen_supervisor_hashes_the_same_blob_it_executes(self):
        h=self.host
        with patch.object(h,'_frozen',None),patch.object(h,'_original_admission',h._original_admission),\
             patch.object(h,'digest',side_effect=AssertionError('Do not hash a separate file read')):
            self.assertEqual(h.load_frozen().EV,h.EVIDENCE)

    def test_stage_identity_binds_plan_marker_profile_and_order(self):
        h=self.host
        plan={'marker_sha256':'1'*64,'attestation_sha256':'2'*64}
        profile=Mock(sha256='3'*64,run_contract_sha256='4'*64)
        row=dict(schema='modly.amd.brick-pbr-stage-ready.v1',stage='native',
                 marker_sha256='1'*64,attestation_sha256='2'*64,
                 profile_sha256='3'*64,run_contract_sha256='4'*64,plan_sha256='5'*64)
        self.assertTrue(h.validate_stage_ready(row,'native',plan,profile,'5'*64))
        for key in row:
            wrong=dict(row);wrong[key]='foreign'
            with self.subTest(key=key),self.assertRaises(ValueError):
                h.validate_stage_ready(wrong,'native',plan,profile,'5'*64)

    def test_binary_go_and_helper_anchor_are_inherited(self):
        h=self.host
        base=h.load_frozen().base
        self.assertEqual(Path(base.SCRIPT).resolve(),ROOT/'scripts/compatibility/amd_wsl_rocdxg_smoke.py')
        stream=Mock()
        with patch.object(base,'healthy_observation',return_value=True):
            base.authorize_go(stream,True,{}, {})
        stream.write.assert_called_once_with(b'GO\n')

    def test_real_command_routes_exact_worker_and_plan_with_bounded_timeout(self):
        h=self.host
        with patch.object(h,'ACTIVE_PLAN_PATH',h.PRIVATE/'execution-plan.json'),\
             patch.object(h,'ACTIVE_PLAN_SHA','6'*64):
            command=h.worker_command(h.to_guest(h.WORKER),h.to_guest(h.EVIDENCE/'run/guest'))
        self.assertIn('450s',command)
        self.assertIn('/opt/modly-wsl/venv/bin/python',command)
        self.assertEqual(command[command.index('--plan-sha256')+1],'6'*64)
        self.assertEqual(command[command.index('--worker')-1],h.to_guest(h.WORKER))

    def test_async_receipt_does_not_block_while_future_pending(self):
        h=self.host
        bridge=Mock();bridge.receipt_dir=Path('C:/synthetic/probe')
        future=Mock();future.done.return_value=False
        base=Mock();base.in_thread.return_value=future
        receipt=h.AsyncHostReceipt(None,base,bridge=bridge,plan={},plan_sha='7'*64,monitor=Mock())
        with patch.object(receipt,'poll_stages'),patch.object(Path,'exists',return_value=True):
            receipt.poll()
            receipt.poll()
        base.in_thread.assert_called_once()
        self.assertFalse(receipt.done)
        bridge.host_receipt.return_value.poll.assert_not_called()

    def test_post_exit_reader_is_fresh_and_does_not_accept_cached_ack(self):
        h=self.host
        bridge=Mock();bridge.root=Path('C:/synthetic/field');bridge.profile=Mock()
        bridge.owner.host_verify.return_value={'same':'identity'}
        data={'verified':True,'metadata':{'artifact_sha256':'8'*64,'artifact_bytes':125305024},'artifact_path':'field'}
        with patch.object(h,'get_bridge',return_value=bridge),\
             patch.object(h,'artifact_module') as artifact:
            artifact.ARTIFACT_NAME='texture-field.safetensors';artifact.verify.return_value=data
            result=h.verify_after_exit(None)
        artifact.verify.assert_called_once()
        self.assertEqual(bridge.owner.host_verify.call_count,2)
        self.assertEqual(result['status'],'PASS')

    def test_complete_frozen_supervisor_lifecycle_and_forced_cleanup_refusal_are_cpu_simulations(self):
        h=self.host;frozen=h.load_frozen();base=frozen.base
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True)
        for forced in (False,True):
            with self.subTest(forced=forced), tempfile.TemporaryDirectory(dir=parent) as temporary:
                ev=Path(temporary);output=ev/'run';world={}
                class Stdin(io.BytesIO):
                    def write(self,data):
                        self.asserted=data
                        world['child'].authorized=data==b'GO\n'
                        return super().write(data)
                class Child:
                    def __init__(self,*args,**kwargs):
                        self.pid=12345;self.stdin=Stdin();self.authorized=False;self.returncode=None
                        world['child']=self
                    def poll(self):
                        if self.authorized:
                            self.returncode=0
                            if not (output/'guest/results.json').exists():
                                (output/'guest').mkdir()
                                base.save_json(output/'guest/results.json',{'classification':'QUALIFIED','SYNTHETIC_CPU_ONLY':True})
                        return self.returncode
                class Job:
                    def attach(self,child):self.child=child
                    def active(self):return 0
                    def finish(self):return {'owned_tree_exit_verified':True,'forced_job_cleanup':forced}
                class Monitor:
                    def __init__(self,*args):self.observations=[]
                    def poll(self,now):self.observations.append({'capture':{'synthetic':True}})
                    def healthy_since(self,*args):return True
                    def latest(self):return {'synthetic':True}
                class Receipt:
                    def __init__(self,*args):self.done=False
                    def poll(self):self.done=world.get('child') is not None and world['child'].authorized
                shim=ModuleType('artifact_io');shim.HostReceipt=Receipt
                shim.verify_on_host=lambda _directory:{'status':'PASS','SYNTHETIC_CPU_ONLY':True}
                helper=SimpleNamespace(windows_observation=lambda _: {},healthy_observation=lambda _:True)
                def sibling(name):return SimpleNamespace(OwnedJob=Job) if name=='amd_forward_owned_job' else helper
                with patch.multiple(frozen,EV=ev,SCRIPT=h.WORKER,
                        preflight=lambda:None,reserve_attempt=lambda path:path.mkdir(),
                        host_admission=lambda *args:{'available_ram_bytes':21*1024**3,'D_free_bytes':60*1024**3},
                        worker_command=lambda *args:['SYNTHETIC_CPU_WORKER'],validate_qualified=lambda _:None),\
                     patch.object(base,'HealthMonitor',Monitor),patch.object(base,'load_sibling',side_effect=sibling),\
                     patch.object(base,'healthy_observation',return_value=True),\
                     patch.object(base,'terminate_owned_distro',return_value={'returncode':0}),\
                     patch.object(frozen.subprocess,'Popen',Child),h.binding('artifact_io',shim):
                    code=frozen.supervise(output)
                result=json.loads((output/'results.json').read_text())
                self.assertEqual(world['child'].stdin.asserted,b'GO\n')
                self.assertEqual(code,1 if forced else 0)
                self.assertEqual(result['classification'],'UNKNOWN' if forced else 'QUALIFIED')
                self.assertTrue(result['observer_healthy'])
                self.assertEqual(result['windows_job_cleanup']['forced_job_cleanup'],forced)


if __name__=='__main__':unittest.main(verbosity=2)
