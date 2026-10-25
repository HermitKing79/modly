"""Scoped routing of qualified stages to Zombie C. Never a generation entrypoint."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import amd_zombie_c_handoff as h

@contextmanager
def argv(arguments):
    old=sys.argv
    try:sys.argv=[str(Path(__file__).resolve()),*arguments];yield
    finally:sys.argv=old

class Bridge:
    def __init__(self,commit,read=h.windows_memory):
        if not re.fullmatch('[0-9a-f]{40}',commit or ''):raise ValueError('Exact stage commit required')
        self.commit=commit;self.read=read;self.installed=False;self._pending_gate=None;self._phase=None;self.gates=0

    def _check_private(self):
        if h.digest(h.ROOT/h.REFERENCE)!=self._authority_sha:raise ValueError('Committed reference authority changed')
        h.verify_private_inputs(h.ROOT,self._authority)

    def _gate(self,label):
        self.gates+=1
        h.guard_reservation(self.read,lambda row:h.save(h.PRIVATE/f'gate-{self.gates:02d}-{label}.json',row),lambda:None)

    def verify_files(self,commit,files=None,*,timeout_seconds=5):
        if commit!=self.commit:raise ValueError('Foreign stage commit denied')
        self._check_private()
        private={self._authority['source_path'],self._authority['input_contract_path']}
        names=h.publication_files(self.sup.COMMITTED_FILES if files is None else files,private,h.CRITICAL)
        result=self._verify_original(commit,names,timeout_seconds=timeout_seconds)
        self._check_private()
        result.update({self._authority['source_path']:self._authority['source_sha256'],
                       self._authority['input_contract_path']:self._authority['input_contract_sha256']})
        if self._pending_gate:
            self._gate(self._pending_gate);self._pending_gate=None
        return result

    def supervise(self,output,commit,*,profile=None):
        allowed={h.CASE/'occupancy'/s/'run' for s in ('dino','sampling','decoder')}|{h.CASE/'shape/run'}
        if Path(output) not in allowed or commit!=self.commit or self._pending_gate:
            raise ValueError('Only one exact C stage may be supervised')
        self._pending_gate=('shape' if Path(output)==h.CASE/'shape/run' else Path(output).parent.name)+'-marker'
        try:
            code=self._supervise_original(output,commit,profile=profile)
            if self._pending_gate is not None:raise RuntimeError('Stage bypassed final committed-file/RAM gate')
            return code
        finally:self._pending_gate=None

    def healthy(self,capture):
        available=self.read()['available_bytes']
        if type(available) is not int or available<h.LIVE_FLOOR:
            raise RuntimeError('Preserved 12 GiB live host floor failed: '+str(available))
        return self._healthy_original(capture)

    def install(self):
        if self.installed:return
        self._authority=h.authority();self._authority_sha=h.digest(h.ROOT/h.REFERENCE)
        import amd_generation_occupancy as occ
        import amd_generation_shape as shape
        import amd_generation_decode as decode
        import amd_sparse_decoder_forward as sup
        self.occ,self.shape,self.decode,self.sup=occ,shape,decode,sup
        occ.configure(h.CASE/'occupancy/source-image.png',h.CASE/'occupancy',
            expected_image_sha=self._authority['source_sha256'],expected_contract_sha=self._authority['input_contract_sha256'])
        self._verify_original=sup.verify_committed_files;sup.verify_committed_files=self.verify_files
        self._supervise_original=sup.supervise;sup.supervise=self.supervise
        self._healthy_original=sup.healthy_observation;sup.healthy_observation=self.healthy
        monitor=sup.WindowsMonitor
        sup.WindowsMonitor=lambda since,send:monitor(since,send,healthy=self.healthy)
        old_preflight=occ._preflight
        def preflight(commit):
            result=old_preflight(commit);self._check_private();self._gate('occupancy-sequence-marker');return result
        occ._preflight=preflight
        old_profile=occ.prepare_profile
        def occupancy_profile(stage):
            p=old_profile(stage);p.worker_script=Path(__file__).resolve()
            p.committed_files=list(dict.fromkeys([*p.committed_files,*h.CRITICAL]));return p
        occ.prepare_profile=occupancy_profile;occ.original.prepare_profile=occupancy_profile
        shape_profile=shape.Runtime.profile
        def profile(runtime):
            p=shape_profile(runtime);p.worker_script=Path(__file__).resolve()
            p.committed_files=list(dict.fromkeys([*p.committed_files,*h.CRITICAL]));return p
        shape.Runtime.profile=profile
        self.installed=True

    def decode_reservation(self,runtime,original,output):
        if Path(output)!=runtime.evidence/'run' or runtime.evidence!=h.CASE/'decode':raise ValueError('Exact C decode output required')
        runtime.host_admission()  # Original live RAM and D: disk guards remain decisive.
        self._gate('geometry-decode-marker')
        return original(output)

    def dispatch(self,name):
        if name not in h.STAGES:raise ValueError('Foreign pipeline stage')
        self.install();self._check_private();self._phase=name
        os.environ['MODLY_ZOMBIE_C_STAGE']=name
        if name=='occupancy':
            return self.occ.main(['--image',str(h.CASE/'occupancy/source-image.png'),'--evidence',str(h.CASE/'occupancy'),'--harness-commit',self.commit])
        if name=='shape':
            with argv(['--occupancy-evidence',str(h.CASE/'occupancy'),'--evidence',str(h.CASE/'shape'),'--harness-commit',self.commit]):
                return self.shape.main()
        if name=='decode':
            runtime=self.decode.Runtime(h.CASE/'shape',h.CASE/'decode',self.commit,'run-zombie-c-60862722-v1',
                                        tile_large_convolutions=True,final_support_profile=None)
            original=runtime.reserve_attempt
            runtime.reserve_attempt=lambda output:self.decode_reservation(runtime,original,output)
            return runtime.supervise()
        import amd_zombie_c_compare as comparison
        return comparison.run(self.commit)

def worker(output):
    commit=os.environ.get('MODLY_ZOMBIE_C_COMMIT','');phase=os.environ.get('MODLY_ZOMBIE_C_STAGE','')
    if not re.fullmatch('[0-9a-f]{40}',commit) or phase not in ('occupancy','shape'):
        raise ValueError('Worker requires exact parent pipeline permit')
    permit_path=h.plain(h.PRIVATE/'pipeline-attempt-started.json')
    if permit_path.stat().st_size>65536:raise ValueError('Bounded parent permit required')
    permit=json.loads(permit_path.read_bytes())
    if (permit.get('harness_commit')!=commit or permit.get('maximum_pipelines')!=1
            or permit.get('maximum_forwards')!=h.FORWARDS or permit.get('source_sha256')!=h.authority()['source_sha256']):
        raise ValueError('Parent pipeline permit differs')
    output=h.plain(output)
    allowed={h.CASE/'occupancy'/s/'run/scratch':n for s,n in (('dino',1),('sampling',22),('decoder',1))} if phase=='occupancy' else {h.CASE/'shape/run/scratch':21}
    if output not in allowed or (output.parent/'results.json').exists():raise ValueError('Exact unused C worker scratch required')
    marker_path=output.parents[1]/'attempt-started.json'
    if marker_path.stat().st_size>65536:raise ValueError('Bounded stage permit required')
    marker=json.loads(marker_path.read_bytes())
    if marker.get('harness_commit')!=commit or marker.get('maximum_forwards')!=allowed[output]:raise ValueError('Stage forward budget differs')
    bridge=Bridge(commit);bridge.install()
    if phase=='occupancy':return bridge.occ.main(['--worker','--output',str(output)])
    with argv(['--worker','--output',str(output)]):return bridge.shape.main()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if not args.worker:
        print('DRY_RUN: scoped worker adapter only; use Invoke-ZombieCGeneration.ps1 after closing apps');return 0
    if args.output is None:parser.error('Owned worker requires --output')
    return worker(args.output)

if __name__=='__main__':raise SystemExit(main())
