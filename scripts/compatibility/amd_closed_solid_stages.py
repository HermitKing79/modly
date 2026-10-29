"""Case routing only. Original stage safety implementation stays unchanged."""
import argparse,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_closed_solid_benchmark as b
import amd_closed_solid_handoff as h
base=b.load('_closed_solid_private_D_stage_namespace',Path(__file__).with_name('amd_zombie_d_stages.py'))
base.h=h;base.__file__=str(Path(__file__).resolve())
class Bridge(base.Bridge):
    def __init__(self,commit,read=None):super().__init__(commit,read=h.windows_memory if read is None else read)
    def install(self):
        if self.installed:return
        if {'amd_generation_occupancy','amd_generation_shape','amd_generation_decode','amd_sparse_decoder_forward'}.intersection(sys.modules):raise RuntimeError('Only one stage adapter in a fresh process')
        super().install()
    def dispatch(self,name):
        if name in ('occupancy','shape'):return super().dispatch(name)
        if name=='decode':
            self.install();self._check_private();self._phase=name
            runtime=self.decode.Runtime(h.CASE/'shape',h.CASE/'decode',self.commit,'run-closed-solid-box-v1',tile_large_convolutions=True,final_support_profile=None)
            original=runtime.reserve_attempt;runtime.reserve_attempt=lambda output:self.decode_reservation(runtime,original,output)
            return runtime.supervise()
        if name=='comparison':
            self._check_private()
            import amd_closed_solid_cpu as cpu
            return cpu.future(self.commit)
        raise ValueError('Only exact closed-solid stages permitted')
base.Bridge=Bridge
worker=base.worker
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args(argv)
    if not a.worker:print('DRY_RUN: scoped adapter only; no model/WSL/marker');return 0
    if a.output is None:p.error('Owned worker scratch required')
    return worker(a.output)
if __name__=='__main__':raise SystemExit(main())
