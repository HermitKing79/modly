"""CPU preparation only for the authenticated brick guided PBR decoder.

There is deliberately no worker, GPU, model-loading or inference CLI route.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'experimental/extensions/trellis2-amd'
EVIDENCE=ROOT/'docs/compatibility/amd-brick-guided-pbr-cpu-evidence'
GEOMETRY=ROOT.parent/'artifacts/amd-brick-neural-completion-c9447dd/neural-output'
GIB=1024**3
LIMITS=dict(allocator_bytes=12*GIB, initial_free_vram_bytes=13*GIB,
    device_reserve_bytes=2*GIB, opaque_headroom_bytes=GIB,
    live_windows_ram_bytes=12*GIB, prelaunch_windows_ram_bytes=37*GIB//2,
    D_free_bytes=50*GIB, tile_rows=32768, native_seconds=60, cpu_seconds=120,
    gpu_seconds=240, host_seconds=480)


class ResourceAdmission(RuntimeError):
    def __init__(self, record):
        self.record=record
        super().__init__('CPU resource admission refused: '+','.join(record['failed']))


def read_resources():
    if sys.platform!='win32':
        raise RuntimeError('This read-only host preparation requires Windows')
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[
            (name,ctypes.c_uint64) for name in ('total','available','page_total',
                'page_available','virtual_total','virtual_available','extended')]
    value=Memory();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    return dict(available_windows_ram_bytes=value.available,D_free_bytes=shutil.disk_usage('D:/').free)


def check_resources():
    measured=read_resources();failed=[]
    if measured['available_windows_ram_bytes']<LIMITS['live_windows_ram_bytes']:
        failed.append('available_windows_ram')
    if measured['D_free_bytes']<LIMITS['D_free_bytes']:
        failed.append('D_free_space')
    record=dict(stage='cpu-only-guided-pbr-preparation',timestamp_utc=datetime.now(timezone.utc).isoformat(),
        measured=measured,required=dict(available_windows_ram_bytes=LIMITS['live_windows_ram_bytes'],
                                      D_free_bytes=LIMITS['D_free_bytes']),failed=failed,admitted=not failed)
    if failed:raise ResourceAdmission(record)
    return record


def cpu_guard(event,args):
    if event=='import' and str(args[0]).split('.')[0] in {
            'torch','triton','trellis2','flex_gemm','o_voxel','cumesh'}:
        raise RuntimeError('CPU preparation cannot import AI/GPU/native runtimes')
    if event.startswith('socket.') or event in ('subprocess.Popen','os.system'):
        raise RuntimeError('CPU preparation cannot launch workers or use the network')


def prepare(profile_name):
    if any(name in sys.modules for name in ('torch','triton','flex_gemm','trellis2')):
        raise RuntimeError('Fresh CPU-only preparation process required')
    sys.addaudithook(cpu_guard)
    resources=check_resources()
    sys.path.insert(0,str(CODE))
    from inference_texture import profile,inputs,admission
    document=profile.contracts.load(GEOMETRY/'run-contract.json').document()
    selected=profile.authenticate(profile_name,document)
    prepared=inputs.prepare(selected)
    plans=admission.memory_plans(selected)
    source_paths=sorted((CODE/'inference_texture').glob('*.py'))+[Path(__file__).resolve()]
    sources={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
             for path in source_paths}
    return dict(schema='modly.amd.brick-guided-pbr-cpu-preparation.v1',
        classification='CPU_PREPARED_NOT_EXECUTED',timestamp_utc=datetime.now(timezone.utc).isoformat(),
        resources=resources,limits=LIMITS,profile=selected.document(),profile_sha256=selected.sha256,
        authenticated_inputs=prepared['metadata'],memory_plans=plans,source_sha256=sources,
        torch_imported=False,gpu_initialized=False,neural_forward_calls=0,model_weights_loaded=False,
        native_bindings_invoked=False,output_material_field_produced=False,
        scope='Preparation is not GPU execution evidence. Independent tests/review and future live admission remain mandatory.')


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--profile',required=True,choices=['brick-1957891-v1'])
    result.add_argument('--output',required=True,type=Path)
    return result


def main():
    args=parser().parse_args();output=args.output.absolute()
    if (output.parent.resolve(strict=True)!=EVIDENCE.resolve(strict=True)
            or output.name!=output.resolve().name or output.suffix!='.json' or os.path.lexists(output)):
        raise ValueError('Fresh exclusive textual preparation receipt in the mission evidence directory required')
    try:record=prepare(args.profile)
    except ResourceAdmission as exc:
        record=dict(classification='BLOCKED',resources=exc.record,gpu_initialized=False,neural_forward_calls=0)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(classification=record['classification'],output=str(output))))
    return 0 if record['classification']=='CPU_PREPARED_NOT_EXECUTED' else 2


if __name__=='__main__':
    raise SystemExit(main())
