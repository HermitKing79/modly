"""Single CPU-only real Torch/Triton guard acceptance; no FlexGEMM import."""
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
STAGE = Path('/home/modly/native-decoder-preparation-6dd94a85')
sys.path.insert(0, str(STAGE / 'site-packages'))
import torch

gpu_attempts = []
def no_gpu(*args, **kwargs):
    gpu_attempts.append('unexpected GPU initialization or metadata')
    raise RuntimeError('CPU acceptance prohibits GPU entry')

assert not torch.cuda.is_initialized()
for name in ('_lazy_init', 'init', 'is_available', 'device_count', 'current_device',
             'get_device_name', 'get_device_properties'):
    setattr(torch.cuda, name, no_gpu)
for name in ('_cuda_init', '_cuda_getDeviceCount'):
    if hasattr(torch._C, name):
        setattr(torch._C, name, no_gpu)

import triton
import triton.testing
spec = importlib.util.spec_from_file_location('import_policy', ROOT / 'scripts/compatibility/amd_wsl_native_import_policy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
scratch = Path(tempfile.mkdtemp(prefix='policy-cpu-', dir=STAGE))
policy = module.ImportPolicy(scratch, writable_roots=(Path(__file__).parent,))
policy.install_tensor_guards(torch)
policy.install_runtime_guards(triton)
policy.install_audit_hook()
policy.install_profile()

@triton.jit
def unexecuted_constructor_fixture():
    pass

autotuner = triton.runtime.Autotuner(unexecuted_constructor_fixture,
                                    unexecuted_constructor_fixture.arg_names,
                                    [triton.Config({}, num_warps=4)], [], None, None)
driver = importlib.import_module('triton.runtime.driver').driver
assert driver._active is None and driver._default is None
rejected = []
def must_reject(label, call):
    try:
        call()
    except module.ImportPolicyError:
        rejected.append(label)
    else:
        raise AssertionError('Boundary escaped: ' + label)

with policy.tensor_mode(torch):
    must_reject('import tensor factory', lambda: torch.empty((1,), device='cpu'))
    policy.set_phase('cpu_meta')
    cpu = torch.empty((1,), device='cpu')
    meta = torch.empty((1,), device='meta')
    assert cpu.device.type == 'cpu' and meta.device.type == 'meta'
    with torch.device('meta'):
        meta_default = torch.empty((1,))
    with torch.device('cpu'):
        cpu_default = torch.empty((1,))
    assert meta_default.device.type == 'meta' and cpu_default.device.type == 'cpu'
    with torch.device('cuda'):
        must_reject('default GPU requested device', lambda: torch.empty((1,)))
    must_reject('GPU requested device', lambda: torch.empty((1,), device='cuda'))

must_reject('JIT run', lambda: unexecuted_constructor_fixture.run())
must_reject('compile', lambda: triton.compile(None))
must_reject('autotuner run', lambda: autotuner.run())
must_reject('helper shared alias', lambda: importlib.import_module('triton.runtime.build').compile_module_from_src('x', 'x'))
must_reject('helper AMD alias', lambda: importlib.import_module('triton.backends.amd.driver').compile_module_from_src('x', 'x'))
must_reject('process', lambda: subprocess.Popen([sys.executable, '-c', 'raise SystemExit(0)']))
must_reject('network', lambda: socket.getaddrinfo('localhost', 1))
must_reject('foreign write', lambda: (scratch.parent / 'policy-unowned-file').write_text('rejected'))
assert len(rejected) == 11
assert not gpu_attempts and not torch.cuda.is_initialized()
assert driver._active is None and driver._default is None
assert not any(name == 'flex_gemm' or name.startswith('flex_gemm.') for name in sys.modules)
assert 'hip_utils' not in sys.modules and '__triton_launcher' not in sys.modules
record = {
    'result': 'PASS', 'torch': torch.__version__, 'torch_distribution': importlib.metadata.version('torch'),
    'hip': torch.version.hip, 'triton': triton.__version__, 'python': sys.version.split()[0],
    'GPU_initialized': torch.cuda.is_initialized(), 'GPU_entry_attempts': gpu_attempts,
    'FlexGEMM_imported': False, 'native_extension_imported': False,
    'Triton_driver_activated': False, 'rejected_boundaries': rejected,
    'implicit_meta_device': meta_default.device.type,
    'implicit_cpu_device': cpu_default.device.type,
    'policy': policy.snapshot(), 'scratch': str(scratch),
}
policy.restore_profile()
policy.restore_runtime_guards()
policy.restore_tensor_guards()
output = Path(__file__).with_suffix('.json')
with output.open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(record), flush=True)
