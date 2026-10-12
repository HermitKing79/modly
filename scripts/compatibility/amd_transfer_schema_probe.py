"""CPU-only characterization of the installed copy boundary; no checkpoint reads."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def main(output):
    output = output.resolve()
    output.relative_to(Path(__file__).resolve().parents[2])
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU probe found an existing GPU context')
    def forbidden(*args, **kwargs):
        raise RuntimeError('CPU probe prohibits GPU initialization')
    torch.cuda._lazy_init = forbidden
    rows = []
    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            rows.append({'op': str(func), 'kwargs': {key: str(value) for key, value in kwargs.items()},
                         'source_device': str(args[0].device) if args and isinstance(args[0], torch.Tensor) else None})
            if any(isinstance(value, torch.device) and value.type == 'cuda' for value in kwargs.values()):
                raise RuntimeError('Recorded intended AMD transfer before dispatch')
            return func(*args, **kwargs)
    result = {'python': sys.version, 'torch': torch.__version__, 'torch_path': torch.__file__,
              'hip': torch.version.hip, 'cuda': torch.version.cuda,
              'schemas': {name: str(getattr(torch.ops.aten, name.split('.')[0]).__getattr__(name.split('.')[1])._schema)
                          for name in ('to.dtype_layout', '_to_copy.default')}}
    # Match production: dispatch mode is installed before inference mode starts.
    with Capture(), torch.inference_mode():
        x = torch.tensor([-.875, -.625, -.375, -.125, .125, .375, .625, .875], dtype=torch.float32).reshape(1,8,1,1,1)
        rows.clear()
        try:
            x.to('cuda:0')
        except RuntimeError as exc:
            result['original_input'] = {'error': str(exc), 'dispatch': list(rows)}
        else:
            raise RuntimeError('Intended AMD copy was not intercepted')
        rows.clear()
        x.to('cpu', copy=True)
        result['original_input_cpu_surrogate'] = {'dispatch': list(rows)}
        rows.clear()
        y = torch.ops.aten._to_copy.default(x, device=torch.device('cpu'))
        result['explicit_input_cpu_surrogate'] = {'dispatch': list(rows), 'shape': list(y.shape),
                'stride': list(y.stride()), 'source_stride': list(x.stride()),
                'dtype': str(y.dtype), 'equal': torch.equal(x, y), 'distinct': y is not x}
        rows.clear()
        z = torch.arange(64, dtype=torch.float32).reshape(1,1,4,4,4)
        rows.clear()
        z.cpu()
        result['original_output_cpu_surrogate'] = {'dispatch': list(rows)}
        rows.clear()
        q = torch.ops.aten._to_copy.default(z, device=torch.device('cpu'))
        result['explicit_output_cpu_surrogate'] = {'dispatch': list(rows), 'shape': list(q.shape),
                'stride': list(q.stride()), 'source_stride': list(z.stride()),
                'dtype': str(q.dtype), 'equal': torch.equal(z, q), 'distinct': q is not z}
    result['mode_nesting'] = 'TorchDispatchMode outside torch.inference_mode'
    result['gpu_initialized'] = torch.cuda.is_initialized()
    result['model_loads'] = 0
    result['gpu_dispatches'] = 0
    yaml = Path(torch.__file__).resolve().parents[1]/'torchgen/packaged/ATen/native/native_functions.yaml'
    result['schema_source'] = {'path': str(yaml), 'sha256': hashlib.sha256(yaml.read_bytes()).hexdigest()}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
