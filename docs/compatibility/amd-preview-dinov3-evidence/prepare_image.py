"""CPU-only snapshot of the observed preview input; never launches a stage."""
import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'api'))


class NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'transformers', 'trellis2',
                                     'flex_gemm', 'o_voxel', 'cumesh', 'nvdiffrast'}:
            raise AssertionError('CPU input snapshot forbids neural/native model imports')


sys.meta_path.insert(0, NoNeuralImports())
from services.amd_completed_result import CompletionContext, _opened, _plain
from services.amd_image_test import _local_test


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main(source, output, observed_path):
    source = _plain(source)
    output = _plain(output)
    observed = json.loads(observed_path.read_bytes())
    assert not output.exists(), 'Exclusive new CPU snapshot directory required'
    assert source.name == observed['filename'], 'Observed preview filename differs'
    with _opened(source, 2 * 1024**2, 0) as (stream, info):
        blob = stream.read(2 * 1024**2 + 1)
        original = dict(path=str(source), bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                        device=info.st_dev, file_id=info.st_ino, links=info.st_nlink)
    result = _local_test(blob)
    assert result['status'] == 'valid' and result['eligible'] is True
    assert result['validation'] == 'original_cpu_input_contract'
    metadata = result['metadata']
    assert len(blob) == observed['bytes']
    for key in ('width', 'height', 'mode', 'format', 'foreground_extent', 'resized_foreground_extent'):
        assert metadata[key] == observed[key], key
    assert metadata['transparent_pixels_present'] is True
    contract = result['input_contract']
    assert contract['input_sha256'] == sha(blob)
    params = dict(image_mode='alpha-only', resolution=512,
                  stage='conditioning-only', include_neg_cond=False)
    context = CompletionContext.new('preview-dino-cpu-admission', blob, params, model_id='trellis2-amd')
    context.validate_request(blob, params)
    assert context.input_sha256 == sha(blob)
    output.mkdir(exist_ok=False)
    with (output / 'source-image.png').open('xb') as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    save(output / 'input-contract.json', dict(source_sha256=sha(blob),
         original_cpu_input_contract=contract, qualification='CPU_ADMISSION_ONLY',
         original_path=original, preview_observation_sha256=sha(observed_path.read_bytes())))
    save(output / 'cpu-result.json', dict(status='PASS_CPU_IMAGE_REVALIDATION',
         image_sha256=sha(blob), metadata=metadata, contract=contract,
         product_context=context.to_dict(), canonical_parameters=params,
         production_ticket_created=False, production_attempt_marker_created=False,
         neural_imports=False, gpu_execution=False, conditioning_output_created=False,
         preview_binding_scope='Observed filename/location/size/dimensions/alpha agree. Fresh exact-byte CPU revalidation; historical in-memory preview response was not exported as a durable authority receipt.'))
    assert (output / 'source-image.png').read_bytes() == blob
    with _opened(source, 2 * 1024**2, 0) as (stream, current):
        assert stream.read(2 * 1024**2 + 1) == blob
        assert (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) == (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    print(json.dumps(dict(status='PASS_CPU_IMAGE_REVALIDATION', image_sha256=sha(blob),
                         source_bytes=len(blob), dimensions=[metadata['width'], metadata['height']],
                         neural_execution=False, production_attempt_reserved=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observed', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output, args.observed)
