"""Authenticated run inputs and bounded shape support; no AI imports at import time."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType

SCHEMA = 'modly.amd.neural-run.v1'
INPUT_NAMES = ('shape-latent-denormalized.npy', 'coords.npy')
# Measured fresh-image support; CPU-qualified under the unchanged live GPU caps.
MAX_INITIAL_ROWS = 4430
MAX_INTERMEDIATE_ROWS = 1_000_000
MAX_FINAL_ROWS = 1_842_968
NATIVE_STAGE = '/home/modly/native-decoder-preparation-6dd94a85'
MAX_CONTRACT_BYTES = 65536


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _path(value):
    if (type(value) is not str or not value.startswith('/') or '\\' in value
            or any(c in value for c in ('\0', '\r', '\n'))):
        raise ValueError('Canonical absolute guest path required')
    path = PurePosixPath(value)
    if path.as_posix() != value or '..' in path.parts or value.startswith('//'):
        raise ValueError('Canonical absolute guest path required')
    return path


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Exact SHA256 required')
    return value


@dataclass(frozen=True)
class RunContract:
    initial_count: int
    input_paths: object
    input_sha256: object
    source_job_sha256: str
    output_directory: PurePosixPath
    scratch_directory: PurePosixPath
    repo_root: PurePosixPath
    maximum_final_rows: int
    maximum_intermediate_rows: int
    native_stage: PurePosixPath
    host_evidence_directory: PurePosixPath | None = None

    def document(self):
        value = {'schema': SCHEMA, 'initial_count': self.initial_count,
                'input_paths': {k: str(v) for k,v in self.input_paths.items()},
                'input_sha256': dict(self.input_sha256), 'source_job_sha256': self.source_job_sha256,
                'output_directory': str(self.output_directory), 'scratch_directory': str(self.scratch_directory),
                'repo_root': str(self.repo_root), 'maximum_final_rows': self.maximum_final_rows,
                'maximum_intermediate_rows': self.maximum_intermediate_rows, 'native_stage': str(self.native_stage)}
        if self.host_evidence_directory is not None:
            value['host_evidence_directory']=str(self.host_evidence_directory)
        return value

    @property
    def sha256(self):
        blob = json.dumps(self.document(), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(blob).hexdigest()


def parse(document):
    fields = {'schema','initial_count','input_paths','input_sha256','source_job_sha256',
              'output_directory','scratch_directory','repo_root','maximum_final_rows',
              'maximum_intermediate_rows','native_stage'}
    if (type(document) is not dict or set(document)-{'host_evidence_directory'} != fields
            or document.get('schema') != SCHEMA):
        raise ValueError('Exact neural run contract schema required')
    for key, low, high in [('initial_count',1,MAX_INITIAL_ROWS),
                            ('maximum_final_rows',MAX_FINAL_ROWS,MAX_FINAL_ROWS),
                            ('maximum_intermediate_rows',MAX_INTERMEDIATE_ROWS,MAX_INTERMEDIATE_ROWS)]:
        if type(document[key]) is not int or not low <= document[key] <= high:
            raise ValueError('Run contract exceeds qualified bounds: '+key)
    if (type(document['input_paths']) is not dict or set(document['input_paths']) != set(INPUT_NAMES)
            or type(document['input_sha256']) is not dict or set(document['input_sha256']) != set(INPUT_NAMES)):
        raise ValueError('Exactly two shape input paths and hashes required')
    paths = {k:_path(v) for k,v in document['input_paths'].items()}
    hashes = {k:_sha(v) for k,v in document['input_sha256'].items()}
    if len(set(paths.values())) != 2:
        raise ValueError('Distinct input files required')
    output, scratch = (_path(document[k]) for k in ('output_directory','scratch_directory'))
    parent = output.parent
    if (parent != scratch.parent or output == scratch or parent.parent != PurePosixPath('/home/modly')
            or not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}', parent.name)
            or parent.name == PurePosixPath(NATIVE_STAGE).name):
        raise ValueError('Distinct sibling output/scratch under one owned /home/modly/run-ID required')
    for path in paths.values():
        if path.is_relative_to(output) or path.is_relative_to(scratch):
            raise ValueError('Inputs cannot be inside newly claimed output or scratch')
    stage = _path(document['native_stage'])
    if str(stage) != NATIVE_STAGE:
        raise ValueError('Only the qualified native staging directory is supported')
    repo=_path(document['repo_root']);host=None
    if document.get('host_evidence_directory') is not None:
        host=_path(document['host_evidence_directory'])
        prefix=repo/'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image'
        if (not host.is_relative_to(prefix) or len(host.relative_to(prefix).parts)<3
                or host.parts[-2:]!=('run','guest')):
            raise ValueError('Host journal must be an exact new-image stage run/guest directory')
    return RunContract(document['initial_count'], MappingProxyType(paths), MappingProxyType(hashes),
                       _sha(document['source_job_sha256']), output, scratch, repo,
                       document['maximum_final_rows'], document['maximum_intermediate_rows'], stage, host)


def load(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_CONTRACT_BYTES:
        raise ValueError('Bounded regular run-contract file required')
    def pairs(rows):
        value = {}
        for key,item in rows:
            if key in value: raise ValueError('Duplicate contract JSON key')
            value[key] = item
        return value
    return parse(json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs))


def read_npy(path, expected_sha256, shape, dtype):
    import numpy as np
    if len(shape)!=2 or type(shape[0]) is not int or not 0<shape[0]<=MAX_INITIAL_ROWS:
        raise ValueError('NPY initial support exceeds qualified input bounds')
    path = Path(path); before = path.stat()
    if (path.is_symlink() or not path.is_file() or before.st_size > MAX_INITIAL_ROWS*32*4+4096
            or digest(path) != expected_sha256):
        raise ValueError('Run input hash, file type or byte bound differs')
    with path.open('rb') as stream:
        version=np.lib.format.read_magic(stream)
        if version not in ((1,0),(2,0)):
            raise ValueError('Unsupported bounded NPY format')
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        header_shape,fortran,header_dtype=reader(stream,max_header_size=4096)
        if header_shape!=shape or fortran or header_dtype!=np.dtype(dtype):
            raise ValueError('Run input header shape, dtype or ordering differs')
        stream.seek(0)
        value = np.load(stream, allow_pickle=False, max_header_size=4096)
        if stream.read(1): raise ValueError('Trailing NPY bytes')
    after = path.stat()
    if ((before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns)
            or value.shape != shape or value.dtype != np.dtype(dtype) or not value.flags.c_contiguous
            or not np.isfinite(value).all()):
        raise ValueError('Run input shape, dtype, contiguity, finite values or preservation differs')
    return value


def load_inputs(contract):
    import amd_shape_decoder_preflight as original
    features = read_npy(contract.input_paths[INPUT_NAMES[0]], contract.input_sha256[INPUT_NAMES[0]],
                        (contract.initial_count,32), 'float32')
    coords = read_npy(contract.input_paths[INPUT_NAMES[1]], contract.input_sha256[INPUT_NAMES[1]],
                      (contract.initial_count,4), 'int32')
    summary = original.validate_arrays(features, coords, count=contract.initial_count)
    summary.update(run_contract_sha256=contract.sha256, source_job_sha256=contract.source_job_sha256,
                   input_paths={k:str(v) for k,v in contract.input_paths.items()},
                   input_sha256=dict(contract.input_sha256))
    return features, coords, summary


def reconstruct_children(parent_coords, logits):
    """Original C2S parent-major nonzero order; child bits x=0,y=1,z=2."""
    import numpy as np
    if (parent_coords.dtype != np.int32 or parent_coords.ndim != 2 or parent_coords.shape[1] != 4
            or logits.dtype != np.float32 or logits.shape != (len(parent_coords),8)
            or not np.isfinite(logits).all()):
        raise ValueError('Original int32 coordinates and exact F32-expanded logits required')
    parents, children = np.nonzero(logits > 0)
    if not 0 < len(parents) <= MAX_FINAL_ROWS:
        raise ValueError('Selected child support outside qualified bound')
    result = parent_coords[parents].copy(); result[:,1:] *= 2
    for axis in range(3): result[:,axis+1] += ((children // (2**axis)) % 2).astype(np.int32)
    return result


def validate_chain(stages, final_coords, initial_count):
    import numpy as np
    if len(stages) != 4 or len(stages[0][0]) != initial_count:
        raise ValueError('Exactly four subdivisions linked to the initial run support required')
    for i,(coords,logits) in enumerate(stages):
        if (not 0 < len(coords) <= MAX_INTERMEDIATE_ROWS or coords.dtype != np.int32
                or coords.shape != (len(coords),4) or (coords[:,0] != 0).any()
                or (coords[:,1:] < 0).any() or (coords[:,1:] >= 32*2**i).any()
                or len(np.unique(coords,axis=0)) != len(coords)):
            raise ValueError('Invalid genuine subdivision coordinates')
        child = reconstruct_children(coords, logits)
        expected = stages[i+1][0] if i < 3 else final_coords
        if not np.array_equal(child, expected):
            raise ValueError('Original C2S child support/order differs from next stage')
    return True
