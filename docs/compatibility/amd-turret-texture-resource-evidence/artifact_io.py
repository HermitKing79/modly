"""Actual six-channel CPU writer and bounded independent Windows receipt."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
sys.path.insert(0, str(ROOT / 'scripts/compatibility'))
WINDOWS_ROOT = 'C:/Users/wattz/Documents/ModlyArtifacts/amd-turret-texture-owner-067f668'
ARTIFACT_ROOT = Path(WINDOWS_ROOT if sys.platform == 'win32' else '/mnt/c/' + WINDOWS_ROOT[3:])
FIELD = ARTIFACT_ROOT / 'field'
IO_DEADLINE = 30.0
MAX_RECEIPT = 128 * 1024


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def durable_json(path, record):
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(str(path))
    pending = path.with_name(path.name + '.publishing')
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(pending, path)


def marker():
    return E / 'neural-attempt.marker'


def validate_ack(ack, metadata):
    if (ack.get('status') != 'PASS' or ack.get('artifact_sha256') != metadata['artifact_sha256']
            or ack.get('artifact_bytes') != metadata['artifact_bytes']):
        raise RuntimeError('Independent host texture-field verification refused')


def check_directory(root=None):
    """Existing host-created exact leaf only; never manufacture inference roots."""
    from amd_texture_field_artifact import approved_root
    from output_owner import ownership
    root = FIELD if root is None else root
    if root not in (FIELD, ARTIFACT_ROOT/'probe'):
        raise ValueError('Foreign output leaf')
    if root.parent != ARTIFACT_ROOT or root != root.resolve(strict=True):
        raise RuntimeError('Texture output mapping differs')
    approved_root(root, authority=ownership(root))
    if any(root.iterdir()):
        raise RuntimeError('Exclusive texture output leaf must be empty before inference')
    if sys.platform != 'win32':
        fs = os.statvfs(root)
        free = fs.f_bavail * fs.f_frsize
    else:
        import shutil
        free = shutil.disk_usage(root).free
    if free < 2 * 1024**3 or not os.access(root, os.W_OK):
        raise RuntimeError('Texture output free-space/permission admission refused')
    return {'root': str(ARTIFACT_ROOT), 'field': str(root), 'free_bytes': free,
            'host_created_existing_leaf': True, 'empty': True}


def write_output(raw_np, features_np, coords_np, provenance, receipt_dir):
    from amd_texture_field_artifact import write
    if provenance.get('mode') != 'production':
        raise RuntimeError('Genuine texture writer requires production provenance')
    began = time.monotonic()
    check_directory()
    from output_owner import ownership, EXPECTED_SHA256
    provenance = dict(provenance, ownership_attestation_sha256=EXPECTED_SHA256)
    result = write(FIELD, raw_np, features_np, coords_np, provenance, authority=ownership(FIELD))
    ready = dict(kind='texture-field', **result, marker_sha256=digest(marker()))
    durable_json(Path(receipt_dir) / 'artifact-ready.json', ready)
    ack = Path(receipt_dir) / 'host-artifact-verified.json'
    while not ack.exists():
        if time.monotonic() - began >= IO_DEADLINE:
            raise TimeoutError('30-second independent texture-field I/O deadline')
        time.sleep(.025)
    if ack.stat().st_size > MAX_RECEIPT:
        raise RuntimeError('Unbounded host texture receipt')
    validate_ack(json.loads(ack.read_text()), result['metadata'])
    if time.monotonic() - began >= IO_DEADLINE:
        raise TimeoutError('30-second texture-field I/O deadline')
    return dict(result, independently_verified_before_release=True)


def verify_on_host(receipt_dir):
    if sys.platform != 'win32':
        raise RuntimeError('Independent texture receipt requires Windows host')
    from amd_texture_field_artifact import verify
    ready_path = Path(receipt_dir) / 'artifact-ready.json'
    if ready_path.stat().st_size > MAX_RECEIPT:
        raise RuntimeError('Unbounded texture artifact receipt')
    ready = json.loads(ready_path.read_text())
    if (ready.get('kind') != 'texture-field' or ready.get('verified') is not True
            or ready.get('marker_sha256') != digest(marker())):
        raise RuntimeError('Texture attempt receipt identity differs')
    expected_guest = '/mnt/c/' + WINDOWS_ROOT[3:] + '/field/'
    if (ready.get('artifact_path') != expected_guest + 'texture-field.safetensors'
            or ready.get('metadata_path') != expected_guest + 'texture-field-metadata.json'):
        raise RuntimeError('Texture receipt output path differs')
    metadata = ready['metadata']
    if (metadata.get('provenance', {}).get('mode') != 'production'
            or metadata.get('channels') != ['red', 'green', 'blue', 'metallic', 'roughness', 'alpha']
            or {name: row.get('shape') for name, row in metadata.get('tensors', {}).items()}
               != {'raw': [1842968, 6], 'features': [1842968, 6], 'coordinates': [1842968, 4]}):
        raise RuntimeError('Texture receipt full support/channel/provenance contract differs')
    from output_owner import host_verify
    ownership_before = host_verify(FIELD, files=True)
    result = verify(FIELD / 'texture-field.safetensors', FIELD, metadata)
    ownership_after = host_verify(FIELD, files=True)
    if result.get('verified') is not True:
        raise RuntimeError('Independent texture reader did not verify')
    return dict(status='PASS', artifact_sha256=metadata['artifact_sha256'],
                artifact_bytes=metadata['artifact_bytes'], Windows_path=result['artifact_path'],
                metadata_path=result['metadata_path'], metadata_verified=True,
                tensor_payload_sha256={name: row['sha256'] for name, row in metadata['tensors'].items()},
                independent_of_worker=True, marker_sha256=ready['marker_sha256'],
                Windows_ownership_before=ownership_before, Windows_ownership_after=ownership_after)


class HostReceipt:
    def __init__(self, output, base):
        self.directory = Path(output) / 'guest'
        self.base = base
        self.future = self.started = None
        self.done = False

    def poll(self):
        if self.done:
            return
        if self.future is None and (self.directory / 'artifact-ready.json').exists():
            self.started = time.monotonic()
            self.future = self.base.in_thread(lambda: verify_on_host(self.directory))
        if self.future is not None:
            if time.monotonic() - self.started >= IO_DEADLINE:
                raise TimeoutError('30-second host texture verification deadline')
            if self.future.done():
                durable_json(self.directory / 'host-artifact-verified.json', self.future.result())
                self.done = True
