"""Exact-leaf field persistence and independent host acknowledgement.

This bridge authorizes no inference. A future worker needs a newly attested
brick leaf; the frozen turret leaf/attestation cannot be reused. CPU fixtures
use a distinct cpu-only-attempt.marker and explicit synthetic provenance.
"""
import hashlib
import os
from pathlib import Path
import sys
import time
from types import ModuleType
from . import artifact, profile as profiles
from ._source import ROOT, load_pinned

IO_DEADLINE = 30.0
MAX_RECEIPT = 128 * 1024
_FROZEN = 'docs/compatibility/amd-turret-texture-decode-resume-evidence/'
PINS = {
    _FROZEN+'artifact_io.py': '4a1d55f77eb3e45e115592e17e0fa4948bb58278fd5a7bf71af561c1ed9543f4',
    _FROZEN+'output_owner.py': '02d862b93ad08edb7830e49169fe6adc9921bebd2755cd94cbce2179e3bf9648',
    _FROZEN+'windows_owner.ps1': '6849a3ac5375cc1693ce306ea28fa4ce9bed32305b8926bc2f5e6034c75f1a94',
}
_private_sys = ModuleType('sys')
vars(_private_sys).update(vars(sys))
_private_sys.path = list(sys.path)
_frozen_receipt = load_pinned('receipt_original', _FROZEN+'artifact_io.py',
    PINS[_FROZEN+'artifact_io.py'], imports={'sys': _private_sys})


def _read(path, maximum=MAX_RECEIPT):
    # Receipt control files use the attested Windows control leaf. They are not
    # field payload filenames and must never borrow a field-file capability.
    field, path = artifact._canonical_field(), Path(path)
    before = field._plain(path, owned=False)
    parent = field._plain(path.parent, directory=True, owned=False)
    if (before.st_dev != parent.st_dev or (before.st_uid, before.st_gid) != (parent.st_uid, parent.st_gid)
            or before.st_size > maximum):
        raise ValueError('Control file ownership/filesystem/byte bound differs')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if field._identity(opened) != field._identity(before):
            raise ValueError('Control file changed during open')
        blob = stream.read(maximum+1)
        after = os.fstat(stream.fileno())
    if (len(blob) > maximum or len(blob) != before.st_size
            or field._identity(before) != field._identity(after)
            or field._identity(before) != field._identity(field._plain(path, owned=False))):
        raise ValueError('Control file changed during bounded read')
    return blob


def _decode(path, maximum=MAX_RECEIPT):
    return artifact._canonical_field()._decode(_read(path, maximum))


def _digest(path):
    return hashlib.sha256(_read(path)).hexdigest()


def _publish(path, value):
    path = Path(path)
    _directory(path.parent)
    if os.path.lexists(path) or os.path.lexists(path.with_name(path.name+'.publishing')):
        raise FileExistsError(str(path))
    artifact._canonical_field()._json_bytes(value, MAX_RECEIPT)
    # The qualified two-sided protocol's exclusive pending file + fsync/rename.
    _frozen_receipt.durable_json(path, value)
    _read(path)


def _directory(path):
    field, path = artifact._canonical_field(), Path(path)
    if not path.is_absolute() or path != path.resolve(strict=True):
        raise ValueError('Canonical existing control/output directory required')
    for parent in (path, *path.parents):
        field._plain(parent, directory=True, owned=False)
        if os.path.lexists(parent/'.git'):
            raise ValueError('Control/output leaves must remain outside Git')
    return path


def _windows(path):
    return path.as_posix() if sys.platform == 'win32' else 'C:/'+path.relative_to('/mnt/c').as_posix()


def _leaves(root, receipt_dir):
    root, receipt_dir = Path(root), Path(receipt_dir)
    _directory(root)
    _directory(receipt_dir)
    if (root.name != 'field' or receipt_dir.name != 'probe'
            or root.parent != receipt_dir.parent or root == receipt_dir):
        raise ValueError('Exact fresh sibling field/probe control leaves required')
    return root, receipt_dir


def _check_directory_resources(root):
    """The frozen writer's unchanged destination free-space/permission gate."""
    if sys.platform != 'win32':
        filesystem = os.statvfs(root)
        free = filesystem.f_bavail * filesystem.f_frsize
    else:
        import shutil
        free = shutil.disk_usage(root).free
    if free < 2 * 1024**3 or not os.access(root, os.W_OK):
        raise RuntimeError('Texture output free-space/permission admission refused')
    return free


def _owner(root, attestation_path=None, attestation_sha256=None):
    script = ROOT / (_FROZEN+'windows_owner.ps1')
    if hashlib.sha256(script.read_bytes()).hexdigest() != PINS[_FROZEN+'windows_owner.ps1']:
        raise ValueError('Original read-only Windows ownership script differs')
    source = _FROZEN+'output_owner.py'
    name = 'receipt_owner_' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    owner = load_pinned(name, source, PINS[source])
    # Exact private adaptation of the frozen two-leaf owner to this new run.
    # snapshot/host_verify/proof/ownership and the script remain unchanged.
    owner.ROOT = root.parent
    if attestation_path is not None:
        owner.PROOF = attestation_path
        owner.EXPECTED_SHA256 = attestation_sha256
    return owner


def mint_host_attestation(root, attestation_path, *, profile, mode='fixture'):
    """Persist actual Windows SID/ACL/file IDs for a fresh exact brick leaf.

    worker_identity is the unchanged expected future UID policy. CPU tests do
    not mint a WSL capability; the genuine factory must verify live UID/mount
    if separately approved future worker execution ever uses this record.
    """
    profile = profiles.require_authenticated(profile)
    attestation_path = Path(attestation_path)
    root, receipt_dir = _leaves(root, attestation_path.parent)
    if (sys.platform != 'win32' or attestation_path.name != 'ownership.json'
            or mode not in ('fixture', 'production') or any(root.iterdir())):
        raise ValueError('Fresh empty Windows field and exact ownership.json required')
    owner = _owner(root)
    snapshots = [(leaf, owner.snapshot(leaf)) for leaf in (root, receipt_dir)]
    windows = root.as_posix()
    if not windows.startswith('C:/Users/wattz/Documents/ModlyArtifacts/'):
        raise ValueError('Fresh artifact leaf must be under the original approved host prefix')
    sid = snapshots[0][1]['current_sid']
    if any(snapshot['current_sid'] != sid for _, snapshot in snapshots):
        raise ValueError('Fresh Windows field/control leaf owners differ')
    value = dict(format='modly-owned-drvfs-v1', worker_identity=[1000, 1000, 1000, 1000],
        windows_owner_sid=sid, windows_user_sid=sid, profile_sha256=profile.sha256,
        run_contract_sha256=profile.run_contract_sha256, mode=mode,
        roots=[dict(windows_path=leaf.as_posix(), guest_path='/mnt/c/'+leaf.as_posix()[3:],
                    host_owner_verified=True, host_snapshot=snapshot) for leaf, snapshot in snapshots])
    _publish(attestation_path, value)
    return dict(path=str(attestation_path), sha256=_digest(attestation_path))


class ReceiptBridge:
    """One profile/root/marker-bound callback and its independent host poller.

    The frozen owner calls its two permitted sibling leaves field and probe.
    Here probe carries only this attempt's bounded receipt/control files.
    """
    def __init__(self, root, receipt_dir, marker_path, marker_sha256, *, profile,
                 attestation_path, attestation_sha256, cancelled=None):
        self.profile = profiles.require_authenticated(profile)
        self.root, self.receipt_dir = _leaves(root, receipt_dir)
        _check_directory_resources(self.root)
        self.marker_path = Path(marker_path)
        self.marker_sha256 = marker_sha256
        self.attestation_path = Path(attestation_path)
        self.attestation_sha256 = attestation_sha256
        self.cancelled = (lambda: False) if cancelled is None else cancelled
        if not callable(self.cancelled):
            raise ValueError('Cancellation requires a callable')
        if self.attestation_path != self.receipt_dir/'ownership.json':
            raise ValueError('Foreign exact-leaf ownership record')
        self._context()
        self.owner = _owner(self.root, self.attestation_path, self.attestation_sha256)
        # Both actual roots pass original Windows ownership checks, or the
        # genuine future worker's live UID/mount/root capability checks.
        self.owner.ownership(self.root)
        self.owner.ownership(self.receipt_dir)

    def _context(self):
        profiles.require_authenticated(self.profile)
        if _digest(self.attestation_path) != self.attestation_sha256:
            raise ValueError('Trusted exact-leaf ownership record changed')
        proof = _decode(self.attestation_path)
        # On Windows this authenticates the fresh root; WSL validates the same
        # saved Windows path via the genuine ownership factory, never a fake uid.
        windows_root = _windows(self.root)
        expected_roots = [(leaf, _windows(leaf)) for leaf in (self.root, self.receipt_dir)]
        if (proof.get('profile_sha256') != self.profile.sha256
                or proof.get('run_contract_sha256') != self.profile.run_contract_sha256
                or proof.get('mode') not in ('fixture', 'production')
                or len(proof.get('roots', [])) != 2
                or any(row.get('windows_path') != windows or row.get('guest_path') != '/mnt/c/'+windows[3:]
                       for row, (_, windows) in zip(proof.get('roots', []), expected_roots))):
            raise ValueError('Foreign root/profile ownership record')
        self.mode = proof['mode']
        marker_name = 'cpu-only-attempt.marker' if self.mode == 'fixture' else 'neural-attempt.marker'
        if self.marker_path != self.receipt_dir/marker_name or _digest(self.marker_path) != self.marker_sha256:
            raise ValueError('Exact independent receipt marker identity differs')
        marker = _decode(self.marker_path, 16384)
        if (marker.get('schema') != 'modly.amd.brick-pbr-receipt-marker.v1'
                or marker.get('mode') != self.mode or marker.get('profile_sha256') != self.profile.sha256
                or marker.get('artifact_root') != windows_root):
            raise ValueError('Receipt marker profile/root/mode differs')

    def _deadline(self, began):
        if self.cancelled():
            raise InterruptedError('Field persistence/host acknowledgement cancelled')
        if time.monotonic()-began >= IO_DEADLINE:
            raise TimeoutError('30-second independent texture-field I/O deadline')

    def _identity(self, metadata):
        provenance = metadata.get('provenance', {})
        if (provenance.get('mode') != self.mode
                or provenance.get('ownership_attestation_sha256') != self.attestation_sha256
                or provenance.get('receipt_marker_sha256') != self.marker_sha256):
            raise ValueError('Artifact mode/ownership/marker provenance binding differs')
        return dict(kind='brick-pbr-field', artifact_path=_windows(self.root)+'/'+artifact.ARTIFACT_NAME,
            metadata_path=_windows(self.root)+'/'+artifact.METADATA_NAME,
            profile_sha256=self.profile.sha256, run_contract_sha256=self.profile.run_contract_sha256,
            marker_sha256=self.marker_sha256, attestation_sha256=self.attestation_sha256,
            artifact_sha256=metadata['artifact_sha256'], artifact_bytes=metadata['artifact_bytes'],
            metadata_sha256=hashlib.sha256(artifact._canonical_field()._json_bytes(metadata)).hexdigest())

    def validate_ack(self, ack, metadata):
        _frozen_receipt.validate_ack(ack, metadata)
        if (any(ack.get(k) != v for k, v in self._identity(metadata).items())
                or ack.get('independent_of_worker') is not True or ack.get('ownership_verified') is not True):
            raise RuntimeError('Independent host profile/path/marker/ownership acknowledgement differs')

    def write_output(self, raw, features, coords, provenance, output, *, profile):
        began = time.monotonic()
        self._deadline(began)
        if profiles.require_authenticated(profile) != self.profile or Path(output) != self.receipt_dir:
            raise ValueError('Foreign callback profile or receipt directory')
        self._context()
        if provenance.get('mode') != self.mode or (self.mode == 'fixture' and provenance.get('synthetic') is not True):
            raise ValueError('Field/attestation fixture or production mode differs')
        provenance = dict(provenance)
        for key, value in (('ownership_attestation_sha256', self.attestation_sha256),
                           ('receipt_marker_sha256', self.marker_sha256)):
            if key in provenance and provenance[key] != value:
                raise ValueError('Caller artifact ownership/marker provenance differs')
            provenance[key] = value
        if any(self.root.iterdir()):
            raise FileExistsError('Exact field leaf must be empty before publication')
        authority = self.owner.ownership(self.root)
        self.owner.ownership(self.receipt_dir)
        _check_directory_resources(self.root)
        result = artifact.write(self.root, raw, features, coords, provenance,
                                authority=authority, profile=self.profile)
        self._deadline(began)
        self._context()
        self.owner.ownership(self.receipt_dir)
        _publish(self.receipt_dir/'artifact-ready.json',
                 dict(self._identity(result['metadata']), verified=True, metadata=result['metadata']))
        ack = self.receipt_dir/'host-artifact-verified.json'
        while not ack.exists():
            self._deadline(began)
            time.sleep(.025)
        self.validate_ack(_decode(ack), result['metadata'])
        self._context()
        self._deadline(began)
        return dict(result, independently_verified_before_release=True)

    def host_receipt(self):
        return HostReceipt(self)


class HostReceipt:
    def __init__(self, bridge):
        if type(bridge) is not ReceiptBridge or sys.platform != 'win32':
            raise ValueError('Independent receipt requires exact Windows host bridge')
        self.bridge, self.done, self.result = bridge, False, None

    def poll(self):
        if self.done:
            return True
        bridge = self.bridge
        ready_path = bridge.receipt_dir/'artifact-ready.json'
        if not ready_path.exists():
            return False
        began = time.monotonic()
        bridge._deadline(began)
        bridge._context()
        ready = _decode(ready_path)
        metadata = ready.get('metadata')
        if (type(metadata) is not dict or ready.get('verified') is not True
                or any(ready.get(k) != v for k, v in bridge._identity(metadata).items())):
            raise ValueError('Ready receipt exact marker/path/profile/hash identity differs')
        before = bridge.owner.host_verify(bridge.root, files=True)
        control_before = bridge.owner.host_verify(bridge.receipt_dir)
        result = artifact.verify(bridge.root/artifact.ARTIFACT_NAME, bridge.root,
                                 metadata, profile=bridge.profile)
        after = bridge.owner.host_verify(bridge.root, files=True)
        control_after = bridge.owner.host_verify(bridge.receipt_dir)
        if before != after or control_before != control_after or result.get('verified') is not True:
            raise ValueError('Windows ownership/file identity changed during independent readback')
        bridge._context()
        bridge._deadline(began)
        ack = dict(bridge._identity(metadata), status='PASS', independent_of_worker=True,
                   ownership_verified=True, windows_ownership_before=before, windows_ownership_after=after)
        bridge.validate_ack(ack, metadata)
        _publish(bridge.receipt_dir/'host-artifact-verified.json', ack)
        bridge._deadline(began)
        self.done, self.result = True, ack
        return True
