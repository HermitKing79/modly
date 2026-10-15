"""Reusable image-to-native-occupancy proof; no shape mesh or full generation claim.

Prepare the CPU image contract first, commit its staged source and this adapter,
then run the existing three supervised workers against that exact harness commit.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_object_occupancy as original

ROOT = original.ROOT
BASE = ROOT / 'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image'
STATIC_FILES = ['scripts/compatibility/amd_generation_occupancy.py',
                'scripts/compatibility/test_amd_generation_occupancy.py']
ENV_IMAGE = 'MODLY_GENERATION_OCCUPANCY_IMAGE'
ENV_EVIDENCE = 'MODLY_GENERATION_OCCUPANCY_EVIDENCE'
ENV_SOURCE_SHA = 'MODLY_GENERATION_OCCUPANCY_IMAGE_SHA256'
ENV_CONTRACT_SHA = 'MODLY_GENERATION_OCCUPANCY_CONTRACT_SHA256'
_profile = original.prepare_profile
_preflight = original.preflight
_contract_sha = None


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def plain_path(value, *, existing=False):
    path = Path(value).absolute()
    if str(path).replace('\\', '/').startswith('//'):
        raise ValueError('Only explicit local paths are accepted')
    for ancestor in (path, *path.parents):
        if ancestor.exists() or ancestor.is_symlink():
            info = ancestor.lstat()
            if ancestor.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Reparse points and symlinks are prohibited')
    resolved = path.resolve(strict=existing)
    if path != resolved:
        raise ValueError('Path aliases are prohibited')
    return resolved


def evidence_path(value, *, existing=False):
    path = plain_path(value, existing=existing)
    try:path.relative_to(plain_path(BASE))
    except ValueError:raise ValueError('Evidence must stay inside the reserved new-image root') from None
    if existing and not path.is_dir():raise ValueError('Evidence must be a directory')
    return path


def read_json(path, limit=8 * 1024**2):
    with plain_path(path, existing=True).open('rb') as stream:blob = stream.read(limit + 1)
    if len(blob) > limit:raise ValueError('JSON exceeds its bounded input limit')
    return json.loads(blob, object_pairs_hook=original.features.unique)


def save_json(path, row):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def image_contract(source, blob):
    image, alpha = original.decode_alpha_image(blob)
    png = io.BytesIO(); image.save(png, format='PNG')
    crop = original.preprocess_cpu(image)
    return image, dict(source_path=source.relative_to(ROOT).as_posix(), source_sha256=digest(blob), source_bytes=len(blob),
        source_size=list(image.size), source_mode=image.mode, pixel_sha256=digest(image.tobytes()),
        input_png_sha256=digest(png.getvalue()), alpha=alpha, crop_size=list(crop.size),
        crop_mode=crop.mode, crop_pixel_sha256=digest(crop.tobytes()),
        preprocessing='original alpha-only preprocess_image; RMBG absent',
        weights_loaded=False, gpu_initialized=False, scope='native-sparse-occupancy-only', full_generation=False)


def source_bytes(path):
    with plain_path(path, existing=True).open('rb') as stream:blob = stream.read(2 * 1024**2 + 1)
    if not blob or len(blob) > 2 * 1024**2:raise ValueError('Source exceeds the inherited 2 MiB image bound')
    return blob


def prepare(image, evidence):
    """Create a fresh CPU-only contract and an exact source copy; load no model."""
    image = plain_path(image, existing=True); evidence = evidence_path(evidence)
    if evidence.exists():raise FileExistsError('Prepare requires an absent evidence leaf')
    blob = source_bytes(image)
    source = evidence / ('source-image' + image.suffix.lower())
    _, row = image_contract(source, blob)
    evidence.mkdir(parents=True, exist_ok=False)
    with source.open('xb') as stream:stream.write(blob); stream.flush(); os.fsync(stream.fileno())
    save_json(evidence / 'input-contract.json', row)
    return row


def selected_image():
    contract = original.CONTRACT
    if _contract_sha is None or digest(contract.read_bytes()) != _contract_sha:
        raise ValueError('Prepared input contract changed')
    row = read_json(contract, 65536)
    image, actual = image_contract(original.SOURCE, source_bytes(original.SOURCE))
    if (row != actual
            or row['source_sha256'] != original.SOURCE_SHA):
        raise ValueError('Prepared image pixels, dimensions, PNG or preprocessing identity changed')
    return image, row


def configure(image, evidence, *, expected_image_sha=None, expected_contract_sha=None):
    """Bind only image/artifact routing; workers receive the same explicit identities."""
    global _contract_sha
    evidence = evidence_path(evidence, existing=True)
    contract = evidence / 'input-contract.json'; row = read_json(contract, 65536)
    source = plain_path(ROOT / row['source_path'], existing=True)
    if source.parent != evidence or not source.name.startswith('source-image'):
        raise ValueError('Prepared source must be owned by this exact evidence leaf')
    supplied = plain_path(image, existing=True)
    source_sha = digest(source_bytes(supplied)); contract_sha = digest(contract.read_bytes())
    if (source_sha != row['source_sha256'] or expected_image_sha is not None and source_sha != expected_image_sha
            or expected_contract_sha is not None and contract_sha != expected_contract_sha):
        raise ValueError('Image or contract differs from the explicit transported identity')
    original.EVIDENCE, original.CONTRACT = evidence, contract
    original.SOURCE, original.SOURCE_SHA = source, source_sha
    _contract_sha = contract_sha
    original.selected_image = selected_image
    original.prepare_profile, original.preflight = prepare_profile, preflight
    original.occupancy.data.write_preview = omit_preview
    selected_image()
    os.environ.update({ENV_IMAGE: str(source), ENV_EVIDENCE: str(evidence),
        ENV_SOURCE_SHA: source_sha, ENV_CONTRACT_SHA: contract_sha})
    return row


def omit_preview(directory, mask):
    """The repeatable handoff consumes native occupancy and requests no voxel mesh."""
    return {'omitted': True,
        'reason': 'repeatable pipeline consumes native occupancy; no diagnostic voxel geometry requested'}


def configure_worker():
    try:values = [os.environ[name] for name in (ENV_IMAGE, ENV_EVIDENCE, ENV_SOURCE_SHA, ENV_CONTRACT_SHA)]
    except KeyError:raise ValueError('Worker requires the complete explicit image/evidence identity') from None
    return configure(values[0], values[1], expected_image_sha=values[2], expected_contract_sha=values[3])


def prepare_profile(stage):
    profile = _profile(stage)
    profile.worker_script = Path(__file__).resolve()
    profile.committed_files = list(dict.fromkeys([*profile.committed_files, *STATIC_FILES]))
    return profile


def required_files(evidence):
    """New static files; consumers retain their inherited computational closure."""
    evidence = evidence_path(evidence, existing=True)
    row = read_json(evidence / 'input-contract.json', 65536)
    source = plain_path(ROOT / row['source_path'], existing=True)
    if source.parent != evidence:raise ValueError('Foreign prepared source path')
    return [*STATIC_FILES, (evidence / 'input-contract.json').relative_to(ROOT).as_posix(),
        source.relative_to(ROOT).as_posix()]


def preflight(commit):
    # Both checks precede the original exclusive sequence-attempt marker.
    original.occupancy.base.verify_committed_files(commit, STATIC_FILES)
    return _preflight(commit)


def verify_sequence(evidence):
    """Reopen the complete saved lineage using original readers, without AI imports."""
    import numpy as np
    evidence = evidence_path(evidence, existing=True)
    contract = read_json(evidence / 'input-contract.json', 65536)
    configure(ROOT / contract['source_path'], evidence)
    attempt = read_json(evidence / 'sequence-attempt-started.json', 65536)
    commit = attempt['harness_commit']
    if type(commit) is not str or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Sequence requires an exact harness commit identity')
    previous = os.environ.get('MODLY_OBJECT_HARNESS_COMMIT')
    os.environ['MODLY_OBJECT_HARNESS_COMMIT'] = commit
    try:
        for stage in original.STAGES:original.successful_stage(stage)
        feature = original.feature_reference(); latent = original.latent_reference()
        decoder = original.successful_stage('decoder'); occupancy = decoder['occupancy_artifacts']
    finally:
        if previous is None:os.environ.pop('MODLY_OBJECT_HARNESS_COMMIT', None)
        else:os.environ['MODLY_OBJECT_HARNESS_COMMIT'] = previous
    for stage in original.STAGES:
        result = read_json(evidence / stage / 'run/results.json')
        if result['process']['job_cleanup'].get('forced_job_cleanup') is not False:
            raise ValueError('Sequence had forced or unverified Job cleanup')
    scratch = evidence / 'decoder/run/scratch'
    if (occupancy.get('readback_verified') is not True or occupancy.get('threshold') != 'decoded > 0'
            or occupancy['provenance']['input'] != latent
            or occupancy['provenance']['harness_commit'] != commit
            or occupancy['provenance']['decoder_checkpoint_sha256'] != decoder['checkpoint_identity']['checkpoint_sha256']
            or read_json(scratch / 'occupancy-provenance.json') != occupancy):
        raise ValueError('Decoder/native-occupancy lineage differs')
    layouts = {'occupancy-logits.npy': ((1, 1, 64, 64, 64), np.dtype('<f4'), 2 * 1024**2),
        'occupancy-mask.npy': ((64, 64, 64), np.dtype('bool'), 512 * 1024),
        'occupancy-coordinates.npy': (None, np.dtype('<i4'), 5 * 1024**2)}
    if set(occupancy['files']) != set(layouts):raise ValueError('Native occupancy tensor closure differs')
    arrays = {}
    for name, (shape, dtype, limit) in layouts.items():
        path = plain_path(scratch / name, existing=True); row = occupancy['files'][name]
        with path.open('rb') as stream:blob = stream.read(limit + 1)
        if (Path(row['path']) != path or len(blob) > limit or len(blob) != row['file_bytes']
                or digest(blob) != row['file_sha256']):raise ValueError('Native occupancy file identity changed')
        array = np.load(io.BytesIO(blob), allow_pickle=False, max_header_size=4096)
        if (array.dtype != dtype or shape is not None and array.shape != shape
                or not np.isfinite(array).all() or list(array.shape) != row['shape']
                or digest(array.tobytes()) != row['payload_sha256']):
            raise ValueError('Native occupancy array identity changed')
        arrays[name] = array
    mask = arrays['occupancy-mask.npy']; coordinates = arrays['occupancy-coordinates.npy']
    xyz = np.argwhere(mask).astype('<i4')
    expected_coordinates = np.concatenate((np.zeros((len(xyz), 1), dtype='<i4'), xyz), axis=1)
    if (not np.array_equal(mask, arrays['occupancy-logits.npy'][0, 0] > 0)
            or not np.array_equal(coordinates, expected_coordinates)):
        raise ValueError('Native zero-threshold occupancy or coordinate ordering changed')
    return dict(feature=feature, latent=latent, occupancy=occupancy, input_contract=contract,
        input_contract_sha256=digest((evidence / 'input-contract.json').read_bytes()),
        evidence=str(evidence), harness_commit=commit, scope='native-sparse-occupancy-only', full_generation=False)


def preserve_root_artifacts(verified):
    evidence = evidence_path(verified['evidence'], existing=True)
    source_rows = {'conditioning.safetensors': verified['feature'],
        **{name: verified['occupancy']['files'][name] for name in ('occupancy-logits.npy', 'occupancy-mask.npy')}}
    # Check every destination before the first exclusive copy.
    destinations = [evidence / name for name in (*source_rows, 'occupancy-root-artifacts.json')]
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError('Canonical occupancy copies must be fresh')
    files = {}
    for name, row in source_rows.items():
        source = plain_path(row['path'], existing=True)
        source.relative_to(evidence / ('dino/run/scratch' if name == 'conditioning.safetensors' else 'decoder/run/scratch'))
        with source.open('rb') as stream:blob = stream.read(8 * 1024**2 + 1)
        if len(blob) != row['file_bytes'] or digest(blob) != row['file_sha256']:
            raise ValueError('Verified source changed before canonical copy')
        target = evidence / name
        with target.open('xb') as stream:stream.write(blob); stream.flush(); os.fsync(stream.fileno())
        if target.read_bytes() != blob:raise ValueError('Canonical occupancy copy readback differs')
        files[name] = dict(path=str(target), source_path=str(source), file_bytes=len(blob), file_sha256=digest(blob))
    receipt = dict(files=files, harness_commit=verified['harness_commit'],
        input_contract_sha256=verified['input_contract_sha256'], source_sha256=verified['input_contract']['source_sha256'],
        scope='native-sparse-occupancy-only', full_generation=False)
    save_json(evidence / 'occupancy-root-artifacts.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path); parser.add_argument('--evidence', type=Path)
    parser.add_argument('--harness-commit'); parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.prepare_only:
        if args.worker or args.output or not args.image or not args.evidence:
            parser.error('--prepare-only requires --image/--evidence and no worker/output')
        row = prepare(args.image, args.evidence)
        print(json.dumps(dict(scope=row['scope'], full_generation=False, input_contract=str(args.evidence / 'input-contract.json'),
            source_sha256=row['source_sha256'], weights_loaded=False, gpu_initialized=False)), flush=True)
        return 0
    if args.worker:
        if args.image or args.evidence or not args.output:parser.error('Worker routing comes only from the explicit environment')
        configure_worker()
    else:
        if not args.image or not args.evidence or not args.harness_commit or args.output:
            parser.error('Execution requires --image, --evidence and --harness-commit')
        configure(args.image, args.evidence)
    forwarded = [str(Path(__file__).resolve())]
    if args.worker:forwarded += ['--worker', '--output', str(args.output)]
    if args.harness_commit:forwarded += ['--harness-commit', args.harness_commit]
    old_argv = sys.argv
    try:
        sys.argv = forwarded
        result = original.main()
    finally:sys.argv = old_argv
    if not args.worker and result == 0:
        preserve_root_artifacts(verify_sequence(original.EVIDENCE))
        print(json.dumps(dict(scope='native-sparse-occupancy-only', full_generation=False,
            canonical_artifacts=str(original.EVIDENCE / 'occupancy-root-artifacts.json'))), flush=True)
    return result


if __name__ == '__main__':raise SystemExit(main())
