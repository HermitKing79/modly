"""One explicitly zombie-bound use of the unchanged standalone WSL decoder.

This adapter authenticates completed, separate upstream roots. It does not open
the product's CPU-only current-image dispatch or change any decoder operation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_generation_decode as historical
import amd_zombie_shape as shape

ROOT = historical.ROOT
PRIVATE = ROOT.parent / 'artifacts/amd-zombie-decode-0468f86'
PUBLIC = ROOT / 'docs/compatibility/amd-zombie-decode-evidence'
EVIDENCE = historical.APPROVED_ROOT / 'zombie-0468f86/decode'
RUN_ID = 'run-zombie-geometry-0468f86-v1'
SHAPE_COMMIT = '35a5b825e3e9128326897a2ce35c87b237fdfa8b'
SOURCE_JOB_SHA = '913ad907c55f1f510cce00d1d7573d518883190628d040506a6bd78578dd54bf'
INPUTS = {
    'shape-latent-denormalized.npy': ('shape-latent-denormalized.npy', '18778e295e7d9405ca2e80b8fde64dacc790696c361f6418436d2e498149b2b3'),
    'coords.npy': ('shape-latent-coordinates.npy', 'c432b04ec70c6200fadf8f5f62ce995514e24bf4194b154ac30a76148a7bb4a2'),
}


def valid_commit(commit):
    if (type(commit) is not str or not re.fullmatch('[0-9a-f]{40}', commit)
            or commit in (SHAPE_COMMIT, shape.OCCUPANCY_COMMIT, shape.prior.DINO_COMMIT)):
        raise ValueError('A distinct exact zombie decoder execution commit is required')


def source_inputs(evidence):
    from amd_generation_occupancy import plain_path
    from amd_shape_decoder_preflight import validate_arrays
    # Admit the lexical and actual root before opening any private source data.
    if Path(evidence) != shape.EVIDENCE or plain_path(evidence, existing=True) != shape.EVIDENCE:
        raise ValueError('Only the exact original zombie Shape-SLat evidence root is admitted')
    receipt = shape.verify_saved(SHAPE_COMMIT)
    review_path = shape.PUBLIC / 'outcome-review.json'
    review = historical.read_json(review_path)
    if (receipt['classification'] != 'PASS' or receipt['harness_commit'] != SHAPE_COMMIT
            or receipt['results_sha256'] != SOURCE_JOB_SHA
            or review['classification'] != 'PASS' or review['blockers']
            or review['execution_harness_commit'] != SHAPE_COMMIT
            or review['adapter_readback_results_sha256'] != SOURCE_JOB_SHA):
        raise ValueError('Original independently reviewed zombie sequence is not complete')
    expected_records = {'results.json', 'committed-harness.json', 'checkpoint-before.json',
        'checkpoint-after.json', 'cpu-gate.json', 'worker.stdout.txt', 'worker.stderr.txt',
        'trace.jsonl', 'windows-periodic.jsonl', 'windows-before.json',
        'windows-cpu-gate.json', 'windows-after-exit.json'}
    if set(review['private_run_evidence_sha256']) != expected_records:
        raise ValueError('Original reviewed supervision evidence closure differs')
    for name, expected in review['private_run_evidence_sha256'].items():
        if historical.digest(plain_path(evidence / 'run' / name, existing=True)) != expected:
            raise ValueError('Original reviewed supervision evidence changed: ' + name)
    marker_path = evidence / 'attempt-started.json'
    marker = historical.read_json(marker_path)
    if (marker['harness_commit'] != SHAPE_COMMIT or marker['maximum_forwards'] != 21
            or marker['maximum_steps'] != 12 or marker['maximum_sequences'] != 1
            or historical.digest(marker_path) != review['exclusive_attempt_marker_sha256']):
        raise ValueError('Original one-sequence marker changed')
    for name in ('host-readback.json', 'independent-readback.json'):
        if historical.read_json(shape.PRIVATE / name) != receipt:
            raise ValueError('Original independent post-exit readback differs')
    reader = historical.load_module('_amd_zombie_original_npy_reader', historical.CODE / 'run_contract.py')
    arrays, references = {}, {}
    for name, (leaf, sha) in INPUTS.items():
        path = plain_path(evidence / 'run/scratch' / leaf, existing=True)
        arrays[name] = reader.read_npy(path, sha, (1000, 4 if name == 'coords.npy' else 32),
                                      'int32' if name == 'coords.npy' else 'float32')
        references[name] = dict(path=path, sha256=sha)
    summary = validate_arrays(arrays['shape-latent-denormalized.npy'], arrays['coords.npy'], count=1000)
    if (summary['coordinates_payload_sha256'] != receipt['readback']['coords_payload_sha256']
            or shape.prior.zombie.SOURCE_SHA != 'db1fbb6bd27717f4b6a0298d64ff1ddff17e80027d782bf5dbfbc959d70fa06e'):
        raise ValueError('Exact source image or original coordinate ordering differs')
    required = sorted(set(shape.CLOSURE + [review_path.relative_to(ROOT).as_posix()]))
    return dict(source_job=evidence / 'run/results.json', source_job_sha256=SOURCE_JOB_SHA,
                count=1000, inputs=references, coords=arrays['coords.npy'], summary=summary,
                required_files=required)


def make_runtime(commit):
    valid_commit(commit)
    # Scope the source-root adaptation to a private copy of the proven host.
    # Shared historical modules, original path policy and decoder stay unchanged.
    host = historical.load_module('_amd_zombie_standalone_host', Path(historical.__file__))
    host.shape_inputs = source_inputs
    host.STATIC_FILES = list(host.STATIC_FILES) + [
        Path(__file__).relative_to(ROOT).as_posix(),
        'scripts/compatibility/test_amd_zombie_decode.py',
        'docs/compatibility/amd-zombie-decode-evidence/cpu-admission-summary.json',
        'docs/compatibility/amd-zombie-decode-evidence/wsl_cpu_probe.py',
    ]
    return host.Runtime(shape.EVIDENCE, EVIDENCE, commit, RUN_ID,
                        tile_large_convolutions=True, final_support_profile=None)


def destination_admission():
    from amd_generation_occupancy import plain_path
    root = plain_path(PRIVATE, existing=True)
    free = shutil.disk_usage(root).free
    if free < 2 * 1024**3:
        raise RuntimeError('Actual private output volume below unchanged 2 GiB disk floor')
    return dict(destination_free_bytes=free, required_destination_free_bytes=2 * 1024**3)


def prepare():
    """Actual input/header/full checkpoint authentication, with no AI imports."""
    runtime = make_runtime('0468f86a13271e401e02e5376dd10377e294c3b6')
    checkpoint = runtime.guest.checkpoint_record(Path(
        'D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors'))
    sys.path.insert(0, str(ROOT / 'api'))
    sys.path.insert(0, str(ROOT / 'experimental/extensions/trellis2-amd'))
    from shape_decoder_loader import ShapeDecoderComponent
    prefix = Path(checkpoint['path']).with_suffix('')
    component = ShapeDecoderComponent(str(prefix.with_suffix('.json')), str(prefix.with_suffix('.safetensors')))
    component._verify_source()
    if component.loaded or component.diagnostics['weights_loaded']:
        raise RuntimeError('CPU preflight unexpectedly loaded weights')
    component.unload()
    if any(name in sys.modules for name in ('torch', 'triton', 'trellis2', 'flex_gemm')):
        raise RuntimeError('CPU-only preparation imported a neural module')
    command = runtime.worker_command(historical.guest_path(historical.GUEST_ENTRY),
                                    historical.guest_path(EVIDENCE / 'run/guest'))
    row = dict(checkpoint=checkpoint, input_summary=runtime.bundle['summary'],
        run_contract=runtime.contract.document(), run_contract_sha256=runtime.contract.sha256,
        source_job_sha256=SOURCE_JOB_SHA, worker_command=command,
        host_admission=runtime.host_admission(), destination_admission=destination_admission(),
        neural_imported=False, model_loaded=False, production_marker_reserved=EVIDENCE.exists())
    if row['production_marker_reserved']:
        raise FileExistsError('Fresh zombie evidence path is already consumed')
    historical.save(PRIVATE / 'cpu-admission.json', row)
    historical.save(PRIVATE / 'probe-run-contract.json', runtime.contract.document())
    summary = dict(private_cpu_admission_sha256=historical.digest(PRIVATE / 'cpu-admission.json'),
        input_summary=runtime.bundle['summary'], source_job_sha256=SOURCE_JOB_SHA,
        source_image_sha256=shape.prior.zombie.SOURCE_SHA,
        shape_execution_commit=SHAPE_COMMIT, checkpoint_sha256=checkpoint['freshly_streamed_sha256'],
        initial_rows=1000, maximum_final_rows=1842968, maximum_intermediate_rows=1000000,
        maximum_tile_rows=32768, final_support_profile=None, large_convolution_tiling=True,
        host_admission=row['host_admission'], destination_admission=row['destination_admission'],
        no_neural_import=True, weights_loaded=False, production_marker_reserved=False)
    historical.save(PUBLIC / 'cpu-admission-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--harness-commit')
    args = parser.parse_args()
    if args.prepare:
        if args.verify or args.harness_commit:
            raise ValueError('Preparation has no execution option')
        row = prepare()
        print(json.dumps(row))
        return 0
    valid_commit(args.harness_commit)
    runtime = make_runtime(args.harness_commit)
    if args.verify:
        row = runtime.verify_outputs()
        historical.save(PRIVATE / 'independent-neural-readback.json', row)
        print(json.dumps(row))
        return 0
    destination_admission()
    return runtime.supervise()


if __name__ == '__main__':
    raise SystemExit(main())
