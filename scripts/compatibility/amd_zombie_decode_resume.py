"""Fresh zombie decoder paths and experiment-only22GiB prelaunch admission.

The qualified standalone engine, input reader, native operations and all live
resource policies are reused unchanged. This does not enable product dispatch.
"""
import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_generation_decode as historical
import amd_zombie_decode as zombie

ROOT = historical.ROOT
PRIVATE = ROOT.parent / 'artifacts/amd-zombie-ram-readiness-91b7455'
PUBLIC = ROOT / 'docs/compatibility/amd-zombie-ram-readiness-3-evidence'
EVIDENCE = historical.APPROVED_ROOT / 'zombie-ram-91b7455/decode'
RUN_ID = 'run-zombie-geometry-ram-91b7455-v1'
PRELAUNCH_RAM = 22 * 1024**3
MEMORY_SOURCE_SHA = '8c6c3a73bb1e58691105f99dd27353ec8fa650e17f0a1045a5b9e45bb86a1e78'
MEMORY_READER = ROOT / 'docs/compatibility/amd-zombie-ram-headroom-evidence/memory_readiness.py'


def verify_readiness():
    summary = historical.read_json(PUBLIC / 'readiness-summary.json')
    raw_path = PRIVATE / 'memory-readiness-private.json'
    if historical.digest(raw_path) != summary['private_memory_receipt_sha256']:
        raise ValueError('Actual three-reading host memory receipt changed')
    raw = historical.read_json(raw_path)
    if (summary['classification'] != 'READY_FOR_FURTHER_ADMISSION'
            or raw['classification'] != summary['classification']
            or summary['required_available_bytes'] != PRELAUNCH_RAM
            or summary['samples'] != raw['samples'] or len(raw['samples']) != 3
            or raw['source_sha256'] != MEMORY_SOURCE_SHA
            or summary['source_sha256'] != MEMORY_SOURCE_SHA
            or summary['qualified_reader'] != MEMORY_READER.relative_to(ROOT).as_posix()
            or historical.digest(MEMORY_READER) != MEMORY_SOURCE_SHA):
        raise ValueError('Qualified actual three-sample memory/source evidence required')
    for sample in raw['samples']:
        if (type(sample['available_physical_bytes']) is not int
                or sample['available_physical_bytes'] < PRELAUNCH_RAM
                or sample['required_available_bytes'] != PRELAUNCH_RAM
                or sample['target_met'] is not True):
            raise ValueError('Every actual available-physical-RAM sample must meet22GiB')
    if (any(raw[name] is not False for name in ('administrator_role_enabled', 'WSL_launched',
            'model_loaded', 'neural_attempt_reserved', 'processes_terminated', 'settings_changed'))
            or type(raw['neural_forwards']) is not int or raw['neural_forwards'] != 0):
        raise ValueError('Read-only normal-user prelaunch evidence required')
    return summary


def reserve_fresh(runtime, original_reserve, output):
    if Path(output) != runtime.evidence / 'run':
        raise ValueError('Exact new owned host run required')
    verify_readiness()
    resource = runtime.host_admission()  # Original12GiB/50GiB gates unchanged.
    admitted = resource['available_ram_bytes'] >= PRELAUNCH_RAM
    historical.save(PRIVATE / 'final-pre-marker-admission.json', dict(
        classification='ADMITTED' if admitted else 'NO-GO: INSUFFICIENT_HEADROOM',
        stage='immediately-before-exclusive-neural-marker', host_resource=resource,
        required_prelaunch_ram_bytes=PRELAUNCH_RAM, harness_commit=runtime.commit,
        run_contract_sha256=runtime.contract.sha256, neural_marker_reserved=False))
    if not admitted:
        raise RuntimeError('Final22GiB prelaunch admission failed; no neural marker reserved: '
                           + str(resource['available_ram_bytes']))
    return original_reserve(output)


def make_runtime(commit):
    zombie.valid_commit(commit)
    verify_readiness()
    route = historical.load_module('_amd_zombie_ram_private_route', Path(zombie.__file__))
    route.EVIDENCE, route.RUN_ID = EVIDENCE, RUN_ID
    route.source_inputs = zombie.source_inputs
    runtime = route.make_runtime(commit)
    private_host = type(runtime).__init__.__globals__
    private_host['STATIC_FILES'] = list(private_host['STATIC_FILES']) + [
        Path(__file__).relative_to(ROOT).as_posix(),
        'scripts/compatibility/test_amd_zombie_decode_resume.py',
        'docs/compatibility/amd-zombie-ram-readiness-3-evidence/readiness-summary.json',
        'docs/compatibility/amd-zombie-ram-headroom-evidence/memory_readiness.py',
    ]
    original_reserve = runtime.reserve_attempt
    runtime.reserve_attempt = lambda output: reserve_fresh(runtime, original_reserve, output)
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness-commit', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    runtime = make_runtime(args.harness_commit)
    if args.verify:
        row = runtime.verify_outputs()
        historical.save(PRIVATE / 'independent-neural-readback.json', row)
        print(row)
        return 0
    from amd_generation_occupancy import plain_path
    if shutil.disk_usage(plain_path(PRIVATE, existing=True)).free < 2 * 1024**3:
        raise RuntimeError('Actual fresh output volume below original2GiB disk floor')
    return runtime.supervise()


if __name__ == '__main__':
    raise SystemExit(main())
