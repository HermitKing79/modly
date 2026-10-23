"""Read-only saved staging classification for the single blocked invocation.

This classifier never imports a fixture/runner or executes native code. It is
deliberately limited to the observed attempt.json.writing-only archive. Any new
stage, worker, protocol or final-attempt file makes it fail closed. It cannot
identify which filesystem operation failed or recover unrecorded elapsed time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-fixture-timing-0c031be'
BASE = '0c031befef0483f4ac3bed05fa35c3546166f255'
SOURCES = (
    'api/services/amd_stage_process.py',
    'api/services/amd_stage_bootstrap.py',
    'api/services/amd_completed_result.py',
    'docs/compatibility/amd-job-stage-orchestrator-evidence/cpu_stage_fixture.py',
    'docs/compatibility/amd-retired-process-evidence/windows_qualification.py',
    'docs/compatibility/amd-fixture-timing-evidence/diagnostic_fixture.py',
    'docs/compatibility/amd-fixture-timing-evidence/diagnostic_process.py',
    'docs/compatibility/amd-fixture-timing-evidence/diagnostic_bootstrap.py',
    'docs/compatibility/amd-fixture-timing-evidence/windows_qualification.py',
)
MAX_STAGING_BYTES = 65536


def digest(blob): return hashlib.sha256(blob).hexdigest()


def write_exclusive(path, row):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def analyze(input_root):
    files = sorted(path for path in input_root.rglob('*') if path.is_file())
    directories = sorted(path for path in input_root.rglob('*') if path.is_dir())
    names = [path.relative_to(input_root).as_posix() for path in files]
    if names != ['attempt.json.writing'] or directories:
        raise RuntimeError('Classifier only accepts the completed staging-only observation; broader evidence requires separate review')
    staging_path = files[0]
    if staging_path.stat().st_size > MAX_STAGING_BYTES:
        raise RuntimeError('Staged attempt exceeds bounded analysis size')
    # Read this raw file once. The exact bytes supply parsing and its manifest.
    staging_blob = staging_path.read_bytes()
    if len(staging_blob) > MAX_STAGING_BYTES:
        raise RuntimeError('Staged attempt changed beyond bounded analysis size')
    complete_json, staged, json_error_type = False, None, None
    try:
        staged = json.loads(staging_blob)
        complete_json = isinstance(staged, dict)
    except (ValueError, UnicodeError) as error:
        json_error_type = type(error).__name__
    admission_blob = (E / 'admission.json').read_bytes()
    cpu_blob = (E / 'cpu-tests.json').read_bytes()
    admission, cpu = json.loads(admission_blob), json.loads(cpu_blob)
    expected = admission.get('source_pins') or {}
    staged = staged if isinstance(staged, dict) else {}
    current = {name: digest((ROOT / name).read_bytes()) for name in SOURCES}
    checks = dict(
        complete_staged_json_object=complete_json,
        exact_nine_reviewed_sources=set(expected) == set(SOURCES),
        staged_admission_cpu_source_maps_equal=staged.get('source_pins') == expected == cpu.get('sources'),
        all_current_sources_match_reviewed_pins=current == expected,
        admission_digest_matches_staging_and_cpu=staged.get('admission_sha256') == digest(admission_blob) == cpu.get('admission_sha256'),
        reviewed_cpu26=(cpu.get('tests') == 26 and cpu.get('classification') == 'PASS_CPU_ONLY'
            and cpu.get('native_pairs') == 0 and len(cpu.get('records') or []) == 2
            and all(row.get('returncode') == 0 for row in cpu.get('records') or [])),
        exact_base=staged.get('base_commit') == admission.get('base_commit') == BASE,
        original_interpreter_hash_bound=staged.get('python_executable_sha256') == admission.get('python_executable_sha256'),
        single_cpu_pair_limit=staged.get('CPU_only') is True and staged.get('maximum_control_pairs') == 1,
        unchanged_limits=(staged.get('deadline_seconds') == 90 and staged.get('stage_timeout_s') == 25
            and staged.get('process_and_job_bytes') == 6 * 1024**3
            and staged.get('ram_floor_bytes') == 12 * 1024**3
            and staged.get('disk_floor_bytes') == 2 * 1024**3),
    )
    manifest = [dict(relative_path='attempt.json.writing', bytes=len(staging_blob), sha256=digest(staging_blob))]
    empty_role = dict(parent_timer_origin_ns=None, parent_deadline_ns=None, last_poll=None,
        timeout_decision_ns=None, parent_popen_elapsed_ns=None, child_popen_elapsed_ns=None,
        script_entry_ns=None, readiness_publication=None, actual_parent_observation_delta_ns=None,
        checkpoint_overhead=None, natural_outcome=None, forced_outcome=None,
        authenticated_fixture_identity=None)
    public = dict(
        schema='modly.amd.fixture-timing.outcome.v1', base_commit=BASE,
        analysis_mode='SAVED_DATA_ONLY_STAGING_ONLY', classification='BLOCKED',
        evidence_integrity='STAGING_AUTHENTICATED' if all(checks.values()) else 'STAGING_AUTHENTICATION_REJECTED',
        checks=checks,
        failure_boundary='Attempt JSON staging exists; public attempt publication did not complete; no worker or fixture evidence exists',
        source_order_basis='Reviewed host source completes attempt publication before owned Job creation or worker launch',
        actual_native_control_pairs=0, recorded_worker_starts=0, recorded_fixture_starts=0,
        recorded_owned_job_creations=0,
        zero_count_limit='Zero native pairs follows the reviewed pre-Job publication boundary; no process/Job history was produced',
        public_attempt_present=False, staging_bytes=len(staging_blob), staged_json_complete=complete_json,
        staging_json_error_type=json_error_type, observed_raw_file_count=1,
        reviewed_cpu_tests=cpu.get('tests'), source_pins=current,
        error_type=None, errno=None, winerror=None, failing_filesystem_operation=None,
        hardlink_publication_begin_ns=None, hardlink_publication_end_ns=None,
        roles=dict(target=dict(empty_role), sentinel=dict(empty_role)),
        host_observer_counters=None, worker_counters=None, bootstrap_counters=None,
        total_runtime_s=None, host_ram_min_bytes=None, process_peak_bytes=None, job_peak_bytes=None,
        staged_prelaunch_resources=staged.get('initial_resources'),
        timing_classification='NO_STARTUP_TIMING_EVIDENCE',
        causal_classification='UNRESOLVED: no exception traceback or errno was saved; no filesystem, sandbox or scheduler cause is established',
        qualification_limits='No lifecycle result, cancellation isolation, WSL, AMD orchestrator, GPU or neural-output qualification',
        no_retry=True,
    )
    private = dict(schema='modly.amd.fixture-timing.outcome-private.v1',
        input_root=str(input_root.resolve()), raw_manifest=manifest,
        observed_directory_count=0, staged_attempt=staged,
        admission_sha256=digest(admission_blob), cpu_receipt_sha256=digest(cpu_blob),
        checks=checks, public=public)
    return private, public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=P / 'cpu-once')
    parser.add_argument('--private-output', type=Path, default=P / 'timing-outcome-private.json')
    parser.add_argument('--public-output', type=Path, default=E / 'timing-evidence.json')
    args = parser.parse_args()
    if not args.input_root.is_dir(): raise RuntimeError('Saved completed invocation directory required')
    if args.private_output.exists() or args.public_output.exists(): raise RuntimeError('Exclusive fresh analysis outputs required')
    private, public = analyze(args.input_root)
    write_exclusive(args.private_output, private)
    public['private_detail_sha256'] = digest(args.private_output.read_bytes())
    write_exclusive(args.public_output, public)
    print(json.dumps(dict(classification=public['classification'], evidence_integrity=public['evidence_integrity'],
                          observed_raw_file_count=1, actual_native_control_pairs=0,
                          checks_passed=all(public['checks'].values()))))


if __name__ == '__main__': main()
