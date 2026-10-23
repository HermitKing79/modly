"""One stdlib-only BEFORE/AFTER identity pair for the final fixture timing study.

Run with ``python -B preservation_audit.py before|after``. This audit does not
import application code or inspect checkpoint headers. Private file tables and
remote captures are written outside the source checkout.
"""
import sys
sys.dont_write_bytecode = True
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time

BASE = '0c031befef0483f4ac3bed05fa35c3546166f255'
E = Path(__file__).resolve().parent
REPO = E.parents[2]
ROOT = REPO.parent
P = ROOT / 'artifacts/amd-fixture-timing-0c031be'
PRIOR_E = REPO / 'docs/compatibility/amd-fixture-startup-evidence'
PRIOR_P = ROOT / 'artifacts/amd-fixture-startup-474fbb4'
RAW27 = ROOT / 'artifacts/amd-retired-process-060d560/cpu-once'
PREFIX = 'docs/compatibility/amd-fixture-timing-evidence/'
REPORT = 'docs/compatibility/amd-fixture-timing.md'
CACHE_ROOTS = (
    'docs/compatibility/amd-turret-texture-decoder-evidence/__pycache__/',
    'docs/compatibility/amd-wsl-turret-neural-decode-evidence/__pycache__/',
    'experimental/extensions/trellis2-amd/__pycache__/',
    'scripts/compatibility/__pycache__/',
)
AUTH = {
    'preservation-before.json': '140eea4f38f20be26ebc60a247a9bf63a492759204250bc3ef6973cbcfb11bcf',
    'preservation-after.json': '103a61559c0002b6e451f9d0bbe5035ecd902d6c3f8e23770585e271abe3a2ff',
    'preservation-review.json': 'fa00113d9e94b6845260cd30be865243b82b9439d13c50f55c81d054728d65ef',
    'timeline-evidence.json': 'faefa6f266c45404ab8d96623c8252447202f7e4ef00f51f34daf05bb32d05ef',
    'independent-review-timeline.json': '83f8b0797d926f6ba079afc4125db26ce880b49db3008ba2607c44919c1e386e',
}
PRIOR_AUDIT_SHA = '04581a66c1114906f8cd434107e3b6c79d1b1876d5574b68e5e946388f504442'
OUTCOME_GIT_SHA = '2131365713ae1f4f5fbe5086d68ea75b193fa7debdff7e5f78d9a3a3879cf87d'
RAW33_AUTH_SHA = '1bbf8a5d1d180e5e5d49422a3d2f37d0f455ae8f20a6fc9d946b5b466bd17d29'



def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def dump(path, value):
    assert not path.exists(), 'Refusing to replace evidence: ' + path.name
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def authenticate(path, expected):
    blob = path.read_bytes()
    assert digest(blob) == expected, 'Authentication failed: ' + path.name
    return json.loads(blob.decode('utf-8-sig'))


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def identity_map(rows):
    result = {os.path.normcase(row['path']): row for row in rows}
    assert len(result) == len(rows)
    return result


def identity_equal(left, right):
    return set(left) == set(right) and all(
        all(left[key][field] == right[key][field]
            for field in ('bytes', 'mtime_ns', 'sha256')) for key in left)


def tree_metadata(root):
    return [{'relative_path': path.relative_to(root).as_posix(),
             'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            for path in sorted(root.rglob('*')) if path.is_file()]


def authenticate_raw27_once(timeline, source_pins):
    out = P / 'historical-raw27-authentication-private.json'
    assert not out.exists(), 'Refusing to rehash historical raw27'
    source = 'docs/compatibility/amd-retired-process-evidence/independent-outcome-review.json'
    working = (REPO / source).read_bytes()
    original = git('show', BASE + ':' + source)
    assert digest(original) == OUTCOME_GIT_SHA
    assert working.replace(b'\r\n', b'\n') == original
    outcome = json.loads(original)
    expected = outcome['raw_file_sha256']
    before = tree_metadata(RAW27)
    assert len(expected) == len(before) == 27
    assert {row['relative_path'] for row in before} == set(expected)
    rows = []; total = 0
    for desc in before:
        rel = desc['relative_path']; start = (RAW27 / rel).stat()
        h = hashlib.sha256()
        with (RAW27 / rel).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk); total += len(chunk)
        end = (RAW27 / rel).stat()
        actual = {'bytes': start.st_size, 'sha256': h.hexdigest()}
        assert actual == expected[rel]
        assert (start.st_size, start.st_mtime_ns) == (end.st_size, end.st_mtime_ns)
        rows.append({'relative_path': rel, 'bytes': actual['bytes'], 'sha256': actual['sha256'], 'mtime_ns': start.st_mtime_ns})
    after = tree_metadata(RAW27)
    assert before == after and total == 52436620
    private = {'schema': 'modly.amd.fixture-timing.historical-raw27-authentication-private.v1',
        'classification': 'PASS', 'mission_base': BASE,
        'original_outcome_git_sha256': OUTCOME_GIT_SHA,
        'original_outcome_working_sha256': digest(working), 'CRLF_only_normalization_match': True,
        'reviewed_startup_timeline_sha256': AUTH['timeline-evidence.json'],
        'timeline_independent_review_sha256': AUTH['independent-review-timeline.json'],
        'file_count': 27, 'total_bytes': total, 'mismatch_count': 0,
        'full_hash_passes': 1, 'files': rows,
        'metadata_before_authentication': before, 'metadata_after_authentication': after,
        'five_source_pins': source_pins}
    dump(out, private)
    dump(E / 'historical-raw27-authentication.json', {key: value for key, value in private.items()
        if key not in ('files', 'metadata_before_authentication', 'metadata_after_authentication')}
        | {'private_detail_sha256': digest(out.read_bytes()),
           'metadata_stable_during_authentication': True,
           'limit': 'Authenticates the saved27-file BLOCKED control attempt only. It does not establish startup cause, Windows qualification or natural target/sentinel settlement.'})


def main(mode):
    assert mode in ('before', 'after')
    assert P.is_dir() and E.is_dir()
    public_path = E / ('preservation-' + mode + '.json')
    private_path = P / ('preservation-' + mode + '-private.json')
    remote_path = P / ('remote-' + mode + '.txt')
    assert not any(path.exists() for path in (public_path, private_path)), 'Single phase already exists; refusing rerun'
    assert remote_path.is_file(), 'Capture remote listing before any file identity reads'
    started = time.monotonic()
    issues = []
    execution_accounting = {
        'completed_before_phase_count': 1,
        'completed_after_phase_count': 1 if mode == 'after' else 0,
        'interrupted_hash_attempt_count': 0,
        'before_successful_remote_reads': 2,
        'before_remote_read_reason': 'Initial elevated read returned stdout; a second elevated read saved the required capture before any scoped identity hashing.',
        'after_successful_remote_reads': 1 if mode == 'after' else 0,
        'failed_remote_requests': 0,
        'historical_raw27_full_hash_phase_count': 1,
    }
    prior = {name: authenticate(PRIOR_E / name, expected) for name, expected in AUTH.items()}
    git_auth = {}
    for name in AUTH:
        working = (PRIOR_E / name).read_bytes()
        blob = git('show', BASE + ':docs/compatibility/amd-fixture-startup-evidence/' + name)
        assert working.replace(b'\r\n', b'\n') == blob
        git_auth[name] = {'working_sha256': digest(working), 'git_blob_sha256': digest(blob), 'CRLF_only_normalization_match': True}
    pa = prior['preservation-after.json']
    pb = prior['preservation-before.json']
    review = prior['preservation-review.json']
    old = authenticate(PRIOR_P / 'preservation-after-private.json', pa['private_detail_sha256'])
    old_before = authenticate(PRIOR_P / 'preservation-before-private.json', pa['authenticated_before_private_sha256'])
    assert pa['authenticated_before_public_sha256'] == AUTH['preservation-before.json']
    assert review['after_public_sha256'] == AUTH['preservation-after.json']
    assert review['before_private_sha256'] == pa['authenticated_before_private_sha256']
    assert review['after_private_sha256'] == pa['private_detail_sha256']
    assert digest((PRIOR_E / 'preservation_audit.py').read_bytes()) == PRIOR_AUDIT_SHA == review['unchanged_audit_source_sha256']
    assert pa['classification'] == pb['classification'] == review['classification'] == 'PASS_WITH_RECORDED_SCOPE'
    assert pa['mission_before_after_all_identities_match'] is True
    assert identity_equal(identity_map(old_before['fresh_file_identities']), identity_map(old['fresh_file_identities']))
    assert len(old['fresh_file_identities']) == 2280 and len(old['remote_heads']) == 121
    raw33 = authenticate(REPO / 'docs/compatibility/amd-retired-process-evidence/evidence-authentication.json', RAW33_AUTH_SHA)
    assert raw33['classification'] == 'PASS_AUTHENTICATED_SAVED_FAILURE_ONLY'
    assert raw33['raw_integrity']['expected_count'] == raw33['raw_integrity']['actual_count'] == 33
    assert raw33['raw_integrity']['missing_extra_size_or_sha256_mismatches'] == 0
    assert review['linked_original_raw33_authentication_public_sha256'] == RAW33_AUTH_SHA
    assert review['historical_cache_directories_preserved'] == 4 and review['historical_cache_files_preserved'] == 20
    timeline = prior['timeline-evidence.json']
    timeline_review = prior['independent-review-timeline.json']
    assert timeline['authentication']['committed_review_sha256'] == OUTCOME_GIT_SHA
    assert any(row['sha256'] == AUTH['timeline-evidence.json'] for row in timeline_review['reviewed_evidence'])
    assert timeline_review['verdicts']['timeline_and_source_interpretation'] == 'PASS'
    # Recheck only the five explicitly pinned production/fixture sources. This
    # reads source bytes, never application imports or checkpoint/model data.
    source_pins = []
    for pin in timeline['authentication']['source_pins']:
        source = pin['source']; blob = (REPO / source).read_bytes()
        committed = git('show', BASE + ':' + source)
        assert digest(blob.replace(b'\r\n', b'\n')) == digest(committed) == pin['attempt_sha256']
        source_pins.append({'source': source, 'sha256': pin['attempt_sha256'], 'base_and_checkout_match': True})
    assert len(source_pins) == 5
    if mode == 'before':
        authenticate_raw27_once(timeline, source_pins)
    raw27_auth_blob = (P / 'historical-raw27-authentication-private.json').read_bytes()
    raw27_auth = json.loads(raw27_auth_blob)
    assert raw27_auth['classification'] == 'PASS' and raw27_auth['file_count'] == 27

    lock = None
    before_private = None
    if mode == 'after':
        lock = json.loads((P / 'before-authentication.json').read_text(encoding='utf-8'))
        authenticate(E / 'preservation-before.json', lock['public_sha256'])
        before_private = authenticate(P / 'preservation-before-private.json', lock['private_sha256'])

    # The old closed table is the exact permitted byte-read set. Its three
    # protected safetensors are generated outputs, never model checkpoints.
    expected = identity_map(old['fresh_file_identities'])
    protected = identity_map(old['protected100'])
    allowed_tensor_outputs = {key for key in protected if Path(key).suffix.lower() == '.safetensors'}
    assert len(allowed_tensor_outputs) == 3
    assert all(Path(key).suffix.lower() not in {'.pt', '.pth', '.ckpt', '.onnx', '.gguf'} for key in expected)
    assert {key for key in expected if Path(key).suffix.lower() == '.safetensors'} == allowed_tensor_outputs
    cache = {}
    bytes_hashed = 0
    for key, desc in expected.items():
        path = Path(desc['path'])
        start = path.stat()
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
                bytes_hashed += len(chunk)
        end = path.stat()
        row = {'path': str(path), 'bytes': start.st_size, 'mtime_ns': start.st_mtime_ns, 'sha256': h.hexdigest(),
               'stable_during_hash': (start.st_size, start.st_mtime_ns) == (end.st_size, end.st_mtime_ns)}
        row['hash_size_mtime_match'] = all(row[field] == desc[field] for field in ('bytes', 'mtime_ns', 'sha256'))
        cache[key] = row
        if not row['stable_during_hash'] or not row['hash_size_mtime_match']:
            issues.append({'kind': 'file_identity_mismatch', 'path': str(path)})
    assert len(cache) == 2280

    def category(name):
        rows = []
        for previous in old[name]:
            row = dict(previous)
            if 'path' in previous:
                row.update(cache[os.path.normcase(previous['path'])])
            else:
                for field in ('metadata', 'wheel'):
                    row[field] = cache[os.path.normcase(previous[field]['path'])]
            if 'descriptor_match' in row:
                row['descriptor_match'] = (row['hash_size_mtime_match'] if 'path' in row else
                    all(row[field]['hash_size_mtime_match'] for field in ('metadata', 'wheel')))
            rows.append(row)
        return rows

    categories = {name: category(name) for name in ('protected100', 'installed_app_inputs12', 'original_profile2103', 'selected_environment42', 'selected_source_backup13')}
    profile_rows = categories['original_profile2103']
    roots = {Path(row['path']).parents[len(PurePosixPath(row['relative_path']).parts) - 1] for row in profile_rows}
    assert len(roots) == 1
    profile_root = roots.pop()
    profile_names = sorted(path.relative_to(profile_root).as_posix() for path in profile_root.rglob('*') if path.is_file())
    profile_set_match = profile_names == old['original_profile_names_normalized']
    if not profile_set_match:
        issues.append({'kind': 'profile_file_set_mismatch'})
    master = next(row for row in categories['installed_app_inputs12'] if row['label'] == 'master')
    assert master['bytes'] == 97073940 and master['sha256'] == '1970bbcc5740dc56c8c9e6e20d97e8afd3176126c57b87d886a034f876eb38ae'

    raw_roots = {'original_raw33': ROOT / raw33['private_raw_root'], 'immediate_raw27': RAW27}
    raw_metadata = {name: tree_metadata(root) for name, root in raw_roots.items()}
    assert len(raw_metadata['original_raw33']) == 33 and len(raw_metadata['immediate_raw27']) == 27
    assert {row['relative_path'] for row in raw_metadata['original_raw33']} == set(raw33['raw_file_sha256'])
    cache_metadata = {name: tree_metadata(REPO / name) for name in CACHE_ROOTS}
    assert sum(map(len, cache_metadata.values())) == 20
    assert raw_metadata['immediate_raw27'] == raw27_auth['metadata_after_authentication']
    raw_match = cache_match = before_after_match = None
    if before_private:
        raw_match = raw_metadata == before_private['preserved_raw_tree_metadata']
        cache_match = cache_metadata == before_private['preserved_historical_cache_metadata']
        before_after_match = identity_equal(cache, identity_map(before_private['fresh_file_identities']))
        if not raw_match:
            issues.append({'kind': 'historical_raw_metadata_mismatch'})
        if not cache_match:
            issues.append({'kind': 'historical_cache_metadata_mismatch'})
        if not before_after_match:
            issues.append({'kind': 'mission_before_after_identity_mismatch'})

    # The one successful listing is captured before this audit starts. Keeping
    # the network request separate prevents a failed request repeating hashes.
    remote_blob = remote_path.read_bytes()
    heads = {}
    for line in remote_blob.decode('utf-8-sig').splitlines():
        commit, ref = line.split()
        assert ref.startswith('refs/heads/') and re.fullmatch('[0-9a-f]{40}', commit) and ref not in heads
        heads[ref] = commit
    baseline_heads = before_private['remote_heads'] if before_private else old['remote_heads']
    assert len(heads) == 122 and BASE in heads.values()
    assert len(baseline_heads) == (122 if mode == 'after' else 121)
    remote_match = all(heads.get(ref) == commit for ref, commit in baseline_heads.items())
    if not remote_match:
        issues.append({'kind': 'remote_baseline_heads_mismatch'})

    # The mission is evidence-only. No tracked production or historical source
    # edit is permitted; the old four cache directories are retained verbatim.
    changed = git('diff', '--name-only', BASE, '--').decode('utf-8').splitlines()
    untracked = git('ls-files', '--others', '--exclude-standard', '--').decode('utf-8').splitlines()
    mission_files = sorted(set(changed) | {name for name in untracked if name == REPORT or name.startswith(PREFIX)})
    unexpected = [name for name in changed if name != REPORT and not name.startswith(PREFIX)]
    unexpected += [name for name in untracked if name != REPORT and not name.startswith(PREFIX) and not name.startswith(CACHE_ROOTS)]
    evidence_files = [name[len(PREFIX):] for name in mission_files if name.startswith(PREFIX)]
    assert all('/' not in name and Path(name).suffix in {'.json', '.py', '.md'} for name in evidence_files)
    if unexpected:
        issues.append({'kind': 'checkout_allowlist_mismatch', 'paths': unexpected})
    source_guard = {'classification': 'PASS' if not unexpected else 'FAIL', 'baseline_commit': BASE,
        'allowed_report': REPORT, 'allowed_evidence_prefix': PREFIX,
        'modified_or_added_production_path_count': len(unexpected),
        'mission_evidence_files_observed': evidence_files,
        'preserved_historical_cache_directory_count': len(CACHE_ROOTS), 'unexpected_paths': unexpected}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    prior_summary = {'public_git_blob_authentication': git_auth, 'after_private_sha256': pa['private_detail_sha256'],
        'source_audit_sha256': PRIOR_AUDIT_SHA, 'classification': pa['classification'],
        'all_2280_old_before_after_identities_equal': True, 'raw33_saved_authentication_sha256': RAW33_AUTH_SHA,
        'raw33_count': 33, 'raw33_payloads_rehashed_by_this_audit': 0,
        'prior_diagnostic_outcome': pa['authenticated_prior_retired_process']['prior_diagnostic_outcome'],
        'reviewed_timeline_sha256': AUTH['timeline-evidence.json'], 'five_source_pins': source_pins,
        'historical_raw27_authentication_private_sha256': digest(raw27_auth_blob)}
    private = {'schema': 'modly.amd.fixture-timing-preservation-' + mode + '-private.v1',
        'created_utc': now, 'mission_base': BASE, **categories,
        'fresh_file_identities': list(cache.values()), 'original_profile_names_normalized': profile_names,
        'preserved_raw_tree_metadata': raw_metadata, 'preserved_historical_cache_metadata': cache_metadata,
        'remote_heads': heads, 'authenticated_prior_startup': prior_summary, 'execution_accounting': execution_accounting,
        'checkout_evidence_only_guard': source_guard, 'issues': issues}
    dump(private_path, private)
    public = {'schema': 'modly.amd.fixture-timing-preservation-' + mode + '.v1',
        'created_utc': now, 'mission_base': BASE, 'classification': 'PASS_WITH_RECORDED_SCOPE' if not issues else 'FAIL',
        'private_detail_sha256': digest(private_path.read_bytes()),
        'audit_source_sha256': digest(Path(__file__).read_bytes()),
        'authenticated_prior_startup': prior_summary,
        'execution_accounting': execution_accounting,
        'protected_files_fresh_full_SHA256_checked': 100, 'protected_files_hash_size_mtime_match': all(row['hash_size_mtime_match'] for row in categories['protected100']),
        'accepted_master_bytes': master['bytes'], 'accepted_master_sha256': master['sha256'], 'accepted_master_fresh_hash_size_mtime_match': master['hash_size_mtime_match'],
        'installed_app_inputs_checked': 12, 'installed_app_inputs_hash_size_mtime_match': all(row['hash_size_mtime_match'] for row in categories['installed_app_inputs12']),
        'original_profile_file_count': 2103, 'original_profile_normalized_file_set_match': profile_set_match,
        'original_profile_hash_size_mtime_match': all(row['hash_size_mtime_match'] for row in profile_rows),
        'selected_environment_check_count': 42, 'selected_environment_checks_match': all(row['descriptor_match'] for row in categories['selected_environment42']),
        'selected_source_backup_check_count': 13, 'selected_source_backup_hash_size_mtime_match': all(row['hash_size_mtime_match'] for row in categories['selected_source_backup13']),
        'fresh_full_hash_unique_file_count': len(cache), 'fresh_full_hash_bytes_read': bytes_hashed,
        'prior_startup_all_identities_match': identity_equal(cache, expected),
        'mission_before_after_all_identities_match': before_after_match,
        'original_raw33_and_immediate_raw27_name_size_mtime_preserved': raw_match,
        'historical_cache_name_size_mtime_preserved': cache_match,
        'historical_raw_metadata_counts': {name: len(rows) for name, rows in raw_metadata.items()},
        'historical_cache_metadata_file_count': sum(map(len, cache_metadata.values())),
        'historical_raw_payload_bytes_read_by_this_audit': 0,
        'checkpoint_or_model_payload_bytes_read': 0, 'checkpoint_header_bytes_read': 0,
        'remote_head_count': len(heads), 'remote_baseline_heads_checked': len(baseline_heads),
        'prior_remote_heads_match': remote_match, 'added_remote_head_count': len(set(heads) - set(baseline_heads)),
        'main_commit': heads['refs/heads/main'], 'remote_capture_sha256': digest(remote_blob),
        'checkout_evidence_only_guard': source_guard, 'issues': [{'kind': row['kind']} for row in issues],
        'scope_limits': [
            'One fresh hash per unique scoped file per phase: protected100, app/master12, original profile2103 with normalized name-set check, selected environment42 descriptors/native bytes and installed source/backups13; 2280 unique files. No application code is executed.',
            'The immediate startup closed preservation pair and committed public receipts authenticate the expected scope. Older trees/reports, original unique400, checkpoint metadata and SDK78 are linked through those receipts rather than freshly enumerated or rehashed.',
            'Three explicitly protected generated safetensors outputs are byte-hashed; no checkpoint/model payload or header is read. No native process-control probe, application, worker, WSL, GPU, Blender, installer, model or historical script is executed.',
            'Historical raw33/raw27 and the four existing cache directories receive name/size/mtime snapshots only in this audit. Raw33 saved authentication remains linked; this audit owns one separate BEFORE full raw27 authentication receipt, using the original committed outcome manifest. No raw payload is read here.',
            'BEFORE captures122 remote heads and checks all121 prior identities. AFTER checks all122 BEFORE identities including main before publication. Publication is outside this audit.',
            'The checkout permits only new JSON/Python/Markdown evidence in this mission directory and its report. Preservation does not upgrade prior BLOCKED Windows controls, viewer outcomes, or startup/runtime qualification.'
        ], 'elapsed_seconds': round(time.monotonic() - started, 3)}
    if lock:
        public['authenticated_before_public_sha256'] = lock['public_sha256']
        public['authenticated_before_private_sha256'] = lock['private_sha256']
    dump(public_path, public)
    if mode == 'before':
        dump(P / 'before-authentication.json', {'public_sha256': digest(public_path.read_bytes()), 'private_sha256': digest(private_path.read_bytes())})
    print(json.dumps({key: public[key] for key in ('classification', 'fresh_full_hash_unique_file_count', 'fresh_full_hash_bytes_read', 'remote_head_count', 'prior_remote_heads_match', 'mission_before_after_all_identities_match', 'elapsed_seconds', 'issues')}))
    return 0 if not issues else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
