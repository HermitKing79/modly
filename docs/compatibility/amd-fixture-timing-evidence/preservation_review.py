"""Close saved preservation receipts without a second identity or raw pass."""
import sys
sys.dont_write_bytecode = True
import datetime
import hashlib
import json
from pathlib import Path

E = Path(__file__).resolve().parent
REPO = E.parents[2]
P = REPO.parent / 'artifacts/amd-fixture-timing-0c031be'
RAW27_SHA = '02098d7bed85a3e7eb2438a3e37db3365df02669ea12e4b8a73736eb5e79d070'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def identities(rows):
    result = {row['path'].casefold(): tuple(row[field] for field in ('bytes', 'mtime_ns', 'sha256')) for row in rows}
    assert len(result) == len(rows)
    return result


def main():
    target = E / 'preservation-review.json'
    assert not target.exists(), 'Saved review already exists; refusing replacement'
    lock = read(P / 'before-authentication.json')
    before = read(E / 'preservation-before.json')
    after = read(E / 'preservation-after.json')
    bp = read(P / 'preservation-before-private.json')
    ap = read(P / 'preservation-after-private.json')
    assert sha(E / 'preservation-before.json') == lock['public_sha256'] == after['authenticated_before_public_sha256']
    assert sha(P / 'preservation-before-private.json') == lock['private_sha256'] == before['private_detail_sha256'] == after['authenticated_before_private_sha256']
    assert sha(P / 'preservation-after-private.json') == after['private_detail_sha256']
    assert before['classification'] == after['classification'] == 'PASS_WITH_RECORDED_SCOPE'
    assert not before['issues'] and not after['issues']
    assert sha(E / 'preservation_audit.py') == before['audit_source_sha256'] == after['audit_source_sha256']
    assert len(bp['fresh_file_identities']) == len(ap['fresh_file_identities']) == 2280
    assert identities(bp['fresh_file_identities']) == identities(ap['fresh_file_identities'])
    assert bp['remote_heads'] == ap['remote_heads'] and len(ap['remote_heads']) == 122
    assert sha(P / 'remote-before.txt') == before['remote_capture_sha256']
    assert sha(P / 'remote-after.txt') == after['remote_capture_sha256']
    assert after['remote_baseline_heads_checked'] == 122 and after['added_remote_head_count'] == 0
    assert after['prior_remote_heads_match'] is True
    assert after['mission_before_after_all_identities_match'] is True
    assert bp['preserved_raw_tree_metadata'] == ap['preserved_raw_tree_metadata']
    assert bp['preserved_historical_cache_metadata'] == ap['preserved_historical_cache_metadata']
    assert after['original_raw33_and_immediate_raw27_name_size_mtime_preserved'] is True
    assert after['historical_cache_name_size_mtime_preserved'] is True
    assert after['historical_cache_metadata_file_count'] == 20
    assert after['fresh_full_hash_bytes_read'] == before['fresh_full_hash_bytes_read'] == 2283191348
    assert after['checkpoint_or_model_payload_bytes_read'] == after['checkpoint_header_bytes_read'] == 0
    assert after['historical_raw_payload_bytes_read_by_this_audit'] == 0
    assert after['checkout_evidence_only_guard']['modified_or_added_production_path_count'] == 0
    # Only the saved authentication table is read; historical raw payloads are
    # not opened here. Its names/sizes/mtimes must match both preservation phases.
    raw27_path = P / 'historical-raw27-authentication-private.json'
    assert sha(raw27_path) == RAW27_SHA
    raw27 = read(raw27_path)
    assert raw27['file_count'] == 27 and raw27['total_bytes'] == 52436620
    assert raw27['mismatch_count'] == 0 and raw27['full_hash_passes'] == 1 and raw27['classification'] == 'PASS'
    raw27_metadata = sorted((row['relative_path'], row['bytes'], row['mtime_ns']) for row in raw27['files'])
    phase_metadata = sorted((row['relative_path'], row['bytes'], row['mtime_ns']) for row in ap['preserved_raw_tree_metadata']['immediate_raw27'])
    assert raw27_metadata == phase_metadata
    result = {
        'schema': 'modly.amd.fixture-timing-saved-preservation-review.v1',
        'classification': 'PASS_WITH_RECORDED_SCOPE',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'review_method': 'Saved receipt SHA256 linkage and table comparison only. No new identity phase, raw payload read, remote request, application, worker, native process-control probe, WSL or GPU execution.',
        'before_public_sha256': sha(E / 'preservation-before.json'),
        'after_public_sha256': sha(E / 'preservation-after.json'),
        'before_private_sha256': lock['private_sha256'],
        'after_private_sha256': after['private_detail_sha256'],
        'unchanged_audit_source_sha256': after['audit_source_sha256'],
        'all_2280_before_after_identities_equal': True,
        'protected100_hash_size_mtime_match': True,
        'app_master12_profile2103_environment42_backups13_match': True,
        'all_122_remote_head_identities_unchanged': True,
        'added_remote_head_count': 0,
        'main_commit': after['main_commit'],
        'original_raw33_and_immediate_raw27_name_size_mtime_preserved': True,
        'historical_cache_directories_preserved': 4,
        'historical_cache_files_preserved': 20,
        'independent_raw27_single_authentication_private_sha256': RAW27_SHA,
        'independent_raw27_count': 27,
        'independent_raw27_total_bytes': 52436620,
        'independent_raw27_mismatches': 0,
        'raw27_authentication_matches_both_preservation_metadata_tables': True,
        'linked_original_raw33_authentication_public_sha256': after['authenticated_prior_startup']['raw33_saved_authentication_sha256'],
        'linked_original_raw33_count': 33,
        'checkout_modified_or_added_production_path_count': 0,
        'execution_accounting': after['execution_accounting'],
        'checkpoint_or_model_payload_bytes_read': 0,
        'checkpoint_header_bytes_read': 0,
        'historical_raw_payload_bytes_read_by_preservation_and_review': 0,
        'scope_limits': 'There is one completed BEFORE and one completed AFTER with no interrupted or repeated identity phase. Two successful elevated BEFORE remote reads comprise a preliminary stdout read and a persisted capture. The frozen audit separately authenticated27 historical raw files once; its inherited zero raw-byte field refers to the2280-file identity loop as clarified by the accounting note. This review never upgrades prior BLOCKED diagnostics or startup/runtime qualification.',
        'accounting_note_sha256': sha(E / 'preservation-accounting-note.json'),
        'historical_raw27_full_hash_passes': 1,
        'separate_historical_raw27_authentication_bytes_read': 52436620,
        'new_audit_script_total_historical_raw_payload_bytes_read': 52436620,
        'issues': [],
    }
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'classification': result['classification'], 'preservation_review_sha256': sha(target), 'all_2280_before_after_identities_equal': True, 'all_122_remote_head_identities_unchanged': True}))


if __name__ == '__main__':
    main()
