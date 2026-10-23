"""Authenticate the saved narrow pair; no fresh protected-file read or probe."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

E = Path(__file__).resolve().parent
REPO = E.parents[2]
P = REPO.parent / 'artifacts/amd-evidence-writer-d784e4b'
BASE = 'd784e4b89e0a05096d2503755c7e931895895d45'
SOURCE = 'docs/compatibility/amd-fixture-timing-evidence/windows_qualification.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().splitlines()


def main():
    before = read(E / 'preservation-before.json')
    after = read(E / 'preservation-after.json')
    bp = read(P / 'preservation-before-private.json')
    ap = read(P / 'preservation-after-private.json')
    lock = read(P / 'preservation-before-lock.json')
    assert sha(E / 'preservation-before.json') == lock['public_sha256'] == after['authenticated_before_public_sha256']
    assert sha(P / 'preservation-before-private.json') == lock['private_sha256'] == before['private_detail_sha256'] == after['authenticated_before_private_sha256']
    assert sha(P / 'preservation-after-private.json') == after['private_detail_sha256']
    assert sha(E / 'capture_preservation.py') == before['capture_source_sha256'] == after['capture_source_sha256']
    assert before['classification'] == after['classification'] == 'PASS_WITH_RECORDED_SCOPE'
    assert not before['issues'] and not after['issues']
    for key in set(bp) - {'phase', 'source_identity'}:
        assert bp[key] == ap[key], key
    assert len(ap['protected100']) == 100 and sum('sha256' in row for row in ap['protected100']) == 78
    assert len(ap['old_private_tree']) == 23
    assert len(ap['historical_cache_metadata']) == 4
    assert sum(map(len, ap['historical_cache_metadata'].values())) == 20
    assert len(ap['remote_heads']) == 123
    assert sha(P / 'remote-before.txt') == before['remote_capture_sha256']
    assert sha(P / 'remote-after.txt') == after['remote_capture_sha256']
    assert ap['partial_attempt'] == bp['partial_attempt'] and len(ap['partial_attempt']) == 1
    assert ap['partial_attempt'][0]['bytes'] == 1991
    assert ap['partial_attempt'][0]['sha256'] == '25ebe42f554489a23815c0e37ba09eae4dd143428481ce3344527cbf384fe6bd'
    assert sha(P / 'original-windows_qualification.git-blob.py') == bp['source_identity']['base_git_blob_sha256'] == ap['source_identity']['base_git_blob_sha256']
    assert bp['source_identity']['working_sha256'] != ap['source_identity']['working_sha256']
    changed = git('diff', '--name-only', BASE, '--')
    assert changed == [SOURCE]
    result = {
        'schema': 'modly.amd.evidence-writer-saved-preservation-review.v1',
        'classification': 'PASS_WITH_RECORDED_SCOPE',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'before_public_sha256': sha(E / 'preservation-before.json'),
        'after_public_sha256': sha(E / 'preservation-after.json'),
        'before_private_sha256': before['private_detail_sha256'],
        'after_private_sha256': after['private_detail_sha256'],
        'capture_source_sha256': before['capture_source_sha256'],
        'prior_saved_2280_closure_authenticated': True,
        'prior_closure_full_rehash_passes': 0,
        'narrow_completed_before_phases': 1, 'narrow_completed_after_phases': 1,
        'all_saved_before_after_preserved_state_equal': True,
        'fresh_protected_metadata_per_phase': 100,
        'efficient_protected_hashes_per_phase': 78,
        'efficient_protected_bytes_hashed_per_phase': before['efficient_protected_bytes_hashed'],
        'original_API_python_sha256': after['original_API_python_sha256'],
        'old_private_evidence_files_preserved': 23,
        'partial_attempt_namespace_size_mtime_file_identity_and_SHA256_preserved': True,
        'partial_attempt_bytes': 1991,
        'partial_attempt_sha256': after['partial_attempt_sha256'],
        'historical_cache_directories_preserved': 4,
        'historical_cache_files_preserved': 20,
        'all_123_remote_heads_preserved_before_publication': True,
        'main_commit': after['main_commit'],
        'authorized_tracked_change': SOURCE,
        'original_source_git_blob_sha256': bp['source_identity']['base_git_blob_sha256'],
        'repaired_source_working_sha256': ap['source_identity']['working_sha256'],
        'original_source_git_blob_backup_authenticated': True,
        'method': 'Saved receipt SHA256 authentication and complete private-table equality only. The sole tracked diff is the authorized diagnostic helper. No protected file or old raw payload reopened; no remote request, filesystem contract, permissions change, Job, fixture, application, WSL or GPU execution.',
        'limits': 'Preservation verifies the recorded narrow scope and authenticates the prior closed 2280-file evidence. It does not requalify original environments/profiles or upgrade any prior BLOCKED outcome. The newly attempted writer contract has a separate outcome review. This receipt precedes publication and any final report addition.',
        'issues': []}
    target = E / 'preservation-review.json'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'classification': result['classification'], 'review_sha256': sha(target),
                      'after_public_sha256': result['after_public_sha256']}))


if __name__ == '__main__':
    main()
