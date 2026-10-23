"""Narrow read-only preservation for the evidence-writer repair; no probes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
E = Path(__file__).resolve().parent
REPO = E.parents[2]
ROOT = REPO.parent
P = ROOT / 'artifacts/amd-evidence-writer-d784e4b'
OLD_E = REPO / 'docs/compatibility/amd-fixture-timing-evidence'
OLD_P = ROOT / 'artifacts/amd-fixture-timing-0c031be'
BASE = 'd784e4b89e0a05096d2503755c7e931895895d45'
SOURCE = 'docs/compatibility/amd-fixture-timing-evidence/windows_qualification.py'
CACHES = ('docs/compatibility/amd-turret-texture-decoder-evidence/__pycache__',
          'docs/compatibility/amd-wsl-turret-neural-decode-evidence/__pycache__',
          'experimental/extensions/trellis2-amd/__pycache__',
          'scripts/compatibility/__pycache__')


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def metadata(path):
    st = path.stat()
    return {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
            'device': st.st_dev, 'file_id': st.st_ino}


def tree(root):
    return [{'relative_path': path.relative_to(root).as_posix(), **metadata(path)}
            for path in sorted(root.rglob('*')) if path.is_file()]


def authenticate_public(name):
    path = OLD_E / name
    working = path.read_bytes()
    blob = git('show', BASE + ':' + path.relative_to(REPO).as_posix())
    assert working.replace(b'\r\n', b'\n') == blob
    return {'working_sha256': sha(working), 'git_blob_sha256': sha(blob),
            'CRLF_only_normalization_match': True}


def capture(mode):
    assert mode in ('before', 'after')
    public_path = E / ('preservation-' + mode + '.json')
    private_path = P / ('preservation-' + mode + '-private.json')
    assert not public_path.exists() and not private_path.exists()
    remote_path = P / ('remote-' + mode + '.txt')
    remote_blob = remote_path.read_bytes()
    heads = dict((ref, commit) for commit, ref in
                 (line.split() for line in remote_blob.decode('utf-8-sig').splitlines()))
    assert len(heads) == 123 and heads['refs/heads/codex/amd-fixture-readiness-timing'] == BASE
    assert heads['refs/heads/main'] == 'f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea'
    names = ('preservation-before.json', 'preservation-after.json',
             'preservation-review.json', 'preservation-accounting-note.json',
             'attempt-staging-preservation.json', 'independent-outcome-review.json',
             'independent-final-review.json', 'admission.json')
    public_auth = {name: authenticate_public(name) for name in names}
    prior_before, prior_after, prior_review = (read(OLD_E / name) for name in names[:3])
    prior_private = {}
    for phase, receipt in (('before', prior_before), ('after', prior_after)):
        path = OLD_P / ('preservation-' + phase + '-private.json')
        assert sha(path.read_bytes()) == receipt['private_detail_sha256']
        prior_private[phase] = read(path)
    assert prior_review['before_private_sha256'] == prior_before['private_detail_sha256']
    assert prior_review['after_private_sha256'] == prior_after['private_detail_sha256']
    assert prior_review['before_public_sha256'] == public_auth['preservation-before.json']['working_sha256']
    assert prior_review['after_public_sha256'] == public_auth['preservation-after.json']['working_sha256']
    assert prior_review['all_2280_before_after_identities_equal'] is True
    def old_map(rows):
        return {row['path'].casefold(): tuple(row[key] for key in ('bytes', 'mtime_ns', 'sha256')) for row in rows}
    assert len(prior_private['after']['fresh_file_identities']) == 2280
    assert old_map(prior_private['before']['fresh_file_identities']) == old_map(prior_private['after']['fresh_file_identities'])
    assert all(heads.get(ref) == commit for ref, commit in prior_private['after']['remote_heads'].items())
    protected = []
    hashed_bytes = 0
    for previous in prior_private['after']['protected100']:
        path = Path(previous['path'])
        row = {'path': str(path), **metadata(path)}
        row['prior_metadata_match'] = all(row[key] == previous[key] for key in ('bytes', 'mtime_ns'))
        assert row['prior_metadata_match']
        if row['bytes'] <= 1024 * 1024:
            row['sha256'] = sha(path.read_bytes())
            hashed_bytes += row['bytes']
            assert row['sha256'] == previous['sha256']
        protected.append(row)
    assert len(protected) == 100
    admission = read(OLD_E / 'admission.json')
    api_python = Path(admission['python_executable'])
    api_identity = {'path': str(api_python), **metadata(api_python), 'sha256': sha(api_python.read_bytes())}
    assert api_identity['sha256'] == admission['python_executable_sha256'] == '5b9eb768cd084d4b7959bc83b1a5111a10f8aaa18e5f4f7ef4187baf85dcf2c3'
    partial_root = OLD_P / 'cpu-once'
    partial = tree(partial_root)
    assert [row['relative_path'] for row in partial] == ['attempt.json.writing']
    partial[0]['sha256'] = sha((partial_root / 'attempt.json.writing').read_bytes())
    assert partial[0]['bytes'] == 1991 and partial[0]['sha256'] == '25ebe42f554489a23815c0e37ba09eae4dd143428481ce3344527cbf384fe6bd'
    caches = {name: tree(REPO / name) for name in CACHES}
    assert sum(map(len, caches.values())) == 20
    for name, rows in caches.items():
        assert [(r['relative_path'], r['bytes'], r['mtime_ns']) for r in rows] == [(r['relative_path'], r['bytes'], r['mtime_ns']) for r in prior_private['after']['preserved_historical_cache_metadata'][name + '/']]
    source_blob = git('show', BASE + ':' + SOURCE)
    source = {'path': SOURCE, **metadata(REPO / SOURCE), 'working_sha256': sha((REPO / SOURCE).read_bytes()),
              'base_git_blob_sha256': sha(source_blob), 'base_git_blob_oid': git('rev-parse', BASE + ':' + SOURCE).decode().strip()}
    if mode == 'before':
        assert (REPO / SOURCE).read_bytes().replace(b'\r\n', b'\n') == source_blob
        with (P / 'original-windows_qualification.git-blob.py').open('xb') as stream:
            stream.write(source_blob)
    old_e = [row for row in tree(OLD_E) if row['relative_path'] != 'windows_qualification.py']
    state = {'old_private_tree': tree(OLD_P), 'old_public_tree_except_authorized_source': old_e,
             'historical_cache_metadata': caches, 'protected100': protected,
             'partial_attempt': partial, 'api_python_identity': api_identity,
             'remote_heads': heads, 'source_identity': source, 'prior_public_authentication': public_auth}
    equal = None
    before_lock = None
    if mode == 'after':
        before_lock = read(P / 'preservation-before-lock.json')
        assert sha((E / 'preservation-before.json').read_bytes()) == before_lock['public_sha256']
        assert sha((P / 'preservation-before-private.json').read_bytes()) == before_lock['private_sha256']
        before = read(P / 'preservation-before-private.json')
        keys = set(state) - {'source_identity'}
        equal = all(state[key] == before[key] for key in keys)
        assert equal
    private = {'schema': 'modly.amd.evidence-writer-preservation-private.v1', 'phase': mode, **state}
    write_new(private_path, private)
    public = {'schema': 'modly.amd.evidence-writer-preservation.v1', 'phase': mode,
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'mission_base': BASE, 'classification': 'PASS_WITH_RECORDED_SCOPE',
              'private_detail_sha256': sha(private_path.read_bytes()),
              'capture_source_sha256': sha(Path(__file__).read_bytes()),
              'authenticated_prior_2280_identity_closure': True,
              'prior_private_table_hashes': {phase: receipt['private_detail_sha256'] for phase, receipt in (('before', prior_before), ('after', prior_after))},
              'prior_public_authentication': public_auth,
              'fresh_protected_metadata_count': 100,
              'protected_metadata_matches_prior': True,
              'efficient_protected_hash_count': sum('sha256' in row for row in protected),
              'efficient_protected_bytes_hashed': hashed_bytes,
              'original_API_python_sha256': api_identity['sha256'],
              'old_private_tree_file_count': len(state['old_private_tree']),
              'historical_cache_directory_count': 4, 'historical_cache_file_count': 20,
              'partial_attempt_file_count': 1, 'partial_attempt_bytes': 1991,
              'partial_attempt_sha256': partial[0]['sha256'],
              'partial_attempt_has_file_identity': True,
              'remote_head_count': len(heads), 'remote_capture_sha256': sha(remote_blob),
              'prior_122_remote_heads_preserved': True, 'main_commit': heads['refs/heads/main'],
              'preserved_before_after_identities_equal': equal,
              'authorized_source_base_git_blob_sha256': source['base_git_blob_sha256'],
              'execution_accounting': {'full_2280_rehash_passes': 0, 'before_successful_remote_reads': 2,
                  'before_remote_reason': 'Initial elevated stdout listing followed by elevated persisted listing.',
                  'failed_remote_requests': 0, 'after_successful_remote_reads': int(mode == 'after'),
                  'filesystem_contract_probes_by_preservation': 0, 'native_fixture_pairs_by_preservation': 0},
              'scope_limits': 'Saved 2280-file closure authenticated through base Git public receipts and hashed private tables; only protected100 metadata and <=1MiB protected files freshly checked. Historical private evidence tree and four cache trees checked by name, size, mtime and file identity; the exact 1991-byte partial is freshly hashed. API executable is hash-only. No environment/profile bulk reads, historical raw rehash, model/checkpoint read, filesystem probe, Job, worker, application, WSL, GPU, permissions change or historical script execution. The authorized diagnostic source is the sole exclusion from old public-tree equality. Prior BLOCKED outcomes remain unchanged.',
              'issues': []}
    if before_lock:
        public['authenticated_before_public_sha256'] = before_lock['public_sha256']
        public['authenticated_before_private_sha256'] = before_lock['private_sha256']
    write_new(public_path, public)
    if mode == 'before':
        write_new(P / 'preservation-before-lock.json', {'public_sha256': sha(public_path.read_bytes()), 'private_sha256': sha(private_path.read_bytes())})
    print(json.dumps({key: public[key] for key in ('phase', 'classification', 'remote_head_count', 'fresh_protected_metadata_count', 'efficient_protected_hash_count', 'old_private_tree_file_count', 'preserved_before_after_identities_equal')}))


if __name__ == '__main__':
    capture(sys.argv[1])
