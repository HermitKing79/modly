"""Read-only Phase22C authentication; output only new evidence JSON."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'amd-static-prop-sbox-uv-pbr-20261031-v1'
PARENT = BASE.parent / 'amd-static-prop-sbox-sampling-aa-20261031-v1'
PROJECT = Path(r'C:\Users\wattz\Documents\S&box Projects\greyfield')
PROOF = BASE.parent.parent / 'proof'
ENGINE = Path(r'C:\Program Files (x86)\Steam\steamapps\common\sbox')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(PROOF), *args], text=True, timeout=15).strip()


def save(name, value):
    with (BASE / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def project_files():
    return {p for p in PROJECT.rglob('*') if p.is_file()
            and '.sbox' not in p.relative_to(PROJECT).parts
            and '.git' not in p.relative_to(PROJECT).parts}


def snapshot():
    prior = json.loads((PARENT / 'preservation-before.json').read_text())
    paths = {Path(p) for p in prior['files']}
    assert len(paths) == 6821
    for name, row in prior['files'].items():
        assert sha(name) == row['sha256'], 'Preserved original changed: ' + name
    publication = json.loads((BASE / 'github-parent-authentication.json').read_text())
    assert publication['commit'] == '9a7efb13495fc9b44d02bbbfe4fb94c5e1534839'
    assert len(publication['files']) == 5
    for row in publication['files']:
        assert row['exact_utf8_content_match'] and sha(PARENT / row['local_name']) == row['sha256']
    assets = json.loads((OLD / 'final-private-test-assets.json').read_text())
    for row in assets['files']:
        assert sha(Path(assets['assets_directory']) / row['path']) == row['sha256']
    captures = json.loads((OLD / 'capture-manifest.json').read_text())
    assert len(captures) == 5
    for row in captures:
        assert sha(row['metadata_path']) == row['metadata_sha256']
        assert sha(row['png_path']) == row['png_sha256']
    expected = {
        'capture-metadata.json': 'b0372a7aa3edd03f3d19872c80050a1474215e0f38bfb3fd8a55f49453465e5b',
        'independent_sampling_oracle.py': 'b0aadae4d6c3ebb859aeada7f135621d219a8cbe26d80124d1296a30e3aab1ae',
        'sampling-root-result.json': 'e988366234ea888028083fffd030ab2f57afdfb08a9efdf2db8a857b5cb4e718',
        'texture/diagnostic_uv_color.png': '35161147faf014e2069660a8681af66762f3d3ad4cc859904087becbb328a6d3',
        'diagnostic_pointwrap_unlit.shader': 'b712ac40cc8445e9830cdddca141d72ee06e4d7b1c2fd258a1fd4a3033ab7982',
    }
    for name, digest in expected.items():
        assert sha(OLD / name) == digest
    result = json.loads((OLD / 'sampling-root-result.json').read_text())
    assert result['status'] == 'NO-GO' and result['union']['eligible_pixels'] == 829668
    paths.update(p for folder in (OLD, PARENT) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    authored = project_files()
    paths.update(authored)
    paths.update(ENGINE / 'bin/managed' / name for name in
                 ('Sandbox.Engine.dll', 'Sandbox.Engine.xml', 'Sandbox.Tools.dll', 'sbox-cli.exe'))
    glb = BASE.parent / 'amd-static-prop-uv-improve-20261031-v1/asset/experimental-static-prop.glb'
    assert sha(glb) == 'afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201'
    record = dict(classification='PASS_PHASE22C_AUTHENTICATED_BASELINE',
                  parent_commit=publication['commit'], historical_phase22_status='NO-GO',
                  historical_phase22_files=len([p for p in OLD.rglob('*') if p.is_file() and '__pycache__' not in p.parts]),
                  files={str(p):dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(paths)},
                  project_files=sorted(map(str, authored)), local_head=git('rev-parse','HEAD'),
                  local_refs=git('show-ref'), local_status=git('status','--short'),
                  excluded='Mutable .sbox caches, .git internals and existing Python bytecode; all authored project files and all Phase22/22B evidence included')
    save('preservation-before.json', record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('files','project_files','local_refs','local_status')} | {'protected_files':len(paths)}))


def verify(label):
    assert re.fullmatch(r'[A-Za-z0-9_-]+', label)
    old = json.loads((BASE / 'preservation-before.json').read_text())
    changed = [p for p,r in old['files'].items() if not Path(p).is_file() or sha(p) != r['sha256']]
    added = sorted(map(str, project_files() - set(map(Path, old['project_files']))))
    historical_before = {Path(p) for p in old['files'] if Path(p).is_relative_to(OLD) or Path(p).is_relative_to(PARENT)}
    historical_current = {p for folder in (OLD, PARENT) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    added_historical = sorted(map(str, historical_current - historical_before))
    git_equal = (git('rev-parse','HEAD') == old['local_head'] and git('show-ref') == old['local_refs']
                 and git('status','--short') == old['local_status'])
    result = dict(classification='PASS' if not changed and not added and not added_historical and git_equal else 'NO-GO',
                  protected_files=len(old['files']), changed_or_missing=changed, new_project_files=added,
                  added_historical_evidence_files=added_historical,
                  local_git_state_unchanged=git_equal, historical_phase22_status='NO-GO')
    save('preservation-' + label + '.json', result)
    print(json.dumps(result))
    if result['classification'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    if sys.argv[1:] == ['snapshot']:
        snapshot()
    elif len(sys.argv) == 3 and sys.argv[1] == 'verify':
        verify(sys.argv[2])
    else:
        raise SystemExit('snapshot | verify LABEL')
