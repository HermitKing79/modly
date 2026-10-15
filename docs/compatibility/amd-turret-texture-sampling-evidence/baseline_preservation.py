"""Passive, narrowly scoped preservation check; no installed package imports."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / 'docs/compatibility'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def identity(path, expected):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        return {'path': str(path), 'matches': False, 'error': 'missing or symlink'}
    before = path.stat()
    result = {'path': str(path), 'bytes': before.st_size,
              'mtime_ns': before.st_mtime_ns, 'sha256': sha(path)}
    after = path.stat()
    result['stable_during_read'] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    result['matches'] = (result['stable_during_read']
        and result['sha256'] == expected['sha256']
        and all(result[k] == expected[k] for k in ('bytes', 'mtime_ns') if k in expected))
    return result


def textsha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest() if value is not None else None


def capture():
    prior = read(DOCS / 'amd-turret-texture-route-evidence/preservation-check.json')
    files = read(DOCS / 'amd-turret-visual-preview-evidence/preservation-before.json')['files']
    protected = [identity(p, row) for p, row in files.items()]
    source = [identity(row['path'], row) for row in prior['source_files']]
    assets = read(DOCS / 'amd-turret-texture-route-evidence/local-assets.json')
    saved = [identity(row['path'], row) for row in assets['saved_inputs'].values()]
    saved.append(identity(assets['image']['path'], assets['image']))
    preview = [identity(row['path'], row) for row in
        read(DOCS / 'amd-turret-visual-preview-evidence/output-verification.json')['outputs'].values()]
    amd_source = Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD')
    backup = identity(amd_source / 'nodes.py.before-profiler-fix',
        {'sha256': '585d81c545cce5ca71fee1cb17f47e65b13c7f98b13a6053e1c1699079d7012f'})
    inventory = read(DOCS / 'amd-cached-neighbour-mixing-evidence/inventory-after.json')
    environments = {}
    for name, root in [('amd', Path('D:/AI/Trellis2AMD/venv')),
        ('modly', Path('C:/Users/wattz/Documents/Modly/extensions/trellis2/venv'))]:
        site = root / 'Lib/site-packages'
        # importlib.metadata reads dist-info text only; no installed modules execute.
        actual = [{'name': d.metadata['Name'], 'version': d.version,
            'metadata_sha256': textsha(d.read_text('METADATA')),
            'record_sha256': textsha(d.read_text('RECORD'))}
            for d in importlib.metadata.distributions(path=[str(site)])]
        expected = inventory[name]['all_distributions']
        sortkey = lambda row: (row['name'].lower(), row['version'])
        native = [identity(row['path'], row)
            for row in prior['windows_environments'][name]['native_binaries']]
        packages = {}
        canonical = lambda name: re.sub(r'[-_.]+', '-', name.lower())
        by_name = {canonical(d.metadata['Name']): d
            for d in importlib.metadata.distributions(path=[str(site)])}
        for package, row in inventory[name]['packages'].items():
            d = by_name.get(canonical(package))
            packages[package] = bool(d and d.version == row['version']
                and d.read_text('WHEEL') == row['wheel'])
        environments[name] = {'root': str(root), 'distribution_count': len(actual),
            'distributions_match': sorted(actual, key=sortkey) == sorted(expected, key=sortkey),
            'pyvenv_cfg_matches': (root / 'pyvenv.cfg').read_text(encoding='utf-8') == inventory[name]['pyvenv_cfg'],
            'selected_package_wheels_match': packages, 'native_binaries': native,
            'broad_recursive_metadata_inventory_performed': False}
    groups = {'protected': protected, 'source': source, 'saved_inputs': saved, 'preview': preview}
    failures = [row['path'] for rows in groups.values() for row in rows if not row['matches']]
    if not backup['matches']:
        failures.append(backup['path'])
    for name, env in environments.items():
        if not (env['distributions_match'] and env['pyvenv_cfg_matches']
            and all(env['selected_package_wheels_match'].values())
            and all(row['matches'] for row in env['native_binaries'])):
            failures.append('environment:' + name)
    return {'status': 'PASS' if not failures else 'FAIL',
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline_commit': '4f845cd306c839d0e833efe488de44b65d82584c',
        'comparison': 'Prior texture-route preservation and its exact preview/input references',
        'files': groups, 'AMD_source_backup': backup, 'windows_environments': environments,
        'unexpected_deltas': failures, 'AI_GPU_native_imports': False, 'WSL_started': False,
        'model_loads': 0, 'builds_or_installs': False,
        'method_note': 'File stability compares size and mtime_ns (access time may change on reads); package names use standard hyphen/underscore/dot normalization. Initial atime-sensitive diagnostic retained separately and superseded.',
        'limitations': 'Selected distribution/package/native metadata and protected-file identity; no broad venv tree scan or fresh WSL inventory. Model checkpoint payload verification belongs to the separately bounded CPU qualification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = capture()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'status': result['status'], 'groups': {k: len(v) for k, v in result['files'].items()},
        'unexpected_deltas': result['unexpected_deltas'], 'torch_imported': 'torch' in sys.modules}))
