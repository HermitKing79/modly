"""Read-only PolyFit 1.6.0 qualification. No installer or native execution path."""
import argparse
import base64
import csv
import hashlib
import io
import json
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / 'artifacts/amd-polyfit-global-proof-20261030-v1'
WHEEL = BASE / 'downloads/polyfit-1.6.0-cp312-cp312-win_amd64.whl'
WHEEL_SHA = 'b16d456c39785886832f0d419a4d6e1f876a26c657d44c5cd1be02ba66e4424f'


def inspect_wheel(path, expected_sha):
    """Inspect bytes only; an internally consistent wheel is not a licence grant."""
    with Path(path).open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != expected_sha:
            raise ValueError('Whole-wheel SHA-256 differs')
        f.seek(0)
        with zipfile.ZipFile(f) as z:
            entries = z.infolist()
            if len(entries) > 64 or sum(e.file_size for e in entries) > 100 * 1024**2:
                raise ValueError('Bounded release inventory required')
            names = set(); folded = set()
            for e in entries:
                p = PurePosixPath(e.filename)
                if (p.is_absolute() or '..' in p.parts or '\\' in e.filename
                        or ':' in e.filename or e.is_dir()
                        or any(x.endswith((' ', '.')) for x in p.parts)
                        or str(p) != e.filename):
                    raise ValueError('Unsafe archive path')
                if e.filename in names or e.filename.casefold() in folded:
                    raise ValueError('Archive path collision')
                if stat.S_ISLNK(e.external_attr >> 16):
                    raise ValueError('Redirected archive path')
                names.add(e.filename); folded.add(e.filename.casefold())
            record = 'polyfit-1.6.0.dist-info/RECORD'
            rows = list(csv.reader(io.StringIO(z.read(record).decode('utf8'))))
            indexed = {}
            for row in rows:
                if len(row) != 3 or row[0] in indexed:
                    raise ValueError('Malformed or duplicate RECORD row')
                indexed[row[0]] = row[1:]
            if set(indexed) != names or indexed[record] != ['', '']:
                raise ValueError('Incomplete RECORD inventory')
            inventory = {}; markers = []
            for name in sorted(names):
                payload = z.read(name); h = hashlib.sha256(payload).hexdigest()
                inventory[name] = dict(bytes=len(payload), sha256=h)
                if name != record:
                    expected_hash, expected_size = indexed[name]
                    encoded = base64.urlsafe_b64encode(bytes.fromhex(h)).rstrip(b'=').decode()
                    if expected_hash != 'sha256=' + encoded or expected_size != str(len(payload)):
                        raise ValueError('Embedded RECORD differs: ' + name)
                if name == 'polyfit/math.dll':
                    for marker in (b'Soplex 3.1.1', b'code\\3rd_party\\soplex\\src\\clufactor.cpp'):
                        if marker in payload:
                            markers.append(marker.decode('ascii'))
            return dict(whole_wheel_sha256=expected_sha, file_count=len(names),
                        expanded_bytes=sum(e.file_size for e in entries), files=inventory,
                        math_DLL_markers=markers, old_soplex_marker_present='Soplex 3.1.1' in markers,
                        native_code_executed=False, installed=False,
                        publisher_signature_or_digest_available=False,
                        integrity_scope='Official HTTPS asset and local SHA-256 plus embedded RECORD; not a reproducible-build attestation')


def license_gate(old_soplex, applicable_grant_evidenced):
    if old_soplex and not applicable_grant_evidenced:
        return dict(decision='NO_GO_BEFORE_INSTALL_OR_NATIVE_EXECUTION', example_authorized=False,
                    reason='Actual wheel identifies SoPlex 3.1.1; applicable ZIB Academic Licence grant not evidenced')
    return dict(decision='UNQUALIFIED_NOT_AN_EXECUTION_AUTHORIZATION', example_authorized=False)


def worker(stage):
    if sys.stdin.buffer.read(4) != b'GO\n':
        raise RuntimeError('Exact owned GO plus EOF required')
    from amd_zombie_d_compare import audit_policy
    sys.addaudithook(audit_policy(BASE))
    if stage == 'authenticate':
        import amd_ai_hard_surface_assessment as old
        old.BASE = BASE
        sys.path.insert(0, 'D:/AI/Trellis2AMD/venv/Lib/site-packages')
        old.authenticate()
    elif stage == 'release-audit':
        import amd_ai_hard_surface_assessment as old
        frozen = old.criteria()
        if json.loads((BASE / 'frozen-phase11-criteria.json').read_text()) != frozen:
            raise ValueError('Frozen criteria differs')
        result = inspect_wheel(WHEEL, WHEEL_SHA)
        result['licence_qualification'] = license_gate(result['old_soplex_marker_present'], False)
        old.save(BASE / 'release-audit.json', result)
    else:
        raise ValueError('Unknown read-only stage')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('authenticate', 'release-audit'))
    parser.add_argument('--worker', choices=('authenticate', 'release-audit'))
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.worker); return 0
    if args.stage:
        import amd_zombie_cpu_mesh as owner
        runs = BASE / 'qualification-stages'
        runs.mkdir(exist_ok=True)
        owner.BASE = runs; owner.__file__ = str(Path(__file__).resolve())
        return owner.supervise(args.stage)
    print('INERT: qualification only; no environment install, PolyFit import or solver entry')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
