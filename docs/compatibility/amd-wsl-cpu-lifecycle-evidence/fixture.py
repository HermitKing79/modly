"""One nonrecursive stdlib Linux CPU fixture; no model or authority producer."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import queue
import stat
import sys
import threading
import time

MEMORY_LIMIT = 6 * 1024**3
ROOT = Path(__file__).resolve().parents[3]


def need(value, message):
    if not value:
        raise ValueError(message)


def decode(blob):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value')
    return json.loads(blob, object_pairs_hook=unique, parse_constant=invalid)


def token(value):
    need(type(value) is str and 0 < len(value) <= 256 and '\0' not in value,
         'Bounded explicit control token required')
    return value


def plain(value):
    path = Path(value)
    need(path.is_absolute() and '..' not in path.parts, 'Explicit absolute private path required')
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            need(not item.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400,
                 'Redirected private path prohibited')
    need(path == path.resolve(), 'Private path alias prohibited')
    return path


def publish(path, row):
    path = plain(path)
    blob = (json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode()
    need(len(blob) <= 65536, 'Fixture JSON publication exceeds bound')
    staging = path.with_name(path.name + '.writing')
    with staging.open('xb') as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    # On failure preserve the original exception and durable staging evidence.
    os.link(staging, path)
    staging.unlink()


def read_json(path):
    path = plain(path)
    if not path.exists() or path.with_name(path.name + '.writing').exists():
        return None
    before = path.stat()
    need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= 65536,
         'Bounded single-link private JSON required')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        blob = stream.read(65537)
        after_open = os.fstat(stream.fileno())
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_nlink')
    need(len(blob) == before.st_size and all(getattr(row, key) == getattr(before, key)
         for row in (opened, after_open, after) for key in fields), 'Private JSON changed during read')
    return decode(blob)


def validate_runtime(row, expected_executable):
    need(type(row) is dict and row.get('uid') == row.get('euid') == 1000
         and row.get('user') == 'modly', 'Original Linux UID-1000 modly user required')
    need(row.get('executable') == expected_executable and row.get('rlimit_as') == [MEMORY_LIMIT, MEMORY_LIMIT],
         'Original Linux Python and inherited 6 GiB address-space limit required')
    pid, identity = row.get('pid'), row.get('identity', {})
    need(type(pid) is int and pid > 0 and identity.get('pid') == pid
         and identity.get('pgrp') == identity.get('session') == pid
         and identity.get('uid') == 1000 and type(identity.get('start_ticks')) is int
         and identity['start_ticks'] > 0 and type(identity.get('ppid')) is int and identity['ppid'] > 0,
         'Genuine original broker-led Linux session identity required')
    token(row.get('boot_id'))
    return row


def collect_runtime():
    need(sys.platform == 'linux', 'Linux fixture only')
    import pwd
    import resource
    source = ROOT / 'api/services/amd_wsl_session.py'
    spec = importlib.util.spec_from_file_location('_original_wsl_identity_parser', source)
    broker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broker)
    uid = os.getuid()
    executable = os.path.realpath('/proc/self/exe')
    expected = os.path.realpath('/usr/bin/python3')
    need(os.path.realpath(sys.executable) == expected, 'Original Linux interpreter selection changed')
    row = dict(pid=os.getpid(), uid=uid, euid=os.geteuid(), user=pwd.getpwuid(uid).pw_name,
        executable=executable, rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)),
        identity=broker._parse_stat(Path('/proc/self/stat').read_text(), uid),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    return validate_runtime(row, expected)


def validate_probe(row, binding):
    need(type(row) is dict and set(row) == {'action', 'context', 'stage_id', 'nonce', 'challenge'},
         'Exact host challenge fields required')
    challenge = token(row['challenge'])
    need(challenge != binding['nonce'] and row == dict(binding, action='PROBE', challenge=challenge),
         'Foreign or repeated-context-nonce host challenge')
    return challenge


def validate_release(row, binding, challenge):
    need(type(row) is dict and row == dict(binding, action='RELEASE', challenge=challenge),
         'Foreign or extra success-release fields')


def receive_cancel(stream, controls, binding):
    try:
        line = stream.readline(65537)
        need(bool(line) and len(line) <= 65536 and line.endswith(b'\n'), 'Control EOF or invalid bounded line')
        need(decode(line) == dict(action='CANCEL', stage_id=binding['stage_id'], nonce=binding['nonce']),
             'Foreign or extra cancellation control fields')
        controls.put('CANCEL')
    except Exception as error:
        controls.put(error)


def main(argv=None, *, stdin=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('target', 'sentinel'))
    parser.add_argument('leaf')
    args = parser.parse_args(argv)
    blob = os.environ['MODLY_COMPLETION_CONTEXT_JSON']
    need(len(blob) <= 65536, 'Product context exceeds bound')
    context = decode(blob)
    keys = {'job_id', 'request_id', 'run_id', 'nonce', 'input_sha256', 'params_sha256',
            'created_ns', 'mode', 'model_id'}
    need(type(context) is dict and set(context) == keys and type(context['created_ns']) is int
         and context['created_ns'] > 0, 'Complete original product context required')
    for key in keys - {'created_ns'}:
        token(context[key])
    binding = dict(context=context, stage_id=token(os.environ['MODLY_STAGE_ID']), nonce=context['nonce'])
    leaf = plain(args.leaf)
    need(leaf.is_dir(), 'Existing owned guest leaf required')
    ack_path = plain(os.environ['MODLY_STAGE_CANCEL_ACK'])
    deadline = time.monotonic() + 24
    def row(**fields):
        return dict(binding, mode=args.mode, **collect_runtime(), monotonic_s=time.monotonic(), **fields)
    publish(leaf / 'ready.json', row())
    controls = queue.Queue(maxsize=1)
    stream = sys.stdin.buffer if stdin is None else stdin
    threading.Thread(target=receive_cancel, args=(getattr(stream, 'raw', stream), controls, binding), daemon=True).start()
    accepted_probe = challenge = None
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError('Linux CPU fixture internal 24-second ceiling')
        try:
            control = controls.get(timeout=.025)
        except queue.Empty:
            control = None
        if isinstance(control, Exception):
            raise ValueError('Invalid or closed authenticated cancellation channel') from control
        if control == 'CANCEL':
            cancelled = row(terminal='cancelled')
            publish(ack_path, cancelled)
            need(read_json(ack_path) == cancelled, 'Durable cooperative cancellation acknowledgement missing')
            publish(leaf / 'completed.json', cancelled)
            return 0
        if args.mode != 'sentinel':
            continue
        release = read_json(leaf / 'release.json')
        if accepted_probe is None:
            need(release is None, 'Success release precedes authenticated alive response')
            probe = read_json(leaf / 'host-challenge.json')
            if probe is not None:
                challenge = validate_probe(probe, binding)
                accepted_probe = probe
                publish(leaf / 'alive-proof.json', row(action='PROBE', challenge=challenge))
        else:
            need(read_json(leaf / 'host-challenge.json') == accepted_probe, 'Accepted host challenge changed or disappeared')
            if release is not None:
                validate_release(release, binding, challenge)
                publish(leaf / 'success.json', row(action='RELEASE', challenge=challenge, terminal='succeeded'))
                return 0


if __name__ == '__main__':
    raise SystemExit(main())
