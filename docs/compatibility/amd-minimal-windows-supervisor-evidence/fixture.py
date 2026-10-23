"""Disposable stdlib-only control worker; never creates processes or native Jobs.

job_nonce is a host-provided stage-launch token, not an observed Windows Job
identity. The unchanged production runner establishes native ownership.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import stat
import sys
import threading
import time


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result


def decode(blob):
    def nonfinite(value):
        raise ValueError('Nonfinite JSON value')
    return json.loads(blob, object_pairs_hook=unique, parse_constant=nonfinite)


def token(value):
    if type(value) is not str or not 0 < len(value) <= 256 or '\0' in value:
        raise ValueError('Bounded explicit stage token required')
    return value


def plain_path(value, *, existing=False):
    path = Path(value).absolute()
    if str(path).replace('\\', '/').startswith('//'):
        raise ValueError('Local private path required')
    for entry in (path, *path.parents):
        if entry.exists() or entry.is_symlink():
            info = entry.lstat()
            if entry.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Redirected private path prohibited')
    if path != path.resolve(strict=existing):
        raise ValueError('Aliased private path prohibited')
    return path


def publish(path, value):
    """Flush an exclusive temporary file, then atomically link without replacing."""
    path = plain_path(path)
    if not path.parent.is_dir():
        raise ValueError('Existing owned publication parent required')
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.pending')
    blob = (json.dumps(value, allow_nan=False, sort_keys=True) + '\n').encode()
    if len(blob) > 65536:
        raise ValueError('Publication exceeds JSON bound')
    with temporary.open('xb') as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    # A failed link retains its durable staging evidence and original exception.
    temporary.unlink()


def read_cancel(stream, controls, stage_id, nonce):
    """One real bootstrap control line; EOF and foreign messages are errors."""
    try:
        line = stream.readline(65537)
        if not line or len(line) > 65536 or not line.endswith(b'\n'):
            raise ValueError('Control EOF or invalid bounded line')
        value = decode(line)
        if value != dict(action='CANCEL', stage_id=stage_id, nonce=nonce):
            raise ValueError('Foreign or extra stage control fields')
        controls.put('CANCEL')
    except Exception as error:
        controls.put(error)


def release_requested(path, expected):
    if not path.exists():
        return False
    # The qualified publisher links the final name before unlinking this exact
    # sibling. Wait for publication completion; retain the single-link check.
    if path.with_name(path.name + '.writing').exists():
        return False
    path = plain_path(path, existing=True)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= 65536:
        raise ValueError('Invalid bounded private release file')
    with path.open('rb') as stream:
        blob = stream.read(65537)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_nlink')
    if (len(blob) != before.st_size or any(getattr(before, key) != getattr(after, key) for key in fields)
            or decode(blob) != expected):
        raise ValueError('Foreign or unstable success release')
    return True


def main(argv=None, *, stdin=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('target', 'sentinel'), required=True)
    parser.add_argument('--private-leaf', required=True)
    parser.add_argument('--release-nonce', required=True)
    parser.add_argument('--job-nonce', required=True)
    args = parser.parse_args(argv)
    context_blob = os.environ['MODLY_COMPLETION_CONTEXT_JSON']
    if len(context_blob) > 65536:
        raise ValueError('Context exceeds JSON bound')
    context = decode(context_blob)
    keys = {'job_id', 'request_id', 'run_id', 'nonce', 'input_sha256', 'params_sha256',
            'created_ns', 'mode', 'model_id'}
    if (type(context) is not dict or set(context) != keys
            or type(context['created_ns']) is not int or context['created_ns'] <= 0):
        raise ValueError('Complete original product context required')
    for key in keys - {'created_ns'}:
        token(context[key])
    stage_id = token(os.environ['MODLY_STAGE_ID'])
    release_nonce, job_nonce = token(args.release_nonce), token(args.job_nonce)
    if len({context['nonce'], release_nonce, job_nonce}) != 3:
        raise ValueError('Independent context, release and stage-launch tokens required')
    leaf = plain_path(args.private_leaf, existing=True)
    if not leaf.is_dir():
        raise ValueError('Existing private worker directory required')
    ack = plain_path(os.environ['MODLY_STAGE_CANCEL_ACK'])
    identity = dict(context=context, stage_id=stage_id, role=args.role, pid=os.getpid(),
                    nonce=context['nonce'], release_nonce=release_nonce, job_nonce=job_nonce)
    publish(leaf / 'ready.json', dict(identity, monotonic_s=time.monotonic()))
    controls = queue.Queue(maxsize=1)
    stream = sys.stdin.buffer if stdin is None else stdin
    stream = getattr(stream, 'raw', stream)
    threading.Thread(target=read_cancel, args=(stream, controls, stage_id, context['nonce']), daemon=True).start()
    expected_release = dict(action='RELEASE', context=context, stage_id=stage_id, pid=identity['pid'],
                            release_nonce=release_nonce, job_nonce=job_nonce)
    while True:
        try:
            control = controls.get(timeout=.025)
        except queue.Empty:
            control = None
        if isinstance(control, Exception):
            raise ValueError('Invalid or closed production control channel') from control
        if control == 'CANCEL':
            # Preserve the bootstrap's exact terminal/context/stage_id/nonce fields.
            publish(ack, dict(identity, terminal='cancelled', monotonic_s=time.monotonic()))
            return 0
        if args.role == 'sentinel' and release_requested(leaf / 'release.json', expected_release):
            publish(leaf / 'released.json', dict(identity, terminal='succeeded', action='RELEASE',
                                                monotonic_s=time.monotonic()))
            return 0


if __name__ == '__main__':
    raise SystemExit(main())
