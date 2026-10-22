"""Disposable stdlib lifecycle payload. No model, mesh or generated asset."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib
import queue
import threading


def save(path, row):
    path = Path(path)
    staging = path.with_name(path.name + '.writing')
    with staging.open('x', encoding='utf-8') as stream:
        json.dump(row, stream); stream.flush(); os.fsync(stream.fileno())
    os.link(staging, path); staging.unlink()


def main(mode):
    context = json.loads(os.environ['MODLY_COMPLETION_CONTEXT_JSON'])
    stage = os.environ['MODLY_STAGE_ID']
    base = Path.cwd()
    script = str(Path(__file__).resolve())
    def stopped():
        if not (base / 'stop.json').exists(): return False
        row = json.loads((base / 'stop.json').read_text())
        if row != {'context': context, 'stage_id': stage}: raise RuntimeError('Foreign stop token')
        return True
    if mode == 'grandchild':
        save(base / 'grandchild-ready.json', {'pid': os.getpid(), 'context': context})
        deadline = time.monotonic() + 30
        while not stopped():
            if time.monotonic() >= deadline: raise TimeoutError('Fixture child deadline')
            time.sleep(.02)
        return
    if mode in ('child', 'daemonizer'):
        child = subprocess.Popen([sys.executable, '-B', '-I', '-S', script, 'grandchild'],
                                 start_new_session=mode == 'daemonizer', close_fds=True)
        save(base / 'child-ready.json', {'pid': os.getpid(), 'grandchild': child.pid, 'context': context})
        if mode == 'child':
            if child.wait(timeout=32) != 0: raise RuntimeError('Grandchild failed')
        return
    if mode not in ('success', 'wait', 'sentinel'): raise ValueError('Exact fixture mode required')
    child_mode = 'daemonizer' if sys.platform == 'linux' else 'child'
    child = subprocess.Popen([sys.executable, '-B', '-I', '-S', script, child_mode], close_fds=True)
    until = time.monotonic() + 5
    while not all((base / name).exists() for name in ('child-ready.json', 'grandchild-ready.json')):
        if time.monotonic() >= until: raise TimeoutError('Fixture descendants did not start')
        time.sleep(.02)
    descendants = json.loads((base / 'child-ready.json').read_text())
    save(base / 'ready.json', {'pid': os.getpid(), 'context': context, 'stage_id': stage,
                             'descendants': descendants, 'synthetic_only': True})
    cancelled = False
    if mode == 'wait':
        message = json.loads(sys.stdin.buffer.readline(65537))
        if message != {'action': 'CANCEL', 'stage_id': stage, 'nonce': context['nonce']}:
            raise ValueError('Fixture requires exact cancellation')
        cancelled = True
    elif mode == 'sentinel':
        controls = queue.Queue()
        def read_cancel():
            try: controls.put(json.loads(sys.stdin.buffer.raw.readline(65537)))
            except Exception: controls.put(None)
        threading.Thread(target=read_cancel, daemon=True).start()
        until = time.monotonic() + 20
        while not (base / 'sentinel-release.json').exists():
            if time.monotonic() >= until: raise TimeoutError('Sentinel release deadline')
            try:
                message = controls.get_nowait()
                if message != {'action': 'CANCEL', 'stage_id': stage, 'nonce': context['nonce']}: raise RuntimeError('Foreign sentinel cancellation')
                cancelled = True
                break
            except queue.Empty: pass
            time.sleep(.02)
        if not cancelled and json.loads((base / 'sentinel-release.json').read_text()) != {'context': context, 'stage_id': stage}:
            raise RuntimeError('Foreign sentinel release')
    else:
        time.sleep(.25)
    save(base / 'stop.json', {'context': context, 'stage_id': stage})
    if child.wait(timeout=4) != 0: raise RuntimeError('Fixture child failed')
    # On Linux the daemonized grandchild is reaped by the broker, not this parent.
    if cancelled:
        save(os.environ['MODLY_STAGE_CANCEL_ACK'], dict(terminal='cancelled', context=context,
                                                       stage_id=stage, nonce=context['nonce']))
    if not cancelled and (base / 'stage-request.json').exists():
        request = json.loads((base / 'stage-request.json').read_text())
        output = base / 'output.json'
        save(output, {'product_context': context, 'synthetic_only': True, 'stage_id': stage})
        save(base / 'stage-result.json', dict(schema='modly.amd.stage-receipt.v1', mode='cpu_lifecycle_test',
            product_context=context, stage_id=stage, status='PASS', predecessor_sha256=request['predecessor_sha256'],
            inputs=request['inputs'], source_sha256=request['source_sha256'],
            outputs=[dict(relative_path='output.json', sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                          bytes=output.stat().st_size, kind='json', schema={})]))


if __name__ == '__main__': main(sys.argv[1])
