"""One nonrecursive synthetic upstream output. No model or stage chain."""
import json
import os
from pathlib import Path
import sys
import time


def publish(path, value):
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('xb') as stream:
        stream.write((json.dumps(value, sort_keys=True)+'\n').encode())
        stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def main(leaf):
    leaf = Path(leaf)
    context = json.loads(os.environ['MODLY_COMPLETION_CONTEXT_JSON'])
    value = dict(context=context, stage_id=os.environ['MODLY_STAGE_ID'],
                 nonce=os.environ['MODLY_STAGE_CONTROL_NONCE'], synthetic=True)
    if value['stage_id'] != 'conditioning' or context['nonce'] == value['nonce']:
        raise ValueError('Independent original product context and ticket nonce required')
    publish(leaf/'ready.json', dict(value, pid=os.getpid()))
    end = time.monotonic()+20  # Remains inside the unchanged 25-second stage deadline.
    while not (leaf/'release.json').exists() or (leaf/'release.json.writing').exists():
        if time.monotonic() >= end: raise TimeoutError('Host native-identity release missing')
        time.sleep(.01)
    if json.loads((leaf/'release.json').read_text()) != value: raise ValueError('Foreign host release')
    publish(leaf/'synthetic-output.json',dict(value,value='synthetic conditioning transport only; no model evidence'))
    return 0


if __name__ == '__main__': raise SystemExit(main(sys.argv[1]))
