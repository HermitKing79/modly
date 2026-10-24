"""Read-only Windows physical-memory admission; never launches a decoder."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PRIVATE = Path('C:/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-zombie-ram-headroom-35b3c70')
REQUIRED = 22 * 1024**3


class Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in ('total', 'available', 'page_total', 'page_available',
                                            'virtual_total', 'virtual_available', 'extended')]


def main():
    if os.name != 'nt' or ctypes.windll.shell32.IsUserAnAdmin():
        raise RuntimeError('Normal non-elevated Windows user context required')
    samples = []
    for index in range(3):
        value = Memory(); value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise ctypes.WinError()
        samples.append(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
            available_physical_bytes=value.available, total_physical_bytes=value.total,
            memory_load_percent=value.load, required_available_bytes=REQUIRED,
            target_met=value.available >= REQUIRED))
        if index < 2:
            time.sleep(4)
    admitted = all(row['target_met'] for row in samples)
    consumers, errors = [], []
    if not admitted:
        import psutil
        for process in psutil.process_iter(['pid', 'name', 'memory_info'], ad_value=None):
            try:
                info = process.info; memory = info['memory_info']
                if memory is not None:
                    consumers.append(dict(pid=info['pid'], name=info['name'],
                        working_set_bytes=memory.rss, private_commit_bytes=getattr(memory, 'private', None)))
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                errors.append(type(exc).__name__)
        consumers.sort(key=lambda row: row['working_set_bytes'], reverse=True)
    row = dict(classification='READY_FOR_FURTHER_ADMISSION' if admitted else 'NO-GO: INSUFFICIENT_HEADROOM',
        samples=samples, minimum_available_bytes=min(s['available_physical_bytes'] for s in samples),
        maximum_available_bytes=max(s['available_physical_bytes'] for s in samples),
        available_span_bytes=max(s['available_physical_bytes'] for s in samples)-min(s['available_physical_bytes'] for s in samples),
        all_three_meet_22GiB=admitted, sample_interval_seconds=4,
        previous_observed_decline_bytes=6700421120, previous_decline_is_not_a_guarantee=True,
        top_process_working_sets=consumers[:20], process_query_errors=errors,
        process_memory_note='Working sets may share pages; private commit is not resident physical RAM. Do not sum as exclusive physical consumption.',
        executable=sys.executable, python=sys.version, administrator_role_enabled=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        WSL_launched=False, model_loaded=False, neural_attempt_reserved=False, neural_forwards=0,
        processes_terminated=False, settings_changed=False)
    with (PRIVATE / 'memory-readiness-private.json').open('x', encoding='utf-8') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({key: row[key] for key in ('classification', 'samples', 'available_span_bytes')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
