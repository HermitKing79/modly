"""Stdlib-only Windows launch gate, invoked with -I -S by the private runner."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback


def guest_path(value):
    value = str(value).replace("\\", "/")
    if value.startswith("/") and not value.startswith("//"):
        return value
    if not value.lower().startswith("c:/"):
        raise ValueError("WSL shared files must remain on the C: volume")
    return "/mnt/c/" + value[3:]


def valid_control(message, spec):
    if not isinstance(message, dict) or (message.get("stage_id"), message.get("nonce")) != (
            spec["stage_id"], spec["nonce"]):
        raise ValueError("Control message does not match stage/nonce")
    if message.get("action") not in ("GO", "START", "PING", "CANCEL"):
        raise ValueError("Unknown stage control action")
    return message["action"]


_ANCHOR = dict(before_ns=time.monotonic_ns(), utc_ns=time.time_ns(), after_ns=time.monotonic_ns())


_EMIT_STATS = dict(count=0, total_ns=0, max_ns=0)
_TIMING_CONTEXT = None


def emit(spec, **fields):
    global _TIMING_CONTEXT
    _TIMING_CONTEXT = dict(stage_id=spec['stage_id'], nonce=spec['nonce'], context=spec['context'])
    began_ns = time.monotonic_ns()
    fields.setdefault('monotonic_ns', began_ns)
    fields.setdefault('clock_anchor', _ANCHOR)
    try:
        print(json.dumps({"stage_id": spec["stage_id"], "nonce": spec["nonce"],
                          "context": spec["context"], **fields}), flush=True)
    finally:
        elapsed_ns = time.monotonic_ns() - began_ns
        _EMIT_STATS['count'] += 1
        _EMIT_STATS['total_ns'] += elapsed_ns
        _EMIT_STATS['max_ns'] = max(_EMIT_STATS['max_ns'], elapsed_ns)


def save_timing():
    # One bounded shutdown sidecar, not a per-protocol-event disk write.
    row = dict(schema='modly.fixture-timing.bootstrap.v1', pid=os.getpid(),
               context=_TIMING_CONTEXT, clock_anchor=_ANCHOR, emit=dict(_EMIT_STATS),
               final_summary_begin_ns=time.monotonic_ns(), final_write_tail_ns=None)
    path = Path.cwd() / ('bootstrap-timing-' + str(os.getpid()) + '.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(row, stream); stream.flush(); os.fsync(stream.fileno())


def read_control(stream, spec):
    line = stream.readline(65537)
    if not line or len(line) > 65536 or not line.endswith(b"\n"):
        raise ValueError("Stage control closed or exceeded limit")
    return valid_control(json.loads(line), spec)


def native_stage(spec):
    controls = queue.Queue(maxsize=64)

    def reader():
        try:
            while True:
                controls.put(read_control(sys.stdin.buffer.raw, spec))
        except Exception:
            controls.put("CANCEL")

    env = os.environ.copy()
    env.update(MODLY_COMPLETION_CONTEXT_JSON=json.dumps(spec["context"]), MODLY_STAGE_ID=spec["stage_id"],
               PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cancel_ack = str(Path(spec["stdout_file"]).with_name("cancel-ack.json"))
    env["MODLY_STAGE_CANCEL_ACK"] = cancel_ack
    forced = cancelled = False
    cancel_at = None
    last_host = time.monotonic()
    deadline = last_host + spec["timeout_s"]
    with open(spec["stdout_file"], "xb", buffering=0) as stdout, open(spec["stderr_file"], "xb", buffering=0) as stderr:
        child = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env, stdin=subprocess.PIPE,
                                 stdout=stdout, stderr=stderr, close_fds=True,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=reader, daemon=True).start()
        emit(spec, event="stage-started", pid=child.pid, timestamp_ns=time.time_ns())
        while child.poll() is None:
            now = time.monotonic()
            try:
                action = controls.get(timeout=.025)
            except queue.Empty:
                action = None
            if action is not None:
                emit(spec, event='control-received', action=action, child_pid=child.pid, timestamp_ns=time.time_ns())
                last_host = now
            if now >= deadline or now - last_host > 15:
                action = "CANCEL"
            if action == "CANCEL" and cancel_at is None:
                cancelled, cancel_at = True, now
                try:
                    child.stdin.write((json.dumps({"action": "CANCEL", "stage_id": spec["stage_id"],
                                                  "nonce": spec["nonce"]}) + "\n").encode())
                    child.stdin.flush()
                except (BrokenPipeError, OSError) as error:
                    # Private stderr/protocol, without hiding the primary error.
                    traceback.print_exc()
                    emit(spec, event='control-error', operation='child.stdin.CANCEL.write_flush',
                         child_pid=child.pid, winerror=getattr(error, 'winerror', None), timestamp_ns=time.time_ns())
            if cancel_at is not None and now - cancel_at >= 3:
                forced = True
                child.kill()
                child.wait(timeout=2)
        emit(spec, event='child-exited', child_pid=child.pid, returncode=child.returncode,
             cancellation_sent=cancelled, forced=forced, timestamp_ns=time.time_ns())
        child.stdin.close()
        returncode = child.wait(timeout=2)
    cooperative = False
    if cancelled and not forced:
        try:
            ack = json.loads(Path(cancel_ack).read_text(encoding="utf-8"))
            cooperative = (ack.get("terminal") == "cancelled" and ack.get("context") == spec["context"]
                           and ack.get("stage_id") == spec["stage_id"] and ack.get("nonce") == spec["nonce"])
        except (OSError, ValueError) as error:
            traceback.print_exc()
            emit(spec, event='ack-error', operation='cancel-ack.read_json',
                 winerror=getattr(error, 'winerror', None), timestamp_ns=time.time_ns())
        emit(spec, event='ack-checked', cooperative=cooperative, timestamp_ns=time.time_ns())
    emit(spec, event="terminal", terminal="cancelled" if cancelled else "succeeded" if returncode == 0 else "failed",
         returncode=returncode, forced=forced, natural=not forced and (not cancelled or cooperative),
         cooperative_cancelled=cooperative, timestamp_ns=time.time_ns())
    return 0


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    action = read_control(sys.stdin.buffer.raw, spec)
    received_ns = time.monotonic_ns()
    emit(spec, received_ns=received_ns, event='control-received', action=action, timestamp_ns=time.time_ns())
    if action != "GO":
        emit(spec, event="terminal", terminal="cancelled", returncode=None, forced=False, natural=True,
             linux_settled=True, reaped_all=True, owned_remaining=[], no_stage_launched=True)
        return 0
    if spec["kind"] == "windows":
        return native_stage(spec)
    raise RuntimeError("Diagnostic copy permits Windows stages only")
    if spec["kind"] != "wsl":
        raise ValueError("Unknown stage backend")
    helper = guest_path(Path(__file__).with_name("amd_wsl_session.py"))
    # The broker creates a second, genuine Linux session gate. Do not buffer or
    # consume further stdin here: the broker receives START/PING/CANCEL directly.
    wsl = str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/wsl.exe")
    child = subprocess.Popen([wsl, "-d", "Modly-AMD-Ubuntu-24.04", "-u", "modly", "--exec",
                              "/usr/bin/python3", "-B", "-I", "-S", helper, guest_path(spec_path)],
                             stdin=sys.stdin.buffer, stdout=sys.stdout.buffer, stderr=sys.stderr.buffer,
                             close_fds=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    result = child.wait()
    emit(spec, event="linux-client-exited", returncode=result, pid=child.pid, timestamp_ns=time.time_ns())
    return result


if __name__ == "__main__":
    primary = None
    try:
        status = main(sys.argv[1])
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            save_timing()
        except BaseException:
            if primary is None:
                raise
            traceback.print_exc()
            if hasattr(primary, 'add_note'):
                primary.add_note('Bootstrap timing persistence failed; evidence incomplete')
    raise SystemExit(status)
