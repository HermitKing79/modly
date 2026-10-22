"""Stdlib Linux ownership broker for one host-pinned, non-adversarial stage.

No distro/global signalling is used. Subreaper ancestry retains daemonized
descendants; pidfds target exact identities. RLIMIT_AS is inherited per process
and aggregate RSS is sampled, rather than an atomic cgroup memory limit.
"""
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import sys
import time

MEMORY_LIMIT = 6 * 1024**3


def _guest_path(value):
    value = str(value).replace("\\", "/")
    if value.startswith("/") and not value.startswith("//"):
        return value
    if not value.lower().startswith("c:/"):
        raise ValueError("Guest shared files must remain on the C: volume")
    return "/mnt/c/" + value[3:]


def _parse_stat(text, uid):
    end = text.rindex(")")
    fields = text[end + 2:].split()
    return {"pid": int(text.split(" ", 1)[0]), "state": fields[0], "ppid": int(fields[1]),
            "pgrp": int(fields[2]), "session": int(fields[3]), "start_ticks": int(fields[19]),
            "rss_pages": int(fields[21]), "uid": uid}


def _read_identity(pid):
    directory = Path("/proc") / str(pid)
    uid = directory.stat().st_uid
    return _parse_stat((directory / "stat").read_text(), uid)


def _snapshot():
    result = {}
    for item in Path("/proc").iterdir():
        if item.name.isdecimal():
            try:
                row = _read_identity(int(item.name))
                result[row["pid"]] = row
            except (FileNotFoundError, ProcessLookupError):
                pass
            except PermissionError:
                # Own-UID descendants must remain inspectable. Foreign procfs
                # restrictions are irrelevant to this broker's ancestry.
                if item.stat().st_uid == 1000:
                    raise
    return result


def _owned_rows(rows, broker_pid, session):
    # Session numbers can be reused after their last member exits. Ancestry to
    # the still-live subreaper is authority; a numeric SID alone never is.
    owned = {}
    parents = {broker_pid}
    changed = True
    while changed:
        changed = False
        for pid, row in rows.items():
            if pid != broker_pid and pid not in owned and row["ppid"] in parents:
                owned[pid] = row
                parents.add(pid)
                changed = True
    if any(row["uid"] != 1000 for row in owned.values()):
        raise RuntimeError("Owned descendant changed UID; settlement cannot be verified")
    return owned


def _key(row):
    return row["pid"], row["start_ticks"], row["uid"]


def _signal_exact(row, sig, broker=None):
    handles = []
    try:
        if _key(_read_identity(row["pid"])) != _key(row):
            raise RuntimeError("Owned process identity changed before signalling")
        handle = os.pidfd_open(row["pid"], 0)
        handles.append(handle)
        if _key(_read_identity(row["pid"])) != _key(row):
            raise RuntimeError("Owned process identity changed after pidfd acquisition")
        broker = broker or _read_identity(os.getpid())
        current = _read_identity(row["pid"])
        chain = [current]
        while current["pid"] != broker["pid"]:
            if current.get("ppid", 0) <= 1 or len(chain) >= 256:
                raise RuntimeError("Live process ancestry does not reach owned broker")
            parent = _read_identity(current["ppid"])
            parent_handle = os.pidfd_open(parent["pid"], 0)
            handles.append(parent_handle)
            # Pin every ancestor and establish each link while that exact
            # parent is still alive. Numeric PPIDs from the earlier snapshot
            # alone cannot authorize a target after intermediate PID reuse.
            again = _read_identity(current["pid"])
            if (_key(again) != _key(current) or again["ppid"] != parent["pid"]
                    or _key(_read_identity(parent["pid"])) != _key(parent)
                    or select.select([parent_handle], [], [], 0)[0]):
                raise RuntimeError("Live process ancestry changed during ownership proof")
            current = parent
            chain.append(current)
        if _key(current) != _key(broker):
            raise RuntimeError("Live broker identity changed during ancestry proof")
        # A second coherent pass rejects reparenting/exits while the first
        # pass walked upward. Ordinary reparenting after this proof preserves
        # ownership under the still-live subreaper.
        for index, expected in enumerate(chain):
            again = _read_identity(expected["pid"])
            if (_key(again) != _key(expected) or index + 1 < len(chain)
                    and again["ppid"] != chain[index + 1]["pid"]
                    or select.select([handles[index]], [], [], 0)[0]):
                raise RuntimeError("Live process ancestry changed before exact signal")
        signal.pidfd_send_signal(handle, sig)
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False
    finally:
        for handle in reversed(handles):
            os.close(handle)


def _configure():
    if sys.platform != "linux" or os.getuid() != 1000 or os.geteuid() != 1000:
        raise RuntimeError("Broker requires Linux UID 1000")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Exact pidfd process ownership is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Could not establish child subreaper")
    current = ctypes.c_int()
    # GET takes a pointer; keep it distinct from SET's integer argtypes.
    libc.prctl.argtypes = None
    if libc.prctl(37, ctypes.byref(current), 0, 0, 0) or current.value != 1:
        raise RuntimeError("Child subreaper state could not be verified")
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if resource.getrlimit(resource.RLIMIT_AS) != (MEMORY_LIMIT, MEMORY_LIMIT):
        raise RuntimeError("Linux address-space limit could not be verified")
    handle = os.pidfd_open(os.getpid(), 0)
    os.close(handle)


def _emit(spec, **fields):
    print(json.dumps({"stage_id": spec["stage_id"], "nonce": spec["nonce"],
                      "context": spec["context"], **fields}), flush=True)


def run(spec):
    _configure()  # No child exists if capability qualification fails.
    broker = _read_identity(os.getpid())
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    stdout_path, stderr_path = _guest_path(spec["stdout_file"]), _guest_path(spec["stderr_file"])
    cancel_ack = str(Path(stdout_path).with_name("cancel-ack.json"))
    env = os.environ.copy()
    env.update(MODLY_COMPLETION_CONTEXT_JSON=json.dumps(spec["context"]), MODLY_STAGE_ID=spec["stage_id"],
               MODLY_STAGE_CANCEL_ACK=cancel_ack, PYTHONDONTWRITEBYTECODE="1",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    gate_r, gate_w = os.pipe()
    stdin_r, stdin_w = os.pipe()
    began = time.monotonic()
    deadline = began + min(float(spec["timeout_s"]), 180)
    # Log descriptors are private to the owned child and its descendants.
    with open(stdout_path, "xb", buffering=0) as stdout, open(stderr_path, "xb", buffering=0) as stderr:
        pid = os.fork()
        if pid == 0:
            try:
                os.close(gate_w)
                os.close(stdin_w)
                os.setsid()
                os.dup2(stdin_r, 0)
                os.dup2(stdout.fileno(), 1)
                os.dup2(stderr.fileno(), 2)
                os.close(stdin_r)
                if os.read(gate_r, 1) != b"G":
                    os._exit(125)
                os.close(gate_r)
                os.chdir(_guest_path(spec["cwd"]))
                os.execvpe(spec["argv"][0], spec["argv"], env)
            except BaseException as exc:
                os.write(2, ("Stage exec failed: " + str(exc) + "\n").encode())
                os._exit(126)
        os.close(gate_r)
        os.close(stdin_r)
        leader = None
        while time.monotonic() < min(deadline, began + 2):
            leader = _read_identity(pid)
            if leader["session"] == pid:
                break
            time.sleep(.005)
        if leader is None or leader["session"] != pid:
            # The child cannot execute without this broker's still-closed gate.
            os.close(gate_w)
            os.close(stdin_w)
            raise RuntimeError("Gated child Linux session could not be established")
        memory = {"rlimit_as_bytes": MEMORY_LIMIT, "aggregate_rss_limit_bytes": MEMORY_LIMIT,
                  "aggregate_rss_peak_bytes": 0, "atomic_cgroup_limit": False}
        _emit(spec, event="ready", uid=1000, broker=broker, leader=leader, boot_id=boot_id, memory=memory)
        last_host = time.monotonic()
        buffer = b""
        gate_open = False
        cancelled = forced = reaped_all = False
        cancel_at = leader_exit_at = None
        returncode = None
        errors = []
        identities = {}
        statuses = []
        remaining = {}
        stdin_open = True
        cleanup_deadline = deadline + 6
        try:
            while True:
                now = time.monotonic()
                # waitpid ECHILD is essential: session emptiness alone misses
                # descendants that created a new session or double-forked.
                reaped_all = False
                while True:
                    try:
                        found, status = os.waitpid(-1, os.WNOHANG)
                    except ChildProcessError:
                        reaped_all = True
                        break
                    if not found:
                        break
                    statuses.append({"pid": found, "status": status, "timestamp_ns": time.time_ns()})
                    if found == pid:
                        returncode = os.waitstatus_to_exitcode(status)
                        leader_exit_at = now
                remaining = _owned_rows(_snapshot(), broker["pid"], leader["session"])
                for row in remaining.values():
                    identities[_key(row)] = row
                rss = sum(row["rss_pages"] for row in remaining.values()) * os.sysconf("SC_PAGE_SIZE")
                rss += _read_identity(broker["pid"])["rss_pages"] * os.sysconf("SC_PAGE_SIZE")
                memory["aggregate_rss_peak_bytes"] = max(memory["aggregate_rss_peak_bytes"], rss)
                if rss > MEMORY_LIMIT:
                    errors.append("Aggregate owned RSS exceeded Linux sampled limit")
                    cancelled = True
                if reaped_all and not remaining:
                    break
                if now >= deadline or now - last_host > 15:
                    cancelled = True
                if leader_exit_at is not None and remaining and now - leader_exit_at > .25:
                    errors.append("Leader exited while owned descendants remained")
                    cancelled = True
                if stdin_open and select.select([0], [], [], .025)[0]:
                    chunk = os.read(0, 4096)
                    if not chunk:
                        stdin_open, cancelled = False, True
                    buffer += chunk
                    if len(buffer) > 65536:
                        raise RuntimeError("Host control exceeded bounded size")
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        message = json.loads(line)
                        if (message.get("stage_id"), message.get("nonce")) != (spec["stage_id"], spec["nonce"]):
                            raise RuntimeError("Linux control stage/nonce mismatch")
                        last_host = now
                        action = message.get("action")
                        if action == "START" and not gate_open and not cancelled:
                            os.write(gate_w, b"G")
                            os.close(gate_w)
                            gate_w = None
                            gate_open = True
                        elif action == "CANCEL":
                            cancelled = True
                        elif action not in ("PING", "START"):
                            raise RuntimeError("Unexpected Linux control action")
                if cancelled and cancel_at is None:
                    cancel_at = now
                    if not gate_open:
                        os.close(gate_w)
                        gate_w = None
                    else:
                        try:
                            os.write(stdin_w, (json.dumps({"action": "CANCEL", "stage_id": spec["stage_id"],
                                                          "nonce": spec["nonce"]}) + "\n").encode())
                        except BrokenPipeError:
                            pass
                if cancel_at is not None and now - cancel_at >= 3:
                    sig = signal.SIGKILL if now - cancel_at >= 4 else signal.SIGTERM
                    for row in remaining.values():
                        if row["state"] != "Z":
                            forced = _signal_exact(row, sig, broker) or forced
                if now >= cleanup_deadline or cancel_at is not None and now - cancel_at >= 6:
                    errors.append("Linux owned descendant settlement deadline exceeded")
                    break
                if not stdin_open:
                    time.sleep(.025)
        except BaseException as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
            # An exceptional broker still attempts exact owned cleanup, without
            # ever pretending that a failed scan established complete ownership.
            forced = True
            for _ in range(40):
                try:
                    rows = _owned_rows(_snapshot(), broker["pid"], leader["session"])
                    for row in rows.values():
                        if row["state"] != "Z":
                            _signal_exact(row, signal.SIGKILL, broker)
                    while True:
                        try:
                            found, status = os.waitpid(-1, os.WNOHANG)
                        except ChildProcessError:
                            reaped_all = True
                            break
                        if not found:
                            break
                    remaining = _owned_rows(_snapshot(), broker["pid"], leader["session"])
                    if reaped_all and not remaining:
                        break
                except BaseException:
                    pass
                time.sleep(.025)
        finally:
            if gate_w is not None:
                os.close(gate_w)
            os.close(stdin_w)
        # A second postscan protects the terminal ack from a stale empty sample.
        remaining = _owned_rows(_snapshot(), broker["pid"], leader["session"])
        settled = reaped_all and not remaining
        cooperative = False
        if cancelled and not forced and gate_open:
            try:
                ack = json.loads(Path(cancel_ack).read_text())
                cooperative = (ack.get("terminal") == "cancelled" and ack.get("context") == spec["context"]
                               and ack.get("stage_id") == spec["stage_id"] and ack.get("nonce") == spec["nonce"])
            except (OSError, ValueError):
                pass
        natural = not forced and not errors and (not cancelled or cooperative or not gate_open)
        _emit(spec, event="terminal", terminal="blocked" if not settled else "cancelled" if cancelled else
              "succeeded" if returncode == 0 and natural else "failed", returncode=returncode,
              linux_settled=settled, reaped_all=reaped_all, owned_remaining=list(remaining.values()),
              forced=forced, natural=natural, cooperative_cancelled=cooperative, uid=1000,
              broker=broker, leader=leader, boot_id=boot_id, process_identities=list(identities.values()),
              reaped=statuses, errors=errors, memory=memory, elapsed_s=time.monotonic() - began,
              timestamp_ns=time.time_ns())
        return 0 if settled else 2


if __name__ == "__main__":
    raise SystemExit(run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))))
