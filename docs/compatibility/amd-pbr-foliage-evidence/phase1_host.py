"""Minimal adapter of the proven CPU owned-job host for one brick import.

--preflight is read-only and reserves no attempt. Actual execution is exclusive:
one Blender process, one exact GO after job attachment, no retry or GPU path.
"""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
E = Path(__file__).resolve().parent
ROOT = E.parents[3]
sys.path.insert(0, str(E))
import phase1_worker as worker

HELPER = E.parent / "amd-brick-pbr-exposure-evidence/render_host.py"
HELPER_SHA = "2d98a925a0b97aee29d21d259e40eeb290ab44386ab035e15a3463c5df0c1418"
BLENDER = Path("C:/Program Files/Blender Foundation/Blender 5.2/blender.exe")
BLENDER_SHA = "284f4041f98e113f3dc10654a7193ffaaa9bfdfec8b87fa116620a48b5f6d4cb"
CAP = 6 * 1024**3
DEADLINE = 180


def load_helper():
    worker.require(worker.digest(worker.plain(HELPER)) == HELPER_SHA, "Historical CPU host pin differs")
    spec = importlib.util.spec_from_file_location("pbr_foliage_phase1_CPU_host", HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.PRIVATE = worker.P
    return helper


def attach_and_go(job, child, record):
    """The only release of this worker's exact binary gate follows attachment."""
    job.attach(child)
    record["attached_before_GO"] = True
    child.stdin.write(b"GO\n")
    child.stdin.flush()
    child.stdin.close()
    record["exact_binary_GO_sent"] = True


def passed(record):
    cleanup = record.get("job_cleanup", {})
    return (record.get("returncode") == 0 and record.get("exact_binary_GO_sent") is True
            and record.get("attached_before_GO") is True and record.get("sources_unchanged") is True
            and record.get("worker_checkpoint_verified") is True
            and 0 <= record.get("worker_execution_seconds", DEADLINE + 1) <= DEADLINE
            and 0 <= record.get("peak_process_committed_bytes", CAP + 1) <= CAP
            and 0 <= record.get("peak_job_committed_bytes", CAP + 1) <= CAP
            and not any(name in record for name in ("error", "measurement_error", "cleanup_error", "preservation_error"))
            and cleanup.get("owned_tree_exit_verified") is True
            and cleanup.get("forced_job_cleanup") is False and cleanup.get("active_after_cleanup") == 0)


def verify_checkpoint(legacy):
    """Host independently checks final receipt/image identities and two-view rig."""
    path = worker.plain(worker.OUT / "result.json")
    worker.require(path.stat().st_size < 256 * 1024, "Bounded final worker checkpoint required")
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    worker.require(checkpoint["classification"] == "PASS" and checkpoint["CPU_only"] is True
                   and checkpoint["fresh_GLTF_imports"] == 1 and checkpoint["same_imported_scene_for_both_views"] is True
                   and checkpoint["sources_unchanged"] is True and checkpoint["glb"]["sha256"] == worker.GLB_SHA
                   and checkpoint["imported_geometry_before"]["oriented_triangle_position_bits_exact"] is True,
                   "Successful exact fresh import checkpoint required")
    worker.validate_two_views(checkpoint["views"])
    for view in checkpoint["views"]:
        image = worker.plain(worker.OUT / (view["name"] + ".png"))
        worker.require(view["file"] == str(image) and image.stat().st_size == view["bytes"]
                       and worker.digest(image) == view["sha256"], "Worker PNG receipt identity differs")
        # Header check avoids importing Blender/numpy or another image runtime into host.
        header = image.read_bytes()[:33]
        worker.require(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR"
                       and header[16:26] == b"\x00\x00\x02\x80\x00\x00\x02\x80\x08\x02",
                       "Host requires 640-square RGB8 PNGs")
    return dict(path=str(path), sha256=worker.digest(path), views=[row["name"] for row in checkpoint["views"]])


def main(preflight=False):
    helper = load_helper()
    legacy = worker.load_legacy()
    worker.plain(worker.P)
    initial = helper.resources()
    worker.require(initial["admitted"], "Pre-marker CPU RAM/disk admission failed: " + json.dumps(initial))
    worker.verify_source_pins(legacy)
    worker.require(worker.digest(worker.plain(BLENDER)) == BLENDER_SHA, "Approved installed Blender binary identity differs")
    paths = worker.source_paths(legacy) + [HELPER, BLENDER, ROOT / "proof/scripts/compatibility/amd_forward_owned_job.py"]
    for path in E.glob("phase1_*.py"):
        compile(path.read_bytes(), str(path), "exec")
    before = worker.identities(paths)
    for path in (worker.OUT, worker.P / "brick-attempt.json", worker.P / "brick-host.json"):
        worker.require(not path.exists(), "No phase1 overwrite or retry: " + str(path))
    command = [str(BLENDER), "--background", "--factory-startup", "--disable-autoexec", "--threads", "4",
               "--python-exit-code", "1", "--python", str(E / "phase1_worker.py"), "--", "--output", str(worker.OUT)]
    if preflight:
        print(json.dumps(dict(classification="PASS", preflight_only=True, no_marker_reserved=True,
                              no_Blender_launched=True, command=command, resources=initial,
                              watched_files=len(paths), GLB_SHA256=worker.GLB_SHA)), flush=True)
        return 0
    final = helper.resources()
    worker.require(final["admitted"], "Final pre-marker CPU RAM/disk admission failed")
    sys.path.insert(0, str(ROOT / "proof/scripts/compatibility"))
    from amd_forward_owned_job import OwnedJob, ExtendedLimits
    worker.OUT.mkdir(exist_ok=False)
    scratch = worker.OUT / "scratch"
    scratch.mkdir(exist_ok=False)
    helper.save(worker.P / "brick-attempt.json", dict(CPU_only=True, command=command, source_before=before,
                source_GLB_sha256=worker.GLB_SHA, fresh_import_limit=1, views=["near", "far"], retry=False,
                memory_cap_bytes=CAP, job_memory_cap_bytes=CAP, deadline_seconds=DEADLINE, initial_resources=final))
    record = dict(CPU_only=True, stage="phase1", command=command, source_before=before,
                  resource_samples=[], initial_resources=final, memory_cap_bytes=CAP,
                  job_memory_cap_bytes=CAP, deadline_seconds=DEADLINE, retry=False)
    job = OwnedJob()
    limits = ExtendedLimits()
    limits.Basic.Flags = 0x2000 | 0x100 | 0x200  # Kill on close; per-process and entire owned-job memory caps.
    limits.ProcessMemory = limits.JobMemory = CAP
    if not job.api.SetInformationJobObject(job.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        job.close()
        raise ctypes.WinError()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               BLENDER_USER_CONFIG=str(scratch), BLENDER_USER_SCRIPTS=str(scratch), BLENDER_USER_EXTENSIONS=str(scratch),
               TEMP=str(scratch), TMP=str(scratch), TMPDIR=str(scratch), CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
               ROCR_VISIBLE_DEVICES="", ONEAPI_DEVICE_SELECTOR="*:cpu", OPENCV_OPENCL_RUNTIME="disabled",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    child = None
    began = time.monotonic()
    last = -1
    try:
        with (worker.OUT / "stdout.txt").open("xb") as stdout, (worker.OUT / "stderr.txt").open("xb") as stderr:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, env=env)
            attach_and_go(job, child, record)
            while child.poll() is None:
                elapsed = time.monotonic() - began
                if elapsed >= DEADLINE:
                    raise TimeoutError("Unchanged 180-second CPU deadline")
                if elapsed - last >= 1:
                    sample = helper.resources()
                    sample["elapsed_seconds"] = elapsed
                    record["resource_samples"].append(sample)
                    last = elapsed
                    worker.require(sample["admitted"], "Live CPU RAM/disk floor failed: " + json.dumps(sample))
                time.sleep(.1)
            record["returncode"] = child.returncode
            record["worker_execution_seconds"] = time.monotonic() - began
            worker.require(record["worker_execution_seconds"] <= DEADLINE, "Blender exceeded the exact 180-second deadline")
            grace = time.monotonic()
            while job.active() and time.monotonic() - grace < 5:
                time.sleep(.025)
            if child.returncode == 0:
                record["worker_checkpoint"] = verify_checkpoint(legacy)
                record["worker_checkpoint_verified"] = True
    except BaseException as error:
        record.update(error=str(error), traceback=traceback.format_exc())
        if child is not None and not record.get("attached_before_GO") and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
            record["unattached_owned_child_terminated"] = True
    finally:
        if child is not None and child.stdin is not None and not child.stdin.closed:
            child.stdin.close()
        try:
            peak = ExtendedLimits()
            if not job.api.QueryInformationJobObject(job.handle, 9, ctypes.byref(peak), ctypes.sizeof(peak), None):
                raise ctypes.WinError()
            record.update(peak_process_committed_bytes=peak.PeakProcessMemory, peak_job_committed_bytes=peak.PeakJobMemory)
        except BaseException as error:
            record["measurement_error"] = str(error)
        try:
            record["job_cleanup"] = job.finish()
            if child is not None:
                child.wait(timeout=5)
        except BaseException as error:
            record["cleanup_error"] = str(error)
        try:
            after = worker.identities(paths)
            record.update(source_after=after, sources_unchanged=before == after)
        except BaseException as error:
            record["preservation_error"] = str(error)
        record.update(elapsed_seconds=time.monotonic() - began)
        record["classification"] = "PASS" if passed(record) else "BLOCKED"
        helper.save(worker.P / "brick-host.json", record)
    print(json.dumps({key: value for key, value in record.items() if key not in ("source_before", "source_after", "resource_samples")}), flush=True)
    return 0 if record["classification"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.preflight))
