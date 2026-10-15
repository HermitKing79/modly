"""Contain one installed Blender learned-colour CPU preview; no automatic retry."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


E = Path(__file__).resolve().parent
ROOT = E.parents[2]
sys.path.insert(0, str(ROOT / "scripts/compatibility"))
from amd_forward_owned_job import OwnedJob, ExtendedLimits

GIB = 1024 ** 3
OUTPUT = ROOT.parent / "artifacts/amd-turret-learned-colour-fa3a98c"
MESH = OUTPUT / "turret-learned-colour.ply"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


class Memory(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in ("total", "available", "page_total", "page_available",
                                           "virtual_total", "virtual_available", "extended")]


if __name__ == "__main__":
    began = time.monotonic()
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    if memory.available < 10 * GIB:
        raise RuntimeError("10GiB RAM admission for optional CPU preview refused")
    if shutil.disk_usage(OUTPUT.parent).free < 2 * GIB:
        raise RuntimeError("2GiB preview disk admission refused")
    blender = Path("C:/Program Files/Blender Foundation/Blender 5.2/blender.exe")
    if not blender.is_file():
        raise RuntimeError("Existing Blender unavailable")
    receipt = json.loads((E / "conversion.json").read_text(encoding="utf-8"))
    if receipt != json.loads((OUTPUT / "conversion.json").read_text(encoding="utf-8")):
        raise RuntimeError("Conversion receipts differ")
    if receipt["classification"] != "PASS" or receipt["geometry_unchanged"] is not True:
        raise RuntimeError("Qualified colour conversion required")
    if MESH.is_symlink() or not MESH.is_file():
        raise RuntimeError("Regular immutable coloured PLY required")
    if digest(MESH) != receipt["coloured_ply"]["sha256"] or MESH.stat().st_size != receipt["coloured_ply"]["bytes"]:
        raise RuntimeError("Coloured PLY identity changed")
    if (E / "preview-attempt.marker").exists():
        raise RuntimeError("No automatic render retry")
    if not OUTPUT.is_dir() or OUTPUT.is_symlink():
        raise RuntimeError("Existing regular conversion output directory required")
    if {path.name for path in OUTPUT.iterdir()} != {"turret-learned-colour.ply", "conversion.json"}:
        raise RuntimeError("Only qualified PLY/receipt may exist before preview")
    scratch = OUTPUT / "blender-scratch"
    scratch.mkdir(exist_ok=False)
    env = dict(os.environ, BLENDER_USER_CONFIG=str(scratch), BLENDER_USER_SCRIPTS=str(scratch),
               BLENDER_USER_EXTENSIONS=str(scratch), TEMP=str(scratch), TMP=str(scratch),
               CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
               ONEAPI_DEVICE_SELECTOR="*:cpu")
    job = OwnedJob()
    limits = ExtendedLimits()
    limits.Basic.Flags = 0x2000 | 0x100  # Kill only owned tree on close + process-memory cap.
    limits.ProcessMemory = 6 * GIB
    if not job.api.SetInformationJobObject(job.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        job.close()
        raise ctypes.WinError()
    command = [str(blender), "--background", "--factory-startup", "--disable-autoexec", "--threads", "4",
               "--python-exit-code", "1", "--python", str(E / "render_preview.py")]
    save(E / "preview-attempt.marker", {
        "CPU_only": True, "command": command, "available_RAM_bytes": memory.available,
        "memory_cap_bytes": 6 * GIB, "deadline_seconds": 180,
        "script_sha256": digest(E / "render_preview.py"), "PLY_sha256": receipt["coloured_ply"]["sha256"],
    })
    record = {"command": command, "CPU_only": True, "available_RAM_bytes": memory.available,
              "memory_cap_bytes": 6 * GIB, "deadline_seconds": 180}
    child, attached = None, False
    try:
        with (E / "preview.stdout.txt").open("xb") as out, (E / "preview.stderr.txt").open("xb") as err:
            child = subprocess.Popen(command, stdout=out, stderr=err, env=env)
            job.attach(child)
            attached = True
            child.wait(timeout=180)
            record["returncode"] = child.returncode
            peak = ExtendedLimits()
            if not job.api.QueryInformationJobObject(job.handle, 9, ctypes.byref(peak), ctypes.sizeof(peak), None):
                raise ctypes.WinError()
            record["peak_committed_bytes"] = peak.PeakProcessMemory
    except BaseException as exc:
        record["error"] = str(exc)
        if child is not None and not attached and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
            record["unattached_owned_child_terminated"] = True
    finally:
        record["job_cleanup"] = job.finish()
        if child is not None:
            child.wait(timeout=5)
        record["elapsed_seconds"] = time.monotonic() - began
        record["PLY_sha256_after_exit"] = digest(MESH)
        record["PLY_unchanged"] = record["PLY_sha256_after_exit"] == receipt["coloured_ply"]["sha256"]
        save(E / "preview-host.json", record)
    raise SystemExit(0 if record.get("returncode") == 0 and record["PLY_unchanged"]
                     and not record["job_cleanup"]["forced_job_cleanup"] else 1)
