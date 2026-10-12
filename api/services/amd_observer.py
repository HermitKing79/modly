"""Established passive Windows GPU observer, reused for AMD preparation IPC."""
import datetime
import json
import os
from pathlib import Path
import subprocess

def _capture(command, timeout):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return {"returncode": result.returncode, "timed_out": False,
                "stdout": result.stdout.decode("utf-8", errors="replace"), "stderr": result.stderr.decode("utf-8", errors="replace")}
    except subprocess.TimeoutExpired as exc:
        return {"returncode": None, "timed_out": True, "error": str(exc)}

def windows_observation(since,timeout=10):
    if not 0<timeout<=10:raise ValueError('Observer timeout must be bounded within ten seconds')
    since = datetime.datetime.fromisoformat(since).isoformat()
    # Passive WMI and System-event log queries; no GPU workload or repair.
    script = (
        "$ErrorActionPreference='Stop'; $result=@{}; "
        "$env:PSModulePath=Join-Path $PSHOME 'Modules'; "
        "try {$diagnostics=Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Diagnostics/Microsoft.PowerShell.Diagnostics.psd1'; "
        "Import-Module $diagnostics -ErrorAction Stop; "
        "$probe=Get-WinEvent -LogName System -MaxEvents 1 -ErrorAction Stop | "
        "Select-Object Id,RecordId,ProviderName,TimeCreated; "
        "$result.observer=@{edition=$PSVersionTable.PSEdition;version=$PSVersionTable.PSVersion.ToString();"
        "diagnostics_module=(Get-Module Microsoft.PowerShell.Diagnostics).Path;system_log_probe=$probe}} "
        "catch {$result.observer_error=$_.Exception.Message;$result.observer_error_id=$_.FullyQualifiedErrorId}; "
        "try {$result.controllers=@(Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion,DriverDate,PNPDeviceID,Status)} "
        "catch {$result.controller_error=$_.Exception.Message}; "
        "try {$events=@(Get-WinEvent -FilterHashtable @{LogName='System';StartTime="
        f"[datetime]'{since}';Id=@(4101,17,18,41)}} -ErrorAction Stop); "
        "$result.events=@($events | Select-Object TimeCreated,Id,ProviderName,Message)} "
        "catch {if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') "
        "{$result.events=@()} else {$result.event_error=$_.Exception.Message}}; "
        "$result | ConvertTo-Json -Depth 6 -Compress")
    executable = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    capture = _capture([str(executable), "-NoProfile", "-NonInteractive",
                                   "-Command", script], timeout)
    if capture["returncode"] == 0:
        try:
            capture["observation"] = json.loads(capture["stdout"])
        except ValueError:
            pass
    return capture


def healthy_observation(capture):
    observation = capture.get("observation", {})
    return (capture.get("returncode") == 0 and capture.get("timed_out") is False and
            isinstance(observation, dict) and "event_error" not in observation and
            "controller_error" not in observation and "observer_error" not in observation and
            isinstance(observation.get("observer"), dict) and
            observation["observer"].get("system_log_probe", {}).get("RecordId", 0) > 0 and
            observation.get("events") == [] and
            isinstance(observation.get("controllers"), list) and
            any("RX 9070 XT" in gpu.get("Name", "") and gpu.get("Status") == "OK"
                for gpu in observation["controllers"]))


