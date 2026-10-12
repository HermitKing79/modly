"""Passive recovery diagnostics. System Python only; never imports GPU packages."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]


def differences(before, after, path=""):
    if type(before) is not type(after):
        return [{"path": path, "before": before, "after": after}]
    result = []
    if isinstance(before, dict):
        for key in sorted(before.keys() | after.keys()):
            if key not in before or key not in after:
                result.append({"path": path + "/" + key,
                               "before": before.get(key), "after": after.get(key)})
            else:
                result += differences(before[key], after[key], path + "/" + key)
    elif isinstance(before, list):
        if len(before) != len(after):
            result.append({"path": path + "/length", "before": len(before), "after": len(after)})
        for index, (left, right) in enumerate(zip(before, after)):
            result += differences(left, right, path + "/" + str(index))
    elif before != after:
        result.append({"path": path, "before": before, "after": after})
    return result


def metadata_rows(venv):
    entries = []
    for directory, dirs, files in os.walk(venv, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            info = path.lstat()
            entries.append([path.relative_to(venv).as_posix(), info.st_size,
                            info.st_mtime_ns, info.st_mode])
    return sorted(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    def save(name, data):
        (output / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    evidence = args.evidence.resolve()
    before = json.loads((evidence / "inventory-before.json").read_text())
    after = json.loads((evidence / "inventory-after.json").read_text())
    result = {"historical_json_differences": differences(before, after),
              "preserved_evidence": {p.name: {"bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(evidence.iterdir()) if p.is_file()}, "scans": []}
    roots = {"amd": Path("D:/AI/Trellis2AMD/venv"),
             "modly": Path("C:/Users/wattz/Documents/Modly/extensions/trellis2/venv")}
    manifests = []
    for index in range(2):
        scan = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        rows = {label: metadata_rows(root) for label, root in roots.items()}
        manifests.append(rows)
        save(f"file-metadata-scan-{index + 1}.json", rows)
        scan["trees"] = {label: {"entries": len(values), "sha256": hashlib.sha256(
            json.dumps(values, separators=(",", ":")).encode()).hexdigest()}
            for label, values in rows.items()}
        result["scans"].append(scan)
    result["passive_scan_differences"] = {}
    for label in roots:
        left = {row[0]: row[1:] for row in manifests[0][label]}
        right = {row[0]: row[1:] for row in manifests[1][label]}
        delta = differences(left, right)
        result["passive_scan_differences"][label] = delta
    save("reconciliation.json", result)
    print(json.dumps({"historical_json_differences": result["historical_json_differences"],
                      "scans": result["scans"],
                      "passive_scan_difference_counts": {k: len(v) for k, v in
                                                           result["passive_scan_differences"].items()}}, indent=2))


if __name__ == "__main__":
    main()
