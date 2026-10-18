"""CPU-only Windows attestation for five fixed original Brick inputs.

Run under the actual Windows file owner. This module invokes the unchanged
SID observer; it does not change ACLs, ownership, inputs, or import Torch.
The JSON evidence stays in the fixed private workspace artifact directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import stat
import subprocess
import sys

SCHEMA = "modly.windows-exact-input-attestation.v1"
CURRENT_SID = "S-1-5-21-3582664715-2766927308-497596791-1001"
WORKSPACE = Path(r"C:\Users\wattz\Documents\ChatGPT\modly")
PRIVATE = WORKSPACE / "artifacts" / "amd-wsl-input-ownership-48ad081"
ATTESTATION = PRIVATE / "windows-input-attestation.json"
CHECK = PRIVATE / "windows-input-attestation-check.json"
OWNER_HELPER = WORKSPACE / "proof" / "docs" / "compatibility" / "amd-turret-texture-decode-resume-evidence" / "windows_owner.ps1"
OWNER_HELPER_SHA256 = "6849a3ac5375cc1693ce306ea28fa4ce9bed32305b8926bc2f5e6034c75f1a94"
POWERSHELL = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
TEXTURE_ROOT = r"C:\Users\wattz\Documents\ModlyArtifacts\amd-brick-material-stage1-df41688\latent"
GEOMETRY_ROOT = r"C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-neural-completion-c9447dd\neural-output"
INPUTS = (
    ("texture", TEXTURE_ROOT + r"\texture-latent.safetensors",
     "4c74343c961e74bf3cfde4c24bd2a21b23063e17f0468b7f3049311fa9dc4289", 1814784),
    ("texture_metadata", TEXTURE_ROOT + r"\texture-latent-metadata.json",
     "25adb1a50b8b31900ebc56d87f45b7fe23dee9305d97b458f70a60799d59e2a0", 39304),
    ("geometry", GEOMETRY_ROOT + r"\shape-decoded.safetensors",
     "d4de05fc82315e44a739c9815a07a531555e108af05c619ea0426d0c630ef861", 104148612),
    ("provenance", GEOMETRY_ROOT + r"\provenance.json",
     "a42340cef604134b2866a3556f6d74c5f631073eb908fb60a64d90139fac6407", 26328),
    ("run_contract", GEOMETRY_ROOT + r"\run-contract.json",
     "4676626d1061e6cd7090b560f952a974a584f8ea9e6218378b7902294b93bd4f", 1449),
)
SYSTEM_SID = "S-1-5-18"
ADMINISTRATORS_SID = "S-1-5-32-544"
TRUSTED_INSTALLER_SID = "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"
TRUSTED_WRITERS = frozenset((CURRENT_SID, SYSTEM_SID, ADMINISTRATORS_SID))
TRUSTED_ANCESTOR_OWNERS = TRUSTED_WRITERS | {TRUSTED_INSTALLER_SID}
# FILE_WRITE_DATA, APPEND_DATA, WRITE_EA, WRITE_ATTRIBUTES, DELETE,
# WRITE_DAC, WRITE_OWNER, GENERIC_ALL, GENERIC_WRITE.
FILE_MUTATION = 0x2 | 0x4 | 0x10 | 0x100 | 0x10000 | 0x40000 | 0x80000 | 0x10000000 | 0x40000000
# Adding a sibling is not replacement of an existing path component.
# DELETE_CHILD, DELETE, WRITE_DAC, WRITE_OWNER, GENERIC_ALL/WRITE do permit
# replacement or changed protection and are rejected for foreign SIDs.
ANCESTOR_MUTATION = 0x40 | 0x10000 | 0x40000 | 0x80000 | 0x10000000 | 0x40000000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def guest_path(windows_path: str) -> str:
    value = PureWindowsPath(windows_path)
    require(value.drive == "C:" and value.is_absolute(), "unexpected Windows drive")
    return "/mnt/c/" + "/".join(value.parts[1:])


def identity(value: os.stat_result) -> dict:
    return {
        "dev": value.st_dev,
        "ino": value.st_ino,
        "mtime_ns": value.st_mtime_ns,
        "nlink": value.st_nlink,
    }


def assert_regular_identity(path: Path, value: os.stat_result) -> None:
    require(stat.S_ISREG(value.st_mode), "input is not a regular file: " + str(path))
    require(value.st_nlink == 1, "hard-linked input: " + str(path))
    require(value.st_ino != 0, "missing file ID: " + str(path))
    require(not (value.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT),
            "reparse input: " + str(path))


def inspect_acl(row: dict, *, is_file: bool) -> list[dict]:
    """Reject untrusted applied ACL mutations; retain safe create-only grants."""
    owner = row.get("owner_sid")
    allowed_owners = {CURRENT_SID} if is_file else TRUSTED_ANCESTOR_OWNERS
    require(owner in allowed_owners, "unexpected owner SID: " + str(row.get("path")))
    require(row.get("reparse") is False, "reparse ACL component")
    require(not row.get("link_type") and not row.get("target"), "linked ACL component")
    attributes = row.get("attributes")
    require(type(attributes) is int and not (attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT),
            "invalid component attributes")
    require(bool(attributes & stat.FILE_ATTRIBUTE_DIRECTORY) != is_file,
            "component kind mismatch")
    rules = row.get("rules")
    require(isinstance(rules, list) and rules and row.get("sddl"),
            "missing DACL evidence")
    create_only = []
    for rule in rules:
        require(rule.get("type") in ("Allow", "Deny"), "unsupported ACL rule type")
        require(type(rule.get("rights")) is int and isinstance(rule.get("sid"), str),
                "invalid numeric ACL rule")
        if rule["type"] != "Allow" or "InheritOnly" in rule.get("propagation", ""):
            continue
        if rule["sid"] in TRUSTED_WRITERS:
            continue
        rights = rule["rights"] & 0xFFFFFFFF
        mask = FILE_MUTATION if is_file else ANCESTOR_MUTATION
        require(not (rights & mask), "foreign applied mutation ACL: " + str(row.get("path")))
        if not is_file and rights & (0x2 | 0x4):
            create_only.append({"path": row["path"], "sid": rule["sid"], "rights": rule["rights"]})
    return create_only


def read_owner(path: Path) -> dict:
    process = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(OWNER_HELPER), "-Root", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    document = json.loads(process.stdout)
    require(document.get("current_sid") == CURRENT_SID, "run as the actual Windows input owner")
    require(document.get("root") == str(path), "observer path mismatch")
    require(document.get("C_filesystem") == "NTFS", "input drive must be NTFS")
    return document


def owner_chain(path: Path, file_document: dict, parent_document: dict) -> list[dict]:
    file_rows = file_document.get("ancestors")
    parent_rows = parent_document.get("ancestors")
    require(isinstance(file_rows, list) and len(file_rows) == 1,
            "exact-file observer shape changed")
    require(isinstance(parent_rows, list), "missing parent observer chain")
    rows = file_rows + parent_rows
    expected = [str(path), *map(str, path.parents)]
    require([row.get("path") for row in rows] == expected, "incomplete or aliased ancestor chain")
    for index, row in enumerate(rows):
        inspect_acl(row, is_file=index == 0)
    return rows


def read_original(path: Path, expected_hash: str, expected_bytes: int) -> tuple[dict, dict]:
    require(str(path.resolve(strict=True)) == str(path), "canonical input alias")
    before = path.lstat()
    assert_regular_identity(path, before)
    require(before.st_size == expected_bytes, "input size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        assert_regular_identity(path, opened)
        require(identity(before) == identity(opened) and before.st_size == opened.st_size,
                "input changed before open")
        remaining = expected_bytes + 1
        byte_count = 0
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            byte_count += len(chunk)
            remaining -= len(chunk)
            digest.update(chunk)
        after_open = os.fstat(stream.fileno())
    require(byte_count == expected_bytes, "input byte count mismatch")
    after = path.lstat()
    assert_regular_identity(path, after)
    require(identity(before) == identity(after_open) == identity(after)
            and before.st_size == after_open.st_size == after.st_size,
            "input identity changed during read")
    require(str(path.resolve(strict=True)) == str(path), "canonical input alias after read")
    require(digest.hexdigest() == expected_hash, "input digest mismatch")
    evidence = {"before": identity(before), "opened": identity(opened),
                "after_open": identity(after_open), "after": identity(after), "stable": True}
    return identity(after), evidence


def snapshot() -> dict:
    require(os.name == "nt", "attest on the Windows host only")
    helper_before = OWNER_HELPER.read_bytes()
    require(hashlib.sha256(helper_before).hexdigest() == OWNER_HELPER_SHA256,
            "unchanged Windows owner observer source pin mismatch")
    parents_before = {root: read_owner(Path(root)) for root in (TEXTURE_ROOT, GEOMETRY_ROOT)}
    files = []
    identities = set()
    for label, windows, digest, size in INPUTS:
        path = Path(windows)
        before_owner = read_owner(path)
        chain = owner_chain(path, before_owner, parents_before[str(path.parent)])
        file_identity, read_evidence = read_original(path, digest, size)
        after_owner = read_owner(path)
        require(before_owner == after_owner, "file owner or ACL changed across read")
        key = file_identity["dev"], file_identity["ino"]
        require(key not in identities, "input file ID alias")
        identities.add(key)
        files.append({
            "label": label, "windows_path": windows, "guest_path": guest_path(windows),
            "sha256": digest, "bytes": size, "windows_identity": file_identity,
            "owner_sid": chain[0]["owner_sid"], "current_sid": CURRENT_SID,
            "ancestors": chain, "read_identity": read_evidence,
        })
    parents_after = {root: read_owner(Path(root)) for root in (TEXTURE_ROOT, GEOMETRY_ROOT)}
    require(parents_before == parents_after, "ancestor owner or ACL changed across reads")
    require(OWNER_HELPER.read_bytes() == helper_before, "observer source changed across reads")
    # Ensure each file still has the recorded identity after collecting every ACL.
    for row in files:
        end = Path(row["windows_path"]).lstat()
        assert_regular_identity(Path(row["windows_path"]), end)
        require(identity(end) == row["windows_identity"] and end.st_size == row["bytes"],
                "input changed after read")
    return {
        "schema": SCHEMA, "current_sid": CURRENT_SID,
        "trusted_input_roots": [guest_path(TEXTURE_ROOT), guest_path(GEOMETRY_ROOT)],
        "observer": {"windows_path": str(OWNER_HELPER), "sha256": OWNER_HELPER_SHA256},
        "files": files,
    }


def fixed_output(path: str | Path, expected: Path) -> Path:
    value = Path(path)
    require(value == expected and str(value) == str(expected), "evidence path is not the fixed private path")
    require(value.parent.resolve(strict=True) == PRIVATE, "private output directory alias")
    if value.exists():
        state = value.lstat()
        assert_regular_identity(value, state)
        require(value.resolve(strict=True) == value, "private evidence path alias")
    return value


def write_json(path: Path, document: dict) -> None:
    # Evidence is numeric metadata; no tensor data is written.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2) + "\n")


def verify(attestation: str | Path = ATTESTATION, expected_sha256: str | None = None) -> dict:
    existing_path = fixed_output(attestation, ATTESTATION)
    raw = existing_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        require(digest == expected_sha256, "private attestation source pin mismatch")
    previous = json.loads(raw)
    current = snapshot()
    require(previous == current, "private Windows attestation changed or no longer matches")
    require(existing_path.read_bytes() == raw, "private attestation changed during verification")
    return {"schema": "modly.windows-exact-input-attestation-check.v1",
            "passed": True, "files": len(current["files"]),
            "attestation_sha256": digest, "current_sid": CURRENT_SID}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "check"))
    parser.add_argument("--output")
    parser.add_argument("--attestation", default=str(ATTESTATION))
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    if args.action == "snapshot":
        require(args.output is not None, "snapshot requires its fixed private output")
        output = fixed_output(args.output, ATTESTATION)
        document = snapshot()
        write_json(output, document)
        print(json.dumps({"schema": SCHEMA, "passed": True, "files": len(document["files"]),
                          "output": str(output)}))
    else:
        result = verify(args.attestation, args.expected_sha256)
        if args.output is not None:
            output = fixed_output(args.output, CHECK)
            write_json(output, result)
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        raise SystemExit(1)

