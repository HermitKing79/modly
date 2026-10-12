"""Fail-closed decoder proof decisions; stdlib-only and free of model imports."""


def validate_identity(archived, current):
    fields = ("config", "file_bytes", "mtime_ns", "header_bytes", "header_sha256", "tensor_count",
              "parameter_count", "tensors", "source_module_sha256", "metadata_only_fixture")
    changed = [field for field in fields if archived.get(field) != current.get(field)]
    if changed or current.get("metadata_only_fixture") is not False:
        raise ValueError("Audited decoder identity differs: " + ", ".join(changed or ["metadata-only fixture"]))


def validate_tensors(expected, actual, device):
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = []
    for key in sorted(set(expected) & set(actual)):
        if (expected[key]["shape"] != actual[key]["shape"] or expected[key]["dtype"] != actual[key]["dtype"]
                or actual[key]["device"] != device or actual[key]["is_meta"] != (device == "meta")):
            mismatched.append({"key": key, "expected": expected[key], "actual": actual[key]})
    if missing or unexpected or mismatched:
        raise ValueError(f"Strict decoder tensor validation failed: missing={missing}; unexpected={unexpected}; mismatched={mismatched}")


def validate_release(release):
    if (release.get("allocated_bytes") != 0 or release.get("reserved_bytes") != 0
            or release.get("model_collected") is not True):
        raise ValueError("Decoder release incomplete: " + str(release))


def gpu_operation_allowed(name):
    # Device copies/allocation and bookkeeping only; no model math.
    return name in {"aten._to_copy.default", "aten.copy_.default", "aten.empty_strided.default",
                    "aten.detach.default", "aten.alias.default"}


def authorize_placement(gate, expected, checkpoint_sha256, worker_sha256):
    if (gate.get("status") != "PASS" or gate.get("gpu_initialized") is not False
            or gate.get("strict") is not True or gate.get("assign") is not True
            or gate.get("missing_keys") != [] or gate.get("unexpected_keys") != []
            or gate.get("parameter_bytes") != expected["tensor_data_bytes"]
            or checkpoint_sha256 != worker_sha256):
        raise ValueError("Gate 1 evidence is incomplete or checkpoint identity changed")
    validate_tensors(expected["tensors"], gate.get("tensors", {}), "cpu")


def classify(record, returncode, timed_out, observer_healthy, preserved):
    try:
        validate_release(record.get("release", {}))
    except ValueError:
        return "PARTIAL"
    passed = (record.get("status") == "PASS" and record.get("gate_one", {}).get("status") == "PASS"
              and record.get("gate_two", {}).get("status") == "PASS" and returncode == 0 and timed_out is False
              and observer_healthy is True and preserved is True)
    return "PASS" if passed else "PARTIAL"
