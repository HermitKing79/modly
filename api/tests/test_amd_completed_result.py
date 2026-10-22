"""CPU-only completion trust and byte-preserving publication tests."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch

try:
    from services import amd_completed_result as completed
except ImportError:
    completed = None


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def glb(document=None):
    document = document or {
        "asset": {"version": "2.0"}, "buffers": [{"byteLength": 44}],
        "bufferViews": [{"buffer": 0, "byteLength": 36},
                        {"buffer": 0, "byteOffset": 36, "byteLength": 6}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                      {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
    }
    data = json.dumps(document, separators=(",", ":")).encode()
    data += b" " * (-len(data) % 4)
    binary = struct.pack("<9f3H", 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2) + b"\0\0"
    return struct.pack("<4sII", b"glTF", 2, 28 + len(data) + len(binary)) + struct.pack("<II", len(data), 0x4E4F534A) + data + struct.pack("<II", len(binary), 0x004E4942) + binary


class CompletedResultTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(completed, "completed-result trust boundary has not been implemented")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "worker"
        self.root.mkdir()
        self.workspace = Path(self.tmp.name) / "workspace"
        self.workspace.mkdir()
        self.image = b"submitted-image"
        self.params = {"steps": 12, "seed": 7}
        self.context = completed.CompletionContext.new("job-test", self.image, self.params, model_id="test-amd/generate")
        self.output = self.root / "result.glb"
        self.output.write_bytes(glb())
        self.worker = self.root / "worker.py"
        self.worker.write_text("# This source is hashed, never executed.\n")
        self.evidence = {}
        for name in ("export", "export_host", "verification", "verification_host"):
            path = self.root / (name + ".json")
            path.write_text(json.dumps({"classification": "PASS", "product_context": self.context.to_dict()}))
            self.evidence[name] = path
        self.receipt = {
            "contract": completed.CONTRACT, "status": "PASS", "product_context": self.context.to_dict(),
            "source_run_id": "original-run", "worker_source_sha256": sha(self.worker.read_bytes()),
            "output": {"path": "result.glb", "bytes": self.output.stat().st_size, "sha256": sha(self.output.read_bytes())},
            "proof_sha256": {key: sha(path.read_bytes()) for key, path in self.evidence.items()},
        }
        self.receipt_path = self.root / "completion.json"

    def authority(self, *, mode=None, blob=None):
        if blob is None:
            blob = json.dumps(self.receipt).encode()
        self.receipt_path.write_bytes(blob)
        pin = completed.FilePin
        authority = completed.CompletionAuthority(
            root=self.root, receipt_path="completion.json", receipt_sha256=sha(blob),
            source_image_sha256=sha(self.image), source_run_id="original-run",
            worker_source=pin(self.worker, sha(self.worker.read_bytes())),
            evidence={key: pin(path, sha(path.read_bytes())) for key, path in self.evidence.items()},
            mode=mode or self.context.mode)
        return completed.CompletedWorkerOutput(str(self.receipt_path), sha(blob)), authority

    def authenticate(self, **kwargs):
        candidate, authority = self.authority(**kwargs)
        return completed.authenticate_completion(self.context, candidate, authority)

    def test_context_and_candidate_strict_roundtrip_and_request_validation(self):
        self.assertEqual(completed.CompletionContext.from_dict(self.context.to_dict()), self.context)
        self.context.validate_request(self.image, {"seed": 7, "steps": 12})
        for image, params in ((b"different", self.params), (self.image, {"steps": 13})):
            with self.assertRaises(ValueError): self.context.validate_request(image, params)
        candidate, _ = self.authority()
        self.assertEqual(completed.CompletedWorkerOutput.from_dict(candidate.to_dict()), candidate)
        for cls, value in ((completed.CompletionContext, self.context.to_dict()),
                           (completed.CompletedWorkerOutput, candidate.to_dict())):
            with self.assertRaises(ValueError): cls.from_dict(dict(value, injected=True))
        with self.assertRaises(ValueError): completed.CompletionContext.new("../escape", self.image, self.params, model_id="test-amd/generate")
        with self.assertRaises(ValueError): completed.CompletionContext.new("job", self.image, {"bad": float("nan")}, model_id="test-amd/generate")

    def test_authenticate_and_publish_original_bytes_with_library_metadata(self):
        original = self.output.read_bytes()
        authenticated = self.authenticate()
        published = completed.promote_completion(authenticated, self.workspace, "Workflows")
        self.assertEqual(published.path, self.workspace / "Workflows/amd-job-test/model.glb")
        self.assertEqual(published.path.read_bytes(), original)
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(published.metadata["artifactId"], self.context.job_id)
        self.assertEqual(published.metadata["versionId"], self.context.run_id)
        self.assertEqual(published.metadata["sourceWorkspacePath"], "Workflows/amd-job-test/model.glb")
        self.assertEqual(published.metadata["provenance"]["mode"], "generated")
        self.assertEqual(json.loads((published.path.parent / "model.json").read_text()), published.metadata)
        self.assertEqual(published.path.stat().st_nlink, 1)
        self.assertFalse(list(self.workspace.rglob("*.part")))
        with self.assertRaises((ValueError, FileExistsError)):
            completed.promote_completion(authenticated, self.workspace, "Workflows")
        self.assertEqual(published.path.read_bytes(), original)
        published.discard()
        self.assertFalse(published.path.exists())

    def test_replay_is_explicit_and_accepts_historical_evidence_only_in_replay_mode(self):
        for path in self.evidence.values(): path.write_text('{"classification":"PASS"}')
        self.receipt["proof_sha256"] = {key: sha(path.read_bytes()) for key, path in self.evidence.items()}
        with self.assertRaises(ValueError): self.authenticate()
        self.context = completed.CompletionContext.new("job-replay", self.image, self.params, mode="replay_fixture", model_id="test-amd/generate")
        self.receipt["product_context"] = self.context.to_dict()
        self.assertEqual(self.authenticate().context.mode, "replay_fixture")
        with self.assertRaises(ValueError): self.authenticate(mode="generated")

    def test_rejects_partial_context_hash_and_receipt_schema_mismatches(self):
        original = json.loads(json.dumps(self.receipt))
        mutations = [lambda r: r.update(status="PARTIAL"), lambda r: r.update(extra=True),
                     lambda r: r.update(source_run_id="old-run"),
                     lambda r: r["product_context"].update(nonce="0" * 32),
                     lambda r: r.update(worker_source_sha256="0" * 64),
                     lambda r: r["output"].update(sha256="0" * 64),
                     lambda r: r["output"].update(bytes=True),
                     lambda r: r["proof_sha256"].update(export="0" * 64)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.receipt = json.loads(json.dumps(original)); mutate(self.receipt)
                with self.assertRaises(ValueError): self.authenticate()
        self.receipt = original
        candidate, authority = self.authority()
        with self.assertRaises(ValueError):
            completed.authenticate_completion(self.context, completed.CompletedWorkerOutput(candidate.receipt_path, "0" * 64), authority)

    def test_rejects_duplicate_json_and_partial_evidence(self):
        blob = json.dumps(self.receipt).replace('"status": "PASS"', '"status":"PASS","status":"PASS"').encode()
        with self.assertRaises(ValueError): self.authenticate(blob=blob)
        for path in self.evidence.values():
            original = path.read_bytes()
            path.write_text(json.dumps({"classification": "PARTIAL", "product_context": self.context.to_dict()}))
            self.receipt["proof_sha256"][path.stem] = sha(path.read_bytes())
            with self.assertRaises(ValueError): self.authenticate()
            path.write_bytes(original); self.receipt["proof_sha256"][path.stem] = sha(original)

    def test_rejects_paths_hardlinks_and_stale_sources(self):
        for relative in ("../result.glb", "sub/../result.glb", "C:/result.glb", "//host/result.glb", "result.glb:stream"):
            with self.subTest(relative=relative):
                self.receipt["output"]["path"] = relative
                with self.assertRaises(ValueError): self.authenticate()
        self.receipt["output"]["path"] = "result.glb"
        linked = self.root / "linked.glb"
        os.link(self.output, linked)
        with self.assertRaises(ValueError): self.authenticate()
        linked.unlink()
        os.utime(self.output, ns=(self.context.created_ns - 10**9,) * 2)
        with self.assertRaises(ValueError): self.authenticate()

    def test_rejects_stale_receipt_and_foreign_candidate_path(self):
        candidate, authority = self.authority()
        os.utime(self.receipt_path, ns=(self.context.created_ns - 10**9,) * 2)
        with self.assertRaises(ValueError): completed.authenticate_completion(self.context, candidate, authority)
        candidate, authority = self.authority()
        foreign = Path(self.tmp.name) / "foreign.json"; foreign.write_bytes(self.receipt_path.read_bytes())
        with self.assertRaises(ValueError):
            completed.authenticate_completion(self.context, completed.CompletedWorkerOutput(str(foreign), candidate.receipt_sha256), authority)

    def test_rejects_corrupt_and_external_glb_and_accessor_overflow(self):
        document = json.loads(glb()[20:20 + struct.unpack_from("<I", glb(), 12)[0]])
        invalid = [b"not-a-glb", glb()[:-1], glb() + b"junk"]
        external = json.loads(json.dumps(document)); external["buffers"][0]["uri"] = "external.bin"
        overflow = json.loads(json.dumps(document)); overflow["accessors"][0]["count"] = 999
        invalid.extend((glb(external), glb(overflow)))
        for blob in invalid:
            with self.subTest(length=len(blob)):
                self.output.write_bytes(blob)
                self.receipt["output"].update(bytes=len(blob), sha256=sha(blob))
                with self.assertRaises(ValueError): self.authenticate()

    def test_cancellation_fault_and_changed_source_never_publish(self):
        authenticated = self.authenticate()
        stopped = threading.Event(); stopped.set()
        with self.assertRaises(InterruptedError):
            completed.promote_completion(authenticated, self.workspace, "Workflows", stopped)
        with patch.object(completed, "_commit", side_effect=OSError("publication fault")):
            with self.assertRaises(OSError): completed.promote_completion(authenticated, self.workspace, "Workflows")
        self.assertFalse(list(self.workspace.rglob("model.glb")))
        self.assertFalse(list(self.workspace.rglob("*.part")))
        self.output.write_bytes(self.output.read_bytes()[:-1] + b"x")
        with self.assertRaises(ValueError): completed.promote_completion(authenticated, self.workspace, "Workflows")
        self.assertFalse(list(self.workspace.rglob("model.glb")))

    def test_cancellation_during_copy_and_discard_preserve_foreign_files(self):
        authenticated = self.authenticate()
        class DuringCopy:
            calls = 0
            def is_set(self):
                self.calls += 1
                return self.calls >= 5
        with patch.object(completed, "_CHUNK", 16):
            with self.assertRaises(InterruptedError):
                completed.promote_completion(authenticated, self.workspace, "Workflows", DuringCopy())
        self.assertFalse(list(self.workspace.rglob("*.part")))
        published = completed.promote_completion(authenticated, self.workspace, "models")
        foreign = published.path.parent / "foreign.txt"; foreign.write_text("keep")
        published.discard()
        self.assertEqual(foreign.read_text(), "keep")

    def test_failed_publication_does_not_delete_a_replaced_foreign_part(self):
        authenticated = self.authenticate()
        def replace_part(part, _final):
            part.unlink()
            part.write_bytes(b"foreign replacement")
            raise OSError("replaced staging path")
        with patch.object(completed, "_commit", side_effect=replace_part):
            with self.assertRaises(OSError): completed.promote_completion(authenticated, self.workspace, "Workflows")
        remaining = list(self.workspace.rglob("*.part"))
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].read_bytes(), b"foreign replacement")

    def test_atomic_commit_removes_only_its_link_when_staging_unlink_fails(self):
        part, final = self.root / "stage.part", self.root / "final.glb"
        part.write_bytes(b"original")
        original_unlink = Path.unlink
        def fail_stage_unlink(path, *args, **kwargs):
            if path == part: raise OSError("injected staging unlink failure")
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", fail_stage_unlink):
            with self.assertRaises(OSError): completed._commit(part, final)
        self.assertFalse(final.exists())
        self.assertEqual(part.read_bytes(), b"original")

    def test_forged_unsealed_result_cannot_publish(self):
        valid = self.authenticate()
        forged = completed.AuthenticatedCompletion(valid.context, valid.path, valid.sha256,
                                                   valid.bytes, valid.source_identity, valid.metadata)
        with self.assertRaises(ValueError): completed.promote_completion(forged, self.workspace, "Workflows")
        self.assertFalse(list(self.workspace.rglob("model.glb")))

    def test_glb_commit_is_last_and_metadata_fault_never_exposes_mesh(self):
        authenticated = self.authenticate()
        seen = []
        original = completed._commit
        def recording_commit(part, final):
            seen.append(final.name)
            if final.name == "model.json":
                self.assertFalse((final.parent / "model.glb").exists())
            else:
                self.assertTrue((final.parent / "model.json").exists())
            return original(part, final)
        with patch.object(completed, "_commit", side_effect=recording_commit):
            published = completed.promote_completion(authenticated, self.workspace, "Workflows")
        self.assertEqual(seen, ["model.json", "model.glb"])
        published.discard()

    def test_discard_preserves_a_replaced_foreign_directory(self):
        published = completed.promote_completion(self.authenticate(), self.workspace, "Workflows")
        original = published.path.parent
        original.rename(original.with_name("moved-owned-directory"))
        original.mkdir()
        published.discard()
        self.assertTrue(original.is_dir())

    def test_existing_legal_collection_with_spaces_is_preserved(self):
        published = completed.promote_completion(self.authenticate(), self.workspace, "My Models")
        self.assertEqual(published.path, self.workspace / "Workflows/My Models/amd-job-test/model.glb")
        published.discard()

    def test_replaced_final_after_commit_is_rejected_and_preserved_on_cancel(self):
        authenticated = self.authenticate()
        original = completed._commit
        for cancel_at_replacement in (False, True):
            with self.subTest(cancel=cancel_at_replacement):
                cancel = threading.Event()
                def replacing_commit(part, final):
                    identity = original(part, final)
                    if final.name == "model.glb":
                        final.unlink(); final.write_bytes(b"foreign replacement after commit")
                        if cancel_at_replacement: cancel.set()
                    return identity
                collection = "Cancelled" if cancel_at_replacement else "Changed"
                with patch.object(completed, "_commit", side_effect=replacing_commit):
                    with self.assertRaises(InterruptedError if cancel_at_replacement else ValueError):
                        completed.promote_completion(authenticated, self.workspace, collection, cancel)
                foreign = self.workspace / "Workflows" / collection / "amd-job-test/model.glb"
                self.assertEqual(foreign.read_bytes(), b"foreign replacement after commit")

    def test_staging_replacement_after_handle_close_is_never_adopted_for_cleanup(self):
        authenticated = self.authenticate()
        original_open = Path.open
        foreign_bytes = b"foreign replacement after staging close"
        for leaf in ("model.glb.part", "model.json.part"):
            with self.subTest(leaf=leaf):
                class ReplacingContext:
                    def __init__(self, stream, path): self.stream, self.path = stream, path
                    def __enter__(self): return self.stream.__enter__()
                    def __exit__(self, *args):
                        result = self.stream.__exit__(*args)
                        self.path.unlink()
                        with original_open(self.path, "wb") as replacement: replacement.write(foreign_bytes)
                        return result
                def opening(path, mode="r", *args, **kwargs):
                    stream = original_open(path, mode, *args, **kwargs)
                    return ReplacingContext(stream, path) if path.name == leaf and mode == "xb" else stream
                collection = "GeometrySwap" if leaf.startswith("model.glb") else "MetadataSwap"
                with patch.object(Path, "open", opening):
                    with self.assertRaises(ValueError):
                        completed.promote_completion(authenticated, self.workspace, collection)
                retained = [p for p in (self.workspace / "Workflows" / collection).rglob("*") if p.is_file()]
                self.assertTrue(any(path.read_bytes() == foreign_bytes for path in retained))


if __name__ == "__main__": unittest.main()
