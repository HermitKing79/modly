"""Host-authenticated completed AMD artifacts; no inference or executable imports.

CompletionAuthority is supplied by trusted host code, never IPC or user params.
Historical proof receipts are accepted only as explicitly labelled fixture replay.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import time
import uuid

CONTRACT = "modly.amd-completed-glb.v1"
_MAX_GLB, _MAX_JSON, _CHUNK = 128 * 1024**2, 4 * 1024**2, 1024**2
_PROOFS = {"export", "export_host", "verification", "verification_host"}
_MODES = {"generated", "replay_fixture"}
_AUTHENTICATED = object()


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    _need(type(value) is str and re.fullmatch("[0-9a-f]{64}", value), "Exact SHA256 required")
    return value


def _name(value):
    _need(type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value), "Safe identifier required")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _request_hashes(image_bytes, params):
    _need(type(image_bytes) is bytes and bool(image_bytes) and type(params) is dict, "Image bytes and parameter object required")
    blob = _canonical(params)
    _need(len(blob) <= 65536, "Parameter object exceeds limit")
    return hashlib.sha256(image_bytes).hexdigest(), hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class CompletionContext:
    job_id: str
    request_id: str
    run_id: str
    nonce: str
    input_sha256: str
    params_sha256: str
    created_ns: int
    mode: str
    model_id: str

    @classmethod
    def new(cls, job_id, image_bytes, params, mode="generated", *, model_id):
        image_hash, params_hash = _request_hashes(image_bytes, params)
        return cls.from_dict(dict(job_id=job_id, request_id=uuid.uuid4().hex, run_id=uuid.uuid4().hex,
                                  nonce=uuid.uuid4().hex, input_sha256=image_hash, params_sha256=params_hash,
                                  created_ns=time.time_ns(), mode=mode, model_id=model_id))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        _need(type(value) is dict and set(value) == set(cls.__dataclass_fields__), "Exact completion context required")
        for key in ("job_id", "request_id", "run_id", "nonce"):
            _name(value[key])
        for key in ("input_sha256", "params_sha256"):
            _sha(value[key])
        _need(type(value["created_ns"]) is int and 0 < value["created_ns"] <= time.time_ns() + 5_000_000_000,
              "Valid request creation time required")
        _need(type(value["mode"]) is str and value["mode"] in _MODES
              and type(value["model_id"]) is str and 0 < len(value["model_id"]) <= 256,
              "Explicit mode and model identity required")
        return cls(**value)

    def validate_request(self, image_bytes, params):
        _need(_request_hashes(image_bytes, params) == (self.input_sha256, self.params_sha256), "Completion request input differs")


@dataclass(frozen=True)
class CompletedWorkerOutput:
    receipt_path: str
    receipt_sha256: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        _need(type(value) is dict and set(value) == set(cls.__dataclass_fields__), "Exact completed-worker envelope required")
        _need(type(value["receipt_path"]) is str and Path(value["receipt_path"]).is_absolute(), "Absolute receipt path required")
        _sha(value["receipt_sha256"])
        return cls(**value)


@dataclass(frozen=True)
class FilePin:
    path: Path
    sha256: str


@dataclass(frozen=True)
class CompletionAuthority:
    root: Path
    receipt_path: str
    receipt_sha256: str
    source_image_sha256: str
    source_run_id: str
    worker_source: FilePin
    evidence: dict[str, FilePin]
    mode: str


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Completed-result publication cancelled")


def _plain(value):
    raw = os.fspath(value)
    _need(isinstance(raw, str) and not any(c in raw for c in ("\0", "\r", "\n"))
          and not raw.replace("\\", "/").startswith("//"), "Explicit local path required")
    path = Path(raw)
    _need(path.is_absolute() and ".." not in path.parts, "Canonical absolute path required")
    for part in path.parts[1:]:
        _need(":" not in part and not part.endswith((".", " ")), "Path aliases and streams prohibited")
        _need(not re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part), "Device names prohibited")
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        _need(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
              "Symlink/reparse paths prohibited")
    _need(path == path.resolve(), "Path aliases prohibited")
    if os.name == "nt":
        import ctypes
        _need(ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor)) in (2, 3, 5, 6), "Local volume required")
    return path


def _child(root, relative):
    _need(type(relative) is str and "\\" not in relative and ":" not in relative, "Relative artifact path required")
    part = PurePosixPath(relative)
    _need(not part.is_absolute() and relative == part.as_posix() and part.parts
          and all(p not in (".", "..") for p in part.parts), "Canonical confined relative path required")
    root = _plain(root)
    path = _plain(root.joinpath(*part.parts))
    _need(path.is_relative_to(root), "Artifact outside trusted root")
    return path


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


@contextmanager
def _opened(path, limit, fresh_ns=0):
    path = _plain(path)
    before = path.stat()
    _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= limit,
          "Bounded single-link regular file required")
    _need(before.st_mtime_ns >= fresh_ns, "Stale completed-result source")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        _need(_identity(opened)[:4] == _identity(before)[:4], "File replaced during open")
        yield stream, before
        _need(_identity(os.fstat(stream.fileno())) == _identity(opened)
              and _identity(_plain(path).stat()) == _identity(before), "File changed while reading")


def _json(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    try:
        return json.loads(blob, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid bounded JSON") from exc


def _pinned_json(pin, fresh_ns=0):
    with _opened(pin.path, _MAX_JSON, fresh_ns) as (stream, _):
        blob = stream.read(_MAX_JSON + 1)
    _need(hashlib.sha256(blob).hexdigest() == _sha(pin.sha256), "Pinned receipt hash differs")
    value = _json(blob)
    _need(type(value) is dict, "Receipt object required")
    return value


def _glb(stream, size):
    _need(stream.read(12) == struct.pack("<4sII", b"glTF", 2, size), "GLB v2 header/length differs")
    length, kind = struct.unpack("<II", stream.read(8))
    _need(kind == 0x4E4F534A and 0 < length <= _MAX_JSON and length % 4 == 0 and length + 28 <= size,
          "Bounded aligned GLB JSON required")
    doc = _json(stream.read(length))
    _need(type(doc) is dict and doc.get("asset", {}).get("version") == "2.0", "glTF 2.0 document required")
    binary, kind = struct.unpack("<II", stream.read(8))
    _need(kind == 0x004E4942 and binary % 4 == 0 and 28 + length + binary == size, "Exact self-contained BIN chunk required")
    pending = [doc]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            _need("uri" not in item and not item.get("extensionsRequired") and not item.get("extensionsUsed"), "External resources/extensions prohibited")
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    buffers = doc.get("buffers")
    _need(type(buffers) is list and len(buffers) == 1 and type(buffers[0].get("byteLength")) is int
          and 0 < buffers[0]["byteLength"] <= binary and binary - buffers[0]["byteLength"] <= 3, "Single embedded buffer required")
    views, accessors = doc.get("bufferViews", []), doc.get("accessors", [])
    _need(type(views) is list and type(accessors) is list and views and accessors, "Mesh accessors required")
    for view in views:
        offset, count = view.get("byteOffset", 0), view.get("byteLength")
        _need(view.get("buffer") == 0 and type(offset) is int and type(count) is int and offset >= 0
              and count > 0 and offset + count <= buffers[0]["byteLength"], "Buffer view bounds differ")
    for accessor in accessors:
        index, count, offset = accessor.get("bufferView"), accessor.get("count"), accessor.get("byteOffset", 0)
        component = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}.get(accessor.get("componentType"))
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}.get(accessor.get("type"))
        _need(type(index) is int and 0 <= index < len(views) and type(count) is int and count > 0
              and type(offset) is int and offset >= 0 and component and width and "sparse" not in accessor, "Accessor layout differs")
        view = views[index]; stride = view.get("byteStride", component * width)
        _need(type(stride) is int and component * width <= stride <= 252 and stride % component == 0
              and offset % component == 0 and offset + (count - 1) * stride + component * width <= view["byteLength"], "Accessor exceeds view bounds")
    meshes = doc.get("meshes")
    _need(type(meshes) is list and meshes, "Actual mesh required")
    for mesh in meshes:
        _need(type(mesh.get("primitives")) is list and mesh["primitives"], "Mesh primitives required")
        for primitive in mesh["primitives"]:
            attrs = primitive.get("attributes", {})
            refs = list(attrs.values()) + ([primitive["indices"]] if "indices" in primitive else [])
            _need("POSITION" in attrs and all(type(i) is int and 0 <= i < len(accessors) for i in refs), "Mesh accessor reference differs")
            _need(accessors[attrs["POSITION"]]["type"] == "VEC3" and accessors[attrs["POSITION"]]["componentType"] == 5126, "F32 mesh positions required")


@dataclass(frozen=True)
class AuthenticatedCompletion:
    context: CompletionContext
    path: Path
    sha256: str
    bytes: int
    source_identity: tuple
    metadata: dict
    _seal: object = None


def authenticate_completion(context, candidate, authority, cancel_event=None):
    _check_cancel(cancel_event)
    context = CompletionContext.from_dict(context.to_dict())
    candidate = CompletedWorkerOutput.from_dict(candidate.to_dict())
    _need(type(authority) is CompletionAuthority and authority.mode == context.mode
          and _sha(authority.source_image_sha256) == context.input_sha256, "Trusted source authority differs")
    receipt_path = _child(authority.root, authority.receipt_path)
    _need(_plain(candidate.receipt_path) == receipt_path and candidate.receipt_sha256 == authority.receipt_sha256,
          "Worker receipt differs from trusted host selection")
    fresh = context.created_ns if context.mode == "generated" else 0
    receipt = _pinned_json(FilePin(receipt_path, authority.receipt_sha256), fresh)
    _need(set(receipt) == {"contract", "product_context", "status", "source_run_id", "worker_source_sha256", "output", "proof_sha256"}
          and receipt["contract"] == CONTRACT and receipt["product_context"] == context.to_dict()
          and receipt["status"] == "PASS" and receipt["source_run_id"] == authority.source_run_id,
          "Incomplete or foreign completion receipt")
    _need(type(authority.source_run_id) is str and authority.source_run_id and set(authority.evidence) == _PROOFS
          and type(receipt["proof_sha256"]) is dict and set(receipt["proof_sha256"]) == _PROOFS, "Exact proof closure required")
    with _opened(authority.worker_source.path, _MAX_JSON) as (stream, _):
        _need(hashlib.file_digest(stream, "sha256").hexdigest() == _sha(authority.worker_source.sha256)
              == receipt["worker_source_sha256"], "Worker source pin differs")
    for name, pin in authority.evidence.items():
        _check_cancel(cancel_event)
        _need(pin.sha256 == receipt["proof_sha256"][name], "Completion proof hash differs")
        proof = _pinned_json(pin, fresh)
        _need(proof.get("classification") == "PASS" and (not fresh or proof.get("product_context") == context.to_dict()),
              "Missing current complete proof")
    output = receipt["output"]
    _need(type(output) is dict and set(output) == {"path", "bytes", "sha256"}
          and type(output["bytes"]) is int and 28 < output["bytes"] <= _MAX_GLB, "Bounded final GLB descriptor required")
    path = _child(authority.root, output["path"])
    with _opened(path, _MAX_GLB, fresh) as (stream, info):
        _need(info.st_size == output["bytes"] and hashlib.file_digest(stream, "sha256").hexdigest() == _sha(output["sha256"]), "Final GLB bytes/hash differ")
        stream.seek(0)
        try:
            _glb(stream, info.st_size)
        except (KeyError, IndexError, TypeError, AttributeError, struct.error) as exc:
            raise ValueError("Invalid GLB structure") from exc
    metadata = dict(contract=CONTRACT, mode=context.mode, product_context=context.to_dict(),
                    source_run_id=authority.source_run_id, input_sha256=context.input_sha256,
                    output_sha256=output["sha256"], output_bytes=output["bytes"], receipt_sha256=candidate.receipt_sha256,
                    proof_sha256=dict(receipt["proof_sha256"]))
    return AuthenticatedCompletion(context, path, output["sha256"], info.st_size, _identity(info), metadata, _AUTHENTICATED)


def _commit(part, final):
    # Linking is atomic and refuses an existing destination on Windows and POSIX.
    part = _plain(part)
    before = part.stat()
    with part.open("rb") as source:
        original = _identity(os.fstat(source.fileno()))
        _need(original[:4] == _identity(before)[:4] and original[-1] == 1, "Staging identity changed")
        os.link(part, final)
        try:
            _need(_identity(_plain(final).stat())[:2] == original[:2]
                  and _identity(_plain(part).stat())[:2] == original[:2], "Commit path replaced")
            source.close()  # Windows refuses unlinking an open Python file handle.
            part.unlink()
            _need(_identity(_plain(final).stat())[:2] == original[:2], "Final path replaced")
            return original[:2]
        except BaseException:
            source.close()
            try:
                if _identity(_plain(final).stat())[:2] == original[:2]:
                    final.unlink()
            except (OSError, ValueError):
                pass
            raise


@dataclass
class PublishedCompletion:
    path: Path
    metadata: dict
    _owned: dict
    _directory_identity: tuple = None

    def discard(self):
        for path, identity in reversed(list(self._owned.items())):
            try:
                if _identity(_plain(path).stat())[:2] == identity[:2]:
                    path.unlink()
            except (OSError, ValueError):
                pass
        try:
            directory = _plain(self.path.parent)
            if _identity(directory.stat())[:2] == self._directory_identity:
                directory.rmdir()
        except (OSError, ValueError):
            pass


def promote_completion(authenticated, workspace, collection, cancel_event=None):
    _need(type(authenticated) is AuthenticatedCompletion and authenticated._seal is _AUTHENTICATED,
          "Authenticated completion required")
    _check_cancel(cancel_event)
    workspace = _plain(workspace)
    _need(workspace.is_dir(), "Existing workspace required")
    _need(type(collection) is str and 0 < len(collection) <= 255 and collection == collection.strip()
          and collection == collection.rstrip(". ") and not re.search(r'[/:*?"<>|\\\x00-\x1f]', collection),
          "Safe collection folder required")
    parent = _child(workspace, "Workflows" if collection == "Workflows" else "Workflows/" + collection)
    parent.mkdir(parents=True, exist_ok=True)
    directory = _child(parent, "amd-" + _name(authenticated.context.job_id))
    directory.mkdir()
    final = directory / "model.glb"
    owned = {}
    published = PublishedCompletion(final, {}, owned, _identity(directory.stat())[:2])
    try:
        part = directory / "model.glb.part"
        with _opened(authenticated.path, _MAX_GLB) as (source, info):
            _need(_identity(info) == authenticated.source_identity, "Authenticated source changed before publication")
            with part.open("xb") as target:
                owned[part] = _identity(os.fstat(target.fileno()))
                digest = hashlib.sha256()
                while block := source.read(_CHUNK):
                    _check_cancel(cancel_event)
                    target.write(block); digest.update(block)
                target.flush(); os.fsync(target.fileno())
                owned[part] = _identity(os.fstat(target.fileno()))
            _need(digest.hexdigest() == authenticated.sha256 and part.stat().st_size == authenticated.bytes, "Published copy differs")
        with _opened(part, _MAX_GLB) as (stream, _):
            _need(hashlib.file_digest(stream, "sha256").hexdigest() == authenticated.sha256, "Copy readback differs")
        metadata = dict(authenticated.metadata, artifactId=authenticated.context.job_id, versionId=authenticated.context.run_id,
                        sourceWorkspacePath=final.relative_to(workspace).as_posix(),
                        provenance=dict(mode=authenticated.context.mode, source_run_id=authenticated.metadata["source_run_id"],
                                        proof_sha256=authenticated.metadata["proof_sha256"]))
        meta_part = directory / "model.json.part"
        metadata_blob = _canonical(metadata)
        with meta_part.open("xb") as stream:
            owned[meta_part] = _identity(os.fstat(stream.fileno()))
            stream.write(metadata_blob); stream.flush(); os.fsync(stream.fileno())
            owned[meta_part] = _identity(os.fstat(stream.fileno()))
        _check_cancel(cancel_event)
        meta_final = directory / "model.json"
        metadata_id = _commit(meta_part, meta_final)
        _need(metadata_id == owned[meta_part][:2], "Metadata staging source replaced")
        owned.pop(meta_part); owned[meta_final] = metadata_id
        _check_cancel(cancel_event)
        # The mesh appears only after its durable provenance; this is the commit point.
        artifact_id = _commit(part, final)
        _need(artifact_id == owned[part][:2], "Artifact staging source replaced")
        owned.pop(part); owned[final] = artifact_id
        _check_cancel(cancel_event)
        for path, size, expected_hash in ((meta_final, len(metadata_blob), hashlib.sha256(metadata_blob).hexdigest()),
                                          (final, authenticated.bytes, authenticated.sha256)):
            with _opened(path, _MAX_GLB) as (stream, info):
                _need(_identity(info)[:2] == owned[path] and info.st_size == size
                      and hashlib.file_digest(stream, "sha256").hexdigest() == expected_hash,
                      "Published file identity or bytes changed")
        _need(_identity(_plain(authenticated.path).stat()) == authenticated.source_identity,
              "Authenticated source changed during publication")
        _check_cancel(cancel_event)
        published.metadata = metadata
        return published
    except BaseException:
        # Only files created by this attempt are eligible; foreign files survive.
        for path in list(owned):
            try:
                if path.name.endswith(".part"):
                    current = _identity(_plain(path).stat())
                    if current[:2] == owned[path][:2]:
                        owned[path] = current
            except (OSError, ValueError):
                pass
        published.discard()
        raise
