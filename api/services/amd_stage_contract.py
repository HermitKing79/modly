"""CPU-only byte validation for host-owned AMD stage contracts.

These bounds describe readable artifacts, not permission to execute a model.
Source-specific neural profile admission remains a separate host obligation.
"""
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import PurePosixPath
import struct

from services.amd_completed_result import (
    CompletionContext, _child, _json as _completion_json, _need, _opened, _sha,
)

STAGE_ORDER = ("preprocess", "dino", "occupancy", "shape", "geometry",
               "texture_sample", "material", "mesh", "colour", "export", "verify")
_MAX_FILE, _MAX_JSON, _CHUNK = 256 * 1024**2, 4 * 1024**2, 1024**2
_WIDTH = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I64": 8, "BOOL": 1, "U8": 1}


def _json(blob):
    value = _completion_json(blob)
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is float:
            _need(math.isfinite(item), "Nonfinite JSON numeric value")
        elif type(item) is dict:
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
    return value


@dataclass(frozen=True)
class Artifact:
    relative_path: str
    sha256: str
    bytes: int
    kind: str
    schema: dict

    def __post_init__(self):
        name = self.relative_path
        _need(type(name) is str and "\\" not in name and ":" not in name
              and not any(c in name for c in ("\0", "\r", "\n")), "Relative artifact path required")
        path = PurePosixPath(name)
        _need(path.parts and not path.is_absolute() and name == path.as_posix()
              and all(p not in (".", "..") for p in path.parts), "Confined canonical artifact path required")
        _sha(self.sha256)
        _need(type(self.bytes) is int and 0 < self.bytes <= _MAX_FILE, "Bounded artifact bytes required")
        _need(self.kind in ("json", "safetensors") and type(self.schema) is dict, "Supported artifact kind/schema required")
        # Detach nested caller dictionaries; revalidation occurs at every read.
        object.__setattr__(self, "schema", _json(json.dumps(self.schema, allow_nan=False)))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        _need(type(value) is dict and set(value) == set(cls.__dataclass_fields__), "Exact artifact descriptor required")
        return cls(**value)


def _context(value):
    if isinstance(value, CompletionContext):
        value = value.to_dict()
    return CompletionContext.from_dict(value)


def _tensor_schema(schema):
    _need(set(schema) == {"tensors", "max_rows", "coordinate_tensor"}, "Exact host tensor schema required")
    rows, tensors, coordinate = schema["max_rows"], schema["tensors"], schema["coordinate_tensor"]
    _need(type(rows) is int and 0 < rows <= 2_000_000, "Host row bound exceeds readable envelope")
    _need(type(tensors) is dict and 0 < len(tensors) <= 64, "Bounded exact tensor names required")
    _need(coordinate is None or type(coordinate) is str and coordinate in tensors, "Named coordinate tensor required")
    for name, spec in tensors.items():
        _need(type(name) is str and 0 < len(name) <= 128 and name != "__metadata__"
              and type(spec) is dict and set(spec) == {"dtype", "shape"}, "Exact host tensor specification required")
        shape, dtype = spec["shape"], spec["dtype"]
        _need(type(dtype) is str and dtype in _WIDTH and type(shape) is list and 1 <= len(shape) <= 5
              and all(type(n) is int and 0 < n <= 2_000_000 for n in shape)
              and shape[0] <= rows and math.prod(shape) * _WIDTH[dtype] <= _MAX_FILE,
              "Tensor shape/dtype exceeds host-qualified bounds")
    if coordinate is not None:
        _need(tensors[coordinate]["dtype"] == "I32" and tensors[coordinate]["shape"][1:] == [4],
              "Coordinates require original I32[N,4]")
    return tensors, coordinate


def _tensor_payload(stream, offset, count, dtype, *, coordinates=False):
    stream.seek(offset)
    digest, seen = hashlib.sha256(), set() if coordinates else None
    remaining, width = count, _WIDTH[dtype]
    while remaining:
        wanted = min(remaining, _CHUNK)
        if coordinates:
            wanted -= wanted % 16
        block = stream.read(wanted)
        _need(len(block) == wanted, "Truncated tensor payload")
        digest.update(block)
        if dtype in ("F32", "F16", "BF16"):
            code, mask = ("I", 0x7f800000) if dtype == "F32" else ("H", 0x7c00 if dtype == "F16" else 0x7f80)
            _need(all(bits & mask != mask for (bits,) in struct.iter_unpack("<" + code, block)), "Nonfinite tensor payload")
        elif dtype == "BOOL":
            _need(all(v in (0, 1) for v in block), "Invalid boolean tensor byte")
        if coordinates:
            for batch, x, y, z in struct.iter_unpack("<4i", block):
                _need(batch == 0 and all(0 <= n < 512 for n in (x, y, z)), "Coordinate batch/spatial bounds differ")
                key = (x * 512 + y) * 512 + z
                _need(key not in seen, "Duplicate sparse coordinate")
                seen.add(key)
        remaining -= wanted
    return digest.hexdigest()


def _safetensors(stream, size, schema, context, expected_coordinates_sha256):
    expected, coordinate = _tensor_schema(schema)
    raw = stream.read(8)
    _need(len(raw) == 8, "Missing safetensors header length")
    length = struct.unpack("<Q", raw)[0]
    _need(2 <= length <= 65536 and length % 8 == 0 and 8 + length < size, "Bounded aligned tensor header required")
    header = _json(stream.read(length))
    _need(type(header) is dict and set(header) == set(expected) | {"__metadata__"}, "Exact tensor name closure required")
    metadata = header["__metadata__"]
    _need(type(metadata) is dict and all(type(k) is str and type(v) is str for k, v in metadata.items()), "String tensor metadata required")
    _need(_json(metadata.get("product_context", "null")) == context.to_dict(), "Artifact product context differs")
    layout, ranges = {}, []
    payload = size - 8 - length
    for name, spec in expected.items():
        row = header[name]
        _need(type(row) is dict and set(row) == {"dtype", "shape", "data_offsets"}
              and type(row["dtype"]) is str and row["dtype"] == spec["dtype"]
              and type(row["shape"]) is list and all(type(n) is int for n in row["shape"])
              and row["shape"] == spec["shape"], "Actual tensor schema differs from host")
        offsets = row["data_offsets"]
        _need(type(offsets) is list and len(offsets) == 2 and all(type(v) is int for v in offsets), "Exact integer tensor offsets required")
        start, stop = offsets
        _need(0 <= start < stop <= payload and stop - start == math.prod(spec["shape"]) * _WIDTH[spec["dtype"]], "Tensor payload bounds differ")
        ranges.append((start, stop))
        layout[name] = (start, stop)
    previous = 0
    for start, stop in sorted(ranges):
        _need(start == previous, "Tensor payload has gaps or overlap")
        previous = stop
    _need(previous == payload, "Trailing tensor bytes prohibited")
    result = {}
    for name, spec in expected.items():
        start, stop = layout[name]
        digest = _tensor_payload(stream, 8 + length + start, stop - start, spec["dtype"], coordinates=name == coordinate)
        result[name] = dict(spec, sha256=digest, finite=True)
    coordinates_sha = result[coordinate]["sha256"] if coordinate is not None else None
    if expected_coordinates_sha256 is not None:
        _need(coordinate is not None and coordinates_sha == _sha(expected_coordinates_sha256), "Actual coordinate payload/order differs")
    return {"tensors": result, "coordinates_sha256": coordinates_sha}


def validate_artifact(root, artifact, context, expected_coordinates_sha256=None):
    """Reopen exact bytes beneath the trusted root; never execute their contents."""
    artifact = Artifact.from_dict(artifact.to_dict())
    context = _context(context)
    path = _child(root, artifact.relative_path)
    limit = _MAX_JSON if artifact.kind == "json" else _MAX_FILE
    fresh_ns = context.created_ns if context.mode == "generated" else 0
    with _opened(path, limit, fresh_ns) as (stream, info):
        _need(info.st_size == artifact.bytes, "Artifact byte count differs")
        digest = hashlib.sha256()
        while block := stream.read(_CHUNK):
            digest.update(block)
        _need(digest.hexdigest() == artifact.sha256, "Artifact SHA256 differs")
        stream.seek(0)
        if artifact.kind == "json":
            _need(not artifact.schema and expected_coordinates_sha256 is None, "JSON cannot supply a tensor schema or coordinates")
            document = _json(stream.read(limit + 1))
            _need(type(document) is dict and document.get("product_context") == context.to_dict(), "JSON artifact product context differs")
            actual = {"document": document}
        else:
            actual = _safetensors(stream, info.st_size, artifact.schema, context, expected_coordinates_sha256)
    return dict(relative_path=artifact.relative_path, sha256=artifact.sha256, bytes=artifact.bytes, kind=artifact.kind, **actual)


def admitted_image(image_bytes, *, preprocess):
    """Validate actual alpha input pixels; source-faithful preprocessing is a stage."""
    _need(type(preprocess) is dict and preprocess == {"image_mode": "alpha-only", "resolution": 512},
          "Explicit qualified alpha-only/512 preprocessing required")
    _need(type(image_bytes) is bytes and 0 < len(image_bytes) <= 2 * 1024**2, "Bounded current image bytes required")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is unavailable; actual image preprocessing is blocked in this runtime") from exc
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            _need(source.mode == "RGBA" and getattr(source, "n_frames", 1) == 1
                  and min(source.size) > 0 and max(source.size) <= 4096
                  and source.width * source.height <= 16 * 1024 * 1024, "Single bounded pre-segmented RGBA image required")
            image = source.copy()
        def mask(value):
            alpha = value.getchannel("A")
            _need(alpha.getextrema()[0] == 0, "Fully transparent background required")
            box = alpha.point(lambda n: 255 if n > 204 else 0).getbbox()
            _need(box is not None and box[2] - box[0] >= 3 and box[3] - box[1] >= 3, "Nondegenerate alpha foreground required")
            return [box[0], box[1], box[2] - 1, box[3] - 1]
        bbox = mask(image)
        scale = min(1, 1024 / max(image.size))
        resized = [int(image.width * scale), int(image.height * scale)]
        _need(min(resized) > 0, "Image becomes degenerate after original resize")
        resized_bbox = mask(image.resize(resized, Image.Resampling.LANCZOS)) if scale < 1 else bbox
        return dict(input_sha256=hashlib.sha256(image_bytes).hexdigest(), source_bytes=len(image_bytes),
                    source_size=list(image.size), source_mode=image.mode, pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    foreground_bbox=bbox, resized_size=resized, resized_foreground_bbox=resized_bbox,
                    preprocessing=dict(preprocess), preprocessing_complete=False, dino_executed=False)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Current alpha image cannot be decoded") from exc
