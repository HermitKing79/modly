"""Headless AMD scaffold. Mock receipts only; upstream factories are deferred."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import uuid
from services.generators.base import BaseGenerator, GenerationCancelled
from services import amd_runtime


def offline_guard(workspace):
    """Confine Python writes; prohibit process launch/network in this worker."""
    workspace = workspace.resolve()
    if sys.platform == "win32":
        import platform
        version = sys.getwindowsversion()
        platform._uname_cache = platform.uname_result(
            system="Windows", node=os.environ.get("COMPUTERNAME", ""), release=str(version.major),
            version=f"{version.major}.{version.minor}.{version.build}", machine="AMD64")
    def writable(path):
        if isinstance(path, int):
            raise RuntimeError("AMD scaffold cannot write through an opaque descriptor")
        try:
            Path(os.fsdecode(path)).resolve().relative_to(workspace)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"AMD scaffold write outside workspace: {path}") from exc
    def audit(event, args):
        if event.startswith("socket.") or event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"):
            raise RuntimeError(f"AMD offline scaffold prohibits {event}")
        if event == "open":
            mode, flags = args[1:3]
            if (mode and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                writable(args[0])
        if event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime"):
            writable(args[0])
        if event in ("os.rename", "os.link", "os.symlink"):
            writable(args[0])
            writable(args[1])
    sys.addaudithook(audit)


class AMDScaffold(BaseGenerator):
    _identity = None
    _config = None
    _tensor_guard = None

    @classmethod
    def worker_identity(cls, model_id):
        cls._config = json.loads(os.environ["MODLY_AMD_CONFIG_JSON"])
        offline_guard(Path(os.environ["WORKSPACE_DIR"]))
        cls._identity, cls._tensor_guard = amd_runtime.worker_preflight(cls._config, model_id)
        return cls._identity

    def load(self):
        if self._identity is None or self._config is None:
            raise ValueError("Verified AMD worker identity required before mock readiness")
        self._checkpoints = amd_runtime.validate_models(self._config["models"], self.MODEL_NODE_ID)
        self._model = "mock-readiness-only"

    def unload(self):
        self._model = None

    def _auto_download(self):
        raise RuntimeError("AMD scaffold is offline; model downloads are disabled")

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        if not isinstance(params, dict) or params.get("mock") is not True:
            raise ValueError("AMD scaffold requires mock=true; inference is not implemented")
        if set(params) - {"mock", "existing_mesh_path", "image_mode"}:
            raise ValueError("Unknown AMD mock parameters; inference settings are prohibited")
        if not self.is_loaded():
            raise ValueError("AMD mock readiness has not been loaded")
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError("Mock route requires nonempty image bytes")
        from alpha_input import validate_mode, decode_alpha_image
        image_mode = validate_mode(params.get("image_mode", "rembg"))
        alpha_input = None
        if image_mode == "alpha-only":
            _, alpha_input = decode_alpha_image(image_bytes)
        mesh = None
        if self.MODEL_NODE_ID == "refine":
            mesh = amd_runtime.absolute_file(params.get("existing_mesh_path"), "Texture input GLB")
            mesh.relative_to(self.outputs_dir.resolve())
            with mesh.open("rb") as stream:
                magic, version, length = struct.unpack("<4sII", stream.read(12))
            if mesh.suffix.lower() != ".glb" or magic != b"glTF" or version != 2 or length != mesh.stat().st_size:
                raise ValueError("Texture mock requires a GLB v2 container header with matching size")
        elif self.MODEL_NODE_ID != "generate":
            raise ValueError("Unknown AMD worker node")
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()
        if progress_cb:
            progress_cb(50, "AMD mock IPC; no geometry or textures generated")
        receipt = {"schema": "modly.amd.mock-receipt.v1", "model_id": self.MODEL_ID,
                   "route": self.MODEL_NODE_ID, "mock": True, "asset_generated": False,
                   "weights_loaded": False, "gpu_initialized": False,
                   "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                   "image_mode": image_mode, "alpha_input": alpha_input,
                   "input_mesh": str(mesh) if mesh else None,
                   "offline_checkpoints": self._checkpoints}
        output = self.outputs_dir / f"amd-{self.MODEL_NODE_ID}-mock-{uuid.uuid4().hex}.json"
        output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return output
