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
    _discovery = None
    _preparation_guards = None
    _pipeline_class = None

    @classmethod
    def worker_identity(cls, model_id):
        cls._config = json.loads(os.environ["MODLY_AMD_CONFIG_JSON"])
        if cls._config.get("mode") == amd_runtime.PREPARATION_MODE:
            from services.amd_offline_guard import block_external_actions
            cls._discovery = []
            sdk = Path(cls._config["python"]).parent / "rocm-sdk.exe"
            verification = block_external_actions(sdk_executable=sdk, discovery_records=cls._discovery,
                                                 scratch_dir=Path(os.environ["WORKSPACE_DIR"]))
            os.environ.update(HIP_VISIBLE_DEVICES="0", ROCR_VISIBLE_DEVICES="0", CUDA_VISIBLE_DEVICES="0")
        else:
            offline_guard(Path(os.environ["WORKSPACE_DIR"]))
        cls._identity, cls._tensor_guard = amd_runtime.worker_preflight(cls._config, model_id)
        if cls._config.get("mode") == amd_runtime.PREPARATION_MODE:
            cls._identity["sdk_verification"] = verification
        return cls._identity

    def load(self):
        if self._identity is None or self._config is None:
            raise ValueError("Verified AMD worker identity required before mock readiness")
        if self._config.get("mode") == amd_runtime.PREPARATION_MODE:
            if self.MODEL_NODE_ID != "generate":
                raise ValueError("Alpha-only geometry preparation does not implement texturing/refine")
            self._model = "geometry-configuration-only; weights-unloaded"
            return
        self._checkpoints = amd_runtime.validate_models(self._config["models"], self.MODEL_NODE_ID)
        self._model = "mock-readiness-only"

    def unload(self):
        if getattr(self, "_preparation", None) is not None:
            self._preparation.close()
            self._preparation = None
        self._model = None

    def prepare_geometry(self, image_bytes, params):
        if (self._config.get("mode") != amd_runtime.PREPARATION_MODE or self.MODEL_NODE_ID != "generate"
                or params != {"image_mode": "alpha-only"} or not self.is_loaded()):
            raise ValueError("Explicit loaded AMD alpha-only preparation worker required")
        if getattr(self, "_preparation", None) is not None:
            self._preparation.close()
            self._preparation = None
        # Reject masks before real imports or factory calls.
        from alpha_input import decode_alpha_image
        from model_assets import validate_geometry_assets
        try:
            image, _ = decode_alpha_image(image_bytes)
            image.close()
            validate_geometry_assets(self._config["geometry"], self._config["source_root"])
        except (ValueError, OSError) as exc:
            raise amd_runtime.PreparationValidationError(str(exc)) from exc
        if type(self)._preparation_guards is None:
            from preparation_guards import PreparationGuards
            type(self)._preparation_guards = PreparationGuards(self._config["source_root"])
        from geometry_preparation import GeometryPreparation
        session = GeometryPreparation(self._config["geometry"], self._config["source_root"],
                                      self._preparation_guards, self._pipeline_class)
        try:
            diagnostics = session.prepare(image_bytes)
            type(self)._pipeline_class = session.pipeline_class
            self._preparation = session
            diagnostics["sdk_discovery"] = self._discovery
            return diagnostics
        except BaseException:
            session.close()
            raise

    def _auto_download(self):
        raise RuntimeError("AMD scaffold is offline; model downloads are disabled")

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        if (self._config or {}).get("mode") == amd_runtime.PREPARATION_MODE:
            raise ValueError("Preparation mode returns diagnostics through prepare_geometry; generation is disabled")
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
