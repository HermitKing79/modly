"""Lazy, CPU-only complete state-dictionary loading from explicit local files."""
import gc
import hashlib
from pathlib import Path
from services.amd_runtime import absolute_file
from services.amd_model_metadata import json_metadata, safetensors_metadata


class StrictLocalComponent:
    def __init__(self, config_path, weights_path, class_factory):
        self.config_path = absolute_file(config_path, "component configuration")
        self.weights_path = absolute_file(weights_path, "component weights")
        self.factory = class_factory
        self.config = json_metadata(self.config_path)
        self.metadata = safetensors_metadata(self.weights_path)
        if self.metadata["metadata_only_fixture"]:
            raise ValueError("Component requires nonempty safetensors tensors")
        self._model = None
        self._failed = False
        self.diagnostics = {"weights_loaded": False, "device": "cpu", "strict": True}

    @property
    def loaded(self):
        return self._model is not None

    def load_cpu(self):
        if self._failed:
            raise RuntimeError("Component loading previously failed; create a new validated session")
        if self.loaded:
            return self._model
        model = None
        state = expected = value = raw = actual = None
        try:
            if json_metadata(self.config_path) != self.config or safetensors_metadata(self.weights_path) != self.metadata:
                raise ValueError("Component configuration/checkpoint metadata changed after preparation")
            import torch
            from safetensors.torch import load_file
            # The factory consumes the verified configuration, never a repository ID.
            with torch.device("meta"):
                model = self.factory(self.config)
            expected = model.state_dict()  # Parameters AND persistent buffers.
            state = load_file(str(self.weights_path), device="cpu")
            missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
            incompatible = [key for key in set(expected) & set(state)
                            if expected[key].shape != state[key].shape or expected[key].dtype != state[key].dtype
                            or state[key].device.type != "cpu" or state[key].is_meta]
            if missing or unexpected or incompatible:
                raise ValueError(f"Complete state dictionary mismatch: missing={missing}; unexpected={unexpected}; shape/dtype/device={sorted(incompatible)}")
            del expected
            state = {key: value.clone() for key, value in state.items()}
            result = model.load_state_dict(state, strict=True, assign=True)
            del state
            if result.missing_keys or result.unexpected_keys:
                raise ValueError("Strict state dictionary load returned incompatible entries")
            # Nonpersistent buffers need an explicit initializer policy; never
            # accept unresolved metadata tensors or silently manufacture values.
            for name, value in list(model.named_parameters()) + list(model.named_buffers()):
                if value.is_meta or value.device.type != "cpu":
                    raise ValueError(f"Parameter/buffer {name} remains on metadata/non-CPU device")
            payload_hashes = {}
            with self.weights_path.open("rb") as stream:
                for name, value in model.state_dict().items():
                    start, end = self.metadata["tensors"][name]["data_offsets"]
                    stream.seek(8 + self.metadata["header_bytes"] + start)
                    raw = stream.read(end - start)
                    actual = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                    if raw != actual:
                        raise ValueError(f"Loaded state dictionary bytes differ from checkpoint: {name}")
                    payload_hashes[name] = hashlib.sha256(actual).hexdigest()
            model.eval().requires_grad_(False)
            self._model = model
            self.diagnostics.update(weights_loaded=True, missing_keys=[], unexpected_keys=[],
                                    entries=len(payload_hashes), payload_sha256=payload_hashes)
            return model
        except BaseException:
            self._failed = True
            model = None
            state = expected = value = raw = actual = None
            gc.collect()
            raise

    def unload(self):
        self._model = None
        self.diagnostics["weights_loaded"] = False
        gc.collect()
