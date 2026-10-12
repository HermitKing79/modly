"""Small ComfyUI interfaces for a dedicated AMD worker, never a ComfyUI process."""
import importlib
from pathlib import Path
import sys
import types
from services.amd_runtime import local_path


class ProgressBar:
    def __init__(self, total, callback=None):
        if not isinstance(total, int) or total <= 0:
            raise ValueError("Progress total must be a positive integer")
        self.total, self.current, self.callback = total, 0, callback

    def update(self, value):
        self.update_absolute(self.current + value)

    def update_absolute(self, value, total=None, preview=None):
        if preview is not None:
            raise NotImplementedError("Headless progress does not implement ComfyUI previews")
        if total is not None:
            if not isinstance(total, int) or total <= 0:
                raise ValueError("Progress total must be a positive integer")
            self.total = total
        if not isinstance(value, int) or value < 0:
            raise ValueError("Progress value must be a nonnegative integer")
        self.current = min(value, self.total)
        if self.callback:
            self.callback(self.current, self.total)


def install_comfy_interfaces(models_root):
    root = local_path(str(models_root), "headless models_root").resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Headless models_root must be an existing local directory")
    names = ("folder_paths", "comfy", "comfy.utils")
    if any(name in sys.modules for name in names):
        raise RuntimeError("Headless interfaces require a fresh worker; existing ComfyUI modules cannot be replaced")
    folder = types.ModuleType("folder_paths")
    folder.models_dir = str(root)
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = ProgressBar
    comfy.utils = utils
    for name, module in zip(names, (folder, comfy, utils)):
        module.__file__ = __file__
        sys.modules[name] = module


def import_pipeline_class(source_root, models_root, route):
    classes = {"generate": ("trellis2_image_to_3d", "Trellis2ImageTo3DPipeline"),
               "refine": ("trellis2_texturing", "Trellis2TexturingPipeline")}
    if route not in classes:
        raise ValueError("Headless route must be generate or refine")
    root = local_path(str(source_root), "headless source_root").resolve(strict=True)
    if not (root / "trellis2/pipelines").is_dir():
        raise ValueError("Headless AMD pipeline source is missing")
    if "trellis2" in sys.modules:
        raise RuntimeError("Headless source identity requires a fresh worker")
    install_comfy_interfaces(models_root)
    sys.path.insert(0, str(root))
    module_name, class_name = classes[route]
    module = importlib.import_module("trellis2.pipelines." + module_name)
    Path(module.__file__).resolve().relative_to(root)
    return getattr(module, class_name)
