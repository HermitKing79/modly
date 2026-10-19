"""Pinned CPU renderer adapter for the seam-aware 1024-square atlas proof.

The unchanged historical renderer supplies its GO gate, private write guard,
GLB export/fresh import, exact oriented-triangle/UV/map checks and CPU EV2 rig.
Only its reference material and receipt descriptions are adapted here.

direct.png is an F32 atlas reference at the SAME UVs, resolution and filtering
as baked.png. Covered texels retain original pre-byte-packing field samples;
derived PNG padding is held fixed. This is not a continuous field shader.
"""
import argparse
import hashlib
import importlib.util
import json
from itertools import islice
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
PROOF_PRIVATE = ROOT / "artifacts/amd-brick-seam-uv-bake-87f3755"
PRIVATE = PROOF_PRIVATE / "render-qualified"
BAKE = PROOF_PRIVATE / "bake"
HISTORICAL = ROOT / "proof/docs/compatibility/amd-brick-uv-bake-evidence/render_patch.py"
HISTORICAL_SHA256 = "96ad2138bb791b12557c9557348f8ad5a351df026774beea158c8f56924127e3"
ATLAS_SIZE = 1024
REFERENCE_SCOPE = (
    "F32 reference uses original pre-quantization source-spatial samples at covered atlas texels. "
    "Reference quantization comparison holds derived padding fixed; padded texels are not independent learned source. "
    "Same atlas resolution, UVs, Linear filtering and EXTEND sampling isolate covered-texel quantization and GLB transport. "
    "Continuous source-surface holdouts and seam checks are separate evidence; this reference is not a continuous field shader."
)
RGB_COMPARISON = (
    "Reference: clamp original encoded F32 covered-texel RGB, exact IEC sRGB decode per texel to F32 scene-linear RGB, "
    "then identical atlas filtering; derived PNG padding is held fixed. "
    "Baked: byte-packed RGB texture with standard sRGB decoding and identical UV/filter settings after fresh GLB import. "
    "This compares covered-texel quantization and transport at fixed atlas resolution, not continuous-field fidelity."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_initial_output(stage, path, live_tempdir=None):
    """Admit no startup content except the exact live empty Blender session."""
    require(stage in ("export", "compare"), "Exact render stage required")
    path = Path(path)
    require(path.is_absolute() and path == PRIVATE / stage, "Exact fresh qualified stage path required")

    def plain(target, directory):
        info = target.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
                "No symlink or reparse startup path: " + str(target))
        require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
                "Plain startup directory/file required: " + str(target))

    for ancestor in (path, *path.parents):
        plain(ancestor, True)
    entries = list(islice(path.iterdir(), 4))
    require(len(entries) <= 3 and all(entry.name in ("scratch", "stdout.txt", "stderr.txt") for entry in entries),
            "Only original host scratch/log entries allowed; no prior outputs")
    session = None
    for entry in entries:
        plain(entry, entry.name == "scratch")
        if entry.name != "scratch":
            continue
        children = list(islice(entry.iterdir(), 2))
        require(len(children) <= 1, "Foreign or excess host scratch children")
        if not children:
            continue
        require(isinstance(live_tempdir, str) and live_tempdir, "Exact live bpy.app.tempdir identity required")
        live = Path(live_tempdir)
        require(live.is_absolute() and live.parent == entry and live == children[0],
                "Only the exact immediate live Blender session child is allowed")
        plain(live, True)
        require(live.resolve(strict=True) == live, "Redirected live Blender session path")
        require(not next(live.iterdir(), None), "Live startup session must be empty")
        session = str(live)
    return {"output": str(path.resolve(strict=True)), "live_session_directory": session,
            "identity_source": "live sys.modules['bpy'].app.tempdir after exact GO" if session else "original empty scratch policy",
            "initial_entries": sorted(entry.name for entry in entries), "session_contents_limit": 0,
            "plain_ancestors_verified": True, "foreign_entries_allowed": False}


def install_output_admission(legacy):
    """Consume the historical exact GO gate once, then assert that admission."""
    original_main, original_await = legacy.main, legacy.await_host
    state = {"admitted": False, "used": False}
    startup = {}

    def verified_gate():
        require(state["admitted"], "Exact historical GO admission required")

    def private_output(stage, path):
        verified_gate()
        module = sys.modules.get("bpy")  # Never import bpy before admission.
        live = getattr(getattr(module, "app", None), "tempdir", None)
        startup.update(validate_initial_output(stage, path, live_tempdir=live))
        print("BRICK_UV_STARTUP_ADMISSION=" + json.dumps(startup, sort_keys=True), flush=True)
        return Path(startup["output"])

    def main(stage, output, save_blend=False):
        require(not state["used"], "One owned render stage per process; no retry")
        state["used"] = True
        original_await()  # The unchanged historical function validates b'GO\\n'.
        state["admitted"] = True
        try:
            return original_main(stage, output, save_blend)
        finally:
            state["admitted"] = False

    legacy.private_output, legacy.await_host, legacy.main = private_output, verified_gate, main
    return startup


def srgb_decode(encoded):
    """IEC/glTF sRGB transfer, followed by one explicit F32 storage rounding."""
    import numpy as np
    values = np.asarray(encoded, dtype=np.float64)
    require(np.isfinite(values).all(), "Finite encoded RGB required")
    values = np.clip(values, 0, 1)
    return np.where(values <= .04045, values / 12.92,
                    ((values + .055) / 1.055) ** 2.4).astype("<f4")


def atlas_payloads(field):
    """Return V-up F32 RGBA images; original input channels remain untouched."""
    import numpy as np
    require(isinstance(field, np.ndarray) and field.dtype == np.float32
            and field.ndim == 3 and field.shape[2] == 6
            and 0 < field.shape[0] <= ATLAS_SIZE and 0 < field.shape[1] <= ATLAS_SIZE
            and np.isfinite(field).all(), "Finite bounded F32[H,W,6] reference atlas required")
    rgb = np.ones((*field.shape[:2], 4), dtype="<f4")
    rgb[:, :, :3] = srgb_decode(field[:, :, :3])
    mr = np.ones_like(rgb)
    mr[:, :, 1] = np.clip(field[:, :, 4], 0, 1)
    mr[:, :, 2] = np.clip(field[:, :, 3], 0, 1)
    return rgb, mr


def float_image(bpy, name, pixels, array_digest):
    """Create an in-memory linear data image, with exact F32 pixel readback."""
    import numpy as np
    require(pixels.dtype == np.float32 and pixels.ndim == 3 and pixels.shape[2] == 4,
            "F32 RGBA image payload required")
    image = bpy.data.images.new(name, width=pixels.shape[1], height=pixels.shape[0],
                                alpha=True, float_buffer=True)
    image.colorspace_settings.name = "Non-Color"
    image.alpha_mode = "STRAIGHT"
    image.pixels.foreach_set(np.ascontiguousarray(pixels).ravel())
    actual = np.empty(pixels.size, dtype="<f4")
    image.pixels.foreach_get(actual)
    require(image.is_float and list(image.size) == [pixels.shape[1], pixels.shape[0]]
            and image.colorspace_settings.name == "Non-Color"
            and np.array_equal(actual.reshape(pixels.shape), pixels), "F32 linear image readback differs")
    return image, {"name": image.name, "shape": list(pixels.shape), "float_buffer": True,
                   "colorspace": image.colorspace_settings.name, "payload_sha256": array_digest(actual),
                   "readback_exact": True, "rows": "V-up; no PNG-style row flip", "alpha": "opaque 1"}


def reference_material(legacy, bpy, obj, values):
    """Same shader defaults as baked, with an unquantized F32 UV atlas input."""
    import numpy as np
    path = legacy.BAKE / "reference-field.npy"
    field = np.load(path, mmap_mode="r", allow_pickle=False)
    require(field.shape == (ATLAS_SIZE, ATLAS_SIZE, 6), "Fixed 1024-square six-channel reference required")
    rgb, mr = atlas_payloads(field)
    material, shader, nodes, links = legacy.new_material(bpy, "F32 covered-texel atlas; fixed derived padding")
    # Retain authentic per-corner source rows as unused provenance, never as RGB.
    require(values.dtype == np.float32 and values.shape == (len(obj.data.vertices), 6)
            and np.isfinite(values).all(), "Authentic F32 source rows required")
    attributes = {}
    for index, channel in enumerate(legacy.CHANNELS):
        attribute = obj.data.attributes.new(name="Original_" + channel, type="FLOAT", domain="POINT")
        attribute.data.foreach_set("value", np.ascontiguousarray(values[:, index]))
        actual = np.empty(len(values), dtype="<f4")
        attribute.data.foreach_get("value", actual)
        require(np.array_equal(actual, values[:, index]), "Unused authentic FLOAT/POINT provenance changed")
        attributes[channel] = legacy.array_digest(actual)
    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    textures, descriptions = {}, {}
    for label, pixels in (("Linear RGB", rgb), ("Metallic Roughness", mr)):
        image, description = float_image(bpy, "F32 atlas " + label, pixels, legacy.array_digest)
        node = nodes.new("ShaderNodeTexImage")
        node.image, node.interpolation, node.extension = image, "Linear", "EXTEND"
        links.new(uv.outputs["UV"], node.inputs["Vector"])
        textures[label], descriptions[label] = node, description
    links.new(textures["Linear RGB"].outputs["Color"], shader.inputs["Base Color"])
    separate = nodes.new("ShaderNodeSeparateColor")
    separate.mode = "RGB"
    links.new(textures["Metallic Roughness"].outputs["Color"], separate.inputs["Color"])
    links.new(separate.outputs["Green"], shader.inputs["Roughness"])
    links.new(separate.outputs["Blue"], shader.inputs["Metallic"])
    require(not shader.inputs["Alpha"].is_linked and shader.inputs["Alpha"].default_value == 1,
            "Reference shader must remain opaque")
    obj.data.materials.append(material)
    return {"reference_kind": "F32 covered-texel spatial atlas with fixed derived PNG padding",
            "reference_scope": REFERENCE_SCOPE, "continuous_source_field_shader": False,
            "reference_field_file": str(path), "reference_field_sha256": digest(path),
            "reference_field_payload_sha256": legacy.array_digest(field), "reference_field_shape": list(field.shape),
            "RGB_conversion": "clamp encoded F32 RGB then IEC sRGB decode per texel; float Non-Color image avoids second decoding",
            "metallic_roughness": "clip source F32 per texel to [0,1]; G=roughness, B=metallic; identical atlas interpolation",
            "reference_images": descriptions, "UV": "UVMap; V-up source arrays", "filter": "Linear", "extension": "EXTEND",
            "raw_FLOAT_POINT_sha256": attributes, "raw_POINT_attributes_used_by_shader": False,
            "original_atlas_alpha_range": [float(field[:, :, 5].min()), float(field[:, :, 5].max())],
            "original_alpha_preserved_in_source_and_unused_POINT_attribute": True, "shader_alpha": "opaque 1"}


def receipt_with_reference(record, adapter_sha256):
    """Replace inherited old RGB prose while preserving all measurements/status."""
    result = dict(record)
    result["renderer_adapter"] = {"file": str(Path(__file__).resolve()), "sha256": adapter_sha256,
                                  "historical_renderer": str(HISTORICAL), "historical_sha256": HISTORICAL_SHA256}
    result["reference_scope"] = REFERENCE_SCOPE
    result["continuous_source_field_shader"] = False
    if "RGB_semantic_difference" in result:
        result["RGB_semantic_difference"] = RGB_COMPARISON
    result["panel_roles"] = {"direct.png": "F32 covered-texel source atlas; fixed derived PNG padding",
                             "baked.png": "Freshly imported standard GLB with byte-packed atlas"}
    return result


def load_renderer():
    """Configure only the pinned API callbacks; never import bpy here."""
    require(HISTORICAL.is_file() and not HISTORICAL.is_symlink(), "Plain historical renderer required")
    require(digest(HISTORICAL) == HISTORICAL_SHA256, "Historical renderer pin differs")
    spec = importlib.util.spec_from_file_location("seam_uv_pinned_renderer", HISTORICAL)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    require(digest(HISTORICAL) == HISTORICAL_SHA256, "Historical renderer changed while loading")
    legacy.PRIVATE, legacy.BAKE = PRIVATE, BAKE
    # Keep the historical loader's first four positional NPY inputs intact.
    legacy.INPUT_NAMES += ("sampled-field.npy", "reference-field.npy")
    adapter = Path(__file__).resolve()
    adapter_sha256 = digest(adapter)
    legacy.BASELINE_HASHES = {**legacy.BASELINE_HASHES, HISTORICAL: HISTORICAL_SHA256, adapter: adapter_sha256}
    legacy.direct_material = lambda bpy, obj, values: reference_material(legacy, bpy, obj, values)
    startup = install_output_admission(legacy)
    original_save_json = legacy.save_json
    legacy.save_json = lambda path, record: original_save_json(path, {**receipt_with_reference(record, adapter_sha256),
                                                                   "startup_output_admission": dict(startup)})
    return legacy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("export", "compare"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-blend", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    require(not args.save_blend or args.stage == "export", "Blend save is export-stage only")
    load_renderer().main(args.stage, args.output, args.save_blend)
