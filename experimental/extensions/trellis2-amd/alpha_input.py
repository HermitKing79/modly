"""Explicit pre-segmented input contract; no model construction or fallback."""
import io

MAX_DIMENSION = 4096
MAX_PIXELS = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024


def validate_mode(mode):
    if mode not in ("rembg", "alpha-only"):
        raise ValueError("Image mode must explicitly be rembg or alpha-only")
    return mode


def _mask(image):
    from PIL import Image
    if not isinstance(image, Image.Image) or image.mode != "RGBA":
        raise ValueError("alpha-only requires a pre-segmented RGBA image; conversion is disabled")
    if (min(image.size) < 1 or max(image.size) > MAX_DIMENSION or
            image.width * image.height > MAX_PIXELS):
        raise ValueError("alpha-only image exceeds dimension/pixel limit")
    alpha = image.getchannel("A")
    if alpha.getextrema()[0] != 0:
        raise ValueError("alpha-only requires fully transparent background pixels")
    # Match the real pipeline's strict alpha > 0.8 * 255 foreground test.
    bbox = alpha.point(lambda value: 255 if value > 204 else 0).getbbox()
    if bbox is None or bbox[2] - bbox[0] < 3 or bbox[3] - bbox[1] < 3:
        raise ValueError("alpha-only requires a nondegenerate foreground above alpha 204")
    return [bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1]


def validate_alpha_image(image):
    from PIL import Image
    bbox = _mask(image)
    # Upstream resizes before computing its crop. A small mask may disappear.
    scale = min(1, 1024 / max(image.size))
    resized_bbox = bbox
    if scale < 1:
        size = (int(image.width * scale), int(image.height * scale))
        if min(size) < 1:
            raise ValueError("alpha-only image becomes degenerate after upstream resize")
        resized_bbox = _mask(image.resize(size, Image.Resampling.LANCZOS))
    return {"mode": "alpha-only", "size": list(image.size), "foreground_bbox": bbox,
            "resized_foreground_bbox": resized_bbox, "rembg_required": False}


def decode_alpha_image(data):
    from PIL import Image
    if not isinstance(data, bytes) or not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("alpha-only image bytes are empty or exceed the byte limit")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("alpha-only requires a single-frame image")
            # Dimensions/mode checked before decompression/copy.
            if (image.mode != "RGBA" or min(image.size) < 1 or max(image.size) > MAX_DIMENSION or
                    image.width * image.height > MAX_PIXELS):
                raise ValueError("alpha-only requires RGBA within dimension/pixel limits")
            result = image.copy()
        return result, validate_alpha_image(result)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("alpha-only image could not be decoded: " + str(exc)) from exc


def preprocess_alpha_image(pipeline, image):
    """Call the real CPU preprocessing method with RMBG deliberately absent.

    This accepts a constructor-created shell; it never calls from_pretrained,
    instantiates an image extractor, or populates model slots. Model loading is
    outside this milestone. The ordinary upstream RMBG factory stays intact.
    """
    validate_alpha_image(image)
    if pipeline.rembg_model is not None:
        raise ValueError("alpha-only boundary requires no constructed RMBG model")
    return pipeline.preprocess_image(image)
