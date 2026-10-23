"""Bounded CPU views of the unchanged native exposed-voxel diagnostic surface.

OBJ coordinates remain x/y/z = tensor D/H/W. W is therefore the displayed
Z/up axis. These views and counts describe occupancy; they do not grade shape.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import amd_sparse_occupancy_data as data

SIZE = (900, 900)
PNG_LIMIT = 2 * 1024**2
SECONDS = 60


def _validate(mask):
    import numpy as np
    if (type(mask) is not np.ndarray or mask.shape != (64, 64, 64)
            or mask.dtype != np.dtype(bool) or not mask.flags.c_contiguous
            or not np.isfinite(mask).all()):
        raise ValueError('Require finite contiguous native bool[64,64,64] mask')


def _diagnostics(mask, check):
    import numpy as np
    check()
    coords = np.argwhere(mask)
    count = len(coords)
    bbox = None
    if count:
        minimum, maximum = coords.min(axis=0), coords.max(axis=0)
        bbox = dict(minimum=minimum.tolist(), maximum=maximum.tolist(),
                    spans=(maximum - minimum + 1).tolist())
    components = data.components(mask)
    check()
    boundary, slices = {}, {}
    for axis, label in enumerate(('D', 'H', 'W')):
        check()
        boundary[label + '-'] = int(np.take(mask, 0, axis=axis).sum())
        boundary[label + '+'] = int(np.take(mask, 63, axis=axis).sum())
        slices[label] = mask.sum(axis=tuple(other for other in range(3) if other != axis)).tolist()
    return dict(native_shape=[64, 64, 64], coordinate_order=['D', 'H', 'W'],
                occupied_count=count, occupancy_fraction=count / 262144,
                components=components, bounding_box=bbox,
                volume_boundary_counts=boundary, slice_counts=slices,
                diagnostic_only=True, threshold_changed=False)


def diagnostics(mask):
    """Actual six-connected components, inclusive bounds and native slice counts."""
    _validate(mask)
    return _diagnostics(mask, lambda: None)


def _cameras():
    a, b, c = 1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(3)
    return {
        'isometric-z-up': dict(screen_right=[-a, a, 0], screen_up=[-b, -b, 2*b],
                               toward_camera=[c, c, c]),
        'plus-x': dict(screen_right=[0, 1, 0], screen_up=[0, 0, 1], toward_camera=[1, 0, 0]),
        'plus-y': dict(screen_right=[-1, 0, 0], screen_up=[0, 0, 1], toward_camera=[0, 1, 0]),
    }


def _view(directory, name, camera, vertices, faces, check):
    from PIL import Image, ImageDraw, PngImagePlugin
    check()
    center = [(min(vertex[axis] for vertex in vertices) + max(vertex[axis] for vertex in vertices)) / 2
              for axis in range(3)]
    points = []
    for vertex in vertices:
        check()
        point = [vertex[axis] - center[axis] for axis in range(3)]
        points.append(tuple(sum(point[axis] * camera[basis][axis] for axis in range(3))
                            for basis in ('screen_right', 'screen_up', 'toward_camera')))
    span = max(max(point[axis] for point in points) - min(point[axis] for point in points)
               for axis in (0, 1))
    if not math.isfinite(span) or span <= 0:
        raise ValueError('Degenerate or nonfinite voxel projection')
    scale = 700 / span
    image = Image.new('RGB', SIZE, '#f3f5f7')
    draw = ImageDraw.Draw(image)

    def depth(face):
        check()
        return sum(points[index - 1][2] for index in face) / 4

    for face in sorted(faces, key=depth):
        check()
        p = [points[index - 1] for index in face]
        a = [p[1][axis] - p[0][axis] for axis in range(3)]
        b = [p[2][axis] - p[0][axis] for axis in range(3)]
        normal = [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
        if normal[2] <= 0:
            continue
        shade = int(100 + 125 * normal[2] / math.sqrt(sum(value*value for value in normal)))
        draw.polygon([(450 + point[0]*scale, 450 - point[1]*scale) for point in p],
                     fill=(shade-30, shade-10, shade), outline=(65, 75, 90))
    label = 'Native 64^3 exposed voxels: ' + name + '; x=D, y=H, z=W (up)'
    draw.text((20, 20), label, fill='#1a2530')
    draw.text((20, 40), 'Diagnostic occupancy only; original cells, faces and winding', fill='#1a2530')
    # A labelled orientation glyph uses the same basis as the voxel projection.
    for axis, axis_label in enumerate(('D (x)', 'H (y)', 'W (z/up)')):
        right, up = camera['screen_right'][axis], camera['screen_up'][axis]
        start = (80, 815)
        endpoint = (80 + 48*right, 815 - 48*up)
        if abs(right) + abs(up) > 0:
            draw.line([start, endpoint], fill='#1a2530', width=2)
            draw.text(endpoint, axis_label, fill='#1a2530')
        else:
            draw.text((20, 865), axis_label + ' points toward camera', fill='#1a2530')
    info = PngImagePlugin.PngInfo()
    info.add_text('coordinate_order', 'x=D,y=H,z=W')
    info.add_text('axis_labels', label)
    info.add_text('camera', json.dumps(camera, sort_keys=True))
    path = directory / ('voxel-preview-' + name + '.png')
    check()
    with path.open('xb') as stream:
        image.save(stream, format='PNG', pnginfo=info)
    check()
    if path.stat().st_size > PNG_LIMIT:
        raise ValueError('Voxel PNG exceeds2MiB cap')
    with Image.open(path) as restored:
        restored.load()
        if restored.size != SIZE or restored.mode != 'RGB' or restored.info['axis_labels'] != label:
            raise ValueError('Voxel view readback differs')
    check()
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                dimensions=list(SIZE), camera=camera, axis_labels=label)


def produce(directory, mask):
    """Retain original gated OBJ/PNG, then add three bounded labelled CPU views."""
    started = time.monotonic()
    _validate(mask)
    directory = Path(directory).resolve(strict=True)
    # FIRST export: unchanged original 50k-face/12MiB-OBJ/2MiB-PNG gates.
    # An original refusal propagates and prevents all extra views.
    original = data.write_preview(directory, mask)

    def check():
        if time.monotonic() - started > SECONDS:
            raise TimeoutError('Voxel preview exceeds60second total CPU budget')

    check()
    vertices, faces = data.surface(mask)
    if (len(vertices), len(faces)) != (original['vertices'], original['faces']):
        raise ValueError('Original voxel surface changed')
    check()
    views = {name: _view(directory, name, camera, vertices, faces, check)
             for name, camera in _cameras().items()}
    actual = _diagnostics(mask, check)
    elapsed = time.monotonic() - started
    if elapsed > SECONDS:
        raise TimeoutError('Voxel preview exceeds60second total CPU budget')
    return dict(original=original, vertices=len(vertices), faces=len(faces), views=views,
                diagnostics=actual, coordinate_order=['D', 'H', 'W'],
                diagnostic_only=True, cpu_only=True, elapsed_seconds=elapsed,
                total_seconds_limit=SECONDS, smoothing=False, threshold_changed=False)
