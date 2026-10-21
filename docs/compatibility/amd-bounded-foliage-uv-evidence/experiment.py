"""One owned CPU selection/UV qualification. Never samples materials or exports."""
import sys
sys.dont_write_bytecode = True
import hashlib, importlib.util, json, os, time
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-bounded-foliage-64d05aa'
OLD = ROOT / 'docs/compatibility/amd-pbr-foliage-evidence'
sys.path.insert(0, str(ROOT / 'docs/compatibility/amd-brick-pbr-mip-evidence'))
from assets import authenticate, load_auth, digest, save, require


def pinned(name, path, sha):
    require(digest(path) == sha, 'Historical source identity: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    require(sys.stdin.buffer.readline(64) == b'GO\n', 'Owned binary GO')
    began = time.monotonic()
    auth = load_auth(); auth.P = P
    sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    sys.path.insert(0, str(E))
    from diagnostics import analyze_topology, coverage_proxies

    nomination_path = E / 'nomination.json'
    require(digest(nomination_path) == 'eecf4d0b880c28aab7fb660a6a573d8498291936ee14ac039e05b09c7572da25', 'Frozen nomination before count')
    nominated = json.loads(nomination_path.read_text())
    roi = tuple(nominated['ROI'])
    require(roi == (346,393,393,436), 'Only one approved smaller ROI')
    _, identities, watched, _ = authenticate()
    before = {str(path): digest(path) for path in watched}
    selector = pinned('bounded_original_selector', OLD / 'foliage_select.py',
        '1e04a68faf7a9e5c23e4f23b1cd5b9b93b6d02b5aee77f89272b403ea358b57a')
    selector.ROI = roi  # Only the visually frozen nomination changes.
    projection = pinned('bounded_original_projection', ROOT / 'docs/compatibility/amd-brick-defect-diagnosis-evidence/projection.py',
        '0ed20e6d03424404f3b3aebc64cae2825a0d1c8d241fd494ce50e742471eb840')
    v = np.load(auth.PINS['vertices'][0], mmap_mode='r', allow_pickle=False)
    f = np.load(auth.PINS['faces'][0], mmap_mode='r', allow_pickle=False)
    require(v.shape == (1957891,3) and v.dtype == np.float32 and f.shape == (4173584,3) and f.dtype == np.int32, 'Full authentic source geometry schema')
    camera = next(row for row in json.loads(auth.PINS['render_metadata'][0].read_text())['views'] if row['name'] == 'pos-x')
    projected = projection.project(v, camera, 1024, 1024)
    out = P / 'run'
    with Image.open(auth.PINS['image_ev2'][0]) as im:
        im.load(); image = im.convert('RGB')
    context = image.copy(); ImageDraw.Draw(context).rectangle(roi, outline='cyan', width=2)
    context.save(out / 'nomination.png')
    try:
        selected, selection = selector.visible_faces(v, f, projected)
    except ValueError as exc:
        # The historical budget exception is preserved verbatim; no reselection.
        save(out / 'result.json', dict(classification='BLOCKED', stage='selection', ROI=list(roi),
            error=str(exc), source_inputs=identities, elapsed_seconds=time.monotonic()-began,
            nomination_SHA256=digest(nomination_path), no_UV_attempt=True))
        print('SELECTION_BLOCKED', str(exc), flush=True)
        return
    save(out/'selection-admission.json', dict(**selection,
        nomination_SHA256=digest(nomination_path),
        original_face_ID_buffer_SHA256=hashlib.sha256(selected.tobytes()).hexdigest()))
    sv, inv = np.unique(f[selected], return_inverse=True)
    sv = sv.astype(np.int64)
    pv = np.array(v[sv]); pf = inv.reshape(-1,3).astype(np.int32)
    require(np.array_equal(pv[pf].view(np.uint32), v[f[selected]].view(np.uint32)), 'Original oriented corners bit-exact')
    require(np.array_equal(sv[pf], f[selected]), 'Complete original source vertex mapping')
    topology = analyze_topology(pf, sv, f)
    coverage = coverage_proxies(projected.screen[f[selected]], roi, nominated['coverage_proxies'])
    require(all(row['covered_pixels'] > 0 for row in coverage['proxies'].values()), 'Both nominated leaf/vine image windows must contain selected projected surface')
    # Exact coordinate-row identity, not an inference from equal row counts.
    common = pinned('bounded_original_common', ROOT / 'docs/compatibility/amd-brick-seam-uv-bake-evidence/common.py',
        'e239c7e031bfa5a072d0c02b4804d96b73ed8ca7dc55145be46882ca0b02e96b')
    require(np.array_equal(common.tensor(auth.PINS['material'][0], 'coordinates', np),
                           common.tensor(auth.PINS['geometry'][0], 'coordinates', np)), 'Entire original material/geometry coordinate ordering')
    direct = np.array(auth.material_features(auth.PINS['material'][0], np)[sv])
    require(np.isfinite(direct).all(), 'Finite original selected material values')

    def array(name, value):
        path = out / (name + '.npy')
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        read = np.load(path, allow_pickle=False)
        require(read.dtype == value.dtype and read.shape == value.shape and read.tobytes() == value.tobytes(), 'Byte-exact NPY readback: ' + name)

    for name, value in [('vertices',pv), ('faces',pf), ('source-face-ids',selected), ('source-vertex-ids',sv), ('direct',direct)]:
        array(name, value)
    selection.update(vertices=len(pv), source_face_ids_sha256=digest(out/'source-face-ids.npy'),
        source_vertex_ids_sha256=digest(out/'source-vertex-ids.npy'), topology=topology,
        coverage=coverage, material_coordinate_identity=True, original_oriented_corner_bits_preserved=True)
    save(out / 'selection.json', selection)  # Durable before the independent UV gate.
    overlay = image.convert('RGBA'); ink = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(ink)
    for triangle in projected.screen[f[selected]]:
        draw.polygon([tuple(x) for x in triangle], fill=(0,220,255,65))
    overlay = Image.alpha_composite(overlay, ink).convert('RGB')
    draw = ImageDraw.Draw(overlay); draw.rectangle(roi, outline='cyan', width=1)
    for key, box in nominated['coverage_proxies'].items():
        draw.rectangle(box, outline='yellow' if key == 'leaf' else 'lime', width=1)
    crop_box = (roi[0]-12, roi[1]-12, roi[2]+12, roi[3]+12)
    sheet = Image.new('RGB',(1200,650),'#20252c'); d=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',17)
    d.text((12,8),'Original +X / fixed ROI and selected original-triangle footprints (cyan)',fill='white',font=font)
    sheet.paste(context.resize((600,600)),(0,45))
    sheet.paste(overlay.crop(crop_box).resize((600,566),resample=Image.Resampling.NEAREST),(600,45))
    d.text((610,614),'Yellow: leaf window; green: curved-vine window. Cropped front surface.',fill='white',font=font)
    sheet.save(out / 'selection.png')

    layout_module = pinned('bounded_original_layout', OLD / 'foliage_layout.py',
        'de7a04fbb1958207ccfe8eee82fb59aa334beefcbbf012df8268804b9a9cceed')
    try:
        result = layout_module.build_layout(pv, pf, selected, sv)
    except ValueError as exc:
        result = dict(classification='BLOCKED', report=dict(reason=str(exc)))
    layout = result.get('layout')
    if layout is not None:
        for name, value in [('uv',layout.uv),('uv-faces',layout.faces),('face-island-ids',layout.face_island_ids),
             ('island-rects',layout.island_rects),('raster-ids',layout.raster_ids),('barycentrics',layout.barycentrics)]:
            array(name,value)
    if 'portable_uv_f32' in result: array('uv-f32',result['portable_uv_f32'])
    require(before == {str(path):digest(path) for path in watched}, 'Original assets unchanged after qualification')
    files = {path.name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path in out.iterdir() if path.suffix in ('.npy','.png')}
    qualified = result['classification']=='GO' and result['report'].get('portable_f32_status')=='GO'
    save(out/'result.json',dict(classification='PASS' if qualified else 'BLOCKED', stage='UV_feasibility_only',
        selection=selection, layout=result['report'], files=files, source_inputs=identities,
        elapsed_seconds=time.monotonic()-began, source_geometry_unchanged=True,
        nomination_SHA256=digest(nomination_path), no_bake_or_export=True))
    print('BOUNDED_FOLIAGE_' + ('PASS' if qualified else 'BLOCKED'), flush=True)


if __name__ == '__main__': main()
