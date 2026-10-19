"""Independent private artifact/image readback and one labelled contact sheet."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,struct,importlib.util,os
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-uv-bake-18c0c92/qualified'

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    if sys.stdin.buffer.readline(64)!=b'GO\n':raise ValueError('Owned host GO required')
    original=ROOT/'docs/compatibility/amd-brick-defect-diagnosis-evidence/diagnose.py'
    if digest(original)!='8fa7c617324ba636856b48f47a995ee64cd2a85f3c26d86d8760ec638b163529':raise ValueError('Original readonly guard')
    spec=importlib.util.spec_from_file_location('readonly_input_verifier',original);auth=importlib.util.module_from_spec(spec);spec.loader.exec_module(auth);auth.P=P
    sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    sys.path.insert(0,str(E));from sampler import pack_native_textures
    b=P/'bake';c=P/'compare';out=P/'finish'
    mesh=auth.PINS['vertices'][0].parent
    v=np.load(mesh/'vertices.npy',mmap_mode='r');f=np.load(mesh/'faces.npy',mmap_mode='r')
    pv,pf,ids,fi,dp=[np.load(b/n,allow_pickle=False) for n in ('vertices.npy','faces.npy','source-vertex-ids.npy','source-face-ids.npy','direct.npy')]
    assert np.array_equal(pv,v[ids]) and np.array_equal(ids[pf],f[fi]),'Original complete ordered triangles/positions changed'
    field=auth.material_features(auth.PINS['material'][0],np)
    assert np.array_equal(dp,field[ids]),'Original unclamped six-channel vertex values differ'
    samples=np.load(b/'sampled-field.npy',allow_pickle=False);mask=np.load(b/'coverage.npy',allow_pickle=False)
    assert samples.shape==(1024,1024,6) and samples.dtype==np.float32 and mask.dtype==np.bool_ and np.isfinite(samples).all()
    packed=pack_native_textures(samples)
    maps={}
    for name,expected in [('base-color.png',packed['base_color_rgba']),('MR.png',packed['metallic_roughness_rgb'])]:
        with Image.open(b/name) as image:
            image.load();actual=np.array(image)[::-1]
        assert np.array_equal(actual[mask],expected[mask]),'Learned covered map bytes differ'
        maps[name]=dict(sha256=digest(b/name),bytes=(b/name).stat().st_size,dimensions=list(actual.shape),covered_pixels_byte_exact=True)
    scenes={}
    for stage in ('export','compare'):
        r=json.loads((P/stage/(stage+'-checkpoint.json')).read_text());assert r['classification']=='PASS' and r['geometry_unchanged'] and r['sources_unchanged']
        scenes[stage]=dict(verified=True,sha256=digest(P/stage/(stage+'-checkpoint.json')))
    images={};arrays={}
    for name in ('direct','baked'):
        path=c/(name+'.png')
        with Image.open(path) as im:
            assert im.format=='PNG' and im.size==(640,640) and im.mode=='RGB';im.load();arrays[name]=np.array(im)
        assert digest(path)==json.loads((c/(name+'-checkpoint.json')).read_text())['sha256']
        images[name]=dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size,dimensions=[640,640])
    sheet=Image.new('RGB',(1280,722),'#20252c');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',20)
    draw.text((18,9),'LEFT: original direct-row learned PBR     RIGHT: UV spatial PBR bake',font=font,fill='white')
    sheet.paste(Image.fromarray(arrays['direct']),(0,43));sheet.paste(Image.fromarray(arrays['baked']),(640,43))
    draw.text((18,690),'Same original triangles, +X camera, CPU lighting and +2 EV. No repairs or generated colours.',font=font,fill='white')
    target=out/'comparison.png';sheet.save(target)
    with Image.open(target) as im:assert im.size==(1280,722);im.load()
    values=samples[mask]
    record=dict(classification='PASS',independent_readback=True,original_vertices_faces_winding_and_order_exact=True,original_unclamped_direct_six_channels_exact=True,
                covered_map_bytes_exact=True,learned_spatial_channel_ranges=[values.min(0).tolist(),values.max(0).tolist()],
                covered_clipped_value_counts=((values<0)|(values>1)).sum(0).tolist(),raw_float_samples_retained=True,alpha_preview='OPAQUE; source alpha retained in original field, sampled floats and RGBA PNG',
                vertices=len(pv),triangles=len(pf),maps=maps,scenes=scenes,images=images,
                contact_sheet=dict(path=str(target),sha256=digest(target),bytes=target.stat().st_size,dimensions=[1280,722]),
                portable_patch=dict(path=str(P/'export/patch.glb'),sha256=digest(P/'export/patch.glb'),bytes=(P/'export/patch.glb').stat().st_size))
    with (out/'verification.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
