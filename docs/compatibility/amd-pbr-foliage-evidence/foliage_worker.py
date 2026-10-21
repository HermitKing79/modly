"""CPU-only fixed foliage selection/UV admission; no material sampling or bake."""
import sys,os,time,json,importlib.util,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-pbr-foliage-ecb15aa'
sys.path.insert(0,str(ROOT/'docs/compatibility/amd-brick-pbr-mip-evidence'))
from assets import authenticate,load_auth,digest,save,require

def main():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Owned binary GO')
    phase1=json.loads((P/'brick/result.json').read_text())
    require(phase1['classification']=='PASS','Genuine brick import must pass first')
    auth=load_auth();auth.P=P/'foliage';sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    sys.path.insert(0,str(E));from foliage_select import visible_faces,ROI
    from foliage_layout import build_layout
    began=time.monotonic();out=P/'foliage/run'
    sheet=Image.new('RGB',(1280,690),'#20252c');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
    draw.text((14,10),'ONE FRESH BLENDER IMPORT: near (left) / fourfold orthographic scale (right)',fill='white',font=font)
    for index,name in enumerate(('near','far')):
        path=P/'brick'/(name+'.png');record=next(row for row in phase1['views'] if row['name']==name)
        require(digest(path)==record['sha256'] and path.stat().st_size==record['bytes'],'Exact authenticated Blender view')
        with Image.open(path) as im:
            im.load();require(im.size==(640,640) and im.mode=='RGB','Actual private Blender images')
            sheet.paste(im,(640*index,50))
    sheet.save(out/'brick-import-comparison.png')
    _,identities,watched,_=authenticate();before={str(p):digest(p) for p in watched}
    v=np.load(auth.PINS['vertices'][0],mmap_mode='r',allow_pickle=False);f=np.load(auth.PINS['faces'][0],mmap_mode='r',allow_pickle=False)
    pp=ROOT/'docs/compatibility/amd-brick-defect-diagnosis-evidence/projection.py'
    require(digest(pp)=='0ed20e6d03424404f3b3aebc64cae2825a0d1c8d241fd494ce50e742471eb840','Original projection pin')
    spec=importlib.util.spec_from_file_location('foliage_original_projection',pp);projection=importlib.util.module_from_spec(spec);sys.modules[spec.name]=projection;spec.loader.exec_module(projection)
    camera=next(x for x in json.loads(auth.PINS['render_metadata'][0].read_text())['views'] if x['name']=='pos-x')
    screen=projection.project(v,camera,1024,1024)
    selected,nomination=visible_faces(v,f,screen)
    source_vertex_ids,inv=np.unique(f[selected],return_inverse=True)
    pv=np.array(v[source_vertex_ids]);pf=inv.reshape(-1,3).astype(np.int32)
    require(np.array_equal(pv[pf].view(np.uint32),v[f[selected]].view(np.uint32)),'Exact original oriented triangle positions')
    require(np.array_equal(source_vertex_ids[pf],f[selected]),'Original vertex/face lineage')
    features=auth.material_features(auth.PINS['material'][0],np);direct=np.array(features[source_vertex_ids])
    # Same original coordinate-row relationship, independently authenticated.
    sys.path.insert(0,str(ROOT/'docs/compatibility/amd-brick-seam-uv-bake-evidence'));from common import tensor
    require(np.array_equal(tensor(auth.PINS['material'][0],'coordinates',np),tensor(auth.PINS['geometry'][0],'coordinates',np)),'Full material coordinate identity')
    for name,array in [('vertices',pv),('faces',pf),('source-face-ids',selected),('source-vertex-ids',source_vertex_ids),('direct',direct)]:
        with (out/(name+'.npy')).open('xb') as stream:np.save(stream,array,allow_pickle=False)
    image=Image.open(auth.PINS['image_ev2'][0]).convert('RGB');draw=ImageDraw.Draw(image);draw.rectangle(ROI,outline='cyan',width=2);image.save(out/'selection.png')
    crop=image.crop((ROI[0]-8,ROI[1]-8,ROI[2]+8,ROI[3]+8));crop.resize((672,640)).save(out/'selection-closeup.png')
    try:
        result=build_layout(pv,pf,selected,source_vertex_ids)
    except ValueError as exc:
        result=dict(classification='BLOCKED',report=dict(reason=str(exc)))
    layout=result.pop('layout',None);result.pop('portable_uv_f32',None)
    if layout is not None:
        for name,array in [('uv',layout.uv),('face-island-ids',layout.face_island_ids),('island-rects',layout.island_rects),('raster-ids',layout.raster_ids),('barycentrics',layout.barycentrics)]:
            with (out/(name+'.npy')).open('xb') as stream:np.save(stream,array,allow_pickle=False)
    require(before=={str(p):digest(p) for p in watched},'Original assets unchanged')
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir() if p.suffix in ('.npy','.png')}
    qualified=result['classification']=='GO' and result['report'].get('portable_f32_status')=='GO'
    save(out/'result.json',dict(classification='PASS' if qualified else 'BLOCKED',phase='CPU_foliage_feasibility_no_bake',nomination=nomination,layout=result['report'],source_inputs=identities,source_geometry_unchanged=True,material_rows_exact=True,files=files,elapsed_seconds=time.monotonic()-began))
    print('FOLIAGE_FEASIBILITY_'+result['classification'],flush=True)

if __name__=='__main__':main()
