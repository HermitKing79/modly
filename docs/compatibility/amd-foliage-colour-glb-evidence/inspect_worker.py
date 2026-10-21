"""Fresh post-import numeric/PNG readback, without Blender or new source samples."""
import sys
sys.dont_write_bytecode=True
import importlib.util,json,time
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import P,ROOT,DECISION,load,load_auth,plain,digest,require,save


def stats(error,np):
    absolute=np.abs(error)
    return dict(rmse=float(np.sqrt(np.mean(error**2))),p95_abs=float(np.percentile(absolute,95)),max_abs=float(absolute.max()))


def main():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Owned binary GO')
    auth=load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from PIL import Image,ImageDraw
    began=time.monotonic();data,identities,watched=load(np)
    before={str(p):digest(p) for p in watched}
    render=P/'render-qualified';out=plain(P/'inspect')
    result=json.loads(plain(render/'result.json').read_text())
    require(result['classification']=='PASS','Genuine Blender comparison required')
    for name in ('imported-corner-linear-COLOR0.npy','imported-corner-source-vertex-ids.npy'):
        path=plain(render/name)
        require(result[name]['sha256']==digest(path) and result[name]['bytes']==path.stat().st_size,'Recorded imported corner array identity')
    actual=np.load(render/'imported-corner-linear-COLOR0.npy',allow_pickle=False)
    ids=np.load(render/'imported-corner-source-vertex-ids.npy',allow_pickle=False)
    require(actual.dtype==np.float32 and actual.shape==(4531,3,3) and np.isfinite(actual).all() and ((actual>=0)&(actual<=1)).all(),'Actual original-ordered imported corner colours')
    require(ids.dtype==np.int64 and np.array_equal(ids,data['source-vertex-ids'][data['faces']]),'Imported original corner lineage')
    qf=np.load(DECISION/'query-faces.npy',allow_pickle=False);bary=np.load(DECISION/'barycentrics.npy',allow_pickle=False)
    raw=np.load(DECISION/'source-spatial.npy',allow_pickle=False)
    linear=np.einsum('qi,qij->qj',bary,actual[qf].astype(np.float64))
    encoded=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
    targets=np.clip(raw.astype(np.float64),0,1)
    base=14531;seam=np.arange(44420).reshape(-1,20)
    cohorts=dict(centroids=np.arange(4531),area_weighted=np.arange(4531,14531),required_surface=np.arange(14531),seam_edge=(seam[:,:10]+base).ravel(),seam_near=(seam[:,10:]+base).ravel())
    metrics={}
    for name,index in cohorts.items():
        row=stats(encoded[index]-targets[index,:3],np)
        row['pass']=row['rmse']<=.02 and row['p95_abs']<=.05 and row['max_abs']<=.15
        metrics[name]=row
    require(all(x['pass'] for x in metrics.values()),'Actual imported colours fail original source-spatial RGB gates')
    fixed=np.array([.00011082693761409814,.999995986777544,1.],np.float64)
    scalar={name:np.sqrt(np.mean((fixed-targets[index,3:])**2,axis=0)).tolist() for name,index in cohorts.items()}
    require(all(max(x)<=.02 for x in scalar.values()),'Fixed factors/OPAQUE fail unchanged scalar gates')
    images={};files={}
    view_receipts={row['name']:row for row in result['views']}
    for name in ('reference-near','imported-near','reference-far','imported-far'):
        path=plain(render/(name+'.png'))
        require(view_receipts[name]['sha256']==digest(path) and view_receipts[name]['bytes']==path.stat().st_size,'Original rendered PNG receipt identity')
        with Image.open(path) as image:image.verify()
        with Image.open(path) as image:
            require(image.size==(640,640) and image.mode=='RGBA','640-square RGBA diagnostic PNG')
            images[name]=np.asarray(image).copy()
        files[name+'.png']=dict(bytes=path.stat().st_size,sha256=digest(path))
    differences={}
    for view in ('near','far'):
        a=images['reference-'+view];b=images['imported-'+view]
        am=a[:,:,3]>=128;bm=b[:,:,3]>=128;union=am|bm
        require(union.any(),'Nonempty rendered foliage')
        iou=float((am&bm).sum()/union.sum());row=stats((a[:,:,:3].astype(float)-b[:,:,:3])/255.,np)
        visible=stats((a[:,:,:3][union].astype(float)-b[:,:,:3][union])/255.,np)
        row.update(silhouette_IoU=iou,visible_RGB=visible)
        require(iou>=.999 and all(x['rmse']<=.02 and x['p95_abs']<=.05 and x['max_abs']<=.15 for x in (row,visible)),'Matched PNG appearance gates failed')
        differences[view]=row
    sheet=Image.new('RGB',(1280,1340),(180,180,180));draw=ImageDraw.Draw(sheet)
    for row,view in enumerate(('near','far')):
        for col,kind in enumerate(('reference','imported')):
            image=Image.fromarray(images[kind+'-'+view]);background=Image.new('RGBA',image.size,(180,180,180,255));background.alpha_composite(image)
            sheet.paste(background.convert('RGB'),(col*640,row*670+30));draw.text((col*640+12,row*670+8),kind+' '+view+' - fixed learned vertex colour',(15,15,15))
    path=out/'comparison.png'
    with path.open('xb') as stream:sheet.save(stream,format='PNG')
    with Image.open(path) as image:image.verify()
    files['comparison.png']=dict(bytes=path.stat().st_size,sha256=digest(path))
    require(before=={str(p):digest(p) for p in watched},'All authenticated sources unchanged')
    save(out/'result.json',dict(classification='PASS',CPU_only=True,existing_queries_rechecked=58951,imported_RGB_source_spatial=metrics,fixed_M_R_OPAQUEalpha_RMSE=scalar,PNG_differences=differences,files=files,source_unchanged=True,elapsed_seconds=time.monotonic()-began,scope='Viewer byte-colour quantisation and transport compared separately; no new sampling, fitting, baking or neural work'))
    print('FOLIAGE_IMPORT_INSPECTION_PASS',flush=True)


if __name__=='__main__':main()
