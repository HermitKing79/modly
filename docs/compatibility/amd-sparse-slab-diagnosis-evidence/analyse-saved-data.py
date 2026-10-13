"""CPU-only read-only analysis of existing artifacts; never load AI checkpoints."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent
occupancy=root/'docs/compatibility/amd-sparse-occupancy-evidence'
latent_path=root/'docs/compatibility/amd-sparse-structure-sampling-evidence/sparse-structure-latent.safetensors'
feature_path=root/'docs/compatibility/amd-dinov3-allocation-evidence/conditioning.safetensors'
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','triton','trellis2','cumesh','flex_gemm','o_voxel','nvdiffrast'):
        raise RuntimeError('Diagnosis prohibits AI/GPU imports')
    if event.startswith('socket.'):raise RuntimeError('Tensor analysis is offline')
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=Path(args[0]);suffix=p.suffix.lower()
        if suffix in ('.safetensors','.pt','.pth','.ckpt','.bin','.gguf') and p not in (latent_path,feature_path):
            raise RuntimeError('Only saved latent/features may be read; no AI weights: '+str(p))
sys.addaudithook(audit)
import numpy as np
from PIL import Image
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def container(path,key,shape,file_sha,limit):
    with path.open('rb') as f:blob=f.read(limit+1)
    assert len(blob)<=limit and hashlib.sha256(blob).hexdigest()==file_sha
    size=struct.unpack('<Q',blob[:8])[0];assert 0<size<=65536
    header=json.loads(blob[8:8+size]);tensor=header[key]
    assert set(header)=={'__metadata__',key} and tensor==dict(dtype='F32',shape=list(shape),data_offsets=[0,math.prod(shape)*4])
    payload=blob[8+size:];assert len(payload)==math.prod(shape)*4
    assert hashlib.sha256(payload).hexdigest()==header['__metadata__']['output_sha256']
    array=np.frombuffer(payload,dtype='<f4').reshape(shape);assert np.isfinite(array).all()
    return array,header['__metadata__']
latent,latent_meta=container(latent_path,'sparse_latent',(1,8,16,16,16),'8a4da87ae1bbc2d5e5186f300adf11597bf986cb58dfb24a564a2eec287b44f2',256*1024)
feature,feature_meta=container(feature_path,'cond',(1,1029,1024),'3f444d42191df07415a8877a3d8d3706cf1b129bedbbb2dfcc0221d5571be56a',5*1024**2)
run=json.loads((occupancy/'run/results.json').read_text());artifact=run['worker']['occupancy_artifacts']
arrays={}
for name,shape,dtype,cap in [('occupancy-logits.npy',(1,1,64,64,64),'<f4',2*1024**2),('occupancy-mask.npy',(64,64,64),'bool',512*1024),('occupancy-coordinates.npy',(1606,4),'<i4',5*1024**2)]:
    p=occupancy/name;assert p.stat().st_size<=cap and sha(p)==artifact['files'][name]['file_sha256']
    a=np.load(p,allow_pickle=False,max_header_size=4096);assert a.shape==shape and a.dtype==np.dtype(dtype)
    arrays[name]=a
logits=arrays['occupancy-logits.npy'][0,0];mask=arrays['occupancy-mask.npy'];coords=arrays['occupancy-coordinates.npy']
assert np.isfinite(logits).all() and np.array_equal(mask,logits>0)
expected=np.concatenate((np.zeros((int(mask.sum()),1),dtype='<i4'),np.argwhere(mask).astype('<i4')),axis=1)
assert np.array_equal(coords,expected)
def stats(a):
    a=a.astype(np.float64)
    return dict(count=a.size,minimum=float(a.min()),maximum=float(a.max()),mean=float(a.mean()),stddev=float(a.std()),
        quantiles={str(q):float(np.quantile(a,q)) for q in (0,.001,.01,.1,.5,.9,.99,.999,1)},
        zeros=int(np.count_nonzero(a==0)),finite=bool(np.isfinite(a).all()))
def geometry(a):
    c=np.argwhere(a)
    return dict(count=len(c),fraction=len(c)/a.size,bbox=None if not len(c) else [c.min(0).tolist(),c.max(0).tolist()],
        extents=None if not len(c) else (c.max(0)-c.min(0)+1).tolist(),
        occupied_per_axis=[a.sum(axis=tuple(j for j in range(3) if j!=i)).tolist() for i in range(3)],
        boundary_counts={n:int(np.take(a,index,axis=axis).sum()) for axis,names in enumerate((('D0','D63'),('H0','H63'),('W0','W63'))) for n,index in zip(names,(0,63))})
thresholds={str(t):geometry(logits>t) for t in (-170,-160,-150,-100,-50,-20,-5,-1,0,1,5,20,50,100)}
z=latent[0];axis_rms={name:float(np.sqrt(np.mean(np.diff(z.astype(np.float64),axis=axis+1)**2))) for axis,name in enumerate(('D','H','W'))}
spatial_std=np.std(z.astype(np.float64),axis=(1,2,3))
tokens=feature[0];patch=tokens[5:].reshape(32,32,1024).astype(np.float64)
input_path=root/'docs/compatibility/amd-dinov3-allocation-evidence/run/scratch/input.png'
assert sha(input_path)==feature_meta['input_png_sha256']
with Image.open(input_path) as im:
    im.load();pixels=np.array(im);assert im.mode=='RGBA' and im.size==(64,64)
    assert hashlib.sha256(im.tobytes()).hexdigest()==feature_meta['input_pixel_sha256']
alpha=pixels[:,:,3];foreground=np.argwhere(alpha>204);bbox=[foreground.min(0).tolist(),foreground.max(0).tolist()]
expected_pixels=np.zeros((64,64,4),dtype=np.uint8)
for y in range(8,56):
    for x in range(8,56):expected_pixels[y,x]=[x*4,y*4,(x+y)*2,255]
assert np.array_equal(pixels,expected_pixels)
source_image=dict(path=str(input_path),sha256=sha(input_path),mode='RGBA',shape=list(pixels.shape),
    foreground_pixels=len(foreground),foreground_bbox_yx=bbox,alpha_values=np.unique(alpha).tolist(),
    exact_fixture_verified=True,description='48x48 opaque square with smooth RGB coordinate gradients on transparent64x64 background; no object or zombie image')
row=dict(classification='UNKNOWN',analysis_scope='CPU only; saved tensors and image; no AI checkpoint load, import or forward',
    input_hashes=dict(latent=sha(latent_path),feature=sha(feature_path),logits=sha(occupancy/'occupancy-logits.npy'),mask=sha(occupancy/'occupancy-mask.npy')),
    latent=stats(latent),latent_channels=[stats(a) for a in z],latent_spatial_stddev_per_channel=spatial_std.tolist(),
    latent_axis_neighbor_rms=axis_rms,latent_rank_channel_matrix=int(np.linalg.matrix_rank(z.reshape(8,-1))),
    logits=stats(logits),positive_logits=stats(logits[mask]),background_logits=stats(logits[~mask]),
    minimum_absolute_logit=float(np.abs(logits).min()),near_zero_count_1=int(np.count_nonzero(np.abs(logits)<=1)),
    threshold_analysis=thresholds,native_geometry=geometry(mask),coordinate_order=['batch','D','H','W'],
    axis_permutation_cannot_change_sorted_extents=True,
    logit_axis_neighbor_rms={name:float(np.sqrt(np.mean(np.diff(logits.astype(np.float64),axis=axis)**2))) for axis,name in enumerate(('D','H','W'))},
    feature=stats(feature),feature_token_mean_range=[float(tokens.mean(-1).min()),float(tokens.mean(-1).max())],
    feature_token_std_range=[float(tokens.std(-1).min()),float(tokens.std(-1).max())],
    feature_unique_tokens=len(np.unique(tokens,axis=0)),feature_patch_neighbor_rms_y=float(np.sqrt(np.mean(np.diff(patch,axis=0)**2))),
    feature_patch_neighbor_rms_x=float(np.sqrt(np.mean(np.diff(patch,axis=1)**2))),
    feature_patch_spatial_std_mean=float(patch.std(axis=(0,1)).mean()),source_image=source_image,
    source_latent_provenance=latent_meta,source_feature_provenance=feature_meta,
    torch_imported='torch' in sys.modules,gpu_initialized=False,model_weights_loaded=False,new_sampling=False,new_mesh=False)
with (evidence/'saved-data-analysis.json').open('x',encoding='utf-8') as f:json.dump(row,f,indent=2)
print(json.dumps(dict(latent=row['latent'],latent_axis_neighbor_rms=axis_rms,latent_rank=row['latent_rank_channel_matrix'],
    logits=row['logits'],near_zero_count_1=row['near_zero_count_1'],threshold_summary={k:{n:v[n] for n in ('count','extents')} for k,v in thresholds.items()},
    feature_token_std_range=row['feature_token_std_range'],feature_unique_tokens=row['feature_unique_tokens'],source_image=source_image,torch_imported=False),indent=2))
