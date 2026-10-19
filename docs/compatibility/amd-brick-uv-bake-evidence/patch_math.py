"""One planar UV chart for an unchanged patch; CPU arithmetic only."""
import numpy as np


def chart(vertices, faces, size=1024, gutter=16):
    if vertices.dtype != np.float32 or faces.dtype != np.int32 or vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('F32 vertices / I32 triangles required')
    if not np.isfinite(vertices).all() or len(faces) == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid original patch')
    yz = vertices[:, 1:].astype(np.float64)
    lo, hi = yz.min(0), yz.max(0)
    if np.any(hi - lo <= 0) or not 0 < gutter < size / 4:
        raise ValueError('Nondegenerate chart and bounded gutter required')
    uv = ((yz - lo) / (hi - lo) * (1 - 2*gutter/size) + gutter/size).astype(np.float32)
    tri = uv[faces].astype(np.float64)
    a,b=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    areas = a[:,0]*b[:,1]-a[:,1]*b[:,0]
    if np.all(areas < -1e-14):
        uv[:,0]=np.float32(1)-uv[:,0]  # Mirror UV only; preserve original negative-X face winding.
        areas=-areas
    if np.any(areas <= 1e-14):
        raise ValueError('Consistent-winding nondegenerate chart required')
    return uv


def rasterize(uv, faces, size):
    """Texel centers, V-up image rows; detect interior overlaps, never drop faces."""
    ids = np.full((size,size), -1, np.int32)
    bary = np.zeros((size,size,3), np.float32)
    for i, face in enumerate(faces):
        t = uv[face].astype(np.float64)*size
        low = np.maximum(0, np.ceil(t.min(0)-.5).astype(int))
        high = np.minimum(size-1, np.floor(t.max(0)-.5).astype(int))
        if np.any(high < low):
            continue
        x,y = np.meshgrid(np.arange(low[0],high[0]+1)+.5, np.arange(low[1],high[1]+1)+.5)
        a,b = t[1]-t[0], t[2]-t[0]
        det = a[0]*b[1]-a[1]*b[0]
        if det <= 1e-12:
            raise ValueError('Positive nondegenerate UV face required')
        rx,ry = x-t[0,0], y-t[0,1]
        w1, w2 = (rx*b[1]-ry*b[0])/det, (a[0]*ry-a[1]*rx)/det
        w = np.stack((1-w1-w2,w1,w2),-1)
        hit = w.min(-1) >= -1e-10
        ys,xs = np.nonzero(hit); ys+=low[1]; xs+=low[0]
        weights = w[hit]
        old = ids[ys,xs]
        both_interior = (old >= 0) & (weights.min(-1)>1e-7) & (bary[ys,xs].min(-1)>1e-7)
        if both_interior.any():
            raise ValueError('Overlapping source faces: planar UV chart is not injective')
        empty = old < 0
        ids[ys[empty],xs[empty]] = i
        bary[ys[empty],xs[empty]] = weights[empty]
    return ids,bary


def assert_injective(uv, faces, area_tolerance=1e-12):
    """Analytic positive-area intersection check, including subtexel overlaps.

    Convex clipping on bbox-overlapping original triangles. Shared original
    edges/vertices have zero intersection area and are permitted.
    """
    triangles=uv[faces].astype(np.float64)
    low,high=triangles.min(1),triangles.max(1)
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    pairs=0
    for i,a in enumerate(triangles):
        candidate=np.flatnonzero((np.minimum(high[i],high[i+1:])-np.maximum(low[i],low[i+1:])>0).all(1))+i+1
        for j in candidate:
            poly=list(a.copy())
            for k in range(3):
                edge=triangles[j,(k+1)%3]-triangles[j,k];origin=triangles[j,k]
                result=[]
                for u,z in zip(poly,poly[1:]+poly[:1]):
                    du,dz=cross(edge,u-origin),cross(edge,z-origin)
                    if du>=0:result.append(u)
                    if (du>=0)!=(dz>=0):result.append(u+(z-u)*(du/(du-dz)))
                poly=result
                if not poly:break
            if len(poly)>=3:
                q=np.asarray(poly)
                area=abs(np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1)))/2
                if area>area_tolerance:raise ValueError('Analytic positive-area UV overlap')
            pairs+=1
    return dict(analytic_bbox_candidate_pairs=pairs,positive_area_overlap_count=0,UV_area_tolerance=area_tolerance)


def bilinear(image, uv):
    """OpenGL normalized UV, V-up unflipped array, clamp edge, texel-center rule."""
    h,w = image.shape[:2]
    q = np.clip(uv*np.array([w,h])-.5, [0,0], [w-1,h-1])
    p = np.floor(q).astype(int); d=q-p
    nx=np.minimum(p[:,0]+1,w-1); ny=np.minimum(p[:,1]+1,h-1)
    return ((1-d[:,0,None])*(1-d[:,1,None])*image[p[:,1],p[:,0]] +
            d[:,0,None]*(1-d[:,1,None])*image[p[:,1],nx] +
            (1-d[:,0,None])*d[:,1,None]*image[ny,p[:,0]] +
            d[:,0,None]*d[:,1,None]*image[ny,nx])


def error(actual, expected):
    d = np.asarray(actual,dtype=np.float64)-expected
    return dict(rmse=float(np.sqrt(np.mean(d*d))), p95_absolute=float(np.quantile(abs(d),.95)),
                max_absolute=float(abs(d).max()), mean_absolute=float(abs(d).mean()))


def srgb_decode(x):
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def srgb_encode(x):
    return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)


def source_padding(values, covered):
    """Original quantization, then Telea only outside the UV raster mask."""
    import cv2
    from sampler import pack_native_textures
    cv2.ocl.setUseOpenCL(False);cv2.setNumThreads(4)
    packed=pack_native_textures(values)
    base,mr=packed['base_color_rgba'],packed['metallic_roughness_rgb']
    original_base,original_mr=base.copy(),mr.copy()
    invmask=(~covered).astype(np.uint8)
    base[:,:,:3]=cv2.inpaint(np.ascontiguousarray(base[:,:,:3]),invmask,3,cv2.INPAINT_TELEA)
    base[:,:,3]=cv2.inpaint(base[:,:,3],invmask,1,cv2.INPAINT_TELEA)
    for col in (1,2):mr[:,:,col]=cv2.inpaint(mr[:,:,col],invmask,1,cv2.INPAINT_TELEA)
    if not np.array_equal(base[covered],original_base[covered]) or not np.array_equal(mr[covered],original_mr[covered]):
        raise ValueError('Padding altered learned covered texels')
    return base,mr,original_base,original_mr
