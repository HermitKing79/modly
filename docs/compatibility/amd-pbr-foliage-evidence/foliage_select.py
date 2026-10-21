"""One fixed original vine/leaf ROI; original faces are selected, never clipped."""
import numpy as np

ROI=(332,384,400,448)

def visible_faces(vertices,faces,projected,max_faces=6000):
    """Original +X centre-depth raster then one edge ring inside the same ROI."""
    screen=projected.screen;x0,y0,x1,y1=ROI
    candidates=[];inside=[]
    for start in range(0,len(faces),65536):
        tri=faces[start:start+65536];xy=screen[tri]
        hit=(xy.max(1)>[x0,y0]).all(1)&(xy.min(1)<[x1,y1]).all(1)
        candidates.extend((np.flatnonzero(hit)+start).tolist())
        full=(xy>=[x0,y0]).all((1,2))&(xy<=[x1,y1]).all((1,2))
        inside.extend((np.flatnonzero(full)+start).tolist())
    if not 0<len(candidates)<=100000:raise ValueError('Bounded fixed ROI candidate count')
    depth=np.full((y1-y0,x1-x0),-np.inf);owner=np.full(depth.shape,-1,np.int32)
    for faceid in candidates:
        tri=faces[faceid];xy=screen[tri]
        low=np.maximum([x0,y0],np.ceil(xy.min(0)-.5).astype(int));high=np.minimum([x1-1,y1-1],np.floor(xy.max(0)-.5).astype(int))
        if (high<low).any():continue
        a,b=xy[1]-xy[0],xy[2]-xy[0];det=a[0]*b[1]-a[1]*b[0]
        if abs(det)<=1e-18:continue
        x,y=np.meshgrid(np.arange(low[0],high[0]+1)+.5,np.arange(low[1],high[1]+1)+.5)
        rx,ry=x-xy[0,0],y-xy[0,1];w1=(rx*b[1]-ry*b[0])/det;w2=(a[0]*ry-a[1]*rx)/det
        weights=np.stack((1-w1-w2,w1,w2),-1);z=(weights*vertices[tri,0]).sum(-1)
        region=np.s_[low[1]-y0:high[1]-y0+1,low[0]-x0:high[0]-x0+1]
        # Strict > retains first original ID at equal-depth ties.
        choose=(weights.min(-1)>=-1e-9)&(z>depth[region]);depth[region][choose]=z[choose];owner[region][choose]=faceid
    inside=np.asarray(inside,np.int32);visible=np.intersect1d(np.unique(owner[owner>=0]),inside)
    if not len(visible):raise ValueError('No complete visible original foliage faces')
    edge_members={}
    for index in inside:
        t=faces[index]
        for k in range(3):edge_members.setdefault(tuple(sorted((int(t[k]),int(t[(k+1)%3])))),[]).append(int(index))
    chosen=set(map(int,visible))
    for index in visible:
        t=faces[index]
        for k in range(3):chosen.update(edge_members[tuple(sorted((int(t[k]),int(t[(k+1)%3]))))])
    selected=np.asarray(sorted(chosen),np.int32)
    if not 20<=len(selected)<=max_faces:raise ValueError('Original ROI face budget: '+str(len(selected)))
    return selected,dict(ROI=list(ROI),candidate_faces=len(candidates),complete_ROI_faces=len(inside),visible_complete_faces=len(visible),selected_faces=len(selected),artificial_crop_boundaries=True,visibility='One original +X centre-depth raster with nearest X; one source-edge ring; no geometry clipping',raster_dimensions=list(owner.shape))
