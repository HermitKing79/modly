"""Pinned previous CPU lifecycle/material reader and fixed output helpers."""
import sys
sys.dont_write_bytecode=True
import gc,json,os
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import *
def reader():return pinned('amended_previous_material_reader',MATERIAL_READER,PINS[MATERIAL_READER])
def start(stage,output):
    r=reader();r.await_host()
    import bpy
    require(Path(output)==P/stage,'Exact private output')
    r.OUT=P/stage;r.initial_output(P/stage,bpy.app.tempdir)
    legacy=r.load_legacy();scratch=legacy.guard_python_writes(P/stage)
    require(bpy.app.background and tuple(bpy.app.version)==(5,2,1),'Pinned Blender5.2.1 background')
    return r,legacy,scratch
def array_file(out,name,value):
    import numpy as np
    path=out/name
    with path.open('xb') as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
    return dict(sha256=digest(path),bytes=path.stat().st_size,shape=list(value.shape),dtype=str(value.dtype))
def record_file(path):return dict(sha256=digest(path),bytes=path.stat().st_size)
def geometry_readback(obj):
    import numpy as np
    m=obj.data
    require(len(m.vertices)==1952142 and len(m.polygons)==4173584 and len(m.loops)==12520752,'Amended exact mesh counts')
    require(not obj.modifiers and not obj.parent and len(m.uv_layers)==0 and not m.has_custom_normals,'Original unmodified no-normal mesh')
    require(all(obj.matrix_world[i][j]==(i==j) for i in range(4) for j in range(4)),'Original identity world transform')
    v=np.empty((len(m.vertices),3),dtype='<f4');f=np.empty((len(m.polygons),3),dtype='<i4');starts=np.empty(len(f),dtype='<i4');total=np.empty(len(f),dtype='<i4');smooth=np.empty(len(f),dtype=np.bool_)
    m.vertices.foreach_get('co',v.ravel());m.loops.foreach_get('vertex_index',f.ravel());m.polygons.foreach_get('loop_start',starts);m.polygons.foreach_get('loop_total',total);m.polygons.foreach_get('use_smooth',smooth)
    require(np.array_equal(starts,np.arange(len(f),dtype='<i4')*3) and (total==3).all() and not smooth.any(),'Original flat ordered triangles')
    return v,f
def colour_bytes(attribute):
    import numpy as np
    a=np.empty((len(attribute.data),4),dtype='<f4');attribute.data.foreach_get('color_srgb',a.ravel())
    require(np.isfinite(a).all() and (a>=0).all() and (a<=1).all(),'Finite unit stored colours')
    result=np.rint(a*255).astype(np.uint8)
    require(np.max(np.abs(a-result.astype(np.float32)/255))<1e-7,'Stored exact byte fractions')
    return result
