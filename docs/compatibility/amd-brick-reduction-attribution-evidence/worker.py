"""Only saved-data attribution. Never simplify, repair, sample AI, or render."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
import common as c
import time,traceback,json

def main():
    c.begin();started=time.monotonic();out=c.P/'diagnose'
    try:
        import numpy as np
        import topology_witness,witness_material
        c.authenticate();data,cv,cf,ids,rgba=c.load(np)
        topology=topology_witness.run(np,data,cv,cf,ids,rgba,out)
        print('TOPOLOGY_WITNESSES_COMPLETE',flush=True)
        material=witness_material.run(np,data,cv,cf,ids,rgba,out)
        print('MATERIAL_PROJECTION_WITNESSES_COMPLETE',flush=True)
        forbidden=[name for name in sys.modules if name.split('.')[0] in {'torch','triton','flex_gemm','o_voxel','trellis2','cupy','bpy'}]
        c.require(not forbidden,'No forbidden model/GPU/renderer module')
        files={p.name:dict(bytes=p.stat().st_size,sha256=c.digest(p)) for p in sorted(out.iterdir()) if p.suffix in {'.json','.png'}}
        c.save(out/'result.json',dict(classification='PASS',scope='Saved-data defect attribution only; rejected candidate remains NO-GO',topology=topology,material_projection=material,files=files,elapsed_seconds=time.monotonic()-started,no_simplification_repair_render_or_new_spatial_samples=True,forbidden_modules=forbidden))
    except BaseException as error:
        c.save(out/'failure.json',dict(classification='BLOCKED',error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started));raise

if __name__=='__main__':main()
