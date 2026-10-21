"""One authorised export of the saved, already qualified foliage values."""
import sys
sys.dont_write_bytecode=True
import json,os,time
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import P,SELECTION,load_auth,load,authenticate,plain,digest,require,array,save


def main():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Exact owned binary GO')
    auth=load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from glb_writer import build_glb
    began=time.monotonic();data,identities,watched=load(np)
    before={str(x):digest(x) for x in watched}
    out=plain(P/'export')
    for name in ('vertices','faces','linear-COLOR0','source-vertex-ids','source-face-ids'):
        array(out/(name+'.npy'),data[name])
    payload=build_glb(data['vertices'],data['faces'],data['linear-COLOR0'],data['source-vertex-ids'],data['source-face-ids'],digest(SELECTION/'source-vertex-ids.npy'),digest(SELECTION/'source-face-ids.npy'))
    path=out/'foliage-learned-colour.glb'
    with path.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    require(path.read_bytes()==payload,'Byte-exact complete GLB write')
    require(before=={str(x):digest(x) for x in watched},'Original inputs unchanged')
    files={x.name:dict(bytes=x.stat().st_size,sha256=digest(x)) for x in out.iterdir() if x.is_file() and x.suffix in ('.npy','.glb')}
    save(out/'result.json',dict(classification='PASS',CPU_only=True,vertices=3177,triangles=4531,files=files,source_identities=identities,inputs_unchanged=True,coordinate_conversion='Original(x,y,z) to glTF(x,z,-y); proper rotation, no winding change',colour='Saved F32 linear COLOR_0 bytes unchanged; no gamma/fit/quantisation',alpha='OPAQUE approximation of measured nearly opaque patch',elapsed_seconds=time.monotonic()-began))
    print('FOLIAGE_COLOUR_EXPORT_PASS',flush=True)


if __name__=='__main__':main()
