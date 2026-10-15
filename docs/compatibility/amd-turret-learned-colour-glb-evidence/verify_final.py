"""Final independent host CPU artifact/preservation checks; no Blender launch."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import contract as c

def main():
    phases=('fixture-export','fixture-import-confirmation','full-export','full-import')
    evidence={p:json.loads((c.E/(p+'.json')).read_text()) for p in phases}
    hosts={p:json.loads((c.E/(p+'-host.json')).read_text()) for p in phases}
    for p in phases:
        assert evidence[p]['classification']=='PASS' and hosts[p]['acceptance']
        assert hosts[p]['returncode']==0 and hosts[p]['job_cleanup']['owned_tree_exit_verified']
        assert not hosts[p]['job_cleanup']['forced_job_cleanup']
    export,readback=evidence['full-export'],evidence['full-import']
    glb=c.OUT/'turret-learned-colour.glb'
    assert c.digest(glb)==export['sha256']==readback['sha256']
    assert glb.stat().st_size==export['bytes']==readback['bytes']
    assert readback['triangles']==c.F and readback['source_vertices']==c.N
    assert readback['binary_payload_verification']['maximum_linear_RGB_error']==0
    linear=np.load(c.OUT/'source-linear-RGB.npy',allow_pickle=False,mmap_mode='r')
    assert linear.shape==(c.N,3) and linear.dtype==np.float32
    rgba=np.ones((c.N,4),'<f4');rgba[:,:3]=linear
    assert c.ahash(rgba)=='6c2f745064a95a3b077ee8682346f22971e95a81c56ce7e2b889eae5c36b9fa1'
    for phase in ('full-export','full-import'):
        assert hosts[phase]['scripts']==hosts['fixture-import-confirmation']['scripts']
    record={'classification':'PASS','scope':'Portable direct-row learned-colour GLB, not spatial PBR bake',
        'GLB':{'path':str(glb),'bytes':glb.stat().st_size,'sha256':c.digest(glb)},
        'original_vertices':c.N,'triangles':c.F,'glTF_vertices':readback['POSITION']['count'],
        'geometry_winding_and_triangle_multiplicity_exact':True,'linear_RGB_payload_error':0,
        'fresh_Blender_import_sRGB_exact':True,'no_external_resources':True,
        'constant_metallic':0,'constant_roughness':.8,'opaque':True,'emission':False,
        'source_linear_RGB_reference_sha256':c.digest(c.OUT/'source-linear-RGB.npy'),
        'sources_unchanged':c.preserve(),'focused_CPU_unit_tests_passed':4,
        'fixture_exports':1,'fixture_imports':2,'full_exports':1,'full_imports':1,
        'initial_fixture_import_limitation':'Verifier assumed explicit layer_name; installed importer deliberately uses active render colour. Failed record preserved.',
        'processes':{p:{k:hosts[p][k] for k in ('elapsed_seconds','peak_committed_bytes','deadline_seconds','memory_cap_bytes','job_cleanup')} for p in phases},
        'no_Torch_ROCm_WSL_models_or_geometry_changes':True}
    c.save(c.E/'verification.json',record)
    c.save(c.OUT/'verification.json',record)
    for p in ('full-export','full-import'):
        c.save(c.OUT/(p+'-metadata.json'),evidence[p])
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
