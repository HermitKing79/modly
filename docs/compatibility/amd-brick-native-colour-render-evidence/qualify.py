"""Fresh CPU-only native-byte/source-material qualification; no Blender import."""
import sys
sys.dont_write_bytecode=True
import json,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import *
from quality import decode,evaluate
def main():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Owned exact binary GO')
    auth=load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)
    import numpy as np
    began=time.monotonic();out=P/'qualify';stage='original_input_authentication'
    try:
        data,ids,watched=load(np);before={str(p):digest(p) for p in watched}
        a={name:np.load(plain(IMPORTED/name),mmap_mode='r',allow_pickle=False) for name in ('actual-vertices.npy','actual-faces.npy','actual-colour-bytes.npy','oracle-colour-bytes.npy')}
        oracle=a['oracle-colour-bytes.npy'];actual=a['actual-colour-bytes.npy'];faces=data['faces'];vertices=data['vertices']
        require(oracle.dtype==actual.dtype==np.uint8 and oracle.shape==(1957891,4) and actual.shape==(12520752,4),'Exact native colour schemas')
        # Use only the historical pure geometry contract, never its failed verifier.
        c=pinned('native_previous_exact_geometry',OLD/'contract.py',PINS[OLD/'contract.py'])
        c.verify_geometry(vertices,faces,a['actual-vertices.npy'],a['actual-faces.npy'])
        for s in range(0,len(faces),32768):
            part=faces[s:s+32768]
            require(np.array_equal(actual[s*3:(s+len(part))*3],oracle[part].reshape(-1,4)),'Exact every original native corner RGBA')
        require((oracle[:,3]==255).all(),'Approved OPAQUE alpha')
        colours=decode(oracle[:,:3]);source=np.load(COLOURS,mmap_mode='r',allow_pickle=False)
        linear_error=colours.astype(np.float64)-source
        loss=dict(linear_RGB_RMSE=float(np.sqrt(np.mean(linear_error**2))),linear_RGB_max_abs=float(np.abs(linear_error).max()),scope='All original source RGB rows; measured only, no new ideal-rounding ceiling')
        del linear_error
        metrics=pinned('native_original_metrics',ROOT/'docs/compatibility/amd-foliage-colour-glb-evidence/verify_glb.py','82ff500153f29a7e1412a88c8ce0ab580278616b23d64a5708f5b2a4dda16ce9')
        stage='all112_existing_source_quality_rows';first,second=evaluate(data,colours,metrics)
        first['scope']='All56910 frozen original source queries; no new positions'
        second['scope']='All2048 frozen interior holdouts; no new positions'
        from independent import verify
        independently=verify(data,oracle)
        independent_rows={name:{k:v for k,v in independently[name].items() if k!='predicted_RGB'} for name in ('original_queries','holdout_queries')}
        save(out/'metric-rows.json',dict(original_queries=first,holdout_queries=second,independent=independent_rows,conversion_loss=loss,limits=dict(RGB_RMSE=.02,RGB_p95=.05,RGB_max=.15,scalar_RMSE=.02)))
        require(first['all_gates_pass'] and second['all_gates_pass'] and independently['all_gates_pass'],'Every unchanged original material gate must pass')
        for own,other in ((first,independently['original_queries']),(second,independently['holdout_queries'])):
            require(set(own['cohorts'])==set(other['cohorts']),'Original cohort identity')
            for name,row in own['cohorts'].items():
                q=other['cohorts'][name];require(row['count']==q['count'] and row['pass']==q['pass'],'Independent cohort decision')
                for label in ('rmse','p95_abs','max_abs'):
                    require(abs(row['RGB'][label]-q['RGB'][label])<=1e-12,'Independent RGB arithmetic')
                    for channel in ('metallic','roughness','alpha'):require(abs(row['scalars'][channel][label]-q['scalars'][channel][label])<=1e-12,'Independent scalar arithmetic')
        require(before=={str(p):digest(p) for p in watched},'Original import and failed evidence unchanged')
        require(not any(n=='bpy' or n.startswith(('torch','triton','flex_gemm')) for n in sys.modules),'No Blender/model import')
        # Predictions are private; public numerical rows contain only aggregate errors.
        for label,entry in (('original',independently['original_queries']),('holdout',independently['holdout_queries'])):
            values=entry.pop('predicted_RGB')
            with (out/(label+'-quantized-predictions.npy')).open('xb') as f:np.save(f,values,allow_pickle=False)
        save(out/'result.json',dict(classification='PASS',import_qualification='PASS',all112_rows_pass=True,original_query_count=56910,holdout_query_count=2048,original_queries=first,holdout_queries=second,independent=independently,conversion_loss=loss,RGBA_corner_bytes_exact=True,original_geometry_exact=True,decode='IEC sRGB byte/255 to scene-linear, float32 table; actual loaded native getter independently rechecked before render',RGB_limits=dict(rmse=.02,p95_abs=.05,max_abs=.15),scalar_RMSE_limit=.02,old_failed_verdict_unchanged=True,no_Blender_import=True,no_new_source_queries=True,scene_SHA256=digest(SCENE),sources_unchanged=True,elapsed_seconds=time.monotonic()-began))
        print('NATIVE_COLOUR_ALL112_PASS',flush=True)
    except BaseException as error:
        save(out/'failure.json',dict(classification='NO-GO',stage=stage,error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-began,retry=False));raise
if __name__=='__main__':main()
