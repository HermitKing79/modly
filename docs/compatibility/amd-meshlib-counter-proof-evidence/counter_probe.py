"""Observe one installed counter binding; one conditional synthetic constructor."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import math,struct,time,traceback
E=Path(__file__).resolve().parent
INITIAL=-12345
MUTATED=-23456

def need(ok,message):
    if not ok:raise ValueError(message)

def validate_state(row):
    need(type(row)is dict and set(row)=={'length','values','getter'},'Exact scalar observation fields')
    n=row['length'];values=row['values'];getter=row['getter']
    need(type(n)is int and 0<=n<=8 and type(values)is list and len(values)==n,'Bounded exact storage length')
    need(all(type(x)is int for x in values),'Native integer storage, not booleans')
    need(type(getter)is dict and type(getter.get('ok'))is bool,'Exact getter observation')
    if n==1:
        need(set(getter)=={'ok','value'} and getter['ok'] is True and type(getter.get('value'))is int
             and getter['value']==values[0],'One stored scalar must have one coherent readable value')
    else:
        need(set(getter)=={'ok','error_type','error'} and getter['ok'] is False
             and type(getter['error_type'])is str and bool(getter['error_type'])
             and type(getter['error'])is str and bool(getter['error']),'Non-single getter must reject with an observed error')
    return True

def validate_bound(row):
    need(type(row)is dict,'Binding observation object')
    for name in ('initial_sentinel','mutated_sentinel','settings_before','settings_after'):
        need(type(row.get(name))is int,'Real native integer: '+name)
    need(row['initial_sentinel']==INITIAL and row['mutated_sentinel']==MUTATED,'Frozen distinct negative sentinels')
    validate_state(row['holder_state'])
    need(row['holder_state']['values']==[MUTATED] and row['settings_before']==INITIAL
         and row['settings_after']==MUTATED,'Settings getter must follow the live single-value storage')
    return True

def _validate_geometry(row):
    for name,want in (('failed_faces',[]),('valid_faces',[0]),('valid_vertices',[0,1,2])):
        actual=row.get(name)
        need(type(actual)is list and all(type(x)is int for x in actual) and actual==want,'Exact synthetic validity: '+name)
    vertices=row.get('vertices');expected=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]
    need(type(vertices)is list and len(vertices)==3,'Exactly three synthetic vertices')
    for actual,want in zip(vertices,expected):
        need(type(actual)is list and len(actual)==3,'Synthetic vertex XYZ schema')
        for value,target in zip(actual,want):
            need(type(value)in (int,float) and math.isfinite(value) and struct.pack('<d',float(value))==struct.pack('<d',target),
                 'Exact promoted original synthetic coordinates and zero signs')
    faces=row.get('faces')
    need(type(faces)is list and len(faces)==1 and type(faces[0])is list and len(faces[0])==3
         and all(type(x)is int for x in faces[0]) and faces[0]in ([0,1,2],[1,2,0],[2,0,1]),'One oriented cyclic synthetic triangle')
    return True

def validate_output(row):
    need(type(row)is dict and type(row.get('constructor_calls'))is int and row['constructor_calls']==1,'Exactly one synthetic constructor')
    need(type(row.get('before'))is int and row['before']==MUTATED,'Observed nonzero pre-call bound sentinel')
    validate_state(row['after'])
    need(row['after']['values']==[0] and type(row.get('settings_after'))is int and row['settings_after']==0,
         'Native output must observably replace the live sentinel with zero')
    _validate_geometry(row)
    return True

class GeometryMismatch(ValueError):
    """A measured contradiction of the frozen synthetic transport contract."""

def main():
    sys.path.insert(0,str(E))
    import common as c
    c.begin();began=time.monotonic();out=c.P/'counter'
    record=dict(classification='BLOCKED',constructor_calls=0,remesh_calls=0,real_patch_reads=0,model_calls=0,
                matrix=[],controls=[],stage='authentication')
    def check():
        if time.monotonic()-began>150:raise TimeoutError('150-second scalar microproof budget')
    def save(name,value):
        check();c.save(out/(name+'.json'),value)
    def files():
        return {p.name:dict(bytes=p.stat().st_size,sha256=c.digest(p)) for p in sorted(out.glob('*.json')) if p.name!='result.json'}
    try:
        c.authenticate();sdk=c.load_sdk_auth();check()
        need(sdk['CPU_import_admitted'] is True and sdk['version']=='3.1.4.297','Exact admitted installed CPU package')
        # No numerical or MeshLib import occurs in this module or before GO/auth.
        import numpy as np
        import meshlib
        from meshlib import mrmeshpy as mr,mrmeshnumpy as mn
        def cpu_only():
            bad=[name for name in sys.modules if name.split('.')[0]in {'torch','bpy','cupy','pycuda','moderngl'}
                 or name.startswith(('meshlib.mrcudapy','meshlib.mrviewerpy'))]
            need(not bad,'No GPU, viewer, model or renderer Python modules')
        cpu_only();known={str(Path(row['path']).resolve()).casefold():row for row in sdk['files'].values()}
        loaded={}
        for name,module in (('meshlib',meshlib),('mrmeshpy',mr),('mrmeshnumpy',mn)):
            path=Path(module.__file__).resolve();row=known.get(str(path).casefold())
            need(row is not None and path.stat().st_size==row['bytes'] and c.digest(path)==row['sha256'],'Loaded reviewed package/module identity')
            loaded[name]=dict(path=str(path),bytes=row['bytes'],sha256=row['sha256'])
        record['loaded_modules']=loaded
        record['package_initialization']='Pinned wrapper adds reviewed DLL search directories and imports CPU mrmeshpy to set package resource/font directories; no viewer/CUDA module is requested.'
        save('module-identities',loaded)
        def observe(holder):
            length=len(holder.values);need(type(length)is int and 0<=length<=8,'At most8 observed scalar storage elements')
            values=[holder.values[i] for i in range(length)]
            need(all(type(value)is int for value in values),'Native integer vector observations')
            try:
                value=holder.value
                if type(value)is not int:raise TypeError('Getter did not return a native Python integer')
                getter=dict(ok=True,value=value)
            except Exception as error:getter=dict(ok=False,error_type=type(error).__name__,error=str(error)[:1000])
            return dict(length=length,values=values,getter=getter)
        def settings_value(settings):
            value=settings.skippedFaceCount;need(type(value)is int,'Settings getter must return its declared native integer');return value
        record['stage']='constructor-helper-matrix'
        for index,(name,args) in enumerate((('default',()),('int0',(0,)),('int1',(1,)),('int2',(2,)),('int1_7',(1,7)))):
            check();holder=mr.int_output(*args);state=observe(holder);row=dict(case=name,state=state)
            record['matrix'].append(row);save('matrix-%02d'%index,row);validate_state(state)
        record['stage']='explicit-vector-controls'
        for size in (0,2):
            holder=mr.int_output();holder.values.resize(size);state=observe(holder)
            row=dict(explicit_length=size,state=state);record['controls'].append(row);save('control-%d'%size,row)
            validate_state(state);need(state['length']==size,'Explicit vector resize must establish the chosen length')
        record['stage']='selected-storage'
        holder=mr.int_output();holder.values.resize(1);resized=observe(holder)
        record['selected_resized']=resized;save('selected-resized',resized);validate_state(resized)
        need(resized['length']==1,'Explicit single storage before setting a scalar')
        holder.value=INITIAL;initial=observe(holder);record['selected_initial']=initial;save('selected-initial',initial)
        validate_state(initial);need(initial['values']==[INITIAL],'.value setter must update the one actual vector element')
        # No resize or reallocation occurs after this pointer is attached.
        settings=mr.MeshBuilder.BuildSettings();settings.skippedFaceCount=holder
        before=settings_value(settings);holder.values[0]=MUTATED
        binding=dict(initial_sentinel=INITIAL,mutated_sentinel=MUTATED,holder_state=observe(holder),settings_before=before,settings_after=settings_value(settings))
        record['binding']=binding;record['stage']='bound-alias';save('binding',binding);validate_bound(binding)
        record['prechecks_passed']=True;cpu_only();check()
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float32);faces=np.array([[0,1,2]],np.int32)
        native_v=vertices.copy();native_f=faces.copy();region=mr.FaceBitSet(1,True)
        settings.allowNonManifoldEdge=True;settings.shiftFaceId=0;settings.region=region
        region_before=np.flatnonzero(mn.getNumpyBitSet(region)).tolist()
        settings_region_before=np.flatnonzero(mn.getNumpyBitSet(settings.region)).tolist()
        record['region_before']=dict(region_before=region_before,settings_region_before=settings_region_before)
        save('region-before',record['region_before'])
        need(region_before==[0] and settings_region_before==[0],'Actual pre-call region and settings binding contain the one synthetic face')
        pre_state=observe(holder);validate_state(pre_state)
        need(pre_state['values']==[MUTATED] and settings_value(settings)==MUTATED,'Same live pre-call binding remains unchanged')
        journal=dict(constructor_call_ordinal=1,holder_before=pre_state,settings_before=settings_value(settings),
                     vertices=vertices.tolist(),faces=faces.tolist(),allowNonManifoldEdge=True,duplicateNonManifoldVertices=False,shiftFaceId=0,
                     region_before=region_before,settings_region_before=settings_region_before)
        save('before-constructor',journal);record['constructor_calls']=1;record['stage']='one-synthetic-constructor';started=time.monotonic()
        mesh=mn.meshFromFacesVerts(native_f,native_v,settings,duplicateNonManifoldVertices=False)
        record['constructor_elapsed_seconds']=time.monotonic()-started;record['constructor_returned']=True;check();cpu_only()
        post=dict(holder_after=observe(holder));record['post_counter']=post;save('post-holder',post['holder_after'])
        post['settings_after']=settings_value(settings);save('post-counter',post)
        failed=np.flatnonzero(mn.getNumpyBitSet(region)).tolist()
        settings_failed=np.flatnonzero(mn.getNumpyBitSet(settings.region)).tolist()
        valid_f=np.flatnonzero(mn.getNumpyBitSet(mesh.topology.getValidFaces())).tolist()
        valid_v=np.flatnonzero(mn.getNumpyBitSet(mesh.topology.getValidVerts())).tolist()
        record['native_validity']=dict(failed_faces=failed,settings_failed_faces=settings_failed,valid_faces=valid_f,valid_vertices=valid_v)
        save('native-validity',record['native_validity'])
        actual_v=np.asarray(mn.getNumpyVerts(mesh))
        if actual_v.shape!=(3,3) or actual_v.dtype not in (np.float32,np.float64):
            record['readback_vertex_schema']=dict(shape=list(actual_v.shape),dtype=str(actual_v.dtype))
            raise GeometryMismatch('Synthetic native vertex schema changed')
        actual_f=[]
        if valid_f==[0]:
            tri=mesh.topology.getTriVerts(mr.FaceId(0));actual_f=[[int(tri[i].get()) for i in range(3)]]
        output=dict(before=pre_state['getter']['value'],after=post['holder_after'],settings_after=post['settings_after'],
                    failed_faces=failed,settings_failed_faces=settings_failed,valid_faces=valid_f,valid_vertices=valid_v,vertices=actual_v.tolist(),faces=actual_f,constructor_calls=1)
        record['output']=output;save('synthetic-readback',output)
        need(settings_failed==failed,'Actual output region and settings-region binding agree')
        need(np.array_equal(native_v.view(np.uint32),vertices.view(np.uint32)) and np.array_equal(native_f,faces),'Synthetic heap inputs unchanged')
        try:_validate_geometry(output)
        except ValueError as error:raise GeometryMismatch(str(error)) from error
        record['stage']='native-counter-write';validate_output(output)
        record.update(classification='PASS',stage='complete',CPU_only_modules_observed=True,
                      interpretation='Only this explicit storage/binding and one synthetic fixture qualify; no real-patch constructor or production patch is authorised.')
        record.update(files=files(),elapsed_seconds=time.monotonic()-began);save('result',record)
    except BaseException as error:
        record.update(classification='NO-GO' if isinstance(error,GeometryMismatch) else 'BLOCKED',error=str(error),traceback=traceback.format_exc(),
                      files=files(),elapsed_seconds=time.monotonic()-began)
        c.save(out/'result.json',record);print('COUNTER_MICROPROOF_STOP '+str(error),flush=True)

if __name__=='__main__':main()
