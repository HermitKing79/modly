"""Windows owner for one published, bounded WSL shape decode and CPU readback.

The existing supervisor owns the Job, health monitor, GO/EOF, stage deadlines,
cleanup grace and named-distro termination. Only run routing/protocol changes.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from types import ModuleType

ROOT=Path(__file__).resolve().parents[2]
APPROVED_ROOT=ROOT/'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image'
CODE=ROOT/'experimental/extensions/trellis2-amd/inference_decoder'
GUEST_ENTRY=CODE/'decode.py'
LEGACY=ROOT/'docs/compatibility/amd-wsl-turret-artifact-writer-ready-evidence/decode.py'
CHECKPOINT_SHA='e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581'
GIT_READ_SECONDS=15
STATIC_FILES=['scripts/compatibility/amd_generation_decode.py','scripts/compatibility/test_amd_generation_decode.py',
    'scripts/compatibility/amd_generation_mesh.py','scripts/compatibility/amd_flexible_dual_grid_cpu.py',
    'scripts/compatibility/amd_generation_shape.py','scripts/compatibility/amd_generation_occupancy.py',
    'scripts/compatibility/amd_wsl_rocdxg_smoke.py','scripts/compatibility/amd_wsl_native_import_policy.py',
    'scripts/compatibility/amd_forward_owned_job.py','scripts/compatibility/amd_bounded_gpu_proof.py',
    'experimental/extensions/trellis2-amd/shape_decoder_loader.py',
    'experimental/extensions/trellis2-amd/strict_loader.py','api/services/amd_model_metadata.py','api/services/amd_runtime.py',
    LEGACY.relative_to(ROOT).as_posix()]
STATIC_FILES += ['scripts/compatibility/'+name+'.py' for name in ('amd_shape_decoder_preflight',
    'amd_shape_sampling_contract','amd_shape_sampling_artifact','amd_feature_checkpoint','amd_decoder_contract')]
STATIC_FILES += ['docs/compatibility/'+name for name in (
    'amd-wsl-turret-artifact-writer-ready-evidence/windows-preservation-before.json',
    'amd-wsl-turret-artifact-writer-ready-evidence/checkpoint-and-resource-preflight.json',
    'amd-wsl-turret-artifact-writer-ready-evidence/preservation-before.json',
    'amd-wsl-packaging-decoder-loading-evidence/dependency-run/results.json',
    'amd-wsl-packaging-decoder-loading-evidence/packaging-staging.json',
    'amd-wsl-native-decoder-preparation-evidence/staging.json',
    'amd-wsl-flexgemm-build-evidence/static-acceptance.json',
    'amd-wsl-neural-decoder-preparation-evidence/decoder-contract-readback.json',
    'amd-shape-decoder-neural-evidence/cpu-probe-4/results.json')]
COPY_LIMITS={'run-contract.json':65536,'phases.jsonl':8*1024**2,'results.json':8*1024**2,
    'provenance.json':1024**2,'shape-decoded.safetensors':256*1024**2}


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read_json(path,limit=8*1024**2):
    from amd_generation_occupancy import plain_path
    from amd_feature_checkpoint import unique
    path=plain_path(path,existing=True)
    if not path.is_file() or path.stat().st_size>limit:raise ValueError('Bounded regular JSON file required')
    return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=unique)


def save(path,row):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(row,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


@contextmanager
def module_binding(name,module):
    old=sys.modules.get(name)
    sys.modules[name]=module
    try:yield
    finally:
        if old is None:sys.modules.pop(name,None)
        else:sys.modules[name]=old


def load_module(name,path):
    old_path=sys.path[:]
    try:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        return module
    finally:sys.path[:]=old_path


def guest_path(path):
    import amd_wsl_rocdxg_smoke as base
    return base.guest_path(Path(path))


def shape_inputs(evidence):
    import numpy as np
    from amd_generation_occupancy import plain_path
    from amd_generation_shape import require_released
    from amd_shape_decoder_preflight import validate_arrays
    evidence=plain_path(evidence,existing=True);evidence.relative_to(APPROVED_ROOT.resolve())
    source=evidence/'run/results.json';before=digest(source);result=read_json(source)
    worker=require_released(result,'shape')
    sampled=worker.get('sampling',{})
    if (worker.get('forward_calls')!=21 or any(sampled.get(k)!=v for k,v in
            {'sequences':1,'started':21,'completed':21,'positive_calls':12,'negative_calls':9}.items())):
        raise ValueError('Source is not one complete original 12-step/21-forward shape sequence')
    arrays={};references={}
    for key,name in [('shape-latent-denormalized.npy','shape-latent-denormalized.npy'),('coords.npy','shape-latent-coordinates.npy')]:
        row=worker['artifacts'][name];path=plain_path(evidence/'run/scratch'/name,existing=True)
        if (Path(row['path'])!=path or row.get('verified') is not True or path.stat().st_size!=row['file_bytes']
                or path.stat().st_size>2*1024**2 or digest(path)!=row['sha256']):
            raise ValueError('Source shape artifact path/hash differs')
        array=np.load(path,allow_pickle=False,max_header_size=4096)
        if (list(array.shape)!=row['shape'] or str(array.dtype)!=row['dtype']
                or hashlib.sha256(array.tobytes()).hexdigest()!=row['payload_sha256']):
            raise ValueError('Source shape artifact payload differs')
        arrays[key]=array;references[key]={'path':path,'sha256':row['sha256']}
    features=arrays['shape-latent-denormalized.npy'];coords=arrays['coords.npy'];count=len(coords)
    summary=validate_arrays(features,coords,count=count)
    required=verify_shape_lineage(evidence,result,worker,features,coords)
    if worker.get('token_count')!=count or digest(source)!=before:raise ValueError('Source job/count changed')
    return {'source_job':source,'source_job_sha256':before,'count':count,'inputs':references,'coords':coords,
        'summary':summary,'required_files':required}


def verify_shape_lineage(evidence,result,worker,features,coords):
    import numpy as np
    import amd_generation_shape as generation
    from amd_shape_sampling_artifact import denormalize
    if generation.classify(worker,True,True,True)!='PASS':raise ValueError('Source shape classification/provenance failed')
    config=read_json(evidence/'pre-execution-admission.json')['context']
    checked=generation.context(Path(config['occupancy_evidence']),Path(config['evidence']),config['harness_commit'])
    if (Path(checked['evidence'])!=evidence or result['process']['harness_commit']!=checked['harness_commit']):
        raise ValueError('Source job is not bound to this fresh published shape context')
    runtime=generation.Runtime(checked);runtime.verify_sampler();bundle=runtime.bundle
    lineage=worker['generation_inputs'];expected=bundle['lineage']
    if (set(lineage)-set(expected)-{'resource_qualification'}
            or any(lineage.get(k)!=v for k,v in expected.items()) or not np.array_equal(coords,bundle['coords'])):
        raise ValueError('Source shape does not retain fresh image/occupancy lineage')
    row=worker['artifacts']['shape-latent-raw.npy'];path=evidence/'run/scratch/shape-latent-raw.npy'
    if (Path(row['path'])!=path or digest(path)!=row['sha256'] or path.stat().st_size!=row['file_bytes']):
        raise ValueError('Source raw shape sample identity changed')
    reader=load_module('_amd_generation_raw_reader',CODE/'run_contract.py')
    raw=reader.read_npy(path,row['sha256'],(len(coords),32),'float32')
    if hashlib.sha256(raw.tobytes()).hexdigest()!=row['payload_sha256']:raise ValueError('Source raw shape payload changed')
    normal=read_json(runtime.historical.PIPELINE)['args']['shape_slat_normalization']
    if not np.array_equal(denormalize(raw,np.asarray(normal['mean'],dtype='f4'),np.asarray(normal['std'],dtype='f4'),
            maximum_tokens=generation.MAX_TOKENS),features):
        raise ValueError('Source shape canonical denormalization differs')
    return runtime.profile().committed_files


def copy_files(source_directory,target,limits,required,owner_uid=None):
    """Stdlib-only transfer; the guest bootstrap attests both directory boundaries."""
    import hashlib,os
    from pathlib import Path
    source_directory=Path(source_directory);target=Path(target);copied={};missing=[]
    for name,limit in limits.items():
        source=source_directory/name
        try:before=source.lstat()
        except FileNotFoundError:
            if name in required:missing.append(name)
            continue
        if (source.is_symlink() or not source.is_file() or before.st_nlink!=1 or not 0<before.st_size<=limit
                or owner_uid is not None and before.st_uid!=owner_uid):
            raise ValueError('Not a bounded plain allowlisted file: '+name)
        h=hashlib.sha256();size=0;destination=target/name
        if name=='phases.jsonl':
            blob=source.read_bytes()
            if destination.read_bytes()!=blob:raise ValueError('Mirrored journal differs from durable guest journal')
            h.update(blob);size=len(blob)
        else:
            with source.open('rb') as src,destination.open('xb') as dst:
                while True:
                    block=src.read(1024**2)
                    if not block:break
                    size+=len(block)
                    if size>limit:raise ValueError('Guest output grew beyond copy bound')
                    h.update(block);dst.write(block)
                dst.flush();os.fsync(dst.fileno())
        after=source.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or size!=before.st_size:
            raise ValueError('Guest output changed during transfer')
        with destination.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=h.hexdigest():raise ValueError('Copied file differs')
        copied[name]={'bytes':size,'sha256':actual}
    if missing:raise ValueError('Missing allowlisted files: '+', '.join(missing))
    return copied


def guest_bootstrap(entry,contract_file,expected_sha,limits):
    """The outer timeout contains this parent, its decoder child and file transfer."""
    import json,os,stat,subprocess,sys
    from pathlib import Path
    sys.path.insert(0,str(Path(entry).parent))
    from run_contract import load
    from artifact_paths import attest_parent,attest_output
    contract=load(contract_file)
    if contract.sha256!=expected_sha:raise ValueError('Transported contract changed')
    Path(str(contract.output_directory.parent)).mkdir(mode=0o700,exist_ok=False)
    attest_parent(contract)
    child=subprocess.run([sys.executable,'-B','-I',entry,'--run-contract',contract_file],check=False)
    source=attest_output(contract);target=Path(str(contract.host_evidence_directory))
    for path in (target,*target.parents):
        info=path.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or path.resolve(strict=True)!=path:
            raise ValueError('Host mirror directory was redirected')
    required=set(limits) if child.returncode==0 else {'run-contract.json','phases.jsonl','results.json'}
    copied=copy_files(source,target,limits,required,os.getuid())
    print('AMD_GENERATION_COPY_RESULT='+json.dumps(copied),flush=True)
    return child.returncode


class PendingHostReceipt:
    """Guest-local writer uses independent post-exit verification, never an early ACK."""
    def __init__(self,output,kind,base):self.done=False
    def poll(self):pass


class BaseProxy:
    def __init__(self,base):self._base=base
    def __getattr__(self,name):return getattr(self._base,name)


class Runtime:
    def __init__(self,shape_evidence,evidence,harness_commit,run_id,*,tile_large_convolutions=False):
        from amd_generation_occupancy import plain_path
        self.evidence=plain_path(evidence);self.evidence.relative_to(APPROVED_ROOT.resolve())
        if self.evidence.name!='decode':raise ValueError('Use a fresh owned decode stage')
        if not re.fullmatch('[0-9a-f]{40}',harness_commit or ''):raise ValueError('Exact published harness commit required')
        if not re.fullmatch('run-[a-zA-Z0-9][a-zA-Z0-9._-]{0,90}',run_id or ''):raise ValueError('Fresh canonical run-ID required')
        self.commit=harness_commit;self.bundle=shape_inputs(shape_evidence)
        if self.evidence==Path(shape_evidence) or self.evidence in Path(shape_evidence).parents:
            raise ValueError('Source and new decode stage must be distinct')
        self.contract_module=load_module('_amd_generation_run_contract',CODE/'run_contract.py')
        with module_binding('run_contract',self.contract_module):
            self.guest=load_module('_amd_generation_guest_decode',GUEST_ENTRY)
        root=PurePosixPath('/home/modly')/run_id
        if type(tile_large_convolutions) is not bool:raise ValueError('Explicit boolean tiling opt-in required')
        self.contract=self.contract_module.parse({'schema':self.contract_module.SCHEMA,'initial_count':self.bundle['count'],
            'input_paths':{k:guest_path(v['path']) for k,v in self.bundle['inputs'].items()},
            'input_sha256':{k:v['sha256'] for k,v in self.bundle['inputs'].items()},
            'source_job_sha256':self.bundle['source_job_sha256'],'output_directory':str(root/'geometry'),
            'scratch_directory':str(root/'scratch'),'repo_root':guest_path(ROOT),
            'maximum_final_rows':1842968,'maximum_intermediate_rows':1000000,
            'native_stage':self.contract_module.NATIVE_STAGE,
            'host_evidence_directory':guest_path(self.evidence/'run/guest'),
            **({'large_convolution_tiling':True} if tile_large_convolutions else {})})
        self.copy_error=None;self.copy_receipt=None;self._supervisor=None

    def host_admission(self):
        import ctypes,shutil
        value=self.supervisor().HostMemory();value.length=ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):raise ctypes.WinError()
        row={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'stage':'supervised-live' if self.evidence.exists() else 'pre-marker',
            'available_ram_bytes':value.available,'D_free_bytes':shutil.disk_usage('D:/').free,
            'required_ram_bytes':12*1024**3,'required_D_free_bytes':50*1024**3}
        row['admitted']=row['available_ram_bytes']>=row['required_ram_bytes'] and row['D_free_bytes']>=row['required_D_free_bytes']
        print('HOST_RESOURCE_ADMISSION='+json.dumps(row),flush=True)
        if self.evidence.exists():
            with (self.evidence/'resource-admission.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
        if not row['admitted']:raise RuntimeError('Host RAM/disk admission failed: '+json.dumps(row))
        return row

    def preflight(self):
        import amd_wsl_rocdxg_smoke as base
        from amd_sparse_decoder_forward import verify_committed_files,COMMITTED_FILES
        if os.environ.get('WORKSPACE_DIR')!=str(ROOT):raise ValueError('Explicit checkout WORKSPACE_DIR required')
        if self.evidence.exists():raise FileExistsError('Decode stage already exists; no automatic retry')
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=GIT_READ_SECONDS).strip()
        if actual!=self.commit:raise ValueError('Run only the exact published current HEAD')
        admission=self.host_admission()
        fresh=shape_inputs(self.bundle['source_job'].parents[1])
        if fresh['source_job_sha256']!=self.contract.source_job_sha256:raise ValueError('Source shape job changed')
        names=sorted(set(STATIC_FILES+list(COMMITTED_FILES)+list(self.bundle['required_files'])+
            [p.relative_to(ROOT).as_posix() for p in CODE.glob('*.py')]))
        verified=verify_committed_files(self.commit,names,timeout_seconds=GIT_READ_SECONDS)
        if digest(ROOT/'scripts/compatibility/amd_wsl_native_import_policy.py')!='8b9207a14757ccda197e11c054d694f9b4ff47006a1cccc46d71f1d494ab96cd':
            raise ValueError('Qualified native policy changed')
        preserved=read_json(LEGACY.parent/'preservation-before.json')
        for field,parent in [('unchanged_control_code',ROOT/'scripts/compatibility'),
                             ('loader_code',ROOT/'experimental/extensions/trellis2-amd')]:
            for name,sha in preserved[field].items():
                if digest(parent/name)!=sha:raise ValueError('Qualified control/loader changed: '+name)
        checkpoint=self.guest.checkpoint_record(Path('D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors'))
        sys.path.insert(0,str(ROOT/'api'));sys.path.insert(0,str(ROOT/'experimental/extensions/trellis2-amd'))
        from shape_decoder_loader import ShapeDecoderComponent
        prefix=Path(checkpoint['path']).with_suffix('')
        component=ShapeDecoderComponent(str(prefix.with_suffix('.json')),str(prefix.with_suffix('.safetensors')))
        component._verify_source()
        if component.loaded or component.diagnostics['weights_loaded']:raise RuntimeError('Metadata preflight loaded a model')
        component.unload()
        accepted=read_json(ROOT/'docs/compatibility/amd-wsl-packaging-decoder-loading-evidence/dependency-run/results.json')
        if (accepted['classification']!='QUALIFIED' or accepted['worker_returncode']!=0 or not accepted['observer_healthy']
                or accepted['windows_job_cleanup']['forced_job_cleanup']):raise ValueError('Qualified CPU dependency evidence missing')
        return {'harness_commit':self.commit,'committed_file_sha256':verified,'run_contract_sha256':self.contract.sha256,
            'source_job_sha256':self.contract.source_job_sha256,'checkpoint_sha256':checkpoint['freshly_streamed_sha256'],
            'distro':base.DISTRO,'model_loaded':False,'host_admission':admission}

    def reserve_attempt(self,output):
        if Path(output)!=self.evidence/'run':raise ValueError('Exact owned host run required')
        self.evidence.mkdir(parents=True,exist_ok=False)
        save(self.evidence/'decode-attempt-started.json',{'harness_commit':self.commit,'run_contract_sha256':self.contract.sha256,
            'source_job_sha256':self.contract.source_job_sha256,'maximum_forwards':1,'guest_script_sha256':digest(GUEST_ENTRY)})
        Path(output).mkdir();(Path(output)/'guest').mkdir()
        save(self.evidence/'run-contract.json',self.contract.document())

    def worker_command(self,script_linux,output_linux):
        import amd_wsl_rocdxg_smoke as base
        if script_linux!=guest_path(GUEST_ENTRY) or output_linux!=guest_path(self.evidence/'run/guest'):
            raise ValueError('Wrong decoder entry or mirror directory')
        bootstrap=(inspect.getsource(copy_files)+'\n'+inspect.getsource(guest_bootstrap)+
            '\nimport json,sys\nraise SystemExit(guest_bootstrap(sys.argv[1],sys.argv[2],sys.argv[3],json.loads(sys.argv[4])))\n')
        return ['wsl.exe','-d',base.DISTRO,'-u','modly','--exec','/usr/bin/env','HSA_ENABLE_DXG_DETECTION=1',
            'LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib:/opt/rocm/lib:/usr/lib/wsl/lib','/usr/bin/timeout','--signal=TERM','--kill-after=5s',
            '450s',base.VENV_PYTHON,'-B','-I','-c',bootstrap,script_linux,guest_path(self.evidence/'run-contract.json'),self.contract.sha256,
            json.dumps(COPY_LIMITS,separators=(',',':'))]

    def copy_guest_output(self,source_directory):
        return copy_files(source_directory,self.evidence/'run/guest',COPY_LIMITS,set(COPY_LIMITS))

    def supervisor(self):
        if self._supervisor is None:
            module=load_module('_amd_generation_legacy_owner',LEGACY)
            module.SCRIPT=GUEST_ENTRY;module.EV=self.evidence
            module.preflight=self.preflight;module.reserve_attempt=self.reserve_attempt;module.worker_command=self.worker_command
            module.host_admission=self.host_admission
            module.validate_qualified=lambda row:self.guest.validate_qualified(row,self.contract)
            module.base=BaseProxy(module.base)
            self._supervisor=module
        return self._supervisor

    def verify_outputs(self):
        import numpy as np
        import amd_generation_mesh as mesh
        target=self.evidence/'run/guest';worker=read_json(target/'results.json')
        self.guest.validate_qualified(worker,self.contract)
        if worker.get('classification')!='QUALIFIED' or read_json(target/'run-contract.json')!=self.contract.document():
            raise ValueError('Guest result or copied run contract differs')
        if digest(self.bundle['source_job'])!=self.contract.source_job_sha256:raise ValueError('Source job changed after decode')
        if any(digest(row['path'])!=self.contract.input_sha256[name] for name,row in self.bundle['inputs'].items()):
            raise ValueError('Source input file changed after decode')
        sidecar=read_json(target/'provenance.json');provenance=sidecar['provenance'];output=worker['output']
        if (sidecar['run_contract']!=self.contract.document() or sidecar['output']!=output
                or provenance['run_contract_sha256']!=self.contract.sha256
                or provenance['source_job_sha256']!=self.contract.source_job_sha256
                or provenance['checkpoint_sha256']!=CHECKPOINT_SHA or provenance['inputs']!=worker['saved_inputs']
                or provenance['input_file_sha256']!=dict(self.contract.input_sha256)
                or any(worker['saved_inputs'].get(k)!=v for k,v in self.bundle['summary'].items())
                or worker['saved_inputs'].get('input_sha256')!=dict(self.contract.input_sha256)):
            raise ValueError('Independent host input/checkpoint/output lineage differs')
        path=target/'shape-decoded.safetensors'
        if output['bytes']!=path.stat().st_size or output['path']!=str(self.contract.output_directory/path.name):
            raise ValueError('Guest output path or bytes differ')
        arrays,children=mesh.read_neural(path,output['sha256'])
        try:
            hashes={k:hashlib.sha256(memoryview(v)).hexdigest() for k,v in arrays.items()}
            if (hashes!=output['tensor_payload_sha256'] or len(arrays['features'])!=output['rows']
                    or len(arrays['subdivision_0_coordinates'])!=self.contract.initial_count
                    or not np.array_equal(arrays['subdivision_0_coordinates'],self.bundle['coords'])
                    or children!=[s['selected_children'] for s in output['subdivisions']]):
                raise ValueError('Independent ten-tensor bytes, counts or initial coordinates differ')
            with path.open('rb') as stream:
                size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(size))
            if (json.loads(header['__metadata__']['provenance'])!=provenance
                    or json.loads(header['__metadata__']['summary'])['subdivisions']!=output['subdivisions']):
                raise ValueError('Independent raw metadata differs from sidecar/result')
        finally:
            for value in arrays.values():value._mmap.close()
        self.copy_receipt={name:{'bytes':(target/name).stat().st_size,'sha256':digest(target/name)} for name in COPY_LIMITS}
        return {'ten_tensors_verified':True,'rows':output['rows'],'sha256':output['sha256'],
            'tensor_payload_sha256':hashes,'run_contract_sha256':self.contract.sha256,
            'source_job_sha256':self.contract.source_job_sha256,'independent_of_worker':True,'model_or_GPU_imported':False}

    def supervise(self):
        module=self.supervisor();receipt_module=ModuleType('artifact_io');receipt_module.HostReceipt=PendingHostReceipt
        with module_binding('artifact_io',receipt_module):code=module.supervise(self.evidence/'run')
        raw=read_json(self.evidence/'run/results.json');record={'classification':'PARTIAL' if raw['classification']=='PARTIAL' else 'FAIL','supervisor':raw['classification'],
            'copy_error':self.copy_error,'harness_commit':self.commit,'run_contract_sha256':self.contract.sha256}
        began=time.monotonic()
        if code==0 and raw['classification']=='QUALIFIED' and self.copy_error is None:
            try:
                record['independent_verification']=self.verify_outputs()
                if raw['elapsed_seconds']+time.monotonic()-began>=480:raise TimeoutError('Complete host acceptance deadline')
                record['classification']='PASS'
            except Exception as exc:record['verification_error']=type(exc).__name__+': '+str(exc)
        record['allowlisted_copy']=self.copy_receipt
        save(self.evidence/'final-result.json',record)
        print('AMD_GENERATION_DECODE_RESULT='+record['classification'],flush=True)
        return 0 if record['classification']=='PASS' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape-evidence',type=Path,required=True);parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--harness-commit',required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--tile-large-convolutions',action='store_true')
    args=parser.parse_args(argv);runtime=Runtime(args.shape_evidence,args.evidence,args.harness_commit,args.run_id,
                                              tile_large_convolutions=args.tile_large_convolutions)
    if args.preflight:print(json.dumps(runtime.preflight()));return 0
    return runtime.supervise()


if __name__=='__main__':raise SystemExit(main())
