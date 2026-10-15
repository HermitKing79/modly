"""Read-only fresh brick texture lineage, source identity and RAM admission."""
import ctypes,hashlib,importlib.metadata,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/compatibility'))
def cpu_guard(event,args):
 if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm','o_voxel','cumesh'} or event in {'socket.connect','socket.bind'}:raise RuntimeError('CPU-only preflight guard')
sys.addaudithook(cpu_guard)
def main():
 import amd_generation_texture as a
 s=a.APPROVED_ROOT/'new-image/occupancy/shape';t=s.parent/'texture'
 art=Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-brick-material-stage1-df41688')
 r=a.Runtime(a.context(s,t,art,'df416889bc65ad43f0f928c563c6d99c64e730a8'))
 identity=r.verify_preflight();profile=r.profile()
 class M(ctypes.Structure):
  _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
 samples=[]
 for i in range(3):
  m=M();m.length=ctypes.sizeof(m)
  if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
  samples.append(dict(available_RAM_bytes=m.available,D_free_bytes=shutil.disk_usage('D:/').free,C_free_bytes=shutil.disk_usage('C:/').free))
  if i<2:time.sleep(1)
 if not all(x['available_RAM_bytes']>=12*1024**3 for x in samples):raise RuntimeError('Original 12GiB RAM admission failed: '+json.dumps(samples))
 if (t/'run').exists() or list((art/'latent').iterdir()):raise ValueError('Unreserved execution and empty actual output required')
 if 'torch' in sys.modules or Path(os.environ['WORKSPACE_DIR']).resolve()!=ROOT:raise RuntimeError('CPU/workspace boundary')
 record=dict(classification='PASS_CPU_PREFLIGHT',torch_imported=False,GPU_initialized=False,checkpoint_loaded=False,checkpoint_sha256=identity['checkpoint_sha256'],checkpoint_bytes=identity['file_bytes'],checkpoint_entries=identity['tensor_count'],checkpoint_dtype=identity['dtypes'],config_sha256=identity['config_sha256'],token_count=identity['token_count'],shape_condition_sha256=identity['shape_condition_sha256'],feature=identity['feature'],shape_lineage=r.bundle['lineage'],prepared_contract_sha256=a.shape._digest(t/'texture-input-contract.json'),persistence_probe_sha256=a.shape._digest(t/'persistence-preflight.json'),published_harness_required=True,source_files={n:a.shape._digest(ROOT/n) for n in profile.committed_files if (ROOT/n).is_file()},profile=dict(cpu_seconds=profile.cpu_seconds,gpu_seconds=profile.gpu_seconds,overall_seconds=profile.overall_seconds,maximum_sequences=profile.maximum_sequences,maximum_steps=profile.maximum_steps,maximum_forwards=profile.maximum_forwards,worker=str(profile.worker_script),supervisor=a.original.supervisor.__file__),runtime_metadata={k:importlib.metadata.version(k) for k in ('torch','numpy','triton-windows','safetensors','psutil')},metadata_only_note='Windows distribution is triton-windows, not triton; no module import or runtime change.',WORKSPACE_DIR=os.environ['WORKSPACE_DIR'],resource_samples=samples,unused_run_path=str(t/'run'),empty_real_artifact_path=str(art/'latent'))
 with (E/'cpu-preflight.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
 print(json.dumps({k:record[k] for k in ('classification','checkpoint_sha256','checkpoint_bytes','checkpoint_entries','token_count','profile','runtime_metadata','resource_samples')}))
if __name__=='__main__':main()
