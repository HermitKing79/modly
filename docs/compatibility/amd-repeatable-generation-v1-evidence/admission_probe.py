"""CPU-only negative capability probe; never presents this as inference.

Runs with the installed Windows AMD interpreter. It validates a genuinely
unused source image, then calls the unchanged production Generate admission
method. No worker identity/model readiness is fabricated or bypassed.
"""
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
EXT=ROOT/'experimental/extensions/trellis2-amd'
SOURCE_REL='docs/compatibility/amd-object-conditioned-occupancy-evidence/image-selection/0e4984a9b3765ce80e9853443f9319ecedf90885c74b56cccfebc09402740f8a.webp'
EXPECTED='a31ba3b084757faa66e4bb91a85bd956fcc1eaaaf298a091fd08044351cd0293'
OUT=Path(sys.argv[1]).resolve()
OUT.relative_to(ROOT.parent/'artifacts')
BLOCKED=('torch','triton','trellis2','cumesh','flex_gemm','o_voxel','nvdiffrast','transformers')
attempts=[]
class NoAI(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in BLOCKED:
   attempts.append(fullname)
   raise RuntimeError('CPU admission probe prohibits AI/native imports: '+fullname)
sys.meta_path.insert(0,NoAI())
sys.path[:0]=[str(EXT),str(ROOT/'api')]
from alpha_input import decode_alpha_image

began=time.monotonic()
blob=(ROOT/SOURCE_REL).read_bytes();assert hashlib.sha256(blob).hexdigest()==EXPECTED
image,alpha=decode_alpha_image(blob)
pixel_sha=hashlib.sha256(image.tobytes()).hexdigest();image.close()
negative=[]
for name,data in [('unreadable',b'not an image'),('empty',b'')]:
 try:decode_alpha_image(data)
 except ValueError as exc:negative.append({'case':name,'rejected':True,'error':str(exc)})
 else:raise AssertionError('Invalid image accepted: '+name)

spec=importlib.util.spec_from_file_location('actual_amd_generator',EXT/'generator.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
generator=module.AMDScaffold(OUT/'no-model-directory',OUT/'no-output-created')
generator.MODEL_NODE_ID='generate'
assert generator._model is None and generator._identity is None
params={'mock':False,'image_mode':'alpha-only'}
try:
 generator.generate(blob,params)
except ValueError as exc:
 rejection=str(exc)
 assert rejection=='AMD scaffold requires mock=true; inference is not implemented'
else:raise AssertionError('Unimplemented non-mock generation unexpectedly accepted')
assert not attempts and not any(name in sys.modules for name in BLOCKED)
assert not (OUT/'no-model-directory').exists() and not (OUT/'no-output-created').exists()
record={'classification':'BLOCKED','probe_type':'CPU input and production method admission, not GPU inference or application IPC',
 'source':{'repo_relative_path':SOURCE_REL,'sha256':EXPECTED,'bytes':len(blob),'pixel_sha256':pixel_sha,
           'subject':'vine-covered brick fragment; materially different from turret','alpha_validation':alpha},
 'runtime_metadata_only':{'python':sys.version.split()[0],'torch_distribution':importlib.metadata.version('torch'),
                          'pillow':importlib.metadata.version('pillow')},
 'production_method':'experimental/extensions/trellis2-amd/generator.py:AMDScaffold.generate',
 'production_method_sha256':hashlib.sha256((EXT/'generator.py').read_bytes()).hexdigest(),
 'request_params':params,'exact_rejection':rejection,'rejected_before_identity_or_model_loading':True,
 'negative_image_cases':negative,'AI_import_attempts':attempts,'model_loads':0,'GPU_initializations':0,
 'neural_forwards':0,'generated_mesh':None,'generated_material':None,'generated_GLB':None,
 'elapsed_seconds':time.monotonic()-began}
with (OUT/'admission.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record),flush=True)
