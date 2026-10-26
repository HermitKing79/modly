"""CPU-only proof witnesses; no Blender, Torch, model or GPU imports."""
import ast,importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.append('D:/AI/Trellis2AMD/venv/Lib/site-packages')
import numpy as np
from PIL import Image
ENTRY=Path(__file__).with_name('amd_cpu_uv_bake.py')
proof=None
if ENTRY.exists():
    spec=importlib.util.spec_from_file_location('_uv_proof_test',ENTRY);proof=importlib.util.module_from_spec(spec);spec.loader.exec_module(proof)

class ExistingDefect(unittest.TestCase):
    def test_legacy_forward_really_passes_one_argument_too_many(self):
        path=Path(__file__).resolve().parents[2]/'api/texture_baker/texture_baker/baker.py'
        tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        ns={'nn':SimpleNamespace(Module=object),'Tensor':object}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),ns)
        baker=ns['TextureBaker']();baker.rasterize=lambda *a:'synthetic-raster'
        with self.assertRaises(TypeError):baker.forward(object(),object(),object(),64)
    def test_proof_module_exists(self):self.assertTrue(ENTRY.is_file(),'CPU proof helper missing')

@unittest.skipUnless(proof,'Proof helper missing')
class ProofWitnesses(unittest.TestCase):
    def test_camera_record_serializes_numpy_scalars_without_losing_values(self):
        row=proof.preview_record('front.png','abc',[np.float32(.5),-4,1],[0,0,np.float32(.25)],1.17)
        self.assertEqual(json.loads(json.dumps(row,allow_nan=False))['camera'],[.5,-4.,1.])
        self.assertEqual(row['target'],[0.,0.,.25])
    def test_audit_rejects_neural_imports_subprocesses_and_foreign_writes(self):
        for event,args in [('import',('torch',)),('subprocess.Popen',('wsl.exe',)),
                           ('socket.connect',('localhost',)),('os.rename',('old','new'))]:
            with self.assertRaises(RuntimeError):proof.audit_policy(event,args)
        with self.assertRaises(ValueError):proof.audit_policy('open',(str(proof.ROOT/'outside-proof.txt'),'w',0))
        with self.assertRaises(ValueError):proof.audit_policy('os.remove',(str(proof.BASE/'source/mesh.ply'),))
    def test_readback_requires_logged_colour_failure_and_keeps_its_value(self):
        self.assertEqual(proof.failed_colour_error('ValueError: Independent linear-colour reference differs: 0.9951595067977905\n'),.9951595067977905)
        for text in ('ValueError: geometry changed','ValueError: Independent linear-colour reference differs: nan',
                     'ValueError: Independent linear-colour reference differs: 0.001'):
            with self.assertRaises(ValueError):proof.failed_colour_error(text)
    def test_geometry_hash_allows_uv_seam_duplicates_but_rejects_surface_changes(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32);f=np.array([[0,1,2],[1,3,2]],np.int32)
        expected=proof.geometry_hash(np,v,f)
        vv=v[f].reshape(-1,3);ff=np.arange(6,dtype=np.int32).reshape(2,3)
        self.assertEqual(expected,proof.geometry_hash(np,vv,ff[::-1][:,[1,2,0]]))
        self.assertNotEqual(expected,proof.geometry_hash(np,v,f[:,[0,2,1]]))
        vv[0,0]=.001;self.assertNotEqual(expected,proof.geometry_hash(np,vv,ff))
    def test_png_has_explicit_srgb_encoding_and_vertical_origin(self):
        rgba=np.ones((2,2,4),np.float32);rgba[0,0,:3]=[.5,0,1];rgba[1,0,:3]=[0,.25,0]
        with tempfile.TemporaryDirectory(dir=proof.BASE) as folder:
            p=Path(folder)/'test.png';proof.write_png(np,p,rgba)
            with Image.open(p) as image:values=np.asarray(image)
            self.assertEqual(values[1,0].tolist(),[188,0,255,255]);self.assertEqual(values[0,0].tolist(),[0,137,0,255])
    def test_uv_validation_rejects_zero_area_and_outside_atlas(self):
        good=np.array([[[0,0],[1,0],[0,1]]],np.float32)
        self.assertEqual(proof.uv_metrics(np,good)['zero_area_triangles'],0)
        with self.assertRaises(ValueError):proof.uv_metrics(np,good*0)
        with self.assertRaises(ValueError):proof.uv_metrics(np,good+2)

if __name__=='__main__':unittest.main()
