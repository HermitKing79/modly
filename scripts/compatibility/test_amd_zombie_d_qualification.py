"""D-only qualification evidence and baseline reuse policies, with no real stage."""
import builtins,contextlib,copy,hashlib,importlib.util,io,json,os,runpy,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
PATH=HERE/'amd_zombie_d_handoff.py'
h=None
if PATH.exists():
    s=importlib.util.spec_from_file_location('_d_qualification_test',PATH);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)

class Exists(unittest.TestCase):
    def test_D_only_handoff_exists(self):self.assertIsNotNone(h,'D-only handoff missing')

@unittest.skipIf(h is None,'not implemented')
class Qualification(unittest.TestCase):
    def test_qualified_evidence_changes_are_rejected(self):
        parent=HERE.parents[2]/'artifacts/amd-zombie-d-policy-tests';parent.mkdir(exist_ok=True)
        folder=Path(tempfile.mkdtemp(dir=parent));(folder/'evidence').mkdir()
        source=folder/'evidence/raster.png';source.write_bytes(b'synthetic qualified raster')
        row={'evidence/raster.png':dict(bytes=source.stat().st_size,sha256=hashlib.sha256(source.read_bytes()).hexdigest())}
        h.verify_qualification(folder,row)
        source.write_bytes(b'modified')
        with self.assertRaises(ValueError):h.verify_qualification(folder,row)
        for invalid in ('../raster.png','C:/foreign.png','evidence\\raster.png'):
            with self.assertRaises(ValueError):h.verify_qualification(folder,{invalid:next(iter(row.values()))})

    def test_actual_isolated_default_cannot_load_models_or_start_stages(self):
        result=subprocess.run([sys._base_executable,'-B','-I','-S',str(PATH)],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads(result.stdout)
        self.assertFalse(plan['neural_execution']);self.assertFalse(plan['marker_reserved'])
        self.assertIn('zombie-d-9ac0364f-v1',plan['case']);self.assertNotIn('zombie-c',plan['case'])

    def test_stale_pipeline_or_neural_output_is_never_reused(self):
        parent=HERE.parents[2]/'artifacts/amd-zombie-d-policy-tests';parent.mkdir(exist_ok=True)
        folder=Path(tempfile.mkdtemp(dir=parent));case=folder/'case';(case/'occupancy').mkdir(parents=True)
        for name in ('source-image.png','input-contract.json'):(case/'occupancy'/name).write_bytes(b'x')
        with patch.multiple(h,CASE=case,PRIVATE=folder/'pipeline',COMPARISON=folder/'compare'):
            h.require_fresh_outputs()
            (case/'occupancy/attempt-started.json').write_bytes(b'{}')
            with self.assertRaises(FileExistsError):h.require_fresh_outputs()

    def test_C_and_A_images_cannot_satisfy_D_authority(self):
        source=h.authority()
        self.assertEqual(source['source_sha256'],'9ac0364f409077cd3fdeb853a1baed3b851c56f4fdc539c9fde73e0e3ab79db9')
        for wrong in ('db1fbb6bd27717f4b6a0298d64ff1ddff17e80027d782bf5dbfbc959d70fa06e',
                      '60862722c989772ad973870890601a6a95ed9a5cd5a9150a378e989db35ad437'):
            altered=dict(source,source_sha256=wrong)
            with patch.object(h.json,'loads',return_value=altered):
                with self.assertRaises(ValueError):h.authority()

@unittest.skipIf(h is None,'not implemented')
class SecondaryC(unittest.TestCase):
    def setUp(self):
        path=HERE/'amd_zombie_d_compare.py';s=importlib.util.spec_from_file_location('_d_secondary_test',path)
        self.p=importlib.util.module_from_spec(s);s.loader.exec_module(self.p)

    def test_existing_C_views_are_verified_without_render_or_runtime_imports(self):
        original=self.p.read(self.p.ORIGINAL/'previews/render-metadata.json')
        before=builtins.__import__
        def guard(name,*args,**kwargs):
            if name.split('.')[0] in {'PIL','numpy','torch','bpy'}:raise AssertionError('Runtime imported')
            return before(name,*args,**kwargs)
        with patch.object(builtins,'__import__',side_effect=guard),patch.object(self.p,'supervise',side_effect=AssertionError('Rendered')):
            metadata,files=self.p.secondary_inputs(original)
        self.assertEqual(len(metadata['views']),8);self.assertEqual(len(files),11)
        self.assertEqual(metadata['source_PLY_sha256'],'a2d015fbcd54f6b1232674c0fef239040501629460ab177dcf826f4b34d3809d')

    def test_secondary_camera_or_source_drift_rejected(self):
        original=self.p.read(self.p.ORIGINAL/'previews/render-metadata.json');read=self.p.read
        for field in ('camera','source','gpu'):
            def altered(path,*args,**kwargs):
                row=read(path,*args,**kwargs)
                if Path(path)==self.p.SECONDARY/'previews/render-metadata.json':
                    if field=='camera':row['views'][0]['ortho_scale']+=.01
                    if field=='source':row['source_PLY_sha256']='0'*64
                    if field=='gpu':row['device']='GPU'
                return row
            with self.subTest(field=field),patch.object(self.p,'read',side_effect=altered):
                with self.assertRaises(ValueError):self.p.secondary_inputs(original)

    def test_D_renderer_rejects_foreign_recovery_mode_before_runtime(self):
        before=builtins.__import__
        def guard(name,*args,**kwargs):
            if name.split('.')[0] in {'bpy','numpy','mathutils'}:raise AssertionError('Renderer imported runtime')
            return before(name,*args,**kwargs)
        with patch.object(sys,'stdin',type('Input',(),{'buffer':io.BytesIO(b'GO\n')})()), \
             patch.object(sys,'argv',['blender','--','--render-recovery']),patch.object(builtins,'__import__',side_effect=guard):
            with self.assertRaisesRegex(ValueError,'no recovery'):
                runpy.run_path(str(HERE/'amd_zombie_d_compare_render.py'),run_name='__main__')

    def test_three_way_sheet_uses_existing_native_C_copies_and_keeps_visual_unassessed(self):
        from PIL import Image
        parent=HERE.parents[2]/'artifacts/amd-zombie-d-policy-tests';parent.mkdir(exist_ok=True)
        folder=Path(tempfile.mkdtemp(dir=parent));base=folder/'D';base.mkdir();out=base/'review';out.mkdir()
        for label in ('original','candidate-D'):(out/label).mkdir()
        p=self.p;original=p.read(p.ORIGINAL/'previews/render-metadata.json')
        cmeta,_=p.secondary_inputs(original)
        for i,row in enumerate(original['views']):
            Image.new('RGB',(768,768),(i,20,30)).save(out/'original'/row['name'])
            Image.new('RGB',(768,768),(i,40,50)).save(out/'candidate-D'/row['name'])
        inputs=dict(immutable_secondary_C=p.snapshot([p.SECONDARY/'previews'/r['name'] for r in cmeta['views']]))
        p.save(base/'comparison-inputs.json',inputs)
        p.save(out/'comparison-review.json',dict(classification='PASS_PAIRED_EVIDENCE_VISUAL_UNASSESSED'))
        with patch.object(p,'BASE',base):
            p.secondary_review()
            result=p.validate_secondary_review()
        self.assertEqual(len(result['views']),8)
        self.assertEqual(result['visual_status'],'UNASSESSED_REQUIRES_HUMAN_VISUAL_REVIEW')
        self.assertIsNone(result['overall_quality_winner'])
        with Image.open(out/'three-way-contact-sheet.png') as sheet:
            self.assertEqual(sheet.size,(1152,3296))
        for row in cmeta['views']:
            self.assertEqual(p.digest(out/'secondary-C'/row['name']),row['sha256'])
        self.assertIn('black speckles',(out/'three-way-review.html').read_text())
        with self.assertRaises(FileExistsError),patch.object(p,'BASE',base):p.secondary_review()

if __name__=='__main__':unittest.main()
