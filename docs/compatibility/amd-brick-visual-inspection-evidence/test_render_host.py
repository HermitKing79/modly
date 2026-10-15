"""Focused CPU checks for the first-render continuation gate; no Blender."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from PIL import Image,ImageDraw
spec=importlib.util.spec_from_file_location('brick_render_host',Path(__file__).with_name('render_host.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
class HostTests(unittest.TestCase):
 def fixture(self,p):
  image=Image.new('RGB',(1024,1024),'#333333');ImageDraw.Draw(image).rectangle((200,200,800,800),fill='#aaaaaa')
  image.save(p/'view-pos-x.png')
  return dict(name='pos-x',PNG='view-pos-x.png',sha256=h.digest(p/'view-pos-x.png'),bytes=(p/'view-pos-x.png').stat().st_size,seconds=5,device='CPU',denoising_use_gpu=False,source_PLY_sha256=h.SOURCE_SHA,vertex_count=1957891,face_count=4173584,geometry_unchanged_after_first_view=True)
 def test_exact_first_image_and_resource_contract(self):
  with tempfile.TemporaryDirectory(dir=h.PRIVATE) as d:
   p=Path(d);r=self.fixture(p);self.assertTrue(h.first_image(p,r)['verified'])
  self.assertEqual((h.CAP,h.RAM_FLOOR,h.DISK_FLOOR,h.DEADLINE),(6*1024**3,12*1024**3,2*1024**3,180))
 def test_foreign_names_hash_and_dimensions_rejected(self):
  with tempfile.TemporaryDirectory(dir=h.PRIVATE) as d:
   p=Path(d);r=self.fixture(p)
   for bad in (dict(r,PNG='../view-pos-x.png'),dict(r,name='neg-x'),dict(r,sha256='0'*64),dict(r,bytes=1),dict(r,device='GPU'),dict(r,vertex_count=4),dict(r,geometry_unchanged_after_first_view=False)):
    with self.assertRaises(ValueError):h.first_image(p,bad)
   Image.new('RGB',(10,10)).save(p/'view-pos-x.png');r.update(sha256=h.digest(p/'view-pos-x.png'),bytes=(p/'view-pos-x.png').stat().st_size)
   with self.assertRaises(ValueError):h.first_image(p,r)
 def test_missing_or_uniform_first_image_rejected(self):
  with tempfile.TemporaryDirectory(dir=h.PRIVATE) as d:
   p=Path(d);r=self.fixture(p);Image.new('RGB',(1024,1024),'black').save(p/'view-pos-x.png')
   r.update(sha256=h.digest(p/'view-pos-x.png'),bytes=(p/'view-pos-x.png').stat().st_size)
   with self.assertRaises(ValueError):h.first_image(p,r)
 def test_resource_failure_preserves_measured_values(self):
  r=h.admission(1,2);self.assertFalse(r['admitted']);self.assertEqual((r['available_ram_bytes'],r['disk_free_bytes']),(1,2))
  self.assertTrue(h.admission(h.RAM_FLOOR,h.DISK_FLOOR)['admitted'])
if __name__=='__main__':unittest.main(verbosity=2)
