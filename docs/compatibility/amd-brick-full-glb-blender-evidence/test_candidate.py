"""Focused pure-CPU index accounting checks; no Blender or original writes."""
import sys
sys.dont_write_bytecode = True
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from candidate import referenced_vertices

class IndexAccounting(unittest.TestCase):
    def test_unused_and_duplicate_positions_not_welded(self):
        f = np.array([[0,1,2],[2,1,0]], dtype='<i4')
        before = f.tobytes(); used, r = referenced_vertices(f, 4, 1)
        self.assertEqual((r['referenced_vertices'], r['unreferenced_vertices']), (3,1))
        self.assertFalse(r['exact_import_vertex_count_gate']); self.assertEqual(f.tobytes(), before)
        self.assertEqual(used.tolist(), [True,True,True,False])
    def test_all_referenced_and_multiplicity(self):
        f = np.array([[0,1,2],[0,1,2],[0,2,3]], dtype='<i4')
        _,r = referenced_vertices(f,4,2)
        self.assertTrue(r['exact_import_vertex_count_gate']); self.assertEqual(r['original_triangles'],3)
    def test_bad_indices(self):
        for f in ([[0,1,-1]], [[0,1,3]]):
            with self.assertRaises(ValueError): referenced_vertices(np.array(f,dtype='<i4'),3)
    def test_malformed_tables(self):
        for f in (np.zeros((1,3),dtype='f4'), np.zeros((1,4),dtype='i4'), np.zeros((0,3),dtype='i4')):
            with self.assertRaises(ValueError): referenced_vertices(f,3)
    def test_bounds(self):
        f = np.array([[0,1,2]],dtype='<i4')
        for n,b in ((True,1),(0,1),(2000001,1),(3,0),(3,32769)):
            with self.assertRaises(ValueError): referenced_vertices(f,n,b)
    def test_actual_host_command_is_cpu_diagnostic_only(self):
        import run_host
        command = run_host.worker_command()
        self.assertEqual(command, ['D:/AI/Trellis2AMD/venv/Scripts/python.exe', '-I', '-B', str(Path(__file__).resolve().parent/'preflight.py')])
        self.assertEqual((run_host.CAP, run_host.DEADLINE), (6*1024**3, 180))
    def test_actual_host_rejects_blender_stages(self):
        import run_host
        for stage in ('import', 'render', 'export'):
            with self.assertRaises(ValueError): run_host.main(stage)

if __name__ == '__main__': unittest.main(verbosity=2)
