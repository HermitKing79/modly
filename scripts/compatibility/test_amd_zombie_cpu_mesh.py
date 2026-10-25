"""CPU-only policy and topology fixtures for saved zombie inspection."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path[:0]=[str(Path(__file__).parent),'D:/AI/Trellis2AMD/venv/Lib/site-packages']
PATH=Path(__file__).with_name('amd_zombie_cpu_mesh.py')
subject=None
if PATH.exists():
    spec=importlib.util.spec_from_file_location('zombie_cpu_mesh',PATH)
    subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)


class Exists(unittest.TestCase):
    def test_cpu_mesh_owner_exists(self):self.assertIsNotNone(subject,'Bounded CPU owner missing')


@unittest.skipIf(subject is None,'Not implemented')
class Policies(unittest.TestCase):
    def test_exact_CPU_resource_floors_admit_and_each_low_value_rejects(self):
        self.assertTrue(subject.admission(12884901888,2147483648)['admitted'])
        self.assertFalse(subject.admission(12884901887,2147483648)['admitted'])
        self.assertFalse(subject.admission(12884901888,2147483647)['admitted'])

    def test_saved_receipts_require_full_success_and_owned_clean_settlement(self):
        final,host,guest,sidecar=subject.receipts()
        subject.validate_receipts(final,host,guest,sidecar)
        for target,key,value in [('final','classification','FAIL'),('host','observer_healthy',False),
            ('host','minimum_host_ram_bytes',12884901887),('guest','forward_calls',2)]:
            rows=copy.deepcopy(dict(final=final,host=host,guest=guest,sidecar=sidecar))
            rows[target][key]=value
            with self.assertRaises(ValueError):subject.validate_receipts(**rows)
        altered=copy.deepcopy(host);altered['windows_job_cleanup']['forced_job_cleanup']=True
        with self.assertRaises(ValueError):subject.validate_receipts(final,altered,guest,sidecar)

    def test_components_preserve_unused_vertices_and_disconnected_surface(self):
        import numpy as np
        faces=np.array([[0,1,2],[2,1,3],[4,5,6]],dtype=np.int32)
        result=subject.components(np,faces,8)
        self.assertTrue(result['components_measured'])
        self.assertEqual(result['surface_components'],2)
        self.assertEqual(result['unused_vertices'],1)
        self.assertEqual(result['largest_component_vertices'],4)
        self.assertEqual(result['largest_component_triangles'],2)

    def test_single_triangle_component_does_not_include_unused_vertex(self):
        import numpy as np
        result=subject.components(np,np.array([[0,2,3]],np.int32),4)
        self.assertEqual(result['surface_components'],1)
        self.assertEqual(result['unused_vertices'],1)
        self.assertEqual(result['largest_component_vertices'],3)


if __name__=='__main__':unittest.main()
