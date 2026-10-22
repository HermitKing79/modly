"""Complete qualification handoff with the genuine pinned local classifier."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
import unittest,hashlib,importlib.util,numpy as np
import patch_contract as pc
from test_patch import edge_index

class IntegrationTests(unittest.TestCase):
    def test_real_classifier_and_inherited_guard_complete_handoff(self):
        oldpath=E.parent/'amd-brick-junction-protected-evidence/patch_contract.py'
        self.assertEqual(hashlib.sha256(oldpath.read_bytes()).hexdigest(),'336e49bd7836f4a66c02ade831d6b37b66f290cb42e3aa30c72e7a009599d60c')
        spec=importlib.util.spec_from_file_location('integration_pinned_classifier',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        size=20;v=np.array([(x,y,0) for y in range(size) for x in range(size)],np.float32);faces=[]
        for y in range(size-1):
            for x in range(size-1):
                a=y*size+x;faces.extend(((a,a+1,a+size+1),(a,a+size+1,a+size)))
        f=np.array(faces,np.int32);n=len(v);keys,order,off,_=edge_index(f,n);inc=np.diff(off)
        _,a=old.classify(v,f,np.arange(n,dtype=np.int32),f,[],keys,inc,n)
        con=np.zeros(len(keys),np.bool_);two=np.flatnonzero(inc==2);con[two]=True
        context=dict(order=order,offsets=off,consistent=con,protected_faces=a['protected_faces'],protected_vertices=a['protected_vertices'],locked_vertices=a['vertex_locks'],global_sheet_labels=a['sheet_labels'],eligible_edge_ids=a['eligible_edge_rows'],face_degree=np.bincount(f.ravel(),minlength=n).astype(np.int32),classify=old.classify,check=lambda:None)
        choice=pc.choose_component(f,keys,order,off,context['eligible_edge_ids']);selection=pc.extract(f,keys,order,off,choice['seed_edge_id'],n)
        rgba=(np.arange(n*4).reshape(n,4)%256).astype(np.uint8);labels=np.zeros(n,np.int32)
        summary,arrays,_=pc.qualify(v,f,rgba,keys,inc,labels,context,choice,selection)
        self.assertTrue(summary['qualified']);self.assertGreater(summary['strict_eligible_edges'],0)
        self.assertGreater(summary['initial_non_touching_opportunities'],0)
        np.testing.assert_array_equal(arrays['source_vertex_ids'][arrays['faces']],f[arrays['source_face_ids']])
        np.testing.assert_array_equal(arrays['vertices'].view(np.uint32),v[arrays['source_vertex_ids']].view(np.uint32))
        np.testing.assert_array_equal(arrays['original_native_RGBA'],rgba[arrays['source_vertex_ids']])
        # Existing global guard facts must survive the complete integration.
        blocked=dict(context,locked_vertices=np.ones(n,np.bool_))
        bad,_,_=pc.qualify(v,f,rgba,keys,inc,labels,blocked,choice,selection)
        self.assertFalse(bad['qualified']);self.assertEqual(bad['strict_eligible_edges'],0)

if __name__=='__main__':unittest.main(verbosity=2)
