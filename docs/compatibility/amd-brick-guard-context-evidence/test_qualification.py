"""The context adapter may change only authenticated choice metadata."""
import sys
sys.dont_write_bytecode=True
import ast,copy,hashlib,unittest,importlib.util
from pathlib import Path
E=Path(__file__).resolve().parent
OLD=E.parent/'amd-brick-census-recovery-evidence/patch_contract.py'

class QualificationSourceTests(unittest.TestCase):
    def test_real_classifier_adapter_handoff_preserves_math_and_locks(self):
        import numpy as np
        saved=sys.modules.get('common')
        spec=importlib.util.spec_from_file_location('common',E/'common.py');c=importlib.util.module_from_spec(spec)
        sys.modules['common']=c
        try:
            spec.loader.exec_module(c)
            spec=importlib.util.spec_from_file_location('synthetic_context_qualification',E/'qualification.py')
            q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q);pc=c.pinnedpc()
            old=c.pin('synthetic_guard_context_classifier',E.parent/'amd-brick-junction-protected-evidence/patch_contract.py',
                      '336e49bd7836f4a66c02ade831d6b37b66f290cb42e3aa30c72e7a009599d60c')
            size=20;v=np.array([(x,y,0) for y in range(size) for x in range(size)],np.float32);faces=[]
            for y in range(size-1):
                for x in range(size-1):
                    a=y*size+x;faces.extend(((a,a+1,a+size+1),(a,a+size+1,a+size)))
            f=np.array(faces,np.int32);n=len(v);left=f[:,[0,1,2]];right=f[:,[1,2,0]]
            flat=(np.minimum(left,right).astype(np.uint64)*np.uint64(n)+np.maximum(left,right).astype(np.uint64)).ravel()
            order=np.argsort(flat,kind='stable').astype(np.int32);keys,starts=np.unique(flat[order],return_index=True)
            offsets=np.append(starts,len(flat)).astype(np.int32);inc=np.diff(offsets)
            _,global_masks=old.classify(v,f,np.arange(n,dtype=np.int32),f,[],keys,inc,n)
            context=dict(order=order,offsets=offsets,consistent=inc==2,protected_faces=global_masks['protected_faces'],
                         protected_vertices=global_masks['protected_vertices'],locked_vertices=global_masks['vertex_locks'],
                         global_sheet_labels=global_masks['sheet_labels'],eligible_edge_ids=global_masks['eligible_edge_rows'],
                         face_degree=np.bincount(f.ravel(),minlength=n).astype(np.int32),classify=old.classify,check=lambda:None)
            original_choice=pc.choose_component(f,keys,order,offsets,context['eligible_edge_ids'])
            selection=pc.extract(f,keys,order,offsets,original_choice['seed_edge_id'],n)
            choice=dict(seed_edge_id=original_choice['seed_edge_id'],selected_component_faces=len(original_choice['component_face_ids']),
                        selected_component_edges=len(original_choice['component_edge_ids']),eligible_face_components=original_choice['eligible_component_count'])
            rgba=(np.arange(n*4).reshape(n,4)%256).astype(np.uint8);labels=np.zeros(n,np.int32)
            expected,expected_arrays,expected_local=pc.qualify(v,f,rgba,keys,inc,labels,context,original_choice,selection)
            actual,arrays,local=q.qualify(v,f,rgba,keys,inc,labels,context,choice,selection)
            self.assertTrue(actual['qualified']);self.assertEqual(actual,expected);self.assertEqual(local,expected_local)
            self.assertEqual(set(arrays),set(expected_arrays))
            for name,array in arrays.items():
                np.testing.assert_array_equal(array.view(np.uint32) if array.dtype==np.float32 else array,
                                              expected_arrays[name].view(np.uint32) if array.dtype==np.float32 else expected_arrays[name])
            blocked,_,_=q.qualify(v,f,rgba,keys,inc,labels,dict(context,locked_vertices=np.ones(n,np.bool_)),choice,selection)
            self.assertFalse(blocked['qualified']);self.assertEqual(blocked['strict_eligible_edges'],0)
        finally:
            if saved is None:sys.modules.pop('common',None)
            else:sys.modules['common']=saved

    def test_only_three_frozen_choice_metadata_expressions_change(self):
        self.assertTrue((E/'qualification.py').is_file(),'Qualification adapter implementation is missing')
        self.assertEqual(hashlib.sha256(OLD.read_bytes()).hexdigest(),'94080df52f5f61084ac304e14d3c499ba02bd805d0f72fbe3592673bba8e34f7')
        old=ast.parse(OLD.read_text(encoding='utf-8'));new=ast.parse((E/'qualification.py').read_text(encoding='utf-8'))
        replacements={
            "len(choice['component_face_ids'])":"choice['selected_component_faces']",
            "len(choice['component_edge_ids'])":"choice['selected_component_edges']",
            "choice['eligible_component_count']":"choice['eligible_face_components']"}
        replacements={ast.dump(ast.parse(k,mode='eval').body):ast.parse(v,mode='eval').body for k,v in replacements.items()}
        changed=[]
        class Normalize(ast.NodeTransformer):
            def visit(self,node):
                match=replacements.get(ast.dump(node))
                if match is not None:changed.append(1);return copy.deepcopy(match)
                return super().visit(node)
        oldfn=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='qualify')
        newfn=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='qualify')
        self.assertEqual(ast.dump(Normalize().visit(oldfn)),ast.dump(newfn))
        self.assertEqual(len(changed),3)
        # Helpers remain the actual authenticated prior implementations.
        bindings={}
        for n in new.body:
            if isinstance(n,ast.Assign):
                for target in n.targets:
                    if isinstance(target,ast.Name):bindings[target.id]=ast.dump(n.value)
        self.assertEqual(bindings['pc'],ast.dump(ast.parse('c.pinnedpc()',mode='eval').body))
        for name in ('need','complete_stars','full_link','greedy_capacity','raw_edge_components'):
            self.assertEqual(bindings[name],ast.dump(ast.parse('pc.'+name,mode='eval').body))

if __name__=='__main__':unittest.main(verbosity=2)
