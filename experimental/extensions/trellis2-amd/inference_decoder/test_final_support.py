"""CPU-only qualification; synthetic data is never claimed as recovered geometry."""
import copy
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_contract as rc
import admission as a
from test_decoder import contract_document


def brick_document():
    d=contract_document()
    d.update(initial_count=4430,maximum_final_rows=1957891,
        input_sha256={'shape-latent-denormalized.npy':'e6a6eec09029db43b7b2356ee71a943ad2f225c45d48a03bfaa519a031b8c808',
                      'coords.npy':'17a8ddc94edaa9da5b02834157eef9f8d4f7f99bfa8e2f4d2b5806f1b1cfe93c'},
        source_job_sha256='b92aade7aca2e832fbf425ec9d4e54d40aa7ffa4b4afdea54c38f71f61b8e62f',
        large_convolution_tiling=True,final_support_profile='brick-1957891-v1')
    return d


class FinalSupportTests(unittest.TestCase):
    def test_explicit_profile_roundtrip_and_historical_document_stability(self):
        c=rc.parse(brick_document())
        self.assertEqual(rc.final_limit(c),1957891)
        self.assertEqual(c.document(),brick_document())
        self.assertEqual(rc.parse(c.document()).sha256,c.sha256)
        self.assertEqual(rc.parse(contract_document()).document(),contract_document())
        self.assertEqual(rc.final_limit(),1842968)

    def test_profile_rejects_foreign_job_hashes_counts_and_missing_optin(self):
        d=brick_document()
        for key,value in [('initial_count',4429),('maximum_final_rows',1957892),
                          ('maximum_final_rows',1842968),('source_job_sha256','f'*64),
                          ('large_convolution_tiling',False),('final_support_profile','other')]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):rc.parse(dict(d,**{key:value}))
        for key in d['input_sha256']:
            bad=copy.deepcopy(d);bad['input_sha256'][key]='f'*64
            with self.assertRaises(ValueError):rc.parse(bad)
        bad=dict(d);bad.pop('final_support_profile')
        with self.assertRaises(ValueError):rc.parse(bad)

    def test_measured_final_plans_60_tiles_and_unchanged_live_admission(self):
        c=rc.parse(brick_document())
        p=a.final_c2s_plan(438684,512,128,1957891,64,contract=c)
        self.assertEqual(p['visible_bytes'],438684*64+1957891*(176+160+128+1024))
        conv=a.chunked_conv_plan(1957891,32768,512**3,contract=c)
        self.assertEqual((conv['chunks'],conv['last_chunk_rows']),(60,24579))
        self.assertEqual(a.output_payload_bytes(1957891,[4430,21391,97313,438684],contract=c),104125380)
        for f,args in [(a.final_c2s_plan,(438684,512,128,1957891,64)),
                       (a.chunked_conv_plan,(1957891,32768,512**3))]:
            with self.assertRaises(a.AdmissionStop):f(*args)
        with self.assertRaises(a.AdmissionStop):a.chunked_conv_plan(1957892,32768,512**3,contract=c)
        events=[]
        h=a.AdmissionHooks(None,None,contract=c,tile_large_convolutions=True,emit=events.append,memory=lambda:dict(
            allocated_bytes=12*a.GIB,reserved_bytes=12*a.GIB,free_bytes=13*a.GIB))
        with self.assertRaises(a.AdmissionStop):h._admit('unchanged',conv)
        self.assertFalse(events[-1]['admitted'])

    def test_exact_parent_child_lineage_and_artifact_identity(self):
        c=rc.parse(brick_document());parents=[4430,21391,97313,438684]
        rc.support_counts(1957891,parents,c)
        for n,p in [(1957890,parents),(1957892,parents),(1957891,[4430,21391,97313,438683])]:
            with self.assertRaises(ValueError):rc.support_counts(n,p,c)
        provenance=dict(run_contract_sha256=c.sha256,source_job_sha256=c.source_job_sha256,
            input_file_sha256=dict(c.input_sha256),final_support_profile=c.final_support_profile,
            checkpoint_sha256=rc.BRICK_CHECKPOINT_SHA256)
        rc.artifact_binding(provenance,c)
        for field in provenance:
            bad=dict(provenance);bad[field]='foreign'
            with self.subTest(field=field),self.assertRaises(ValueError):rc.artifact_binding(bad,c)

    def test_synthetic_c2s_order_finite_validation_and_default_chain(self):
        import numpy as np
        p=np.array([[0,1,2,3],[0,4,5,6]],np.int32)
        logits=np.full((2,8),-1,np.float32);logits[0,[0,7]]=1;logits[1,2]=1
        np.testing.assert_array_equal(rc.reconstruct_children(p,logits),[[0,2,4,6],[0,3,5,7],[0,8,11,12]])
        logits[0,0]=np.nan
        with self.assertRaises(ValueError):rc.reconstruct_children(p,logits)

    def test_both_fourth_c2s_admissions_record_counts_before_rejection(self):
        # Scalar-only lifecycle fixture; no authentic learned mask is available.
        c=rc.parse(brick_document());events=[];samples=iter([
            dict(allocated_bytes=1684097024,reserved_bytes=3451912192,free_bytes=11531587584),
            dict(allocated_bytes=12*a.GIB,reserved_bytes=12*a.GIB,free_bytes=13*a.GIB)])
        fake_torch=SimpleNamespace(bool='bool',int32='int32',equal=lambda x,y:x is y)
        h=a.AdmissionHooks(None,fake_torch,contract=c,tile_large_convolutions=True,
                           emit=events.append,memory=lambda:dict(next(samples)))
        h.sequence=[('next',None,None,'conv2',3)];h.up_states={3:{'completed':0}}
        coords=SimpleNamespace(dtype='int32',device='cpu')
        class Mask:
            shape=(438684,8);dtype='bool';device='cpu'
            def sum(self):return SimpleNamespace(item=lambda:1957891)
        sub=SimpleNamespace(feats=Mask(),coords=coords)
        x=SimpleNamespace(feats=SimpleNamespace(shape=(438684,512),device='cpu'),coords=coords,
                          spatial_shape=(256,256,256),get_spatial_cache=lambda key:None)
        block=SimpleNamespace(out_channels=64,channels=128,_forward=lambda:None)
        module=SimpleNamespace(forward=lambda:None)
        site=('blocks.3.4.updown',module,block,3)
        # Each pre-hook samples once for its durable count and once for admission.
        good=dict(allocated_bytes=1684097024,reserved_bytes=3451912192,free_bytes=11531587584)
        bad=dict(allocated_bytes=12*a.GIB,reserved_bytes=12*a.GIB,free_bytes=13*a.GIB)
        samples=iter([good,good,good,bad])
        with patch.object(h,'_sparse',return_value=(438684,512)):
            h._up_pre(site,module,(x,sub))
            self.assertEqual(h.up_states[3]['pending_width'],64)
            h.up_states[3]['completed']=1;x.feats.shape=(438684,128)
            with self.assertRaises(a.AdmissionStop):h._up_pre(site,module,(x,sub))
        self.assertEqual([e['call'] for e in events if e['event']=='c2s-count-before-admission'],[1,2])
        self.assertEqual([e['admitted'] for e in events if e['event']=='admission'],[True,False])
        self.assertTrue(all(e['children']==1957891 for e in events if e['event']=='c2s-count-before-admission'))
        h.remove();self.assertNotIn('mask',h.up_states[3])

    def test_complete_profile_receipt_rejects_foreign_large_tile_support(self):
        import decode
        c=rc.parse(brick_document());n=1957891
        parents=rc.BRICK_PARENT_ROWS;children=(*parents[1:],n)
        row=dict(decoder_cpu='PASS',native_import='PASS',forward_calls=1,convolutions_completed=40,
            final_convolution_chunks=60,final_convolution_rows=n,native_map_calls=5,
            final_convolution_patch_restored=True,large_convolution_patch_restored=True,
            large_tiled_convolutions=[dict(site=site,rows=438684,in_channels=128,out_channels=512 if i==5 else 128,
                tile_rows=32768,chunks=14,rows_completed=438684,global_map_reused=i>0,completed=True)
                for i,site in enumerate(a.LARGE_TILED_SITES)],
            subdivision_operations=[dict(site=f'blocks.{i}.{(4,16,8,4)[i]}.updown',call=call,
                parents=parent,children=children[i]) for i,parent in enumerate(parents) for call in (1,2)],
            output=dict(rows=n,verified=True,readback_verified_before_release=True,
                subdivisions=[dict(parents=p,selected_children=ch) for p,ch in zip(parents,children)]),
            CPU_model_collected=True,CPU_parameters_collected=True,CPU_component_collected=True,
            memory_release=dict(allocated=0,reserved=0),workspace_clear_calls=1,
            runtime_dtype_counts=dict(BF16=216,F32=76),constructed_on_meta=True,buffers_count=0,
            decoder_diagnostics=dict(strict=True,roundtrip_verified=True,entries=292,
                dtype_counts=dict(F16=216,F32=76),missing_keys=[],unexpected_keys=[]),
            policy_final=dict(rejected_calls=0),run_contract_sha256=c.sha256,source_job_sha256=c.source_job_sha256)
        decode.validate_qualified(row,c)
        bad=copy.deepcopy(row)
        for r in bad['large_tiled_convolutions']:
            r.update(rows=438683,rows_completed=438683)
        with self.assertRaises(RuntimeError):decode.validate_qualified(bad,c)
        for key,value in [('final_convolution_chunks',59),('convolutions_completed',39),
                          ('memory_release',dict(allocated=1,reserved=0))]:
            with self.assertRaises(RuntimeError):decode.validate_qualified(dict(row,**{key:value}),c)

    def test_final64_original_arithmetic_cross_tile_and_sixty_row_ranges(self):
        from test_large_tiling import coordinates,independent_neighbors,coordinate_math,pinned_functions
        import torch
        from chunked_final import row_chunked_explicit
        self.assertFalse(torch.cuda.is_initialized())
        with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU entry forbidden')):
            coords=coordinates();neighbors=independent_neighbors(coords)
            feats=((torch.arange(len(coords)*64).reshape(-1,64)%8)-4).to(torch.bfloat16)
            weight=torch.zeros(64,3,3,3,64,dtype=torch.bfloat16)
            weight[:,1,1,1,0]=.25;weight[:,0,1,1,1]=.5
            bias=torch.arange(64).to(torch.bfloat16)*.25
            reference=coordinate_math(feats,coords,weight,bias)
            native_python=pinned_functions()['_sparse_submanifold_conv_forward']
            torch.testing.assert_close(native_python(feats,{'neighbor_map':neighbors},weight,bias),reference,rtol=0,atol=0)
            actual=row_chunked_explicit(torch,feats,neighbors,weight,bias,3,lambda *args:None)
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
        ranges=[(start,min(1957891,start+32768)) for start in range(0,1957891,32768)]
        self.assertEqual(len(ranges),60);self.assertEqual(ranges[-1],(1933312,1957891))
        self.assertEqual(sum(end-start for start,end in ranges),1957891)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
