"""Small CPU arithmetic/admission proofs; no models, native packages or GPU work."""
import ast
import builtins
from contextlib import ExitStack
import copy
import hashlib
import importlib
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE))
PINNED_SOURCE=ROOT/'docs/compatibility/amd-wsl-native-decoder-preparation-evidence/pinned-linux-submanifold_conv3d.py.txt'
SOURCE_SHA256='67641d13e06ca135a4f90eb71a01f7d8d32448bb5aa7c31b858c69368811fb6b'
OFFSETS=tuple(product((-1,0,1),repeat=3))
SENTINEL=0xffffffff
SITES=('blocks.2.8.conv2',*(f'blocks.3.{i}.conv' for i in range(4)),'blocks.3.4.conv1')


def pinned_functions():
    """Compile only genuine pure-Torch methods; never import the native module."""
    if hashlib.sha256(PINNED_SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA256:
        raise AssertionError('Pinned source fixture identity changed')
    tree=ast.parse(PINNED_SOURCE.read_text())
    nodes=[]
    for name in ('_compute_neighbor_cache_torch','_sparse_submanifold_conv_forward'):
        node=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name))
        node.decorator_list=[];nodes.append(node)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[]))
    namespace=dict(torch=torch,SubMConv3dNeighborCache=dict,
                   spconv=SimpleNamespace(ALGORITHM=0),Algorithm=SimpleNamespace(EXPLICIT_GEMM=0))
    exec(compile(module,str(PINNED_SOURCE),'exec'),namespace)
    return namespace


def coordinates():
    # Deliberately not spatially sorted. Row0 has neighbours in later tiles;
    # the isolated last row exercises sentinel-only off-centre positions.
    return torch.tensor([[0,1,1,1],[0,1,1,2],[0,0,1,1],[0,2,1,1],
                         [0,1,0,1],[0,1,2,1],[0,4,4,4]],dtype=torch.int32)


def independent_neighbors(coords):
    lookup={tuple(row):index for index,row in enumerate(coords.tolist())}
    rows=[]
    for batch,x,y,z in coords.tolist():
        rows.append([lookup.get((batch,x+dx,y+dy,z+dz),SENTINEL) for dx,dy,dz in OFFSETS])
    return torch.tensor(rows,dtype=torch.uint32)


def untiled_math(feats,neighbor,weight,bias):
    """Independent untiled im2col construction and original GEMM equation."""
    n,ci=feats.shape
    columns=torch.zeros(n,27,ci,dtype=feats.dtype)
    for row,indices in enumerate(neighbor.tolist()):
        for kernel,index in enumerate(indices):
            if index!=SENTINEL:columns[row,kernel]=feats[index]
    result=columns.reshape(n,27*ci) @ weight.reshape(weight.shape[0],27*ci).T
    return result if bias is None else result+bias


def coordinate_math(feats,coords,weight,bias):
    """Direct high-precision coordinate convolution; uses no neighbour table."""
    lookup={tuple(row):index for index,row in enumerate(coords.tolist())}
    outputs=[]
    for batch,x,y,z in coords.tolist():
        value=torch.zeros(weight.shape[0],dtype=torch.float64)
        for (dx,dy,dz),(kw,kh,kd) in zip(OFFSETS,product(range(3),repeat=3)):
            source=lookup.get((batch,x+dx,y+dy,z+dz))
            if source is not None:value+=weight[:,kw,kh,kd,:].double() @ feats[source].double()
        if bias is not None:value+=bias.double()
        outputs.append(value)
    return torch.stack(outputs).to(feats.dtype)


class CPUOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert not torch.cuda.is_initialized()
        cls.stack=ExitStack();cls.previous_threads=torch.get_num_threads();torch.set_num_threads(1)
        def forbidden(*args,**kwargs):raise AssertionError('CPU proof forbids GPU, model loading and compilation')
        for owner,name in ((torch.cuda,'_lazy_init'),(torch,'load'),(torch,'compile')):
            cls.stack.enter_context(patch.object(owner,name,forbidden))
        original_import=builtins.__import__
        def cpu_import(name,*args,**kwargs):
            if name.split('.')[0].lower() in {'flex_gemm','cumesh','nvdiffrast','trellis2','triton','flash_attn'}:
                raise AssertionError('CPU proof forbids native/model package import: '+name)
            return original_import(name,*args,**kwargs)
        cls.stack.enter_context(patch.object(builtins,'__import__',cpu_import))

    @classmethod
    def tearDownClass(cls):
        assert not torch.cuda.is_initialized()
        cls.stack.close();torch.set_num_threads(cls.previous_threads)

    def tearDown(self):self.assertFalse(torch.cuda.is_initialized())


class ArithmeticTests(CPUOnly):
    def test_coordinate_reference_matches_pinned_torch_neighbor_order(self):
        coords=coordinates();neighbor=independent_neighbors(coords)
        genuine=pinned_functions()['_compute_neighbor_cache_torch'](coords,(1,128,5,5,5),(3,3,3),(1,1,1))['neighbor_map']
        self.assertTrue(torch.equal(neighbor,genuine))
        self.assertTrue(any(index>=2 and index!=SENTINEL for index in neighbor[0].tolist()))
        self.assertEqual(sum(index==SENTINEL for index in neighbor[-1].tolist()),26)

    def test_f32_128_to128_and512_match_untiled_and_coordinate_math(self):
        tiler=importlib.import_module('chunked_final').row_chunked_explicit
        coords=coordinates();neighbor=independent_neighbors(coords);n=len(coords)
        generator=torch.Generator().manual_seed(20261015)
        feats=torch.randn(n,128,generator=generator)*.125
        pinned=pinned_functions()['_sparse_submanifold_conv_forward']
        for co in (128,512):
            weight=torch.randn(co,3,3,3,128,generator=generator)*.03125
            for bias in (None,torch.linspace(-.125,.125,co)):
                with self.subTest(co=co,bias=bias is not None):
                    expected=untiled_math(feats,neighbor,weight,bias)
                    direct=coordinate_math(feats,coords,weight,bias)
                    torch.testing.assert_close(expected,direct,rtol=1e-5,atol=1e-5)
                    torch.testing.assert_close(pinned(feats,{'neighbor_map':neighbor},weight,bias),expected,rtol=1e-5,atol=1e-5)
                    for tile in (1,2,4,7):
                        events=[]
                        actual=tiler(torch,feats,neighbor,weight,bias,tile,lambda *args:events.append(args))
                        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
                        torch.testing.assert_close(actual,direct,rtol=1e-5,atol=1e-5)
                        self.assertEqual(events,[(phase,start,min(n,start+tile)) for start in range(0,n,tile) for phase in ('entry','return')])

    def test_bf16_exact_representable_128_to128_and512_with_bias_and_sentinels(self):
        tiler=importlib.import_module('chunked_final').row_chunked_explicit
        coords=coordinates();neighbor=independent_neighbors(coords);n=len(coords)
        feats=((torch.arange(n*128).reshape(n,128)%8)-4).to(torch.bfloat16)
        for co in (128,512):
            weight=torch.zeros(co,3,3,3,128,dtype=torch.bfloat16)
            for output in range(co):
                for kernel in (0,4,13,22):
                    kw,kh,kd=tuple(product(range(3),repeat=3))[kernel]
                    weight[output,kw,kh,kd,(output+kernel)%128]=.25
            for bias in (None,(torch.arange(co)%4).to(torch.bfloat16)*.25):
                expected=coordinate_math(feats,coords,weight,bias)
                torch.testing.assert_close(untiled_math(feats,neighbor,weight,bias),expected,rtol=0,atol=0)
                for tile in (1,2,4):
                    actual=tiler(torch,feats,neighbor,weight,bias,tile,lambda *args:None)
                    # Small integer inputs, four quarter-valued products and
                    # quarter-valued bias make every result BF16-representable.
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_tiler_rejects_wrong_dimension_dtype_tile_and_neighbor_contract(self):
        module=importlib.import_module('chunked_final');admission=importlib.import_module('admission')
        feats=torch.ones(7,128);neighbor=independent_neighbors(coordinates());weight=torch.ones(128,3,3,3,128)
        for tile in (0,True,32769):
            with self.subTest(tile=tile),self.assertRaises(admission.AdmissionStop):
                module.row_chunked_explicit(torch,feats,neighbor,weight,None,tile,lambda *args:None)
        for f,m,w,b in ((feats.double(),neighbor,weight,None),(feats,neighbor.long(),weight,None),
                         (feats,neighbor[:,:26],weight,None),(feats,neighbor,weight[:,:,:,:,:127],None),
                         (feats,neighbor,weight,torch.zeros(127))):
            with self.assertRaises(admission.AdmissionStop):module.row_chunked_explicit(torch,f,m,w,b,2,lambda *args:None)


class PlannerTests(CPUOnly):
    def planner(self):
        module=importlib.import_module('admission')
        self.assertTrue(hasattr(module,'large_chunked_conv_plan'),'bounded large-support planner is missing')
        return module,module.large_chunked_conv_plan

    def test_measured438684_plan_preserves_global_map_and_full_output(self):
        admission,planner=self.planner();n=438684;tile=32768;ci=128;volume=11280384
        for co in (128,512):
            extra=n*(16*ci+4*co)
            plan=planner(n,ci,co,tile,volume,cached=False,block_extra=extra)
            expected=dict(im2col_bytes=tile*27*ci*2,gathered_bytes=tile*27*ci*2,
                          neighbor_map_bytes=n*27*4,hash_bytes=2*n*8,
                          long_index_bytes=2*tile*27*8,mask_bytes=tile*27,
                          output_bytes=n*co*2,tile_output_bytes=tile*co*2,block_extra_bytes=extra)
            for key,value in expected.items():self.assertEqual(plan[key],value,key)
            self.assertEqual(plan['visible_bytes'],sum(expected.values()))
            self.assertEqual((plan['rows'],plan['tile_rows'],plan['chunks'],plan['last_chunk_rows']),(n,tile,14,12700))
            cached=planner(n,ci,co,tile,volume,cached=True,block_extra=extra)
            self.assertEqual(cached['neighbor_map_bytes'],0);self.assertEqual(cached['hash_bytes'],0)
            self.assertEqual(cached['visible_bytes'],plan['visible_bytes']-expected['neighbor_map_bytes']-expected['hash_bytes'])
        self.assertEqual((admission.MAX_ROWS,admission.FINAL_ROWS,admission.TILE_ROWS),(1000000,1842968,32768))
        self.assertEqual(tuple(admission.LARGE_TILED_SITES),SITES)

    def test_planner_rejects_expanded_rows_dimensions_tiles_volume_and_nonintegers(self):
        admission,planner=self.planner();valid=(438684,128,128,32768,11280384)
        for index,values in ((0,(0,True,1000001)),(1,(64,127,256,True)),(2,(64,129,1024,True)),
                             (3,(0,True,32769)),(4,(0,True,512**3+1))):
            for value in values:
                args=list(valid);args[index]=value
                with self.subTest(index=index,value=value),self.assertRaises(admission.AdmissionStop):planner(*args)
        for extra in (-1,True,1.0):
            with self.assertRaises(admission.AdmissionStop):planner(*valid,block_extra=extra)
        with self.assertRaises(admission.AdmissionStop):planner(*valid,cached=1)

    def test_live_admission_preserves_padding_allocator_reserves_and_refuses_insufficient(self):
        admission,planner=self.planner();n=438684
        plan=planner(n,128,128,32768,11280384,block_extra=n*(16*128+4*128))
        padded=(plan['visible_bytes']*5+3)//4
        memory=dict(allocated_bytes=admission.GIB,reserved_bytes=admission.GIB,
                    free_bytes=13*admission.GIB,total_bytes=16*admission.GIB)
        events=[];hooks=admission.AdmissionHooks(None,torch,emit=events.append,memory=lambda:memory)
        hooks._admit('large-positive',plan)
        self.assertEqual(events[-1]['padded_bytes'],padded);self.assertTrue(events[-1]['admitted'])
        for values in (dict(memory,free_bytes=padded+admission.FREE_RESERVE-1),
                       dict(memory,reserved_bytes=admission.ALLOCATOR_LIMIT-padded-admission.OPAQUE_HEADROOM+1)):
            hooks.memory=lambda:values
            with self.assertRaises(admission.AdmissionStop):hooks._admit('large-negative',plan)
        self.assertEqual((admission.ALLOCATOR_LIMIT,admission.INITIAL_FREE_REQUIRED,admission.FREE_RESERVE,admission.OPAQUE_HEADROOM),
                         (12*1024**3,13*1024**3,2*1024**3,1024**3))


class TorchCPUFacade:
    """Only the device identity guard is mapped to CPU for lifecycle fixtures."""
    def __getattr__(self,name):return getattr(torch,name)
    def device(self,*args,**kwargs):
        if args in (('cuda',0),('cuda:0',)):return torch.device('cpu')
        return torch.device(*args,**kwargs)


def lifecycle_fixture():
    admission=importlib.import_module('admission')
    functions=pinned_functions();owner=type('PinnedOwner',(),{
        '_sparse_submanifold_conv_forward':staticmethod(functions['_sparse_submanifold_conv_forward'])})
    memory=lambda:dict(allocated_bytes=admission.GIB,reserved_bytes=admission.GIB,
                       free_bytes=13*admission.GIB,total_bytes=16*admission.GIB)
    hooks=admission.AdmissionHooks(None,TorchCPUFacade(),memory=memory,tile_large_convolutions=True)
    coords=coordinates();neighbor=independent_neighbors(coords);cache={'neighbor_map':neighbor}
    hooks.large_neighbor_map=neighbor;hooks.large_neighbor_coords=coords
    class Conv(torch.nn.Module):
        def __init__(self,co):
            super().__init__();self.weight=torch.zeros(co,3,3,3,128,dtype=torch.bfloat16)
            self.weight[:,1,1,1,0]=.25;self.bias=torch.zeros(co,dtype=torch.bfloat16)
        def forward(self,x):return owner._sparse_submanifold_conv_forward(x.feats,x.cache,self.weight,self.bias)
    modules={name:Conv(512 if index==5 else 128) for index,name in enumerate(SITES)}
    handles=[]
    def original_admission_post(module,args,output):
        hooks.convolution_completed+=1;hooks.active_conv=None
    for module in modules.values():handles.append(module.register_forward_hook(original_admission_post))
    x=SimpleNamespace(feats=torch.ones(len(coords),128,dtype=torch.bfloat16),coords=coords,spatial_shape=(5,5,5),cache=cache)
    events=[]
    return SimpleNamespace(admission=admission,owner=owner,hooks=hooks,modules=modules,x=x,cache=cache,
                           facade=TorchCPUFacade(),events=events,handles=handles,
                           emit=lambda event,**values:events.append(dict(event=event,**values)))


class LifecycleTests(CPUOnly):
    def fixture(self):
        self.assertTrue((HERE/'chunked_large.py').exists(),'scoped large-support patch is missing')
        fixture=lifecycle_fixture();self.addCleanup(lambda:[handle.remove() for handle in fixture.handles])
        fixture.module=importlib.import_module('chunked_large')
        return fixture

    def start(self,fixture,*,name=SITES[0],index=0):
        hooks=fixture.hooks;conv=fixture.modules[name];n=len(fixture.x.coords)
        hooks.convolution_completed=33+index;hooks.active_conv=(name,conv,fixture.x)
        hooks.large_conv_plan=fixture.admission.large_chunked_conv_plan(n,128,512 if index==5 else 128,
            32768,125,cached=index>0,block_extra=0)
        return conv

    def test_only_six_sites_are_scoped_reuse_one_global_map_and_restore_source(self):
        fixture=self.fixture();owner=fixture.owner;name='_sparse_submanifold_conv_forward'
        descriptor=owner.__dict__[name]
        large=fixture.module.LargeSupportConvolutionPatch(owner,fixture.modules,fixture.hooks,fixture.facade,fixture.emit).install()
        self.addCleanup(large.remove)
        self.assertIs(owner.__dict__[name],descriptor)
        with self.assertRaises(fixture.admission.AdmissionStop):
            large.forward(fixture.x.feats,fixture.cache,fixture.modules[SITES[0]].weight,fixture.modules[SITES[0]].bias)
        for index,site in enumerate(SITES):
            conv=self.start(fixture,name=site,index=index)
            output=conv(fixture.x)
            torch.testing.assert_close(output,torch.full((7,512 if index==5 else 128),.25,dtype=torch.bfloat16),rtol=0,atol=0)
            self.assertIs(owner.__dict__[name],descriptor)
            self.assertIsNone(large.active)
            self.assertEqual(fixture.hooks.convolution_completed,34+index)
        fixture.module.validate_records(large.records)
        self.assertEqual([row['global_map_reused'] for row in large.records],[False,True,True,True,True,True])
        self.assertTrue(all(row['rows_completed']==7 and row['chunks']==1 for row in large.records))
        self.assertIs(large.neighbor_cache,fixture.cache)
        large.remove();self.assertEqual(large.handles,[]);self.assertIsNone(large.neighbor_cache)
        self.assertIs(owner.__dict__[name],descriptor)

    def test_constructor_rejects_foreign_source_nonstatic_site_set_and_tile(self):
        fixture=self.fixture();cls=fixture.module.LargeSupportConvolutionPatch
        def construct(owner=fixture.owner,modules=fixture.modules,tile=32768):
            return cls(owner,modules,fixture.hooks,fixture.facade,fixture.emit,tile_rows=tile)
        for tile in (0,True,32767,32769):
            with self.assertRaises(fixture.admission.AdmissionStop):construct(tile=tile)
        for modules in (dict(list(fixture.modules.items())[:-1]),
                        {**fixture.modules,'foreign':fixture.modules[SITES[0]]},
                        {key:fixture.modules[SITES[0]] for key in SITES}):
            with self.assertRaises(fixture.admission.AdmissionStop):construct(modules=modules)
        foreign=type('Foreign',(),{'_sparse_submanifold_conv_forward':staticmethod(lambda *args:None)})
        with self.assertRaises(fixture.admission.AdmissionStop):construct(owner=foreign)
        nonstatic=type('NonStatic',(),{'_sparse_submanifold_conv_forward':pinned_functions()['_sparse_submanifold_conv_forward']})
        with self.assertRaises(fixture.admission.AdmissionStop):construct(owner=nonstatic)

    def test_rejects_out_of_order_disabled_and_foreign_replacement_before_call(self):
        fixture=self.fixture();large=fixture.module.LargeSupportConvolutionPatch(
            fixture.owner,fixture.modules,fixture.hooks,fixture.facade,fixture.emit).install()
        self.addCleanup(large.remove)
        conv=self.start(fixture,name=SITES[1],index=1)
        with self.assertRaises(fixture.admission.AdmissionStop):conv(fixture.x)
        conv=self.start(fixture);fixture.hooks.tile_large_convolutions=False
        with self.assertRaises(fixture.admission.AdmissionStop):conv(fixture.x)
        fixture.hooks.tile_large_convolutions=True
        descriptor=fixture.owner.__dict__[large.name]
        setattr(fixture.owner,large.name,staticmethod(lambda *args:None))
        with self.assertRaises(fixture.admission.AdmissionStop):conv(fixture.x)
        setattr(fixture.owner,large.name,descriptor)
        self.assertEqual(large.records,[])

    def test_foreign_neighbor_coords_features_weight_bias_and_plan_fail_closed(self):
        for changed in ('map','coords','features','weight','bias','plan'):
            with self.subTest(changed=changed):
                fixture=self.fixture();descriptor=fixture.owner.__dict__['_sparse_submanifold_conv_forward']
                large=fixture.module.LargeSupportConvolutionPatch(fixture.owner,fixture.modules,fixture.hooks,fixture.facade,fixture.emit).install()
                conv=self.start(fixture);large._enter(SITES[0],conv,(fixture.x,))
                feats,cache,weight,bias=fixture.x.feats,fixture.cache,conv.weight,conv.bias
                if changed=='map':cache={'neighbor_map':fixture.cache['neighbor_map'].clone()}
                elif changed=='coords':fixture.x.coords=fixture.x.coords.clone()
                elif changed=='features':feats=feats.clone()
                elif changed=='weight':weight=weight.clone()
                elif changed=='bias':bias=bias.clone()
                else:fixture.hooks.large_conv_plan={**fixture.hooks.large_conv_plan,'visible_bytes':1}
                try:
                    with self.assertRaises(fixture.admission.AdmissionStop):large.forward(feats,cache,weight,bias)
                finally:large.remove()
                self.assertIs(fixture.owner.__dict__[large.name],descriptor)
                self.assertEqual(large.handles,[])

    def test_repeated_call_or_foreign_reused_cache_is_rejected_and_cleanup_restores(self):
        for foreign_cache in (False,True):
            with self.subTest(foreign_cache=foreign_cache):
                fixture=self.fixture();large=fixture.module.LargeSupportConvolutionPatch(
                    fixture.owner,fixture.modules,fixture.hooks,fixture.facade,fixture.emit).install()
                self.addCleanup(large.remove)
                self.start(fixture)(fixture.x)
                if foreign_cache:
                    self.start(fixture,name=SITES[1],index=1)
                    fixture.x.cache={'neighbor_map':fixture.hooks.large_neighbor_map}
                else:self.start(fixture)
                with self.assertRaises(fixture.admission.AdmissionStop):fixture.modules[SITES[1] if foreign_cache else SITES[0]](fixture.x)
                large.remove();self.assertIs(fixture.owner.__dict__[large.name],large.original_descriptor)

    def test_partial_coverage_or_mutated_completed_receipt_is_rejected(self):
        fixture=self.fixture();validate=fixture.module.validate_records
        with self.assertRaises(fixture.admission.AdmissionStop):validate([])
        records=[dict(site=name,rows=438684,in_channels=128,out_channels=512 if i==5 else 128,
                      tile_rows=32768,chunks=14,rows_completed=438684,global_map_reused=i>0,completed=True)
                 for i,name in enumerate(SITES)]
        validate(records)
        for key,value in (('chunks',13),('rows_completed',438683),('completed',False),('global_map_reused',False),('tile_rows',32769)):
            altered=copy.deepcopy(records);altered[1][key]=value
            with self.subTest(key=key),self.assertRaises(fixture.admission.AdmissionStop):validate(altered)


if __name__=='__main__':unittest.main(verbosity=2)
