"""Focused CPU tests for preserved texture inputs and positional C2S guides."""
from pathlib import Path
import ast
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import amd_texture_decoder_inputs as inputs


class SourceTensor(np.ndarray):
    """Tiny NumPy equivalents for the inspected source's tensor operations."""
    def clone(self): return self.copy()
    def detach(self): return self
    def sum(self, dim=-1): return np.asarray(self).sum(axis=dim).view(SourceTensor)
    def nonzero(self): return np.argwhere(np.asarray(self)).view(SourceTensor)


class SourceSparse:
    def __init__(self, feats, coords, shape=None):
        self.feats=np.asarray(feats).view(SourceTensor); self.coords=np.asarray(coords).view(SourceTensor)
        self._shape=shape; self._scale=(1,1,1); self.device='cpu'
    def get_spatial_cache(self, _): return None


class SourceTorch:
    Size=tuple
    @staticmethod
    def repeat_interleave(value, repeats, dim=0, output_size=None):
        result=np.repeat(np.asarray(value),np.asarray(repeats),axis=dim).view(SourceTensor)
        if output_size is not None: assert len(result)==output_size
        return result
    @staticmethod
    def arange(count, device=None): return np.arange(count).view(SourceTensor)


class GuideTests(unittest.TestCase):
    def test_bfloat16_conversion_is_exact_and_preserves_original_bytes(self):
        bits = np.array([[0x0000, 0x8000, 0x3f80, 0xbf80, 0x0001, 0x8001, 0x4000, 0xc000]], dtype='<u2')
        before = bits.tobytes()
        values = inputs.bf16_to_f32(bits)
        np.testing.assert_array_equal(values.view('<u4'), bits.astype('<u4') << 16)
        self.assertEqual(bits.tobytes(), before)
        self.assertEqual(values.dtype, np.dtype('<f4'))
        for bad in (np.array([[0x7f80]], dtype='<u2'), np.array([[0x7fc1]], dtype='<u2')):
            with self.assertRaises(ValueError): inputs.bf16_to_f32(bad)

    def test_vectorized_children_match_independent_scalar_x_fastest_order(self):
        parents = np.array([[0, 3, 1, 4], [0, 0, 2, 1], [0, 2, 0, 0]], dtype='<i4')
        values = np.array([[0, -0., 1, -1, 2, 3, -3, 4], [1, 0, 0, 1, 0, 0, 1, 0], [-1]*8], dtype='<f4')
        expected = []
        offsets = [(0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)]
        for parent, row in zip(parents, values):
            for value, offset in zip(row, offsets):
                if value > 0:
                    expected.append([int(parent[0]), *[2*int(c)+d for c,d in zip(parent[1:], offset)]])
        actual = inputs.reconstruct_children(parents, values)
        np.testing.assert_array_equal(actual, np.array(expected, dtype='<i4'))
        self.assertFalse(actual.flags.writeable)
        for wrong in (parents.astype('<i8'), parents[:, ::-1], parents.copy()):
            if wrong.dtype == np.dtype('<i4') and wrong.flags.c_contiguous: wrong[0,0] = 1
            with self.assertRaises(ValueError): inputs.reconstruct_children(wrong, values)

    def test_paths_map_windows_drives_without_starting_wsl(self):
        self.assertEqual(str(inputs.platform_path('C:/Users/a/file', platform='linux')), '/mnt/c/Users/a/file')
        self.assertEqual(str(inputs.platform_path('/mnt/c/Users/a/file', platform='win32')).replace('\\','/'), 'C:/Users/a/file')
        with self.assertRaises(ValueError): inputs.platform_path('relative/file', platform='linux')

    def test_genuine_four_guides_match_every_saved_coordinate_in_exact_order(self):
        result = inputs.read_inputs()
        self.assertEqual(result['denormalized'].shape, (3743,32))
        self.assertEqual(result['denormalized'].dtype, np.dtype('<f4'))
        self.assertEqual(result['coords'].dtype, np.dtype('<i4'))
        self.assertEqual(result['geometry_coords'].shape, (1842968,4))
        self.assertTrue(result['metadata']['verified'])
        self.assertEqual(result['metadata']['parent_rows'], [3743,17015,79169,378891])
        self.assertEqual(result['metadata']['child_rows'], [17015,79169,378891,1842968])
        self.assertTrue(all(row['child_order_exact'] for row in result['metadata']['guides']))
        # Execute the exact, SHA-pinned upstream forward body with only its CPU
        # topology operations represented by NumPy; no model/native import.
        source=Path(inputs.platform_path(inputs.SOURCE_ROOT))/'trellis2/modules/sparse/spatial/spatial2channel.py'
        tree=ast.parse(source.read_bytes())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SparseChannel2Spatial')
        forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        forward.returns=None
        for arg in forward.args.args: arg.annotation=None
        namespace={'torch':SourceTorch,'SparseTensor':SourceSparse,'ValueError':ValueError,'range':range,'tuple':tuple}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[forward],type_ignores=[])),str(source),'exec'),namespace)
        c2s=type('OriginalC2S',(),{'factor':2,'forward':namespace['forward']})()
        validator = inputs.GuideOrderValidator(result)
        for stage, guide in enumerate(result['guides']):
            self.assertEqual(guide['logits_bf16'].dtype, np.dtype('<u2'))
            self.assertFalse(guide['logits_bf16'].flags.writeable)
            children = result['guides'][stage+1]['coords'] if stage < 3 else result['geometry_coords']
            original=c2s.forward(SourceSparse(np.zeros((len(guide['coords']),8),dtype='<f4'),guide['coords']),
                                 SourceSparse(guide['logits_f32']>0,guide['coords']))
            np.testing.assert_array_equal(original.coords,children)
            for _ in range(2):
                validator.before_c2s(stage, guide['coords'], guide['logits_f32'] > 0, guide['coords'])
                validator.after_c2s(stage, children)
        self.assertEqual(validator.finalize(result['geometry_coords'])['calls'], [2,2,2,2])
        bad = inputs.GuideOrderValidator(result)
        guide = result['guides'][0]
        for parent, mask, coords, cached in ((guide['coords'][::-1].copy(), guide['logits_f32']>0, guide['coords'], False),
                                           (guide['coords'], ~(guide['logits_f32']>0), guide['coords'], False),
                                           (guide['coords'], guide['logits_f32']>0, guide['coords'][::-1].copy(), False),
                                           (guide['coords'], guide['logits_f32']>0, guide['coords'], True)):
            with self.assertRaises(ValueError): bad.before_c2s(0, parent, mask, coords, cached=cached)
        with self.assertRaises(ValueError): bad.finalize(result['geometry_coords'])
        with self.assertRaises(ValueError): bad.before_c2s(1,result['guides'][1]['coords'],result['guides'][1]['logits_f32']>0,result['guides'][1]['coords'])
        bad.before_c2s(0,result['guides'][0]['coords'],
                       result['guides'][0]['logits_f32']>0,result['guides'][0]['coords'])
        with self.assertRaises(ValueError): bad.after_c2s(0,result['guides'][1]['coords'][::-1].copy())
        with self.assertRaises(ValueError): bad.before_c2s(0,result['guides'][0]['coords'],result['guides'][0]['logits_f32']>0,result['guides'][0]['coords'])

    def test_source_pin_rejects_replacement(self):
        with self.assertRaises((ValueError, FileNotFoundError)): inputs.verify_source_identity(Path(__file__).parent)

    def test_strict_metadata_layout_payload_and_size_rejections(self):
        for bad in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '['*18+'0'+']'*18):
            with self.assertRaises(ValueError): inputs._decode(bad)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory).resolve()/'sample.safetensors'
            raw=np.array([[1]],dtype='<f4').tobytes()
            layout={'x':('F32',(1,1))}; hashes={'x':hashlib.sha256(raw).hexdigest()}
            header={'__metadata__':{},'x':{'dtype':'F32','shape':[1,1],'data_offsets':[0,4]}}
            def encode(row, payload=raw):
                h=json.dumps(row,separators=(',',':')).encode();h+=b' '*(-len(h)%8)
                b=struct.pack('<Q',len(h))+h+payload;path.write_bytes(b)
                return hashlib.sha256(b).hexdigest(),len(b)
            digest,size=encode(header)
            self.assertEqual(inputs._container(path,digest,size,layout,hashes)[1]['x'][0,0],1)
            with self.assertRaises(ValueError): inputs._read(path,size,'0'*64)
            with self.assertRaises(ValueError): inputs._read(path,size-1,digest)
            header['x']['data_offsets'][0]=False
            digest,size=encode(header)
            with self.assertRaises(ValueError): inputs._container(path,digest,size,layout,hashes)
            header['x']['data_offsets'][0]=0
            digest,size=encode(header,raw+b'x')
            with self.assertRaises(ValueError): inputs._container(path,digest,size,layout,hashes)
            digest,size=encode(header)
            with self.assertRaises(ValueError): inputs._container(path,digest,size,layout,{'x':'0'*64})

    def test_module_and_real_read_do_not_import_torch_or_model_native_packages(self):
        code = "import sys; import amd_texture_decoder_inputs as m; m.read_inputs(); assert not any(x.split('.')[0] in ('torch','triton','trellis2','flex_gemm','cumesh','o_voxel') for x in sys.modules)"
        process = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == '__main__': unittest.main()
