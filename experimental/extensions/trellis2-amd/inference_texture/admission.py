"""Authenticated F16 guided-PBR admissions; no Torch, model or native imports."""
from math import prod
import ast
from hashlib import sha256
from pathlib import Path

from ._source import load_pinned
from .profile import require_authenticated
from . import inputs

HISTORICAL = 'docs/compatibility/amd-turret-texture-decode-resume-evidence/'
HISTORICAL_SHA256 = '99c1078cbcb4b026feeec3f085976586f1118812bdd3d4b5fbe0e76c47cbbbbe'
historical = load_pinned('pbr_f16_admission', HISTORICAL + 'admission.py',
                         HISTORICAL_SHA256, imports={'amd_texture_decoder_inputs': inputs})
AdmissionStop, require = historical.AdmissionStop, historical.require
GIB, MAX_ROWS, TILE_ROWS = historical.GIB, historical.MAX_ROWS, historical.TILE_ROWS
ALLOCATOR_LIMIT = historical.ALLOCATOR_LIMIT
INITIAL_FREE_REQUIRED = historical.INITIAL_FREE_REQUIRED
FREE_RESERVE, OPAQUE_HEADROOM = historical.FREE_RESERVE, historical.OPAQUE_HEADROOM
COUNTS, WIDTHS = historical.COUNTS, historical.WIDTHS
conv_plan, c2s_plan = historical.conv_plan, historical.c2s_plan
LARGE_TILED_SITES = ('blocks.2.8.conv2', *(f'blocks.3.{i}.conv' for i in range(4)),
                     'blocks.3.4.conv1')
FINAL_SITE = 'blocks.3.4.conv2'
NATIVE_SITES = ('blocks.0.0.conv', 'blocks.0.4.conv2', 'blocks.1.16.conv2',
                LARGE_TILED_SITES[0], FINAL_SITE)
EXPLICIT_SOURCE_SHA256 = '67641d13e06ca135a4f90eb71a01f7d8d32448bb5aa7c31b858c69368811fb6b'


def qualify_cache_type(cache_type):
    """Authenticate the genuine cache class without importing any native code."""
    require(type(cache_type) is type and cache_type.__name__ == 'SubMConv3dNeighborCache'
            and cache_type.__module__ == 'flex_gemm.ops.spconv.submanifold_conv3d',
            'genuine original neighbor-cache class required')
    methods = ('__init__', '__getitem__', '__setitem__')
    source_path = None
    for name in methods:
        method = cache_type.__dict__.get(name)
        require(callable(method) and hasattr(method, '__code__'), 'original cache methods missing')
        path = Path(method.__code__.co_filename)
        require(source_path in (None, path), 'cache methods have foreign origins')
        source_path = path
    blob = source_path.read_bytes()
    require(sha256(blob).hexdigest() == EXPLICIT_SOURCE_SHA256, 'cache source identity differs')
    node = next(n for n in ast.parse(blob).body
                if isinstance(n, ast.ClassDef) and n.name == 'SubMConv3dNeighborCache')
    for name in methods:
        fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source_path), 'exec'), namespace)
        actual, expected = cache_type.__dict__[name].__code__, namespace[name].__code__
        require((actual.co_code, actual.co_consts, actual.co_names, actual.co_argcount)
                == (expected.co_code, expected.co_consts, expected.co_names, expected.co_argcount),
                'cache method implementation differs from pinned original')
    return cache_type


def neighbor_from_cache(cache, cache_type):
    require(type(cache) is cache_type and set(vars(cache)) == {'neighbor_map'},
            'original explicit-GEMM neighbor-cache instance required')
    return cache['neighbor_map']


def final_c2s_plan(n, feature_channels, skip_channels, children, out_channels, *, profile):
    p = require_authenticated(profile)
    require(all(type(v) is int for v in (n, feature_channels, skip_channels, children, out_channels))
            and (n, feature_channels, skip_channels, children, out_channels)
            == (p.parent_rows[-1], 512, 128, p.final_rows, 64),
            'fourth guided C2S differs from authenticated brick support')
    return historical._c2s_bytes(n, feature_channels, skip_channels, children, out_channels)


def chunked_conv_plan(n, tile, volume, *, profile):
    p = require_authenticated(profile)
    require(type(n) is int and n == p.final_rows and type(tile) is int and tile == TILE_ROWS
            and type(volume) is int and 0 < volume <= 512**3,
            'final F16 convolution support/tile/spatial volume differs')
    # The original two-byte final plan, including the full native map/hash,
    # full output and conservative downstream live-buffer allowance.
    plan = dict(im2col_bytes=tile*27*64*2, gathered_bytes=tile*27*64*2,
                neighbor_map_bytes=n*27*4, hash_bytes=2*n*8,
                long_index_bytes=2*tile*27*8, mask_bytes=tile*27,
                output_bytes=n*64*2, tile_output_bytes=tile*64*2,
                block_extra_bytes=n*(16*64+4*64))
    plan['visible_bytes'] = sum(plan.values())
    plan.update(rows=n, tile_rows=tile, chunks=(n+tile-1)//tile,
                last_chunk_rows=(n-1)%tile+1, in_channels=64, out_channels=64,
                spatial_volume=volume, cached=False)
    return plan


def large_chunked_conv_plan(n, ci, co, tile, volume, cached=False, block_extra=0, *, profile):
    p = require_authenticated(profile)
    require(all(type(v) is int for v in (n, ci, co, tile, volume, block_extra))
            and n == p.parent_rows[-1] and ci == 128 and co in (128, 512)
            and tile == TILE_ROWS and 0 < volume <= 256**3 and block_extra >= 0
            and type(cached) is bool, 'large F16 convolution support/widths/budget differ')
    # Global native map and all retained source feature tables stay global.
    # Only the proven helper's gather/index/im2col/GEMM temporaries are tiled.
    t = min(n, tile)
    plan = dict(im2col_bytes=t*27*ci*2, gathered_bytes=t*27*ci*2,
                neighbor_map_bytes=0 if cached else n*27*4,
                hash_bytes=0 if cached else 2*n*8,
                long_index_bytes=2*t*27*8, mask_bytes=t*27,
                output_bytes=n*co*2, tile_output_bytes=t*co*2,
                block_extra_bytes=block_extra)
    plan['visible_bytes'] = sum(plan.values())
    plan.update(rows=n, in_channels=ci, out_channels=co, tile_rows=tile,
                chunks=(n+tile-1)//tile, last_chunk_rows=(n-1)%tile+1,
                spatial_volume=volume, cached=cached)
    return plan


def memory_plans(profile):
    """CPU arithmetic only; retained live GPU allocations must still be measured."""
    p = require_authenticated(profile)
    convolutions, expansions = [], []
    for stage, count in enumerate(COUNTS[:4]):
        n, ci = p.parent_rows[stage], WIDTHS[stage]
        entries = [(f'blocks.{stage}.{j}.conv', n, ci, ci, n*32*ci,
                    not (stage == 0 and j == 0), (32*2**stage)**3)
                   for j in range(count)]
        co = WIDTHS[stage+1]*8
        entries.append((f'blocks.{stage}.{count}.conv1', n, ci, co,
                        n*(16*ci+4*co), True, (32*2**stage)**3))
        nchild, ci = p.child_rows[stage], WIDTHS[stage+1]
        entries.append((f'blocks.{stage}.{count}.conv2', nchild, ci, ci,
                        nchild*(16*ci+4*ci), False, (64*2**stage)**3))
        for site, rows, cin, cout, extra, cached, volume in entries:
            if site == FINAL_SITE:
                plan = chunked_conv_plan(rows, TILE_ROWS, volume, profile=p)
            elif site in LARGE_TILED_SITES:
                plan = large_chunked_conv_plan(rows, cin, cout, TILE_ROWS, volume,
                                              cached, extra, profile=p)
            else:
                plan = conv_plan(rows, cin, cout, volume, cached, extra)
            padded = (plan['visible_bytes']*5+3)//4
            convolutions.append(dict(site=site, **plan, padded_bytes=padded,
                maximum_retained_bytes=ALLOCATOR_LIMIT-OPAQUE_HEADROOM-padded,
                minimum_free_bytes=padded+FREE_RESERVE))
        args = (p.parent_rows[stage], ci*8, WIDTHS[stage], nchild, ci)
        plan = final_c2s_plan(*args, profile=p) if stage == 3 else c2s_plan(*args)
        expansions.extend(dict(stage=stage, call=call, **plan) for call in (1, 2))
    return dict(convolution_plans=convolutions, c2s_plans=expansions,
                expected_native_map_calls=5, large_tiles=84, final_tiles=60,
                output_payload_bytes=p.final_rows*64,
                retained_memory_is_measured_at_runtime=True,
                spatial_volumes_are_nominal_upper_bounds=True)


class AdmissionHooks(historical.AdmissionHooks):
    """Reuse source-pinned F16 architecture, completion and admission machinery."""
    def __init__(self, decoder, torch, *, profile, guidance, cache_type, emit=None, memory=None):
        self.profile = require_authenticated(profile)
        self.cache_type = qualify_cache_type(cache_type)
        super().__init__(decoder, torch, emit=emit, memory=memory, guidance=None)
        self.guidance = guidance
        self.guide_order = inputs.GuideOrderValidator(guidance, profile=self.profile)
        self.tile_large_convolutions = True
        self.large_neighbor_map = self.large_neighbor_coords = self.large_conv_plan = None
        self.final_conv_plan = None

    def _sparse(self, x, width=None, final=False):
        n, c = x.feats.shape
        require((0 < n <= MAX_ROWS or (final and n == self.profile.final_rows and c in (16, 64)))
                and (width is None or c == width), 'F16 feature rows/width mismatch')
        require(tuple(x.coords.shape) == (n, 4) and x.coords.dtype == self.torch.int32,
                'coordinate rows/shape/dtype mismatch')
        require(x.feats.dtype == self.torch.float16 and x.feats.is_contiguous()
                and x.coords.is_contiguous(), 'features/coords must be contiguous F16/int32')
        require(x.feats.device.type == 'cuda' and x.feats.device.index in (0, None)
                and x.coords.device == x.feats.device, 'features/coords must share cuda:0 device')
        require(tuple(x.shape) == (1, c) and len(x.spatial_shape) == 3
                and all(type(v) is int and v > 0 for v in x.spatial_shape),
                'unexpected sparse batch/spatial shape')
        return n, c

    def _conv_pre(self, site, module, args):
        name, _, block, kind, stage = site
        require(len(args) == 1 and self.active_conv is None and self.convolution_completed < 40
                and self.sequence[self.convolution_completed][1] is module,
                f'{name}: convolution order mismatch')
        if kind == 'conv2':
            require(self.up_states[stage]['completed'] == 2, f'{name}: both C2S calls required')
        x = args[0]
        ci = WIDTHS[stage+1] if kind == 'conv2' else WIDTHS[stage]
        co = WIDTHS[stage+1]*8 if kind == 'conv1' else ci
        final = name == FINAL_SITE
        n, _ = self._sparse(x, ci, final=final)
        expected = self.profile.child_rows[stage] if kind == 'conv2' else self.profile.parent_rows[stage]
        require(n == expected and self.rows in (None, n), f'{name}: authenticated support differs')
        require(tuple(module.weight.shape) == (co, 3, 3, 3, ci)
                and tuple(module.kernel_size) == (3, 3, 3)
                and tuple(module.dilation) == tuple(module.stride) == (1, 1, 1),
                f'{name}: weight/kernel/dilation/stride mismatch')
        require(module.weight.dtype == self.torch.float16 and module.weight.device == x.feats.device
                and (module.bias is None or (module.bias.dtype == self.torch.float16
                                             and module.bias.device == x.feats.device)),
                f'{name}: original F16 weights/bias required')
        cache = x.get_spatial_cache('SubMConv3d_neighbor_cache_3x3x3_dilation(1, 1, 1)')
        if cache is not None:
            neighbor = neighbor_from_cache(cache, self.cache_type)
            require(tuple(neighbor.shape) == (n, 27) and neighbor.dtype == self.torch.uint32
                    and neighbor.device == x.feats.device and neighbor.is_contiguous(),
                    f'{name}: invalid cached global neighbor map')
        extra = n*(16*ci+4*block.mlp[0].out_features) if kind == 'conv' else n*(16*ci+4*co)
        self.rows = n
        if final:
            require(ci == co == 64 and cache is None, 'final support/cache differs')
            plan = chunked_conv_plan(n, TILE_ROWS, prod(x.spatial_shape), profile=self.profile)
            self.final_conv_plan = plan
        elif name in LARGE_TILED_SITES:
            index = LARGE_TILED_SITES.index(name)
            require((cache is None) == (index == 0), 'large cache creation/reuse order differs')
            if index:
                require(cache['neighbor_map'] is self.large_neighbor_map
                        and x.coords is self.large_neighbor_coords,
                        'large cached map/coordinates differ from admitted native result')
            plan = large_chunked_conv_plan(n, ci, co, TILE_ROWS, prod(x.spatial_shape),
                                          cache is not None, extra, profile=self.profile)
            self.large_conv_plan = plan
        else:
            plan = conv_plan(n, ci, co, prod(x.spatial_shape), cache is not None, extra)
        self._admit(name, plan)
        self.active_conv = (name, module, x)

    def _up_pre(self, site, module, args):
        name, _, block, stage = site
        state = self.up_states[stage]
        require(len(args) == 2 and self.active_conv is None and state['completed'] < 2
                and self.convolution_completed < 40
                and self.sequence[self.convolution_completed][3:] == ('conv2', stage),
                f'{name}: C2S call order mismatch')
        x, sub = args
        call = state['completed']
        n, _ = self._sparse(x, block.out_channels*8 if call == 0 else block.channels)
        require(sub is not None and tuple(sub.feats.shape) == (n, 8)
                and sub.feats.dtype == self.torch.bool and sub.feats.device == x.feats.device
                and sub.coords.dtype == self.torch.int32 and sub.coords.device == x.coords.device
                and self.torch.equal(sub.coords, x.coords), f'{name}: original boolean guide mask required')
        cached = x.get_spatial_cache('channel2spatial_2') is not None
        self.guide_order.before_c2s(stage, x.coords.detach().cpu().numpy(),
            sub.feats.detach().cpu().numpy(), sub.coords.detach().cpu().numpy(), cached=cached)
        children = int(sub.feats.sum().item())
        require(n == self.profile.parent_rows[stage] and children == self.profile.child_rows[stage]
                and not cached, f'{name}: authenticated C2S support/cache differs')
        if call:
            require(state['mask'] is sub.feats and state['parents'] == n
                    and state['children'] == children and self.torch.equal(state['coords'], x.coords),
                    f'{name}: second C2S changed mask/parent support')
        args = (n, block.out_channels*8, block.channels, children, block.out_channels)
        plan = final_c2s_plan(*args, profile=self.profile) if stage == 3 else c2s_plan(*args)
        self.emit(dict(event='c2s-count-before-admission', site=name, call=call+1,
                       parents=n, children=children, source=historical.source_identity(module.forward),
                       mask_source=historical.source_identity(block._forward), memory=self.memory(),
                       profile=self.profile.sha256, authentic_guided_mask=True))
        self._admit(name, plan)
        state.update(mask=sub.feats, coords=sub.coords, parents=n, children=children,
                     pending_width=x.feats.shape[1]//8)

    def remove(self):
        super().remove()
        self.large_neighbor_map = self.large_neighbor_coords = self.large_conv_plan = None
        self.final_conv_plan = None
        self.decoder = None
        self.sequence.clear()
        self.up_states.clear()
