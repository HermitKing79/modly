"""Pinned decoder's explicit-GEMM/C2S admission. Imports no Torch; never runs a probe."""
from math import prod
from hashlib import sha256
from pathlib import Path

GIB = 1024**3
MAX_ROWS = 1_000_000
FINAL_ROWS = 1_842_968
TILE_ROWS = 32_768
ALLOCATOR_LIMIT = 12 * GIB
INITIAL_FREE_REQUIRED = 13 * GIB  # Root's separate admission before loading/execution.
FREE_RESERVE = 2 * GIB
OPAQUE_HEADROOM = GIB
COUNTS = (4, 16, 8, 4, 0)
WIDTHS = (1024, 512, 256, 128, 64)


class AdmissionStop(RuntimeError):
    """The genuine decode stopped; its evidence must be reported as PARTIAL."""


def require(ok, reason):
    if not ok:
        raise AdmissionStop('PARTIAL: ' + reason)


def source_identity(forward):
    path=Path(forward.__code__.co_filename)
    return {'module':forward.__module__,'qualname':forward.__qualname__,
            'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest()}


def conv_plan(n, ci, co, volume, cached=False, block_extra=0):
    require(0 < n <= MAX_ROWS and ci > 0 and co > 0 and 0 < volume < 2**64 and block_extra >= 0,
            'convolution rows/channels/spatial volume outside admitted bounds')
    # Wheel source: im2col[mask] = feats[neighbor_map.long()[mask]]. Both features
    # coexist; full and masked long indices may coexist too. Hash ratio is 2.
    p = dict(im2col_bytes=n*27*ci*2, gathered_bytes=n*27*ci*2,
             neighbor_map_bytes=0 if cached else n*27*4,
             long_index_bytes=2*n*27*8, mask_bytes=n*27,
             hash_bytes=0 if cached else 2*n*((4 if volume < 2**32 else 8)+4),
             output_bytes=n*co*2, block_extra_bytes=block_extra)
    p['visible_bytes'] = sum(p.values())
    p.update(rows=n, in_channels=ci, out_channels=co, spatial_volume=volume, cached=cached)
    return p


def c2s_plan(n, feature_channels, skip_channels, children, out_channels):
    require(0 < n <= MAX_ROWS and 0 < children <= min(8*n, MAX_ROWS)
            and feature_channels > 0 and skip_channels > 0 and out_channels > 0
            and feature_channels % 8 == skip_channels % 8 == 0,
            'C2S parent/child rows or channel widths outside admitted bounds')
    return _c2s_bytes(n, feature_channels, skip_channels, children, out_channels)


def final_c2s_plan(n, feature_channels, skip_channels, children, out_channels):
    require(all(type(v) is int for v in (n,feature_channels,skip_channels,children,out_channels))
            and 0 < n <= MAX_ROWS and 0 < children <= min(8*n, FINAL_ROWS)
            and (feature_channels,skip_channels,out_channels)==(512,128,64),
            'fourth C2S differs from the proven full support/widths')
    return _c2s_bytes(n, feature_channels, skip_channels, children, out_channels)


def _c2s_bytes(n, feature_channels, skip_channels, children, out_channels):
    # Source has two expansions. Per expansion: parent clone/count/reduction,
    # nonzero indices, new coords, parent/sub indices and arithmetic temporaries.
    # Include both gathered features, downstream skip repeat, norm32/SiLU,
    # conv2 output and residual addition even when chunking reduces their peaks.
    p = dict(geometry_bytes=n*64+children*176,
             gathered_bytes=children*(feature_channels//8+skip_channels//8)*2,
             skip_repeat_bytes=children*out_channels*2,
             block_extra_bytes=children*out_channels*16)
    p['visible_bytes'] = sum(p.values())
    p.update(parents=n, children=children, feature_channels=feature_channels,
             skip_channels=skip_channels, out_channels=out_channels)
    return p


def chunked_conv_plan(n, tile, volume):
    require(type(n) is int and 0<n<=FINAL_ROWS and type(tile) is int and 0<tile<=TILE_ROWS
            and type(volume) is int and 0<volume<=512**3,
            'final chunked convolution support/tile/spatial volume differs')
    # Global map and original ratio-2 hash; only im2col/gather/index/mm are tiled.
    p=dict(im2col_bytes=tile*27*64*2,gathered_bytes=tile*27*64*2,
           neighbor_map_bytes=n*27*4,hash_bytes=2*n*8,
           long_index_bytes=2*tile*27*8,mask_bytes=tile*27,
           output_bytes=n*64*2,tile_output_bytes=tile*64*2,
           block_extra_bytes=n*(16*64+4*64))
    p['visible_bytes']=sum(p.values())
    p.update(rows=n,tile_rows=tile,chunks=(n+tile-1)//tile,last_chunk_rows=(n-1)%tile+1,
             in_channels=64,out_channels=64,spatial_volume=volume,cached=False)
    return p


def output_payload_bytes(final_rows, parent_rows):
    require(len(parent_rows)==4 and all(type(n) is int and 0<n<=MAX_ROWS for n in parent_rows), "invalid output parent rows")
    require(type(final_rows) is int and 0<final_rows<=FINAL_ROWS, "invalid final output rows")
    return final_rows*(7*4+4*4)+sum(parent_rows)*(8*2+4*4)


class AdmissionHooks:
    def __init__(self, decoder, torch, emit=None, memory=None):
        self.decoder, self.torch = decoder, torch
        self.emit = emit or (lambda event: None)
        self.memory = memory or self._memory
        self.handles, self.sequence, self.up_states = [], [], {}
        self.active_conv, self.rows, self.convolution_completed = None, None, 0
        self.subdivision_records = []
        self.final_neighbor_map = None

    def _memory(self):
        free, total = self.torch.cuda.mem_get_info(0)
        return dict(free_bytes=free, total_bytes=total,
                    allocated_bytes=self.torch.cuda.memory_allocated(0),
                    reserved_bytes=self.torch.cuda.memory_reserved(0))

    def _admit(self, name, plan):
        memory = self.memory()  # Retained caches/model/live buffers are included now.
        padded = (plan['visible_bytes']*5+3)//4
        allowed = (max(memory['allocated_bytes'], memory['reserved_bytes']) + padded
                   + OPAQUE_HEADROOM <= ALLOCATOR_LIMIT
                   and memory['free_bytes'] >= padded+FREE_RESERVE)
        self.emit(dict(event='admission', site=name, admitted=allowed, rows=self.rows,
                       padded_bytes=padded, plan=plan, memory=memory))
        require(allowed, f'{name}: visible allocation plus reserve/headroom exceeds memory limits')
        return memory

    def _sparse(self, x, width=None, final=False):
        n, c = x.feats.shape
        require((0 < n <= MAX_ROWS or (final and 0<n<=FINAL_ROWS and c in (16,64)))
                and (width is None or c == width), 'feature rows/width mismatch')
        require(tuple(x.coords.shape) == (n, 4) and x.coords.dtype == self.torch.int32,
                'coordinate rows/shape/dtype mismatch')
        require(x.feats.dtype == self.torch.bfloat16 and x.feats.is_contiguous()
                and x.coords.is_contiguous(), 'features/coords must be contiguous BF16/int32')
        require(x.feats.device.type == 'cuda' and x.feats.device.index in (0, None)
                and x.coords.device == x.feats.device, 'features/coords must share cuda:0 device')
        require(tuple(x.shape) == (1, c) and len(x.spatial_shape) == 3
                and all(v > 0 for v in x.spatial_shape), 'unexpected sparse batch/spatial shape')
        return n, c

    def install(self):
        require(not self.handles and not self.sequence, 'hooks may only be installed once')
        d = self.decoder
        require(tuple(d.num_blocks) == COUNTS and tuple(d.model_channels) == WIDTHS
                and len(d.blocks) == 5 and d.pred_subdiv, 'decoder architecture differs from pinned parent')
        up_modules = []
        for i, count in enumerate(COUNTS):
            require(len(d.blocks[i]) == count+(i < 4), 'decoder block count mismatch')
            for j in range(count):
                block = d.blocks[i][j]
                self.sequence.append((f'blocks.{i}.{j}.conv', block.conv, block, 'conv', i))
            if i < 4:
                block = d.blocks[i][-1]
                require(block.channels == WIDTHS[i] and block.out_channels == WIDTHS[i+1]
                        and block.pred_subdiv and block.updown.factor == 2,
                        'C2S width/mask/factor mismatch')
                for kind in ('conv1', 'conv2'):
                    self.sequence.append((f'blocks.{i}.{count}.{kind}', getattr(block, kind), block, kind, i))
                up_modules.append((f'blocks.{i}.{count}.updown', block.updown, block, i))
        found = list(d.named_modules())
        require([(name, m) for name, m in found if type(m).__name__ == 'SparseConv3d']
                == [(s[0], s[1]) for s in self.sequence], 'original 40 SparseConv3d sites mismatch')
        require([(name, m) for name, m in found if type(m).__name__ == 'SparseChannel2Spatial']
                == [(s[0], s[1]) for s in up_modules], 'original four updown modules mismatch')
        try:
            for site in self.sequence:
                self.handles.append(site[1].register_forward_pre_hook(lambda m, args, s=site: self._conv_pre(s, m, args)))
                self.handles.append(site[1].register_forward_hook(lambda m, args, out, s=site: self._conv_post(s, m, args, out)))
            for site in up_modules:
                self.up_states[site[3]] = dict(completed=0)
                self.handles.append(site[1].register_forward_pre_hook(lambda m, args, s=site: self._up_pre(s, m, args)))
                self.handles.append(site[1].register_forward_hook(lambda m, args, out, s=site: self._up_post(s, m, args, out)))
        except BaseException:
            self.remove()
            raise
        return self

    def _conv_pre(self, site, module, args):
        name, _, block, kind, stage = site
        require(self.active_conv is None and self.convolution_completed < 40
                and self.sequence[self.convolution_completed][1] is module, f'{name}: convolution order mismatch')
        if kind == 'conv2':
            require(self.up_states[stage]['completed'] == 2, f'{name}: both C2S calls must complete first')
        x = args[0]
        ci = WIDTHS[stage+1] if kind == 'conv2' else WIDTHS[stage]
        co = WIDTHS[stage+1]*8 if kind == 'conv1' else ci
        final = name=='blocks.3.4.conv2'
        n, _ = self._sparse(x, ci, final=final)
        require(self.rows in (None, n), f'{name}: unexpected support rows')
        require(tuple(module.weight.shape) == (co, 3, 3, 3, ci)
                and tuple(module.kernel_size) == (3, 3, 3) and tuple(module.dilation) == (1, 1, 1)
                and tuple(module.stride) == (1, 1, 1), f'{name}: weight/kernel/dilation/stride mismatch')
        require(module.weight.dtype == self.torch.bfloat16 and module.weight.device == x.feats.device
                and (module.bias is None or module.bias.dtype == self.torch.bfloat16), f'{name}: weights must be BF16')
        cache = x.get_spatial_cache('SubMConv3d_neighbor_cache_3x3x3_dilation(1, 1, 1)')
        if cache is not None:
            neighbor = cache['neighbor_map'] if isinstance(cache, dict) else cache.neighbor_map
            require(tuple(neighbor.shape) == (n, 27) and neighbor.dtype == self.torch.uint32
                    and neighbor.device == x.feats.device and neighbor.is_contiguous(), f'{name}: invalid cached neighbor_map')
        # ConvNeXt: full norm output, float32 norm temporaries, two hidden MLP
        # buffers, projected output and residual; conservative despite chunk65536.
        extra = n*(16*ci+4*block.mlp[0].out_features) if kind == 'conv' else n*(16*ci+4*co)
        self.rows = n
        if final:
            require(0<n<=FINAL_ROWS and ci==co==64 and cache is None, 'final full support/cache differs')
            plan=chunked_conv_plan(n,TILE_ROWS,prod(x.spatial_shape))
        else:
            plan=conv_plan(n, ci, co, prod(x.spatial_shape), cache is not None, extra)
        self._admit(name,plan)
        self.active_conv = (name, module, x)

    def _conv_post(self, site, module, args, out):
        name, _, _, kind, stage = site
        try:
            require(self.active_conv == (name, module, args[0]), f'{name}: missing admitted convolution')
            width = WIDTHS[stage+1]*8 if kind == 'conv1' else (WIDTHS[stage+1] if kind == 'conv2' else WIDTHS[stage])
            n, _ = self._sparse(out, width, final=name=='blocks.3.4.conv2')
            require(n == self.rows and self.torch.equal(out.coords, args[0].coords), f'{name}: convolution changed support')
            self.convolution_completed += 1
            self.emit(dict(event='conv_completed', site=name, rows=n, memory=self.memory()))
        finally:
            self.active_conv = None

    def _up_pre(self, site, module, args):
        name, _, block, stage = site
        state = self.up_states[stage]
        require(self.active_conv is None and state['completed'] < 2 and self.convolution_completed < 40
                and self.sequence[self.convolution_completed][3:] == ('conv2', stage), f'{name}: C2S call order mismatch')
        x, sub = args
        call = state['completed']
        n, _ = self._sparse(x, block.out_channels*8 if call == 0 else block.channels)
        require(sub is not None and tuple(sub.feats.shape) == (n, 8)
                and sub.feats.dtype == self.torch.bool and sub.feats.device == x.feats.device
                and sub.coords.dtype == self.torch.int32 and sub.coords.device == x.coords.device
                and self.torch.equal(sub.coords, x.coords), f'{name}: authentic boolean mask/parent coords mismatch')
        children = int(sub.feats.sum().item())  # Before original nonzero/repeat/coords/gather.
        self.emit(dict(event='c2s-count-before-admission',site=name,call=call+1,
            parents=n,children=children,children_per_parent_mean=children/n,
            children_per_parent_maximum=8,mask_shape=list(sub.feats.shape),mask_dtype=str(sub.feats.dtype),
            source=source_identity(module.forward),mask_source=source_identity(block._forward),
            memory=self.memory(),row_limit=MAX_ROWS,zero_children=children==0,
            over_row_limit=children>MAX_ROWS,
            row_limits_would_reject=not (0<n<=MAX_ROWS and 0<children<=min(8*n,MAX_ROWS)),
            bounded_fourth_support=(stage==3 and 0<n<=MAX_ROWS and 0<children<=min(8*n,FINAL_ROWS)
                and (block.out_channels*8,block.channels,block.out_channels)==(512,128,64)),
            feature_channels=block.out_channels*8,skip_channels=block.channels,out_channels=block.out_channels,
            parent_spatial_shape=list(x.spatial_shape)))
        require(x.get_spatial_cache('channel2spatial_2') is None, f'{name}: unexpected cache would bypass learned mask')
        if call:
            require(state['mask'] is sub.feats and state['parents'] == n and state['children'] == children
                    and self.torch.equal(state['coords'], x.coords), f'{name}: second C2S changed mask/parent support')
        planner = final_c2s_plan if stage==3 else c2s_plan
        plan = planner(n, block.out_channels*8, block.channels, children, block.out_channels)
        self._admit(name, plan)
        state.update(mask=sub.feats, coords=sub.coords, parents=n, children=children, pending_width=x.feats.shape[1]//8)

    def _up_post(self, site, module, args, out):
        name, _, _, stage = site
        state = self.up_states[stage]
        n, _ = self._sparse(out, state['pending_width'], final=stage==3)
        require(n == state['children'], f'{name}: output disagrees with genuine mask sum')
        state['completed'] += 1
        self.rows = n
        record = dict(event='c2s_completed', site=name, call=state['completed'],
                      parents=state['parents'], children=n, memory=self.memory())
        self.subdivision_records.append(record)
        self.emit(record)

    def assert_complete(self):
        require(self.convolution_completed == 40 and len(self.subdivision_records) == 8
                and all(s['completed'] == 2 for s in self.up_states.values()), 'genuine decoder call sequence incomplete')

    def remove(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.active_conv = None
        self.final_neighbor_map = None
        for state in self.up_states.values():
            state.pop('mask', None)
            state.pop('coords', None)
