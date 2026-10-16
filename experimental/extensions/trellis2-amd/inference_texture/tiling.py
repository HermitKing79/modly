"""Seven authenticated F16 sites reuse the unchanged genuine row helper."""
from math import prod

from ._source import load_pinned
from .profile import require_authenticated
from .source_identity import qualify_function
from .admission import (historical, HISTORICAL, require, TILE_ROWS, LARGE_TILED_SITES,
                        FINAL_SITE, large_chunked_conv_plan, chunked_conv_plan, neighbor_from_cache)

HELPER_SHA256 = '707d73d7331a4896bffeb851c2aa144175841cd4752a4497760d9f6a44ea2d39'
_frozen = load_pinned('pbr_f16_row_helper', HISTORICAL + 'chunked_final.py',
                      HELPER_SHA256, imports={'admission': historical})
SOURCE_SHA256 = _frozen.SOURCE_SHA256
row_chunked_explicit = _frozen.row_chunked_explicit


def _expected_record(site, profile, *, final=False):
    p = require_authenticated(profile)
    n = p.final_rows if final else p.parent_rows[-1]
    index = LARGE_TILED_SITES.index(site) if not final else 0
    return dict(site=site, rows=n, in_channels=64 if final else 128,
                out_channels=64 if final else (512 if index == 5 else 128),
                tile_rows=TILE_ROWS, chunks=(n+TILE_ROWS-1)//TILE_ROWS,
                rows_completed=n, global_map_reused=(not final and index > 0), completed=True)


def validate_records(records, *, profile):
    require(type(records) is list and len(records) == 6, 'six large F16 records required')
    for site, record in zip(LARGE_TILED_SITES, records):
        require(record == _expected_record(site, profile), 'incomplete/altered large F16 record')
    return True


def validate_final_record(record, *, profile):
    require(record == _expected_record(FINAL_SITE, profile, final=True),
            'incomplete/altered final F16 record')
    return True


class _ConvolutionPatch:
    """The original class descriptor is replaced only during admitted sites."""
    final = False

    def __init__(self, owner, modules, hooks, torch, emit, *, profile):
        self.profile = require_authenticated(profile)
        self.sites = (FINAL_SITE,) if self.final else LARGE_TILED_SITES
        require(tuple(modules) == self.sites and len({id(m) for m in modules.values()}) == len(self.sites),
                'patch requires exact original F16 modules in original order')
        require(hooks.profile == self.profile, 'patch/admission profile differs')
        self.owner, self.modules, self.hooks, self.torch, self.emit = owner, dict(modules), hooks, torch, emit
        self.name = '_sparse_submanifold_conv_forward'
        self.original_descriptor = owner.__dict__[self.name]
        original = getattr(owner, self.name)
        require(isinstance(self.original_descriptor, staticmethod),
                'genuine pinned explicit-GEMM source identity differs')
        qualify_function(original, SOURCE_SHA256, 'SubMConv3dFunction',
                         '_sparse_submanifold_conv_forward', defaults=(None,))
        require(original.__globals__.get('torch') is torch
                and original.__globals__.get('spconv') is hooks.spconv
                and original.__globals__.get('Algorithm') is hooks.spconv.Algorithm
                and hooks.spconv.Algorithm.EXPLICIT_GEMM == 'explicit_gemm',
                'original explicit-GEMM runtime globals differ from admitted boundary')
        self.handles, self.records = [], []
        self.active, self.neighbor_cache = None, None
        self.invoked = False

    @property
    def chunks(self):
        return sum(row['chunks'] for row in self.records)

    @property
    def rows_completed(self):
        return sum(row['rows_completed'] for row in self.records)

    def install(self):
        require(not self.handles and self.active is None and not self.records, 'patch installed twice')
        try:
            for site, module in self.modules.items():
                self.handles.append(module.register_forward_pre_hook(
                    lambda m, args, s=site: self._enter(s, m, args)))
                # Torch's always_call restores the descriptor even when a forward
                # fails; outer finally remains authoritative for pre-hook failures.
                self.handles.append(module.register_forward_hook(
                    lambda m, args, output, s=site: self._exit(s, m, args, output), always_call=True))
        except BaseException:
            self.remove()
            raise
        return self

    def _enter(self, site, module, args):
        index = len(self.records)
        completed = 39 if self.final else 33+index
        require(self.active is None and index < len(self.sites) and site == self.sites[index]
                and module is self.modules[site] and len(args) == 1
                and self.hooks.active_conv == (site, module, args[0])
                and self.hooks.convolution_completed == completed,
                'F16 patch lifecycle/site/order differs')
        if self.final:
            require(self.hooks.up_states[3]['completed'] == 2, 'final C2S calls incomplete')
        require(self.owner.__dict__[self.name] is self.original_descriptor,
                'foreign explicit-GEMM replacement')
        self.active, self.invoked = (site, module, args[0]), False
        setattr(self.owner, self.name, staticmethod(self.forward))

    def forward(self, feats, neighbor_cache, weight, bias=None):
        require(self.active is not None and not self.invoked, 'foreign/repeated tiled call')
        site, module, x = self.active
        expected = _expected_record(site, self.profile, final=self.final)
        require(self.hooks.active_conv == self.active and feats is x.feats
                and weight is module.weight and bias is module.bias
                and tuple(feats.shape) == (expected['rows'], expected['in_channels'])
                and feats.dtype == self.torch.float16 and feats.device == self.torch.device('cuda', 0)
                and tuple(weight.shape) == (expected['out_channels'], 3, 3, 3, expected['in_channels']),
                'foreign F16 input/weight/bias')
        neighbor = neighbor_from_cache(neighbor_cache, self.hooks.cache_type)
        admitted_map = self.hooks.final_neighbor_map if self.final else self.hooks.large_neighbor_map
        require(neighbor is admitted_map, 'map is not the admitted genuine global native result')
        if not self.final:
            require(x.coords is self.hooks.large_neighbor_coords,
                    'large map coordinates differ from original native input')
        if self.records:
            require(neighbor_cache is self.neighbor_cache, 'foreign reused neighbor cache')
        else:
            self.neighbor_cache = neighbor_cache
        plan = self.hooks.final_conv_plan if self.final else self.hooks.large_conv_plan
        if self.final:
            expected_plan = chunked_conv_plan(expected['rows'], TILE_ROWS, prod(x.spatial_shape), profile=self.profile)
        else:
            require(plan is not None, 'missing admitted large plan')
            extra = expected['rows']*(32*128 if site.endswith('.conv')
                                     else 16*128+4*expected['out_channels'])
            expected_plan = large_chunked_conv_plan(expected['rows'], 128, expected['out_channels'],
                TILE_ROWS, prod(x.spatial_shape), bool(self.records), extra, profile=self.profile)
        require(plan == expected_plan, 'tiled admission plan differs')
        allocation = {key: value for key, value in plan.items()
                      if key.endswith('_bytes') and key != 'visible_bytes'}
        # The native map now exists and is in the observed retained baseline.
        # Do not remove the full output or downstream allowance before allocation.
        allocation.pop('neighbor_map_bytes')
        allocation.pop('hash_bytes')
        allocation['visible_bytes'] = sum(allocation.values())
        self.hooks._admit(site+'-full-output-and-first-tile', allocation)
        row = dict(expected, chunks=0, rows_completed=0, completed=False)
        self.records.append(row)
        self.invoked = True

        def observe(phase, start, end):
            if phase == 'entry':
                tile_plan = {key: value for key, value in allocation.items()
                             if key not in ('output_bytes', 'visible_bytes')}
                tile_plan['visible_bytes'] = sum(tile_plan.values())
                self.hooks._admit(site+'-tile-'+str(start), tile_plan)
            else:
                require(phase == 'return', 'unexpected helper observation')
                row['chunks'] += 1
                row['rows_completed'] += end-start
            self.emit(('final' if self.final else 'large')+'-tile-'+phase,
                      site=site, start=start, end=end, rows=end-start,
                      chunk=row['chunks']+(phase == 'entry'), memory=self.hooks.memory(),
                      host_return_is_not_sync=True)

        result = row_chunked_explicit(self.torch, feats, neighbor, weight, bias, TILE_ROWS, observe)
        require(row['chunks'] == plan['chunks'] and row['rows_completed'] == plan['rows'],
                'incomplete F16 tile coverage')
        row['completed'] = True
        return result

    def _exit(self, site, module, args, output):
        # always_call also runs when an earlier pre-hook failed before our entry.
        if self.active is None:
            return
        try:
            require(self.active == (site, module, args[0]), 'foreign F16 return')
            if output is not None:
                require(self.invoked and self.records[-1]['completed']
                        and self.hooks.convolution_completed == (40 if self.final else 33+len(self.records)),
                        'F16 return/coverage/order differs')
        finally:
            setattr(self.owner, self.name, self.original_descriptor)
            self.active = None

    def assert_complete(self):
        if self.final:
            require(len(self.records) == 1, 'one final F16 call required')
            validate_final_record(self.records[0], profile=self.profile)
        else:
            validate_records(self.records, profile=self.profile)
        require(self.active is None and self.owner.__dict__[self.name] is self.original_descriptor,
                'F16 descriptor was not restored')

    def remove(self):
        if self.active is not None:
            setattr(self.owner, self.name, self.original_descriptor)
        self.active = self.neighbor_cache = None
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.modules.clear()
        self.hooks = self.torch = self.emit = None


class LargeSupportConvolutionPatch(_ConvolutionPatch):
    """Exactly six sites, 84 F16 tiles over one full 438684-row global map."""


class FinalConvolutionPatch(_ConvolutionPatch):
    """Final64 site, 60 F16 tiles over its own full 1957891-row native map."""
    final = True

    def __init__(self, owner, module, hooks, torch, emit, *, profile):
        super().__init__(owner, {FINAL_SITE: module}, hooks, torch, emit, profile=profile)
