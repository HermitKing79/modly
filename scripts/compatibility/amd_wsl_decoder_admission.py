"""Visible Linux EXPLICIT_GEMM arithmetic; native allocation entry stays NO-GO.

Pure CPU planners for the pinned decoder. They are not installed hooks, do not
select an algorithm, and never authorize an operation. Library indexing/BLAS
workspace is unbounded by the inspected sources; extra headroom cannot prove it.
"""
from amd_shape_decoder_admission import Memory, GIB, admit, checked_product, token_count

CHANNELS = (1024, 512, 256, 128, 64)
BLOCKS = (4, 16, 8, 4)
UNKNOWN_EXPLICIT = ('ATen advanced-indexing workspace', 'ATen addmm/mm and HIP BLAS workspace',
                    'allocator rounding and surrounding neural lifetimes')
UNKNOWN_SUBDIVISION = ('ATen sum/nonzero/repeat_interleave/index workspace',
                       'learned support count and geometry cache provenance')


class UnknownWorkspace(RuntimeError):
    """A visible allocation estimate cannot authorize native execution."""


def convolution_sites():
    sites = []
    for stage, count in enumerate(BLOCKS):
        c, next_c = CHANNELS[stage:stage+2]
        for block in range(count):
            sites.append(dict(name=f'blocks.{stage}.{block}.conv', weight=f'blocks.{stage}.{block}.conv.weight',
                              stage=stage, support_stage=stage, in_channels=c, out_channels=c))
        for suffix, ci, co, support in (('conv1', c, 8*next_c, stage), ('conv2', next_c, next_c, stage+1)):
            name = f'blocks.{stage}.{count}.{suffix}'
            sites.append(dict(name=name, weight=name+'.weight', stage=stage,
                              support_stage=support, in_channels=ci, out_channels=co))
    return sites


def subdivision_sites():
    return [dict(name=f'blocks.{stage}.{count}.updown', stage=stage, input_support_stage=stage,
                 output_support_stage=stage+1, maximum_children_per_parent=8,
                 source_calls=('h = self.updown(h, subdiv_binarized)', 'x = self.updown(x, subdiv_binarized)'))
            for stage, count in enumerate(BLOCKS)]


def hash_layout(tokens, spatial_volume):
    token_count(tokens)
    if type(spatial_volume) is not int or not 0 < spatial_volume < 2**64:
        raise ValueError('Spatial volume must leave the all-ones hash-key sentinel unused')
    key_size = 4 if spatial_volume < 2**32 else 8
    capacity = checked_product(tokens, 2)
    # The pinned init_hashmap default returns keys plus uint32 index values.
    return dict(capacity=capacity, key_itemsize=key_size, value_itemsize=4,
                key_bytes=checked_product(capacity, key_size), value_bytes=checked_product(capacity, 4),
                greatest_domain_key=spatial_volume-1, greatest_source_index=tokens-1,
                key_sentinel=2**(8*key_size)-1, value_sentinel=2**32-1,
                probe_termination_attested=False)


def explicit_visible(*, site, tokens, spatial_volume):
    candidates = [row for row in convolution_sites() if row['name'] == site]
    if len(candidates) != 1: raise ValueError('Foreign fixed-decoder convolution site')
    source = candidates[0]
    layout = hash_layout(tokens, spatial_volume)
    v, ci, co = 27, source['in_channels'], source['out_channels']
    # Original Linux Python branch: zeros im2col and advanced-index gather,
    # then F16 addmm/mm. Count upper bounds for simultaneously retained tensors.
    buffers = dict(hash_keys=layout['key_bytes'], hash_values=layout['value_bytes'],
                   neighbor_map=checked_product(tokens,v,4), mask=checked_product(tokens,v),
                   neighbor_long=checked_product(tokens,v,8), masked_indices=checked_product(tokens,v,8),
                   im2col=checked_product(tokens,v,ci,2), gathered_features=checked_product(tokens,v,ci,2),
                   output=checked_product(tokens,co,2))
    return dict(operation=site, branch='EXPLICIT_GEMM', source=source, tokens=tokens,
                hash_key_itemsize=layout['key_itemsize'], hash_layout=layout, buffers=buffers,
                visible_bytes=sum(buffers.values()), unknown_workspace=list(UNKNOWN_EXPLICIT),
                whole_call_bound=False, executable_admission=False)


def subdivision_visible(*, stage, parent_tokens, child_tokens, input_channels, cached=False):
    if type(stage) is not int or not 0 <= stage < 4: raise ValueError('Foreign subdivision stage')
    token_count(parent_tokens); token_count(child_tokens)
    if child_tokens > checked_product(parent_tokens,8): raise ValueError('Subdivision exceeds eight children per parent')
    if type(input_channels) is not int or input_channels not in (CHANNELS[stage],8*CHANNELS[stage+1]):
        raise ValueError('Foreign C2S input width')
    if type(cached) is not bool: raise ValueError('Explicit geometry-cache state required')
    # Per actual updown invocation, not a whole C2S block. Six arithmetic
    # intermediates across three axes are conservatively counted together.
    geometry = 0 if cached else checked_product(parent_tokens,32)+checked_product(child_tokens,88)
    gather = checked_product(child_tokens,16)
    features = checked_product(child_tokens,input_channels//8,2)
    return dict(operation=subdivision_sites()[stage]['name'], parent_tokens=parent_tokens,
                child_tokens=child_tokens, input_channels=input_channels, output_channels=input_channels//8,
                cached=cached, geometry_bytes=geometry, gather_index_bytes=gather, feature_bytes=features,
                visible_bytes=geometry+gather+features, per_source_invocation=True,
                unknown_workspace=list(UNKNOWN_SUBDIVISION), whole_call_bound=False, executable_admission=False)


def before_explicit_convolution(operation, *, memory, site, tokens, spatial_volume):
    plan = explicit_visible(site=site,tokens=tokens,spatial_volume=spatial_volume)
    admit(memory,plan['visible_bytes'],site)
    raise UnknownWorkspace('EXPLICIT_GEMM rejected before original allocation: '+', '.join(UNKNOWN_EXPLICIT))


def before_subdivision(operation, *, memory, stage, parent_tokens, child_tokens, input_channels, cached=False):
    plan = subdivision_visible(stage=stage,parent_tokens=parent_tokens,child_tokens=child_tokens,
                               input_channels=input_channels,cached=cached)
    admit(memory,plan['visible_bytes'],plan['operation'])
    raise UnknownWorkspace('C2S rejected before original allocation: '+', '.join(UNKNOWN_SUBDIVISION))
