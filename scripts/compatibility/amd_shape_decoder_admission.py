"""Pre-allocation contracts for the pinned four-stage neural shape decoder.

No GPU queries, model calls, allocation or mutation occur in these planners.
These prospective planners are NOT installed decoder hooks or GPU authorization.
A future approved worker would need measured allocator/free-memory state and
validated source-shape metadata before calling them before source operations.
Synthetic tests prove ordering only for the supplied callbacks. They do not
establish complete native workspace bounds. Source arithmetic is not modified.
"""
from dataclasses import dataclass

GIB=1024**3
ALLOCATOR_CAP=12*GIB
INITIAL_FREE_MIN=13*GIB
MAX_TOKENS=1_000_000
OUTPUT_LIMIT=256*1024**2
HEADROOM=GIB
INT32_MAX=2**31-1
CHANNELS=(1024,512,256,128,64)


def integer(value,label,maximum=INT32_MAX):
    if type(value) is not int or not 0<=value<=maximum:raise ValueError('Invalid/overflow '+label)
    return value


def checked_product(*values):
    result=1
    for v in values:
        result*=integer(v,'allocation factor')
        if result>2**63-1:raise ValueError('Allocation-size overflow')
    return result


@dataclass(frozen=True)
class Memory:
    allocated:int
    reserved:int
    available:int

    def validate(self):
        for name in ('allocated','reserved','available'):integer(getattr(self,name),name,2**63-1)
        if not self.allocated<=self.reserved<=ALLOCATOR_CAP:raise ValueError('Allocator state exceeds unchanged12GiB cap')


def initial_admission(memory):
    memory.validate()
    if memory.available<INITIAL_FREE_MIN:raise ValueError('Initial available VRAM below unchanged13GiB admission')


def admit(memory,projected,label):
    memory.validate();integer(projected,'projected bytes',2**63-1)
    # Reserved includes model, inputs, live caches, prior subdivisions and
    # allocator-retained pages. Count it once; do not assume caches are free.
    need=projected+HEADROOM
    if memory.reserved+need>ALLOCATOR_CAP or need>memory.available:
        raise ValueError('Pre-allocation memory rejection: '+label)
    return dict(operation=label,projected_additional_bytes=projected,workspace_headroom_bytes=HEADROOM,
                retained_reserved_bytes=memory.reserved,ceiling_bytes=ALLOCATOR_CAP)


def token_count(n):
    integer(n,'token count',MAX_TOKENS)
    if not n:raise ValueError('Empty sparse support cannot be admitted')


def neighbor_visible_bytes(tokens):
    # Postprocess2 adds reduced code, old/new segments and <=27 valid entries
    # per block. This estimate requires a separately checked B1>=64 dispatch.
    # Sorting/cumsum internal workspace is not included or attested here.
    return checked_product(tokens,680)+112+120*((tokens+63)//64)+8


def convolution(memory,*,tokens,in_channels,out_channels,splitk,stage):
    token_count(tokens);integer(stage,'stage',4)
    for value,label in ((in_channels,'input channels'),(out_channels,'output channels'),(splitk,'splitk')):integer(value,label)
    c=CHANNELS[stage]
    # All source convs are 3x3x3 submanifold, unit stride/dilation, F16.
    # C2S conv1 occurs BEFORE subdivision and has8*next_channels outputs.
    valid={(c,c)}
    if stage<4:valid.add((c,8*CHANNELS[stage+1]))
    if (in_channels,out_channels) not in valid or splitk not in (1,2,4):
        raise ValueError('Foreign sparse-convolution contract')
    checked_product(tokens,27) # Native uint32 indexing stays bounded.
    native_neighbors=neighbor_visible_bytes(tokens)
    output=checked_product(tokens,out_channels,2)
    split_output=checked_product(tokens,out_channels,4,splitk) if splitk>1 else 0
    split_reduction=checked_product(tokens,out_channels,4) if splitk>1 else 0
    # Cache specializations attested here assume pointer_range<=INT32_MAX.
    if max(output,split_output)>INT32_MAX:raise ValueError('Unverified large-pointer specialization')
    # Proposed allowance for surrounding LayerNorm/MLP intermediates. This
    # is not a measured whole-call bound. An eventual worker must enforce
    # the unchanged allocator cap; this planner does not configure it.
    neural_intermediates=checked_product(tokens,in_channels,24)
    plan=admit(memory,native_neighbors+output+split_output+split_reduction+neural_intermediates,'sparse-convolution-stage'+str(stage))
    plan.update(tokens=tokens,native_neighbor_bytes=native_neighbors,output_bytes=output,
                split_accumulator_bytes=split_output,split_reduction_bytes=split_reduction,
                neural_intermediate_allowance_bytes=neural_intermediates,
                native_workspace_policy='Unknown internal workspace; future worker must enforce unchanged allocator ceiling. Not an execution admission proof.')
    return plan


def subdivision(memory,*,stage,parent_tokens,child_tokens,input_channels,cached=False):
    integer(stage,'subdivision stage',3);token_count(parent_tokens);token_count(child_tokens)
    integer(input_channels,'input channels')
    if type(cached) is not bool:raise ValueError('Cache-hit flag must be explicit bool')
    maximum=checked_product(parent_tokens,8)
    if child_tokens>maximum:raise ValueError('Subdivision count exceeds exact8-child contract')
    if input_channels not in (CHANNELS[stage],8*CHANNELS[stage+1]):
        raise ValueError('Foreign C2S feature width')
    output_channels=input_channels//8
    # Source sum/nonzero, cloned/expanded coordinates, indices, offset and
    # gather intermediates; cache hits still allocate output features.
    geometry=0 if cached else checked_product(parent_tokens,32)+checked_product(child_tokens,88)
    # Exact source always evaluates idx*8+subidx, even with cached geometry.
    # Conservatively count both int64 multiplication and addition results.
    gather=checked_product(child_tokens,16)
    coordinates=geometry+gather
    features=checked_product(child_tokens,output_channels,2)
    next_c=CHANNELS[stage+1]
    # Prospective splitK<=4 contract. A future native gate must reject any
    # larger specialization before output allocation; this is not installed.
    next_split=checked_product(child_tokens,next_c,4,5)
    next_convolution=neighbor_visible_bytes(child_tokens)+checked_product(child_tokens,26*next_c)+next_split
    # Admission for output support is checked before nonzero/repeat/gather,
    # including its upcoming conv2. Never clamp a learned mask to meet it.
    plan=admit(memory,coordinates+features+next_convolution,'subdivision-stage'+str(stage))
    plan.update(parent_tokens=parent_tokens,child_tokens=child_tokens,resolution=64*2**stage,
                coordinate_and_index_bytes=coordinates,geometry_generation_bytes=geometry,
                gather_index_bytes=gather,feature_bytes=features,cached=cached,
                next_splitk_upper_bound=4,next_split_accumulator_and_reduction_bytes=next_split,
                output_channels=output_channels)
    return plan


def before_operation(planner,operation,**kwargs):
    """The only ordering contract: plan/validate before original allocation."""
    plan=planner(**kwargs)
    return operation(),plan


def output_admission(*,tokens,subdivision_tokens):
    token_count(tokens)
    if len(subdivision_tokens)!=4:raise ValueError('Four source subdivision records required')
    for n in subdivision_tokens:token_count(n)
    # F32[N,7]+I32[N,4], four F16[N_stage,8]+I32[N_stage,4], bounded metadata.
    bytes_=checked_product(tokens,44)+sum(checked_product(n,32) for n in subdivision_tokens)+1024**2
    if bytes_>OUTPUT_LIMIT:raise ValueError('Bounded artifact size exceeded')
    return bytes_
