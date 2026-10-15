"""Output-row tiling of the pinned explicit-GEMM path; no source-file modification."""
from hashlib import sha256
from pathlib import Path
from math import prod
from admission import require, FINAL_ROWS, TILE_ROWS, chunked_conv_plan

SOURCE_SHA256='67641d13e06ca135a4f90eb71a01f7d8d32448bb5aa7c31b858c69368811fb6b'
SITE='blocks.3.4.conv2'

def row_chunked_explicit(torch,feats,neighbor_map,weight,bias,tile_rows,observe):
    n,ci=feats.shape
    co,kw,kh,kd,wci=weight.shape
    v=kw*kh*kd
    require(type(tile_rows) is int and 0<tile_rows<=TILE_ROWS and n>0 and ci==wci
            and (kw,kh,kd)==(3,3,3),'invalid tile/features/weight contract')
    require(feats.dtype in (torch.float16,torch.float32) and weight.dtype==feats.dtype
            and weight.device==feats.device and weight.is_contiguous() and feats.is_contiguous(),
            'feature/weight layout/device/dtype differs')
    require(tuple(neighbor_map.shape)==(n,v) and neighbor_map.dtype==torch.uint32
            and neighbor_map.device==feats.device and neighbor_map.is_contiguous(), 'global neighbour map contract')
    require(bias is None or (tuple(bias.shape)==(co,) and bias.dtype==feats.dtype and bias.device==feats.device),
            'bias contract differs')
    output=torch.empty((n,co),device=feats.device,dtype=feats.dtype)
    matrix=weight.view(co,v*ci).transpose(0,1)  # Exact original layout; no new permutation.
    for start in range(0,n,tile_rows):
        end=min(n,start+tile_rows)
        observe('entry',start,end)  # Live admission before any tile allocation.
        flat=neighbor_map[start:end].view(-1)  # Original global row indices, never rebased.
        im2col=torch.zeros(((end-start)*v,ci),device=feats.device,dtype=feats.dtype)
        mask=flat!=0xffffffff
        im2col[mask]=feats[flat.long()[mask]]  # FULL source, including cross-tile neighbours.
        im2col=im2col.view(end-start,v*ci)
        result=torch.addmm(bias,im2col,matrix) if bias is not None else torch.mm(im2col,matrix)
        output[start:end].copy_(result)
        del result,im2col,mask,flat  # No previous-tile references at the next allocation.
        observe('return',start,end)
    return output

class FinalConvolutionPatch:
    """Replacement exists only between the genuine final module's pre/post hooks."""
    def __init__(self,owner,module,hooks,torch,emit):
        self.owner,self.module,self.hooks,self.torch,self.emit=owner,module,hooks,torch,emit
        self.name='_sparse_submanifold_conv_forward'
        self.original_descriptor=owner.__dict__[self.name]
        self.original=getattr(owner,self.name)
        require(isinstance(self.original_descriptor,staticmethod)
                and sha256(Path(self.original.__code__.co_filename).read_bytes()).hexdigest()==SOURCE_SHA256,
                'genuine pinned explicit-GEMM source identity differs')
        self.handles=[];self.active=False;self.calls=0;self.chunks=0;self.rows_completed=0

    def install(self):
        require(not self.handles and not self.active,'final patch installed twice')
        try:
            self.handles.append(self.module.register_forward_pre_hook(self._enter))
            self.handles.append(self.module.register_forward_hook(self._exit))
        except BaseException:
            self.remove();raise
        return self

    def _enter(self,module,args):
        require(not self.active and self.calls==0 and len(args)==1
                and self.hooks.active_conv==(SITE,self.module,args[0])
                and self.hooks.convolution_completed==39 and self.hooks.up_states[3]['completed']==2,
                'final patch lifecycle/site/order differs')
        require(self.owner.__dict__[self.name] is self.original_descriptor,'foreign explicit-GEMM replacement')
        self.active=True
        setattr(self.owner,self.name,staticmethod(self.forward))

    def forward(self,feats,neighbor_cache,weight,bias=None):
        require(self.active and self.calls==0 and self.hooks.active_conv is not None,'foreign/repeated final call')
        name,module,x=self.hooks.active_conv
        require(name==SITE and module is self.module and feats is x.feats
                and weight is module.weight and bias is module.bias
                and tuple(feats.shape)==(FINAL_ROWS,64) and feats.dtype==self.torch.float16
                and feats.device==self.torch.device('cuda',0),'foreign final input/weight/bias')
        neighbor=neighbor_cache['neighbor_map']
        require(neighbor is self.hooks.final_neighbor_map,'map is not the admitted genuine global native result')
        plan=chunked_conv_plan(FINAL_ROWS,TILE_ROWS,prod(x.spatial_shape))
        # Native map now exists. Include full output and conservative downstream
        # allowance; retained model/input/caches are observed by _admit, not guessed.
        allocation={k:v for k,v in plan.items() if k.endswith('_bytes') and k!='visible_bytes'}
        allocation.pop('neighbor_map_bytes');allocation.pop('hash_bytes')
        allocation['visible_bytes']=sum(allocation.values())
        self.hooks._admit('final-full-output-and-first-tile',allocation)
        self.calls+=1
        def observe(phase,start,end):
            if phase=='entry':
                tile_plan={k:v for k,v in allocation.items() if k not in ('output_bytes','visible_bytes')}
                tile_plan['visible_bytes']=sum(tile_plan.values())
                self.hooks._admit('final-tile-'+str(start),tile_plan)
            else:
                self.chunks+=1;self.rows_completed+=end-start
            self.emit('final-tile-'+phase,start=start,end=end,rows=end-start,chunk=self.chunks+(phase=='entry'),
                      memory=self.hooks.memory(),host_return_is_not_sync=True)
        output=row_chunked_explicit(self.torch,feats,neighbor,weight,bias,TILE_ROWS,observe)
        require(self.chunks==57 and self.rows_completed==FINAL_ROWS,'incomplete tile coverage')
        return output

    def _exit(self,module,args,output):
        require(self.active and self.calls==1,'final patch return without genuine call')
        setattr(self.owner,self.name,self.original_descriptor)
        self.active=False

    def remove(self):
        if self.active:
            setattr(self.owner,self.name,self.original_descriptor)
            self.active=False
        for h in reversed(self.handles):h.remove()
        self.handles.clear()
