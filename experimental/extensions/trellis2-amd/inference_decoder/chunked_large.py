"""Opt-in row tiling of the six pinned 128-channel tail convolutions only."""
from math import prod
from hashlib import sha256
from pathlib import Path
from admission import require, MAX_ROWS, TILE_ROWS, LARGE_TILED_SITES, large_chunked_conv_plan
from chunked_final import SOURCE_SHA256, row_chunked_explicit


class LargeSupportConvolutionPatch:
    def __init__(self,owner,modules,hooks,torch,emit,tile_rows=TILE_ROWS):
        require(tuple(modules)==LARGE_TILED_SITES and len({id(m) for m in modules.values()})==6,
                'large patch requires exactly the six original sites')
        require(type(tile_rows) is int and tile_rows==TILE_ROWS,
                'large patch requires the reviewed 32768-row budget')
        self.owner,self.modules,self.hooks,self.torch,self.emit=owner,dict(modules),hooks,torch,emit
        self.tile_rows=tile_rows;self.name='_sparse_submanifold_conv_forward'
        self.original_descriptor=owner.__dict__[self.name]
        original=getattr(owner,self.name)
        require(isinstance(self.original_descriptor,staticmethod)
                and sha256(Path(original.__code__.co_filename).read_bytes()).hexdigest()==SOURCE_SHA256,
                'genuine pinned explicit-GEMM source identity differs')
        self.handles=[];self.active=None;self.invoked=False;self.records=[];self.neighbor_cache=None

    def install(self):
        require(not self.handles and self.active is None and not self.records,'large patch installed twice')
        try:
            for name,module in self.modules.items():
                self.handles.append(module.register_forward_pre_hook(lambda m,args,s=name:self._enter(s,m,args)))
                self.handles.append(module.register_forward_hook(lambda m,args,out,s=name:self._exit(s,m,args,out)))
        except BaseException:
            self.remove();raise
        return self

    def _enter(self,name,module,args):
        index=len(self.records)
        require(self.hooks.tile_large_convolutions and self.active is None and index<6
                and name==LARGE_TILED_SITES[index] and module is self.modules[name] and len(args)==1
                and self.hooks.active_conv==(name,module,args[0])
                and self.hooks.convolution_completed==33+index,
                'large patch lifecycle/site/order differs')
        require(self.owner.__dict__[self.name] is self.original_descriptor,'foreign explicit-GEMM replacement')
        self.active=(name,module,args[0]);self.invoked=False
        setattr(self.owner,self.name,staticmethod(self.forward))

    def forward(self,feats,neighbor_cache,weight,bias=None):
        require(self.active is not None and not self.invoked,'foreign/repeated large call')
        name,module,x=self.active
        require(self.hooks.active_conv==self.active and feats is x.feats
                and weight is module.weight and bias is module.bias
                and feats.ndim==2 and feats.shape[1]==128 and 0<feats.shape[0]<=MAX_ROWS
                and feats.dtype==self.torch.bfloat16 and feats.device==self.torch.device('cuda',0),
                'foreign large input/weight/bias')
        co=512 if name==LARGE_TILED_SITES[-1] else 128
        require(tuple(weight.shape)==(co,3,3,3,128),'large weight dimensions differ')
        neighbor=neighbor_cache['neighbor_map']
        require(neighbor is self.hooks.large_neighbor_map and x.coords is self.hooks.large_neighbor_coords,
                'map is not the admitted genuine global native result')
        reused=bool(self.records)
        if reused:require(neighbor_cache is self.neighbor_cache,'foreign reused neighbor cache')
        else:self.neighbor_cache=neighbor_cache
        n=feats.shape[0];plan=self.hooks.large_conv_plan
        require(plan is not None and plan==large_chunked_conv_plan(n,128,co,self.tile_rows,
                prod(x.spatial_shape),reused,plan['block_extra_bytes']), 'large admission plan differs')
        allocation={k:v for k,v in plan.items() if k.endswith('_bytes') and k!='visible_bytes'}
        # Genuine full map now exists. Other retained global maps remain in the
        # measured baseline. Drop only this already allocated map/hash allowance.
        allocation.pop('neighbor_map_bytes');allocation.pop('hash_bytes')
        allocation['visible_bytes']=sum(allocation.values())
        self.hooks._admit(name+'-full-output-and-first-tile',allocation)
        row=dict(site=name,rows=n,in_channels=128,out_channels=co,tile_rows=self.tile_rows,
                 chunks=0,rows_completed=0,global_map_reused=reused,completed=False)
        self.records.append(row);self.invoked=True
        def observe(phase,start,end):
            if phase=='entry':
                tile_plan={k:v for k,v in allocation.items() if k not in ('output_bytes','visible_bytes')}
                tile_plan['visible_bytes']=sum(tile_plan.values())
                self.hooks._admit(name+'-tile-'+str(start),tile_plan)
            else:
                row['chunks']+=1;row['rows_completed']+=end-start
            self.emit('large-tile-'+phase,site=name,start=start,end=end,rows=end-start,
                      chunk=row['chunks']+(phase=='entry'),memory=self.hooks.memory(),host_return_is_not_sync=True)
        result=row_chunked_explicit(self.torch,feats,neighbor,weight,bias,self.tile_rows,observe)
        require(row['chunks']==plan['chunks'] and row['rows_completed']==n,'incomplete large tile coverage')
        row['completed']=True
        return result

    def _exit(self,name,module,args,output):
        require(self.active==(name,module,args[0]) and self.invoked and self.records[-1]['completed']
                and self.hooks.convolution_completed==33+len(self.records),'large return/order differs')
        setattr(self.owner,self.name,self.original_descriptor)
        self.active=None

    def remove(self):
        if self.active is not None:
            setattr(self.owner,self.name,self.original_descriptor)
        self.active=None;self.neighbor_cache=None
        for handle in reversed(self.handles):handle.remove()
        self.handles.clear()


def validate_records(records):
    require(type(records) is list and len(records)==6,'six large tiled convolutions required')
    rows=records[0].get('rows')
    require(type(rows) is int and 0<rows<=MAX_ROWS,'invalid large tiled support')
    for i,(name,row) in enumerate(zip(LARGE_TILED_SITES,records)):
        require(row==dict(site=name,rows=rows,in_channels=128,out_channels=512 if i==5 else 128,
                tile_rows=TILE_ROWS,chunks=(rows+TILE_ROWS-1)//TILE_ROWS,rows_completed=rows,
                global_map_reused=i>0,completed=True),'incomplete or altered large tiled convolution record')
