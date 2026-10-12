"""Bounded allocator state/origin capture and weak Python-tensor observations."""
import hashlib
import json
from pathlib import Path
import weakref

HISTORY=dict(enabled='all',context='state',stacks='all',max_entries=512,clear_history=True,
             compile_context=False,global_record_annotations=False)
MAX_BYTES=16*1024**2


def integer(value):
    if type(value) is not int or value<0:raise ValueError('Invalid allocator integer')
    return value


def frames(rows):
    if type(rows) is not list or len(rows)>256:raise ValueError('Allocator stack exceeds bound')
    for row in rows:
        if (type(row) is not dict or any(type(row.get(k)) is not str or len(row[k])>32768 for k in ('filename','name'))
                or type(row.get('line')) is not int):raise ValueError('Malformed allocator frame')
    return rows


def summarize(snapshot,allocated,reserved,owners):
    integer(allocated);integer(reserved)
    if type(owners) is not list or len(owners)>8192:raise ValueError('Observed owner bound exceeded')
    segments=snapshot.get('segments');traces=snapshot.get('device_traces')
    if type(segments) is not list or len(segments)>4096 or type(traces) is not list or len(traces)>1:
        raise ValueError('Allocator snapshot device/segment bound exceeded')
    if any(type(rows) is not list or len(rows)>512 for rows in traces):raise ValueError('Allocator trace bound exceeded')
    for rows in traces:
        for row in rows:frames(row.get('frames',[]))
    totals=dict(active_allocated=0,active_awaiting_free=0,inactive=0);active=[];count=0;reserved_total=0
    for segment in segments:
        if type(segment.get('device')) is not int or segment['device']!=0:raise ValueError('Snapshot contains unexpected GPU device')
        blocks=segment.get('blocks')
        if type(blocks) is not list:raise ValueError('Malformed allocator blocks')
        count+=len(blocks)
        if count>16384:raise ValueError('Allocator block bound exceeded')
        sizes=0;segment_active=0;segment_allocated=0
        for block in blocks:
            size=integer(block['size']);address=integer(block['address']);requested=integer(block['requested_size'])
            state=block['state']
            if state not in totals or requested>size:raise ValueError('Invalid allocator block state/size')
            totals[state]+=size;sizes+=size
            if state!='inactive':segment_active+=size
            if state=='active_allocated':segment_allocated+=size
            stack=frames(block.get('frames',[]))
            if state!='inactive':
                matches=[owner for owner in owners if address<=owner['storage_address']<address+size]
                active.append(dict(address=address,size=size,requested_size=requested,state=state,frames=stack,
                                   observed_python_owners=matches))
        if (sizes!=integer(segment['total_size']) or segment_allocated!=integer(segment['allocated_size'])
                or segment_active!=integer(segment['active_size'])):raise ValueError('Allocator segment totals mismatch')
        reserved_total+=sizes
    if totals['active_allocated']!=allocated or reserved_total!=reserved:
        raise ValueError('Snapshot does not reconcile with measured allocated/reserved bytes')
    return dict(active_allocated_bytes=totals['active_allocated'],active_awaiting_free_bytes=totals['active_awaiting_free'],
                inactive_bytes=totals['inactive'],reserved_bytes=reserved_total,segments=len(segments),blocks=count,
                active_blocks=active,observed_python_owner_count=len(owners),
                interpretation='Allocation origins and observed weak Python wrappers; absence of a wrapper does not establish native ownership.')


class WeakOwners:
    def __init__(self,limit=8192):
        if type(limit) is not int or not 0<limit<=8192:raise ValueError('Invalid weak registry limit')
        self.entries=[];self.current={};self.limit=limit
    def track(self,tensor,label):
        if getattr(getattr(tensor,'device',None),'type',None)!='cuda':return
        if tensor.device.index!=0:raise ValueError('Observed tensor on unexpected device')
        ident=id(tensor);existing=self.current.get(ident)
        if existing is not None and existing['ref']() is tensor:return
        if len(self.entries)>=self.limit:raise ValueError('Observed tensor registry bound exceeded')
        row={'ref':weakref.ref(tensor),'label':str(label)[:160]}
        self.entries.append(row);self.current[ident]=row
    def living(self):
        rows=[]
        for entry in self.entries:
            tensor=entry['ref']()
            if tensor is not None:
                storage=tensor.untyped_storage()
                rows.append(dict(label=entry['label'],shape=list(tensor.shape),dtype=str(tensor.dtype),
                    storage_address=storage.data_ptr(),storage_bytes=storage.nbytes()))
                storage=tensor=None
        return rows


class AllocationDiagnostics:
    def __init__(self,scratch):
        self.scratch=Path(scratch);self.owners=WeakOwners();self.begun=False;self.captured=False
    def begin(self,torch):
        if self.begun or not torch.cuda.is_initialized():raise RuntimeError('Allocation history requires authorized GPU initialization once')
        self.begun=True;torch.cuda.memory._record_memory_history(**HISTORY)
    def observe(self,values,label):
        for tensor in values:self.owners.track(tensor,label)
    def capture(self,torch,allocated,reserved):
        if not self.begun or self.captured:raise RuntimeError('Allocator snapshot permit absent/consumed')
        self.captured=True
        snapshot=torch.cuda.memory._snapshot(augment_with_fx_traces=False)
        owners=self.owners.living();summary=summarize(snapshot,allocated,reserved,owners)
        settings={'blas_workspace_bytes':torch._C._cuda_getCublasWorkspaceSize(),
                  'blaslt_workspace_bytes':torch._C._cuda_getCublasLtWorkspaceSize()}
        payload=dict(snapshot=snapshot,observed_python_owners=owners,summary=summary,
                     workspace_settings=settings,observed_tensor_records=len(self.owners.entries),history=HISTORY)
        blob=json.dumps(payload,allow_nan=False).encode('utf-8')
        if len(blob)>MAX_BYTES:raise ValueError('Serialized allocator snapshot exceeds 16 MiB bound')
        path=self.scratch/'allocator-release.json'
        with path.open('xb') as stream:
            stream.write(blob);stream.flush()
            import os
            os.fsync(stream.fileno())
        torch.cuda.memory._record_memory_history(enabled=None,context=None,stacks='all',max_entries=512)
        return dict(snapshot_verified=True,summary=summary,workspace_settings=settings,
                    snapshot_path=str(path),snapshot_bytes=len(blob),snapshot_sha256=hashlib.sha256(blob).hexdigest(),
                    observed_tensor_records=len(self.owners.entries),history_disabled=True)
