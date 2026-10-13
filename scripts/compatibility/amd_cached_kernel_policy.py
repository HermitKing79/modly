"""Reuse existing cached-only seams, selecting a leaf and exact owned buffers."""
from amd_native_convolution_policy import NativePolicy

class CachedKernelPolicy(NativePolicy):
    def select_leaf(self,leaf):
        import triton
        if type(leaf) is not triton.runtime.JITFunction or self.leaf is not None or leaf.device_caches or leaf.cache_key!=self.meta['jit_source_key']:raise RuntimeError('Foreign/already-used cached leaf')
        self.leaf=leaf;self.buffers={}

    def bind(self,buffers):
        if self.buffers:raise RuntimeError('Repeated owned-buffer binding')
        expected={'input':((2049,1024),self.torch.float16),'weight':((1024,27,1024),self.torch.float16),'bias':((1024,),self.torch.float16),
            'neighbor':((2049,27),self.torch.uint32),'sorted_idx':((2049,),self.torch.int64),'output':((2,2049,1024),self.torch.float32),
            'valid_kernel':((599,),self.torch.int32),'valid_kernel_seg':((34,),self.torch.int32)}
        if set(buffers)!=set(expected):raise RuntimeError('Foreign cached-kernel buffers')
        for name,(shape,dtype) in expected.items():
            t=buffers[name];self.tensor(t,shape,dtype)
            if type(t) is not self.torch.Tensor or t.storage_offset()!=0 or t.data_ptr()%16 or t.untyped_storage().nbytes()!=t.numel()*t.element_size():raise RuntimeError('Noncanonical cached-kernel pointer')
            self.buffers[name]=(t,t.untyped_storage(),shape,dtype)

    def launch_once(self,buffers):
        self.permit.take('convolution')
        for name,(original,storage,shape,dtype) in self.buffers.items():
            t=buffers[name];self.tensor(t,shape,dtype)
            if type(t) is not self.torch.Tensor or t._cdata!=original._cdata or t.untyped_storage()._cdata!=storage._cdata or t.storage_offset()!=0 or t.untyped_storage().nbytes()!=t.numel()*t.element_size():raise RuntimeError('Foreign/altered cached-kernel input')
        if set(buffers)!=set(self.buffers) or not self.buffers:raise RuntimeError('Missing exact kernel buffers')
        return self.leaf[(264,2)](buffers['input'],buffers['weight'],buffers['bias'],buffers['neighbor'],buffers['sorted_idx'],buffers['output'],
            2049,11,1024,1024,27,64,128,32,2,True,buffers['valid_kernel'],buffers['valid_kernel_seg'],
            num_warps=4,num_ctas=1,num_stages=2,waves_per_eu=3)

    def release(self):
        self.buffers.clear();super().release()

    def close(self):
        if hasattr(self,'buffers'):self.buffers.clear()
        super().close()
