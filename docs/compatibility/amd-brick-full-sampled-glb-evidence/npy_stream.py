"""Exclusive, bounded NPY backing files; callers retain original row order."""
import sys
sys.dont_write_bytecode=True
import os
import numpy as np

def create(path,dtype,shape):
    dtype=np.dtype(dtype)
    if dtype not in (np.dtype('<f4'),np.dtype('u1')) or len(shape) not in (1,2) or not all(type(n) is int and 0<n<=2_000_000 for n in shape):raise ValueError('Bounded proof NPY structure')
    if len(shape)==2 and shape[1]!=3:raise ValueError('Only RGB VEC3')
    with path.open('xb') as out:
        np.lib.format.write_array_header_1_0(out,dict(descr=np.lib.format.dtype_to_descr(dtype),fortran_order=False,shape=shape));offset=out.tell();out.truncate(offset+int(np.prod(shape))*dtype.itemsize);out.flush();os.fsync(out.fileno())
    return np.memmap(path,mode='r+',offset=offset,dtype=dtype,shape=shape)

def sync(path,view):
    view.flush()
    with path.open('r+b') as stream:os.fsync(stream.fileno())
