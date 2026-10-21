"""Pure deterministic position holdout on previously frozen original faces."""
import sys
sys.dont_write_bytecode = True
import numpy as np

COHORT_ORDER = ('global','brick','foliage','rear_x','backfacing_x','boundary',
                'nonmanifold','high_variation',*(f'octant_{i}' for i in range(8)))


def build(vertices, faces, regions, old_world, seed=20261023):
    """Freeze 64 faces and two new positions per each of 16 prior face pools.

    This is a position holdout on previously fixed faces, not a face-disjoint
    holdout. No observed error, corrected prediction or source sampler enters
    the draw. A collision after final F32 rounding blocks the entire result;
    there is no redraw, removal, refill or alternative seed.
    """
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(type(seed) is int and seed >= 0, 'Nonnegative integer seed required')
    require(isinstance(vertices,np.ndarray) and vertices.dtype == np.float32
            and vertices.ndim == 2 and vertices.shape[1] == 3
            and 0 < len(vertices) <= 2_000_000, 'Finite F32[N,3] original vertices required')
    require(isinstance(faces,np.ndarray) and faces.dtype == np.int32
            and faces.ndim == 2 and faces.shape[1] == 3
            and 0 < len(faces) <= 4_173_584, 'Original I32[F,3] faces required')
    require(isinstance(old_world,np.ndarray) and old_world.dtype == np.float32
            and old_world.ndim == 2 and old_world.shape[1] == 3
            and len(old_world)+2048 <= 65_536, 'Bounded F32[M,3] prior positions required')
    require(isinstance(regions,dict) and len(regions) == 16
            and all(isinstance(name,str) and name for name in regions), 'Exactly sixteen named frozen regions required')
    for start in range(0,len(vertices),32768):
        require(np.isfinite(vertices[start:start+32768]).all(), 'Nonfinite original geometry')
    require(np.isfinite(old_world).all(), 'Nonfinite prior world position')
    for start in range(0,len(faces),32768):
        block = faces[start:start+32768]
        require(np.all(block >= 0) and np.all(block < len(vertices)), 'Original face index outside vertex bounds')
    order = COHORT_ORDER if set(regions) == set(COHORT_ORDER) else tuple(sorted(regions))
    pools = {}
    for name in order:
        pool = regions[name]
        require(isinstance(pool,np.ndarray) and pool.dtype == np.int32 and pool.ndim == 1,
                'Frozen region IDs must be original I32 vectors')
        require(len(pool) >= 64, 'Frozen region underfilled: '+name)
        require(np.all(pool >= 0) and np.all(pool < len(faces)), 'Frozen region face ID outside bounds')
        pool = np.sort(pool)
        require(np.all(pool[1:] != pool[:-1]), 'Duplicate frozen region face ID')
        pools[name] = pool
    unique_faces = np.unique(np.concatenate(list(pools.values())))
    areas = np.empty(len(unique_faces),np.float64)
    # Validate every frozen pool face, including faces a later random draw omits.
    for start in range(0,len(unique_faces),32768):
        xyz = vertices[faces[unique_faces[start:start+32768]]].astype(np.float64)
        areas[start:start+len(xyz)] = .5*np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)
    require(np.isfinite(areas).all() and np.all(areas > 0), 'Frozen pool contains zero-area or invalid triangle')
    query_face = np.empty(2048,np.int32)
    bary = np.empty((2048,3),np.float64)
    cohorts = {}
    epsilon = np.finfo(np.float64).eps
    for index,name in enumerate(order):
        pool = pools[name]
        weights = areas[np.searchsorted(unique_faces,pool)]
        pool_area = float(weights.sum(dtype=np.float64))
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,index,0])))
        chosen = np.sort(rng.choice(pool,size=64,replace=False,p=weights/pool_area)).astype(np.int32)
        begin,end = index*128,(index+1)*128
        query_face[begin:end] = np.repeat(chosen,2)
        interior_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,index,1])))
        uv = np.clip(interior_rng.random((64,2,2),dtype=np.float64),epsilon,1-epsilon)
        root = np.sqrt(uv[:,:,0])
        block = bary[begin:end].reshape(64,2,3)
        block[:,:,0] = 1-root
        block[:,:,1] = root*(1-uv[:,:,1])
        block[:,:,2] = root*uv[:,:,1]
        cohorts[name] = dict(query_start=begin,query_stop=end,pool_count=int(len(pool)),
                             selected_face_count=64,query_count=128,pool_area=pool_area,
                             selected_face_ids=chosen.tolist())
    require(np.all((bary > 0) & (bary < 1)) and np.all(np.abs(bary.sum(axis=1)-1) <= 2e-15),
            'Strict interior barycentric weights required')
    world = np.einsum('ni,nij->nj',bary,vertices[faces[query_face]].astype(np.float64)).astype(np.float32)
    require(np.isfinite(world).all(), 'Nonfinite final F32 holdout position')
    # Numeric float tuple equality/hash treats +0 and -0 identically. Old
    # duplicates are allowed; any new-vs-old or new-vs-new match is rejected.
    occupied = set(map(tuple,old_world))
    for point in map(tuple,world):
        require(point not in occupied, 'Final F32 world position collision; no redraw permitted')
        occupied.add(point)
    return dict(query_face=query_face,bary=bary,kinds=np.tile(np.array([4,5],np.int8),1024),
                world=world,cohorts=cohorts,
                draw_rule=dict(scope='position holdout on previously fixed faces; not face-disjoint',
                    seed=seed,generator='PCG64',numpy=np.__version__,cohort_order=list(order),
                    face_seed='SeedSequence([seed,cohort_index,0])',
                    interior_seed='SeedSequence([seed,cohort_index,1])',
                    selection='64 area-weighted unique faces; ascending original face IDs',
                    interior='two points per face; uniforms[64,2,2] C order; (1-sqrt(u),sqrt(u)*(1-v),sqrt(u)*v)',
                    endpoint_guard='uniforms clipped to [F64 epsilon,1-epsilon]',
                    final_world='F64 barycentric einsum then F32 conversion',
                    collision_rule='exact numeric F32 positions; signed zeros equal; zero collisions; no redraw',
                    queries=2048))
