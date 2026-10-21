# Full GLB Blender pre-import gate

Starting commit df195c49ae162c9fa9e3a2280608dd37eeeb76f8. CPU only.

The requested exact 1,957,891 imported vertices is mandatory. Stock Blender
5.2.1 importer builds its mesh from the sorted distinct triangle index values,
even with merge_vertices=False. Preliminary original index membership found
1,952,142 referenced and 5,749 unreferenced vertices. This is a deterministic
viewer/count mismatch, not an AMD, material or damaged-surface result.

Before any Blender import, one owned CPU worker authenticates all original
inputs, previous independent proof, installed importer source and binary;
independently reopens the entire unchanged GLB with the previously qualified
streaming reader; then counts all referenced source indices in bounded chunks.
The diagnosis receipt distinguishes successful CPU investigation from BLOCKED
import/visual acceptance. No import/render candidate is authorised after this
exact-count mismatch. No vertex restoration, importer edits, GLB re-export or
acceptance relaxation.

Retain 6 GiB process AND Job caps, 180 seconds, 12 GiB live available Windows
RAM and 2 GiB destination free. Fresh diagnostic marker/output only; binary GO
after Job attachment, periodic health checks, natural process/Job settlement,
source identity comparison before/after. No production import marker.

Seven focused CPU tests cover unused/all-used indices, original multiplicity,
unchanged inputs, malformed/index/batch bounds and actual host command/stage
rejection. Independent review binds exact
source hashes and installed importer audit before the owned diagnostic capture.

If the mismatch is verified, publish BLOCKED and request a future surface-only
acceptance contract (all original oriented triangles and colours; omit only
unreferenced points) as the cheapest next decision. Do not infer that a render
or imported material has already been qualified. No images are produced.
