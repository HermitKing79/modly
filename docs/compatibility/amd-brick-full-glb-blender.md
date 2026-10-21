# Full-resolution Blender import: BLOCKED before launch

Base: `df195c49ae162c9fa9e3a2280608dd37eeeb76f8`.
Branch: `codex/amd-brick-full-glb-blender`.

The installed Blender 5.2.1 importer cannot satisfy the requested exact
1,957,891 imported-vertex count for this unchanged GLB. No Blender import or
render was attempted, and no production import marker was consumed.

The complete GLB still contains all **1,957,891 POSITION/COLOR_0 rows** and
**4,173,584 original oriented triangles**. Its triangles reference only
**1,952,142 distinct vertex indices**, leaving **5,749 unreferenced points**
(about 0.294%). Independently counting the actual GLB index accessor and the
original mesh face table gave identical results.

Installed `io_scene_gltf2/blender/imp/mesh.py:304–307` unconditionally computes
`np.unique(indices, return_inverse=True)` and appends `vs[unique_indices]`.
At line 430 it creates exactly that many mesh vertices. `merge_vertices=False`
does not prevent this: the merge option is considered later, at line 396.
The stock importer offers no unused-triangle-vertex preservation option.
Inverse index remapping preserves indexed triangle order; this is not evidence
that surface triangles would be lost. Actual import, shader interpretation,
colour quantisation and rendering remain untested.

The exact authenticated GLB is 97,073,940 bytes, SHA-256
`1970bbcc5740dc56c8c9e6e20d97e8afd3176126c57b87d886a034f876eb38ae`.
The earlier full-file F32 geometry/colour proof remains valid. This milestone
does not reinterpret that successful export as a failure or blame AMD.

## Evidence and limits

- [Installed importer audit](amd-brick-full-glb-blender-evidence/importer-audit.json)
- [Focused CPU checks](amd-brick-full-glb-blender-evidence/focused-tests.json)
- [Pre-execution review](amd-brick-full-glb-blender-evidence/independent-candidate-review.json)
- [Actual CPU measurements and decision](amd-brick-full-glb-blender-evidence/outcome.json)
- [Independent outcome review](amd-brick-full-glb-blender-evidence/independent-outcome.json)
- [Preservation before](amd-brick-full-glb-blender-evidence/preservation-before.json)
- [Preservation after](amd-brick-full-glb-blender-evidence/preservation-after.json)

Only a read-only CPU diagnostic is executed. The original 6 GiB process AND
Job ceilings, 180-second deadline, 12 GiB live Windows available-RAM floor and
2 GiB destination-free floor remain enforced. Importer allocations are source
projections, not measured Blender peaks; no render admission is claimed.
Private original assets, GLB, F32 colours, protected100, selected environment
identities, source edits/backups and historical evidence are preserved. No
geometry, material, source environment or importer changes; no WSL/model/GPU
work; no new images or model files are published.

Seven focused checks passed, with no skips. The owned diagnostic completed in
1.625 seconds; its process tree settled naturally in 2.985 seconds. Measured
peak committed memory was 128,192,512 bytes for the process and 129,024,000
bytes for its Job. Minimum observed available Windows RAM was 20,710,375,424
bytes; destination free space remained at least 113,092,775,936 bytes. The Job
was empty before cleanup; no forced termination. These are **CPU diagnostic**
measurements, not Blender import/render measurements. There are no new images
or inspection scene to view from this blocked milestone.

**Next decision:** separately approve a triangle-preserving viewer contract
which permits Blender to omit only the 5,749 unreferenced points while requiring
all 4,173,584 oriented triangles, their positions and colours to survive.
Then one bounded genuine import and separately admitted render can be attempted.
Do not re-export or repair the asset to manufacture the original vertex count.
