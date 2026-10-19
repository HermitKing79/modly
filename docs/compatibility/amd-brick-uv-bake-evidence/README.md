# Bounded CPU UV patch proof

**Actual result: BLOCKED at analytic positive-area overlap in the planar UV
chart.** No real texture sampling, maps, GLB, Blender launch or render occurred.
The sampler and exporter/parser checks below describe implemented synthetic
qualification and the unexecuted downstream design, not completed artifacts.

This is an experiment on one unchanged subset of the authenticated brick mesh.
It does not call the GPU exporter, repair geometry, or perform a full-model bake.
Private assets stay outside Git.

`source-semantics.json` pins the actual installed sources. The CPU port preserves
native neighbour order, sparse support renormalization, six-channel sampling and
byte packing. Native GPU floating-point bitwise parity is not established.
The selected triangles remain on the original source surface, so the native
closest-point projection is mathematically the identity here. This does not
establish equivalence for a simplified or otherwise changed mesh.

The initial signed-normal nomination failed before UV creation or baking.
`nomination-correction.json` supersedes only its selection/path details in
`probe-contract.json`: a genuine visible negative-X-winding sheet is selected;
UV U is mirrored without changing original geometry or winding. The original
failure, bounded nine-ray observation and byte-exact source snapshots remain
in the top-level private experiment directory. The corrected attempt uses its
fresh `qualified` child directory. No failed stage directory can be reused.

Focused checks use the existing CPU scientific interpreter, not Torch:

```powershell
& 'D:\AI\Trellis2AMD\venv\Scripts\python.exe' -I -B -c "import sys,unittest;sys.path.insert(0,r'docs/compatibility/amd-brick-uv-bake-evidence');unittest.TextTestRunner().run(unittest.defaultTestLoader.discover(r'docs/compatibility/amd-brick-uv-bake-evidence','test_*.py'))"
```

Each real stage uses `run_host.py bake`, `export`, or `compare`, and requires the
exact approved base/experiment branch, fresh destination, resource floors,
Windows Job ownership before binary GO, a 6 GiB process ceiling and 180-second
deadline. Stages are not automatic retries. Inspect the preceding result first.

The unexecuted quality score would compare the packed texture against continuous source spatial
samples at every face centroid and 10,000 area-weighted points. RGB filtering
decodes sRGB texels to linear before bilinear filtering and re-encodes for the
encoded-RGB metric. This is a base-level/LOD0 score, not a mipmap-quality proof.
The older direct-row preview has different spatial and gamma interpolation;
its render difference is reported separately, not treated as bake error.

The unexecuted texture/export design retains RGB in the native encoded representation, MR is Non-Color with
G=roughness/B=metallic, and the material is opaque. Original learned alpha
would remain in float data and the source RGBA PNG; an opaque GLB exporter may omit
unused alpha. No occlusion, emission or invented details are introduced.
