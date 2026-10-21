# Actual native colour qualification and one saved-scene CPU view

Base4c38a7a00e2fffa692eb40c4942f786ca5b07bc6. No GLB import command is
reachable. All original failed verifier/source/receipts remain unchanged.

One fresh owned CPU qualification authenticates the complete original GLB,
F32 colours, saved scene SHA67d9a943..., four imported NPY readbacks and all
source-spatial references. It rechecks every original referenced position,
ordered triangle and all12,520,752 actual cornerRGBAbytes against the saved
native oracle. Native quantisation loss is measured, without a new ceiling.

The stored colour_srgb byte fractions are decoded by the IEC sRGB piecewise
function to an F32 scene-linear table. This models the standard BYTE_COLOR
decode and is not claimed bit-identical to every native LUT/working-space
operation. The actual native .color getter is separately checked below.
All56,910 old source queries and2,048 frozen interior holdouts are replayed;
the original64+48 metric rows, RGBRMSE<=0.02/p95<=0.05/max<=0.15 and separate
metallic/roughness/OPAQUE-alpha RMSE<=0.02 remain unchanged. No old prediction
bit-equality test is reused for deliberately quantised inputs. Source targets,
face IDs, barycentrics, support and clipping semantics remain frozen.

The original evaluator and a separate independent evaluator both calculate
all112 rows. The latter builds its own cohorts, barycentric interpolation,
RGB transfer, statistics and scalar tests; it never calls the historical
evaluator. Double-precision arithmetic differences may be <=1e-12 (three
bounded products/sums and percentile interpolation); this is not a material
quality threshold. A single failing quality row stops before rendering.

After fresh admission, one Blender5.2.1 worker directly opens the unchanged
saved scene, never glTF-imports it. It verifies all1,952,142 referenced vertices,
all4,173,584 original oriented triangles, flat shading and original material.
All actual stored cornerbytes must equal the qualified saved bytes. It reads
the actual native scene-linear getter in bulk and proves repeated corners of
each original vertex have bit-identical RGB. The actual getter representation
is replayed against all112 original quality rows before rendering, using the
independent evaluator and unchanged gates. No fitted float/ULP bound or colour
replacement is introduced. This also covers native LUT/working-space effects.

One +X orthographic640x64032-sample CyclesCPU view, OpenImageDenoiseCPU,
fourthreads, no adaptive sampling, original exact camera/studio, AgX/None/+2EV.
The imported flat geometry and materials/bytes are rechecked after rendering.
A fresh owned CPU image reader reopens the PNG and creates one labelled
private contextual comparison against the previous original-mesh PBR+2EV PNG.
Different colour sampling/gamma and constant material factors prohibit a
pixel-perfect equivalence claim. Independent visual review is required.

All stages retain6GiB process AND Job caps,180seconds,12GiB live Windows
available RAM and2GiB destination free, fresh exclusive paths/markers, exact
binary GO after Job attachment, continuous observation and natural empty Job.
Render also requires the authentic previous import peaks<=5GiB; measured prior
import1,684,721,664B and original full-model PBR render2,979,557,376B support
the budget. Neither historical measurement is reported as this render's peak.
No retry, neural execution, geometry modification, export, baking or package
change. Images/model arrays stay outside Git; only source-safe evidence ships.
