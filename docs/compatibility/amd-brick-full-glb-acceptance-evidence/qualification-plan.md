# Approved compacted Blender import and one CPU view

Base a258c4eb97c239e3ec766648e71ca59b51ce1de9. The user's amendment
permits omission of exactly 5,749 original triangle-unreferenced points.
The unchanged source retains 1,957,891 POSITION/COLOR_0 rows. The imported
mesh must retain exactly 1,952,142 referenced vertices and all 4,173,584
triangles, with bit-exact source positions and exact face/corner order,
winding and multiplicity under the installed sorted-unique index remap.

One stock Blender5.2.1 GLB import only. No weld, repair, colour replacement,
additional removal or changed importer. Before that call, derive the complete
original-to-imported map and construct a bounded native BYTE_COLOR oracle:
32,768 POINT values per batch using the installed native colour setter.
Require the imported CORNER attribute to use the identical RNA value class.
Read all actual stored RGBA bytes; every original corner must match its
source vertex's native oracle bytes exactly. Quantify expected byte-storage
loss against original linear F32 separately with the independent reader;
encoded max error <=0.5/255 +2e-6, without relaxed prior RGB quality gates.
Native byte association is exact; no one-byte transport tolerance.

Save the verified import privately as inspected.blend and save four NPY
readback arrays. A separate owned CPU reader independently repeats complete
original-index, position-bit, oriented triangle and native-byte checks before
any render. The original glTF constant material/shader, opacity and two-sided
behaviour must pass the previous material-reader contract unchanged.

Each worker retains 6GiB process AND Job caps,180 seconds,12GiB live available
Windows RAM and2GiB destination free. Exclusive per-stage marker/output,
binary GO after Job attachment, periodic observation, natural empty Job and
unchanged source identities are required. No retry. The prior installed
importer audit projects1.37GB visible/native geometry and normal-array storage;
Python readback adds at most about0.6GB for corner colours/temporaries. Startup,
native validation and allocator overhead remain uncertain and are contained by
the hard caps; these projections are not actual Blender measurements.

Separate render admission: import and independent verification must pass;
measured import process AND Job peaks must leave >=1GiB below the6GiB cap.
Previous original full-mesh PBR render measured2,979,557,376 committed bytes
for1024px/32 samples. One new640x64032-sample +X Cycles CPU view only,
OpenImageDenoise CPU, four threads, no adaptive sampling, accepted exact camera
and neutral lights, AgX/None/+2EV/gamma1. Render opens the saved private
.blend; it never reimports the GLB. No optional isometric view is admitted.
Keep imported flat shading and original colours/materials unchanged.

Fresh image reopening produces one private labelled comparison to the
previous original-geometry PBR +2EV image. Different colour sampling and
gamma semantics make this contextual, not pixel-exact equivalence. An
independent visual review is required; existing geometry defects are not
new export defects. No source/model/image payloads go to the public repo.

Focused CPU tests cover exact sorted remapping, corner/winding/multiplicity,
forbidden merges, malformed bounds, independent colour oracle association
and actual supervisor commands/limits. Independent candidate/host review
binds exact source/test/plan hashes before the import. If any stage fails,
preserve evidence and stop without another import or render.
