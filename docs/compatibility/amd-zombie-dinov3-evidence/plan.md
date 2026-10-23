# Zombie standalone conditioning plan

Approved base: `b8afdd58c3d01af264b92af5f4406fc8096dfb6d`.

Use the existing standalone DINO worker and Windows Job supervisor, not the
production stage runner. Add only an experiment-specific image/artifact adapter.
The authenticated private snapshot is immutable; copy its exact PNG bytes into
fresh owned scratch for the original worker's input record. Preserve original
alpha preprocessing, strict F32 checkpoint load, one conditioning call, feature
writer, binary authorisation and approved BLAS workspace teardown.

1. Authenticate input/CPU receipts, source/config/checkpoint, protected state and
   historical branch heads. Record the correct prefix-plus-header hash domain.
2. Focused CPU tests: exact image/pixels/preprocessing, foreign/changed inputs,
   exclusive paths/writes, unchanged profile/defaults, writer/readback and failed
   completion rejection. No model loading or GPU initialisation.
3. Independently review exact candidate and CPU results. Commit and verify the
   execution revision before the original host consumes its exclusive marker.
4. At most one original supervised run, CPU120/GPU180/overall360 seconds. Retain
   original host-RAM and initial device-memory admission, healthy observers,
   baseline checks and natural Job settlement. The legacy DINO Job has kill on
   close; unlike later decoder experiments it has no process/Job memory cap or
   allocator-fraction ceiling. Do not claim otherwise or change that contract.
5. Independently reopen the fresh finite F32 [1,1029,1024] artifact after exit,
   verify input/model/source/commit provenance, measured zero allocated/reserved
   Torch bytes, teardown, preservation and publish only source-safe evidence.

No occupancy, shape, texture, GLB, WSL, production authority or Generate enablement.
Any failed attempt stops without retry. Private image/features remain outside Git.
