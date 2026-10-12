# CPU regression development evidence

The initial 11 transfer tests failed with the explicit assertion that the exact
single-use `TensorTransfers` permit was missing. After implementing that permit,
the combined transfer/forward/loading-contract run passed 32 tests.

A new classification test failed because a record without both exact transfer
counts was incorrectly classified PASS. Requiring integer input/output counts of
one corrected it. The first captured CPU verification then passed 38 tests.

Independent review identified missing stride binding and different characterization
mode nesting. The stride test failed while valid new stride metadata was absent
from the contract. After binding input/output strides and matching production
nesting, the saved reviewed CPU verification passed all 39 tests. Tests used only
CPU descriptors and tiny synthetic checkpoints; the schema probe blocked GPU
initialization and recorded false for its GPU-initialized flag.
