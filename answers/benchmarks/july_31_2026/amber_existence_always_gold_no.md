# AMBER existence questions are always gold-no (benchmark fact)

Date: 2026-07-31

In `data/amber/data/annotations.json`, every entry with type
`discriminative-hallucination` (mapped to category `existence` by
`_amber_discriminative_qtype` in `src/dataset.py`) has `truth: "no"`.

Counts: 4924 / 4924 existence annotations are gold-no; 0 are gold-yes.

Attribute and relation do have both labels (attribute 3814 yes / 3814 no;
relation 975 yes / 689 no including the bare `relation` type).

So zero existence gold-yes on the AMBER-450 pin is **not** a pin/sampling bug;
it matches the full benchmark.
