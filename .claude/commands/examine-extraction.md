---
description: Have the examiner interrogate an extraction spec before it is planned. Usage /examine-extraction <ext_id>
---

Extraction id: $1

Read `extractions/$1_extraction.md` before anything else. If it does not exist or is an
unfilled template, say so and stop.

Delegate to the examiner subagent. An extraction spec is examined on different questions than a
design spec — there is no prediction table. Focus on:

- **Independence structure.** For every pair of id sets, is disjoint/nested/identical stated,
  and does the stated structure actually support each line under "comparisons this must support
  later"? A nested pair cannot yield independent estimates and no later analysis recovers it.
- **What already exists.** Is anything here already on disk, or invariant to the parameter being
  varied so that no new computation is needed at all?
- **What this forecloses.** Is it complete? Caches invalidated, files overwritten, namespace
  collisions, reversibility.
- **Namespacing.** Where a cache key omits a parameter that changes its contents, is the fix
  stated as a prerequisite?
- **Scope.** Does anything in here compare two arms or produce a number Alex would read as
  evidence? If so it belongs in a design spec instead.

Questions and factual discrepancies only. Do not propose the extraction Alex is missing and do
not rank what would be useful to extract.
