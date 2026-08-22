<!-- Last updated: 2026-08-17 — harness template; delete this line in your copy -->

# Extraction spec — <ext_id>

Copy to `extractions/<ext_id>.md`, or into a dated subdirectory as
`extractions/<date>/<ext_id>.md`. Three fields. Delete every angle-bracket
placeholder; the pre-write hook rejects plans whose spec still contains them.

There is one pre-drawn table, under field 2, and it is optional — the prose is the spec. It has
no column for output paths, cache layout, or forward-pass counts, because those are the plan's
to resolve rather than yours to state. See "Not your job".

Short on purpose. A bad extraction costs compute; a bad design costs a belief. The gate is
correspondingly light, and everything an implementer works out by reading the repo is absent —
see "Not your job" at the bottom.

---

## Scope check

Covers **producing and storing primitives**: activations, per-layer stacks, attention weights,
per-head values, residual streams, extracted directions, caches.

Does not cover **any comparison, or any number you would read as evidence** — those need
`designs/<exp_id>.md`, whether or not they need a GPU.

The test: **would a different value change what you believe?** Producing a tensor cannot come
out wrong in a way that changes a belief; measuring something can.

---

## 1. What data

Plain English. What primitives, over what items, at what sizes, for which models. A paragraph
is enough.

If items are paired or grouped in a way that matters — the same image under two captions, the
same question under two prefixes, several items sharing a source object — say so. Pairing that
is not stated tends not to survive into what gets written.

<answer>

## 2. How the sets relate

The field that cannot be fixed later. Everything else here is a rerun; this is a rerun you do
not know you need.

For every pair of sets: disjoint, nested, or identical, and why. State it as intent — what
produces the property is the plan's problem, not your sentence to write.

| Set A | Set B | Disjoint, nested, or identical | Why |
|---|---|---|---|
| <set> | <set> | ... | ... |
| ... | ... | ... | ... |
| ... | ... | ... | ... |

Then, in one line: what is held constant across the sets, and what varies. If more than one
thing varies, say so deliberately rather than by omission.

<answer>

## 3. What this must support later

No hypothesis needed. What must you be *able* to compute once these exist.

This is what field 2 gets checked against. A comparison that varies two things at once under
your stated structure will not answer what you want, and here is where that costs ten minutes
instead of a re-extraction.

- <comparison>

---

## Code gaps

**Filled by an agent, not by you.** Leave it empty; an agent that reads the repo fills it in.

Places where the repo cannot produce what fields 1–3 ask for as written: an argument that
exists at one layer and is not forwarded by the layer above it, a cache key that omits a
parameter that changes its contents, an entry point that does not accept a structure named in
field 1. Found by reading the code, with file and line.

"Not your job" below already assigns *resolving* these to the plan. This section is where they
are made visible in the spec rather than surfacing only inside the plan, so that a gap is a
recorded fact rather than something you have to notice or patch yourself.

The boundary is the same one that governs the whole spec: a gap belongs here only if closing it
changes nothing in fields 1, 2, or 3. Anything that would change them comes back to you as a
question.

- <gap — where the code stops short, with file and line — what it blocks, or "none">

---

## Not your job

The examiner is silent on these and the plan resolves them. Do not fill them in:

- Namespacing, slugs, cache layout, key composition, output paths
- Which repo parameter carries which of your requirements
- What already exists on disk and should be reused rather than regenerated
- Forward-pass counts, wall clock, disk, GPU memory feasibility
- Verification checks and manifest design
- Whether existing code produces the structure you asked for, or needs new code to

Two standing rules mean you never write them either. Extractions are additive: nothing already
on disk becomes unreachable or unreproducible. If that is impossible, the plan stops and asks
rather than proceeding. Write a line here only to *override* those defaults.

The plan comes back to you in one case: when resolving something above would change fields 1,
2, or 3.
