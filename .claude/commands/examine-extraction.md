---
description: Have the examiner check an extraction spec against the ways extracted data silently stops answering the question. Usage /examine-extraction <ext_id>
---

Extraction id: $1

Read `extractions/$1_extraction.md`. If it does not exist or is an unfilled template, say so and
stop.

Delegate to the examiner subagent under the constraints below.

## What this examination is for

One thing: **will this data still answer the question when Alex gets to it.** Not whether the
code is right, not whether the spec is well organised, not whether the plan will be efficient.
A bad extraction that is merely inefficient costs a rerun. A bad extraction that quietly cannot
support its downstream comparison costs weeks, because it produces artifacts that look correct,
get used, and confound everything after them.

Every check below is a way that happens.

## The checks

Read fields 1 and 2 against field 3. Consult the code for facts about what the extraction path
actually stores; do not consult it to evaluate whether the code currently implements what Alex
asked for. Missing code is a coding task.

**1. What varies between arms.** For each comparison in field 3, does exactly one thing differ
between the things compared? Sets that are both disjoint and different sizes vary identity and
size together, and no later analysis separates them.

**2. Pairing and clustering.** If the design is paired — same image, two captions — is the
pairing preserved in what gets stored, or does it become two unpaired piles? And are items
clustered such that effective n falls below nominal n: images reused across items, shared object
classes, one template generating many items. Clustering changes every reliability and interval
computed later, and it cannot be recovered if the grouping is not recorded.

**3. Granularity of what is stored.** Read the extraction path and state what it actually
writes — which token position, which layers, pooled or per-position, which forward pass. Then
ask whether that granularity supports field 3. Storing pooled or sliced activations forecloses
every finer analysis, silently. Note that position interacts with length: two conditions of
different lengths read at "the last token" are read at different positions.

**4. Overlap and leakage.** Between the sets themselves, and between the extraction items and
any benchmark Alex will later evaluate on.

Two further checks, raised **only if flagrantly violated**, in one line each, no question:

**5. Nuisance variables riding the intended contrast** — length, lexical overlap, answer
distribution, source, template. At extraction time the only question is whether the nuisance is
recorded, since you can only condition on what was stored.

**6. Provenance of the labels** — what produced the ground truth, and whether that contaminates
the comparison.

Those two are usually design questions surfacing early. If field 3 is thin because the
experiment is not formed yet, they are not answerable and you should not press them.

## Scope

Silence on: namespacing, slugs, cache layout, paths, cost, GPU memory, verification design, spec
organisation, whether existing code implements the requested structure, and whether a sentence
sits in the right field. These are the plan's, and raising them is what makes an examination
unusable.

Where field 3 is too thin to check fields 1 and 2 against, say that in one line rather than
inventing the missing intent. That is a real finding: it means the extraction is waiting on the
experiment being formed, and Alex should hear it plainly.

## Output

Hard cap: **three-five questions.** Fewer is better. Zero is a legitimate result and should be stated
in one line.

**Blocking** — questions, each with the fact behind it and a file:line citation where one
exists.

**Corrections** — factual errors as statements, never as questions. Omit if none.

Then stop. No summary, no file list, no note on what else you considered.
