# answers/ — the shared scratchpad

Agents write here. This is where a derivation, a mechanism write-up, or a factual lookup
goes so that it survives the session and can be read with a markdown renderer instead of a
terminal.

It is also the one place in this repo where the belief/artifact line has to be held by
convention rather than by permissions, so the rule is stated here and repeated in every
agent file.

## What belongs here

**Concepts.** Derivations, worked synthetic examples, statistical and geometric background,
method explanations. Tutor output lives here. This is the highest-value use of the directory:
a derivation you had to chase once should not have to be chased again.

**Mechanism.** How a piece of the codebase works. What a function computes, where a hook
attaches, how a mask is built, what a config field controls. The
`attention_knockout/july_20_2026/` notes are the model for this — they are documentation of
what the code does, verified by reading it.

**Facts about runs.** What was run, with which command, where the outputs landed, what the
headline numbers were, what deviated. Facts, not readings.

**Literature summaries.** What a paper claims, measures, and demonstrates.

## What does not belong here

**Interpretations of your results.** What a number means, what a pattern indicates, which
explanation the evidence favours, what the finding is. Those live in
`analysis/<run_id>_reading.md`, they are written by Alex, and no agent writes them.

`answers/interpretations/` is denied to agents in `settings.json` specifically so that the
obvious workaround has a wall in front of it.

**Recommendations about what to run next.** Ranking, prioritising, or proposing experiments.
Those are design decisions and they go in `designs/`.

**Appraisal of literature.** Whether a method is genuinely novel, whether a claim is
supported, whether it is worth building on. Summary yes; verdict no.

The distinction in one line: **this directory holds things that would be true regardless of
what your experiment found.** If a note would have to be rewritten because a result came out
differently, it is a reading and it does not go here.

## Layout

```
answers/<topic>/<month>_<day>_<year>/<descriptive_name>.md
```

Topic directories are reused where one already fits; a new topic gets a new directory. Dates
use the existing convention, e.g. `july_28_2026`. Check the date before writing rather than
assuming it.

Filenames are self-describing and readable without the directory path in hand.
`noise_floor_and_disattenuation.md`, not `answer_3.md`.

## Format

Every note opens with a title line, the date, and one sentence on what question it answers.

Mathematics is written in LaTeX between `$` or `$$` delimiters. These files are meant to be
read in a markdown preview pane, not in a terminal — that is the point of writing them to
disk. Terminal responses summarise; the file carries the derivation.

Where a note depends on a fact read from the codebase, cite the file and line so the note can
be re-verified when the code moves.
