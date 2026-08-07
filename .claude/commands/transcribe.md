---
description: Transcribe a scanned handwritten PDF or image to markdown. Usage /transcribe <path in learning/scans/>
---

<!-- Last updated: 2026-08-05 -->

Path: $ARGUMENTS

Read the file directly and transcribe it to markdown. Do not summarize, correct, or reorganize —
the goal is a faithful transcription of what's on the page, not an improved version of it.

Math notation goes in LaTeX, `$` for inline and `$$` for display, matching the source as closely
as your reading of it allows.

**Where you cannot read something with confidence — a symbol, a subscript, a crossed-out line —
mark it `[illegible]` or `[unclear: your best guess]` rather than silently filling it in.** This
file exists to be a ground-truth check against agent-generated derivations elsewhere in this
workflow. A transcription that quietly guesses defeats that purpose worse than one with visible
gaps.

Write the result to `learning/scans/transcripts/<same base filename>.md`. If a transcript
already exists at that path, ask before overwriting rather than replacing it silently.