# POPE Existence Subsets Rebuild: POPE-30-yes and POPE-30-no

Date: 2026-07-22
Status: approved by Alex in chat; ready for implementation
Scope: dataset files only. CPU-only, minutes. No GPU, no runs. Plan B (`pope_yes_no_windowed_steering_control_plan_2026-07-22.md`) depends on this plan completing and its verification output being reported.

## Motivation (recorded)

The current `pinned_pope_existence_yes_30.json` selected "first 10 gold=yes per split," but POPE's gold=yes questions are identical across the random/popular/adversarial split files (the splits differ only in negative-object sampling). The pin therefore contains 10 unique (image, question) pairs, each triplicated — confirmed on disk and confirmed in run results (every count in the run summaries is a multiple of 3). Going forward the project focuses on existence (object detection) sycophancy only. Decisions from chat: rebuild in place per Alex's convention (pins under `data/pope/` are the living record); stratified negatives (10 per split) for the no-set; both sets image-matched.

## Superseded pin archive (verbatim ids of the OLD pinned_pope_existence_yes_30.json)

The completed runs under `pope30_windowed_steering` and `pope30_existence_yes_baseline` used these ids; the file content is being replaced, so the old id list is archived here for provenance:

random: pope_random_00000, 00002, 00004, 00006, 00008, 00010, 00012, 00014, 00016, 00018
popular: pope_popular_00000, 00002, 00004, 00006, 00008, 00010, 00012, 00014, 00016, 00018
adversarial: pope_adversarial_00000, 00002, 00004, 00006, 00008, 00010, 00012, 00014, 00016, 00018

(10 unique inputs; the popular and adversarial blocks duplicate the random block's images and questions.)

## Task 1 — POPE-30-yes: edit `data/pope/pinned_pope_existence_yes_30.json` IN PLACE

- Selection rule: first 30 gold=yes items from the RANDOM split, in `data/pope/pinned_eval_ids.json` order (the random split is the canonical source since yes-questions are identical across splits).
- New file content: `benchmark`, `n: 30`, `generated` date, `selection` (the rule above, written out), `supersedes` note ("2026-07-19 triplicated version; ids archived in pope_yes_no_30_dataset_rebuild_plan_2026-07-22.md"), `ids` (all `pope_random_*`), `items` (per item: id, image_id, question text, gold) for self-documentation, and `content_hash` (sha256 over the canonical items list).
- Inline asserts (fail loudly): n == 30; all gold == "yes"; all (image_id, question) pairs unique; count of distinct images reported (expected roughly 10-12 at ~3 yes-questions per image — if the source does not look like that, stop and report rather than proceed).
- By construction the first 10 items coincide with the 10 unique inputs of the superseded pin.

## Task 2 — POPE-30-no: create `data/pope/pinned_pope_existence_no_30.json`

- Selection rule: 10 gold=no items from EACH split (random, popular, adversarial), restricted to images in the POPE-30-yes image set, taken in `pinned_eval_ids.json` order within each split. If a (image_id, questioned-object) pair collides with one already selected from another split, skip it and take the next candidate in order.
- Same schema as Task 1 (including per-item records, split field, content_hash).
- Inline asserts: n == 30; all gold == "no"; all (image_id, question) unique; 10 per split; image set is a subset of the POPE-30-yes image set. Report the per-split object lists in the verification output so Alex can eyeball negative difficulty.

## Task 3 — Augmented prompt files

- Build `data/pope/augmented_pope30_yes.jsonl` and `data/pope/augmented_pope30_no.jsonl` from the two pins using the existing augment builder with the same template set as before (`leading_clauses_v1+filler_clauses_v1`), same schema as the existing augmented files.
- Leave `data/pope/augmented_pope30.jsonl` on disk untouched; it is superseded and must not be referenced by any new run.

## Verification output (report to Alex before Plan B launches)

A short printed table: for each of the two pins — n, gold composition, unique (image, question) count, distinct image count, image-overlap between sets, per-split counts (no-set), content hashes; plus first 3 example questions per set.

## Open questions

1. Raw POPE source location and `pinned_eval_ids.json` are referenced by the old pin's selection note and assumed present under `data/pope/`; if the eval-ids file does not contain gold=no orderings per split, fall back to the raw split files' native order and record the deviation in the pin's `selection` field.
