# Do POPE runs need the same pin-file fix as CHAIR?

Date: 2026-07-31

No for this grid. `run_steering_visual_reasoning_validation.sh` calls POPE with
`--limit 200` and **no** `--subset_ids_file`. It loads questions from tracked
files already in `git archive` / the working-tree tarball:

- `data/pope/output/coco/coco_pope_{random,popular,adversarial}.json`

AMBER uses `data/amber/pinned_amber_disc_450.json` (tracked + in tarball).
CHAIR uses `data/chair/pinned_chair_500.json` (was gitignored; now overlaid).

POPE also needs COCO **val2014** images on NFS (from `setup_vlm`); that is
separate from pin JSON.
