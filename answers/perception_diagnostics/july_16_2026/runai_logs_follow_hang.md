# `runai … logs --follow` appears to hang

Often it’s **connected and waiting**, not dead — especially during long quiet steps (AMBER GDrive download, COCO train2014 ~13 GiB, HF snapshot). Those print little until a phase finishes.

## What to do

1. **Ctrl+C** is safe — it only stops your log viewer, not the job.
2. Snapshot instead of follow:
```bash
runai training standard logs pd-s2-stage -p nlm-mh --tail=100
```
3. Confirm the job is still alive:
```bash
runai workload list -p nlm-mh
```
`Running` + `1/1` GPU → still working.
4. Progress without logs (WinSCP NFS):
   - `vlm_hallucination/data/amber/` (zip / `images/`)
   - `vlm_hallucination/data/coco/train2014.zip` size growing
   - `hf_cache/` growth
5. Re-follow later when you expect chatter (direction extract prints more).

If status flips to **Failed** / **Completed**, `--follow` may hang looking for a live stream — use `--tail` / `--previous` instead.
