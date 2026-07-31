# Why A5 audit responses look blank

**Not an audit bug.** Blank `<pre></pre>` / empty `response` fields are copied from the source 2026-06-22 Qwen2.5-VL AMBER `responses.json` files.

Facts:
- 38/300 audit rows have empty responses (12.7%); all are Qwen2.5-VL.
- Spot-checks match source: `response == ""` in the original cell files.
- Every Qwen2.5 AMBER cell that day (including `no_intervention`) has the **same 65/450** empty IDs (symmetric diff vs baseline = 0).
- Those same IDs are **non-empty** on LLaVA baseline; images exist on disk.

**Root cause (Romanus, 2026-07-16):** that eval co-scheduled LLaVA and Qwen on the same GPU and hit OOMs; failed Qwen generations were recorded as empty strings. Fixed item set across all Qwen cells is consistent with per-item OOM failures rather than intervention-specific collapse.

So ~most of the gallery “failures” are empty generations from that Qwen2.5 run, not `_normalize_yes_no` failing on long answers. Of 49 `parsed=None`: 38 blank + 11 non-empty unparseable prose.
