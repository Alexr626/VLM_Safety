# Did the 5-candidate mock dry-run call a real MLLM?

**No.** Stages 1–4 were invoked with `--provider mock`. `MLLMClient` short-circuits in that mode and returns canned schema-valid JSON with `model="mock"` and zero token counts — no Anthropic/OpenAI HTTP calls, no image upload.

Stage 0 and stage 5 are deterministic (COCO annotations / assemble) and never call an MLLM.

Call logs under `data/vti/v2/calls/stage{1,2,3,4}.jsonl` from that run record `model: mock` and `input_tokens`/`output_tokens` = 0.
