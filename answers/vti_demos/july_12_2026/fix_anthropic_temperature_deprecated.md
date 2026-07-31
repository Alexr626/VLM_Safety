# Fix: Anthropic opus-4-8 rejects `temperature`

**Error:** `anthropic.BadRequestError: temperature is deprecated for this model` on stage 2 (`claude-opus-4-8`). Stage 1 (sonnet) had accepted `temperature=0.0`.

**Fix:** `MLLMClient` no longer sends `temperature` to Anthropic unless explicitly set. OpenAI still defaults to `0.0`.

**Resume:** Stage 1 already has 4 passes — re-run from stage 2.
