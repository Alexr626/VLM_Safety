# What is SDPA here? (2026-07-20)

**SDPA** = **Scaled Dot-Product Attention**, specifically PyTorch’s fused kernel
`torch.nn.functional.scaled_dot_product_attention`.

In this repo / experiment it means the **default attention backend** HuggingFace
uses for LLaVA and Qwen2.5-VL decoder layers when flash-attn is not installed
(our env intentionally has no flash-attn). The model computes attention via
SDPA instead of a hand-written “eager” loop that materializes the full
attention matrix.

That matters for knockout because:

- SDPA accepts a **4D additive attention mask** (what we inject with −∞ / `dtype.min` on blocked edges).
- SDPA does **not** return per-edge attention weights, so the eager-mode verify
  pass temporarily falls back to manual/eager attention with
  `output_attentions=True` to check that blocked edges are actually zero.
