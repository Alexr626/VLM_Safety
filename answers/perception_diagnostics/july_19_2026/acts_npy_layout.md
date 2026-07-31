# How `acts/*.npy` are saved (perception dumps)

**Date:** 2026-07-19 (paths updated)

Each file under `…/dumps/<run>/<steering_setting>/acts/` is **one (item, prompt-condition) forward** under that setting’s intervention. It is **not** one layer per file.

- **Filename:** `{item_id}__{condition_id}.npy` (e.g. `amber_disc_00498__neutral.npy`)
- **Contents:** last **prefill** token hidden state at **every layer**, including embedding as row 0
- **Shape:** `(n_layers + 1, hidden_dim)` fp16  
  - Qwen2.5-VL-7B: `(29, 3584)` → 28 decoder layers + embed  
  - LLaVA-1.5-7B: `(33, 4096)` → 32 decoder layers + embed
- **Intervention:** applies to the whole setting directory (hooks on during capture). Example: `additive_layer_0.5` = additive × layer × β=0.5. Rotation × layer × 0.5 is `rotation_layer_0.5`.

Companion: `norms/{same_stem}.npy` is `(n_layers+1, seq_len)` per-position ℓ2 norms for the full prefill.
