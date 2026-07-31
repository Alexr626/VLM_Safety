# Confirmation: textual VTI extraction site = post-residual (layer)

**Date:** 2026-06-28

**Yes.** Offline textual direction extraction uses **post-residual stream** activations — the per-layer `outputs.hidden_states` tuple from a full `forward_vl` / model forward — not isolated MLP submodule outputs.

In this repo: `evaluation/interventions/vti/directions.py` → `get_hiddenstates()` reads `hidden_states[layer][0, -1, :]` from `wrapper.forward_vl(...)`.

In the authors' reference: `VTI/vti_utils/icv_utils.py` → `get_hiddenstates()` uses `model(..., output_hidden_states=True).hidden_states` and takes the last token per layer — same semantics.

Inference `hook_site` (`mlp` vs `layer`) is a separate choice at eval time; it does not change where directions are fit.
