# Which of the 4 plan sanity checks were rerun on POPE-120? (2026-07-20)

For the POPE-120 assertive protocol, only Stage-0 baseline dumps + flip-set computation ran. No knockout cells were run on this protocol.

## The four plan checks

| # | Check | Rerun on POPE-120? | Result |
|---|--------|--------------------|--------|
| 1 | Tokenization invariant (all items × conditions × models) | **No** | Not evaluated. Asserted inside `run_knockout.py` before the window sweep; knockout not started for POPE-120. |
| 2 | First-token decidedness ≥ ~99% on baseline manifests | **Yes** | **Pass** for both models. LLaVA 0.9917 (n_parseable=360); Qwen 1.0000 (n_parseable=351). |
| 3 | Full-depth anchor (≥ ~80% agreement with neutral on flip set) | **No** | Not evaluated on POPE-120. Requires all-layers clause knockout. Old AMBER+POPE-30 LLaVA result was ~51% (failed); that is a different dataset/protocol. |
| 4 | Eager-mode zero-attention assertion | **No** | Not re-run on POPE-120. Lives in knockout verify + `tests/test_attention_knockout.py`. |

## Related Stage-0 gates that *were* run (not in the four above)

- Flip-set size ≥ 15: LLaVA **fail** (union=6); Qwen **pass** (union=24).
- Prefix length filler_b vs assertive: **pass** (8=8 both models).
- filler_b vs neutral answer agreement (extra): LLaVA 0.975; Qwen 1.000.
