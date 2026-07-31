# Text-based steering (VTI) beta sweep — where results live & manager summary

## What was tested (plain language)

We applied **text-based activation steering** (VTI, from the Visual and Textual Intervention paper) to several vision-language models. The steering nudges the model’s internal text decoder activations along a learned “anti-hallucination” direction, extracted from paired clean vs. hallucinated demo captions.

The main knob is **β (beta)** — steering strength from 0.1 (weak) to 1.0 (strong). The VTI paper’s recommended setting is **β = 0.4**.

We tested three steering geometries at two hook sites:

| Short name | Meaning |
|---|---|
| **additive @ MLP** | Add the direction at the MLP sub-block output |
| **additive @ layer** | Add the direction at the full decoder-layer output |
| **uniform rotation @ MLP** | Rotate activations in the plane spanned by the direction |

**Not included in the main β grids:** `uniform_rotation @ layer` — this variant often collapses outputs (empty or very short captions) and was studied separately as a diagnostic sweep.

**Models:** LLaVA-1.5-7B, Qwen-VL-Chat, Qwen2-VL-7B, Qwen2.5-VL-7B

**Benchmarks:**
- **POPE** — yes/no object-presence questions (accuracy is the headline metric)
- **CHAIR** — open-ended image captions; **CHAIRᵢ** (hallucination rate, lower is better) is the headline metric
- **AMBER** — discriminative yes/no questions about existence, attributes, and relations

---

## Where the result files are

All results follow this layout:

```
evaluation/results/<run_date>/<model_short>/<benchmark>/<intervention>__b<beta>/
    metric_summary.json    ← headline metrics (start here)
    responses.json         ← per-sample model outputs
```

### Run 1 — POPE β grid (most complete, 4 models)

- **Date folder:** `evaluation/results/2026-06-19/`
- **Driver script:** `evaluation/run_scripts/run_vti_pope_beta_grid.sh`
- **Scale:** 200 samples per POPE split (random / popular / adversarial) × 10 β values × 3 methods × 4 models = **360 cells**
- **Example path:**
  `evaluation/results/2026-06-19/llava-1.5-7b-hf/pope_random/vti_textual_additive_mlp__b0.4/metric_summary.json`
- **Baselines:** `.../no_intervention/metric_summary.json` (no `__b` suffix)
- **Run log:** `evaluation/results/_logs/beta_grid_20260619_110137.log`

### Run 2 — CHAIR + AMBER β grid (hallucination-focused benchmarks)

- **Date folder:** `evaluation/results/2026-06-22/`
- **Driver script:** `evaluation/run_scripts/run_exp1_repro_grid.sh` (via `run_beta_grid_local.sh`)
- **Scale:** pinned subsets — CHAIR n=500, AMBER n=450; β ∈ {0.1…0.9} (β=1.0 not run here)
- **Example path:**
  `evaluation/results/2026-06-22/llava-1.5-7b-hf/chair/vti_textual_additive_mlp__b0.4/metric_summary.json`
- **Pre-built summary (facts only, full tables):**
  `evaluation/results/2026-06-22/_diagnostic_summary_chair_amber.md`
- **Machine-readable twin:**
  `evaluation/results/2026-06-22/_diagnostic_summary_chair_amber.json`

**Completion note:** LLaVA has the full β grid on both CHAIR and AMBER. Qwen2.5-VL has only partial Exp-1 cells (through β≈0.4).

### Run 3 — Rotation-strength diagnostic (layer-site uniform rotation)

Separate fine-grained β sweep for `uniform_rotation @ layer` (the variant that can collapse outputs):

- **Sweep JSONs:** `evaluation/results/2026-06-22/<model>/{chair|amber}_rotation_strength/sweep_uniform_rotation_layer_n{N}.json`
- **Also summarized in:** `_diagnostic_summary_chair_amber.md` § Experiment 2

### Run 4 — Small local smoke (July 2026, not a full grid)

- **Date folder:** `evaluation/results/2026-07-02/`
- Only a handful of textual cells (n=25 AMBER, n=? CHAIR) for layer-site methods — useful for debugging, not for reporting headline numbers.

---

## Manager-friendly summary of findings

### POPE (object presence, n=200/split, June 19)

Split-averaged accuracy across random/popular/adversarial splits:

| Model | Baseline | Best method & β | Best acc | Δ vs baseline | Acc @ β=0.4 (paper) |
|---|---|---|---|---|---|
| LLaVA-1.5-7B | 84.7% | additive @ layer, β=0.9 | 88.5% | +3.8 pts | 85.3% |
| Qwen-VL-Chat | 85.7% | additive @ layer, β=0.8 | 86.3% | +0.7 pts | 85.8% |
| Qwen2-VL-7B | 88.0% | additive @ MLP/layer, β=0.2 | 88.3% | +0.3 pts | 88.0% |
| Qwen2.5-VL-7B | 86.3% | uniform_rotation @ MLP, β=0.7 | 88.2% | +1.8 pts | 87.2% |

**Takeaway:** On POPE, steering gives modest gains. Stronger β (0.7–1.0) helps LLaVA most; newer Qwen models are already strong baselines and move less. The paper’s β=0.4 is near-optimal but not always the peak.

### CHAIR (caption hallucination, n=500, June 22) — LLaVA only (full grid)

Baseline: **CHAIRᵢ = 17.0%** (hallucinated objects per caption).

| Method | Best β for ↓CHAIRᵢ | CHAIRᵢ at best β | Δ vs baseline |
|---|---|---|---|
| additive @ layer | 0.9 | 16.1% | −0.9 pts |
| additive @ MLP | 0.3 | 16.5% | −0.5 pts |
| uniform_rotation @ MLP | 0.8 | 15.1% | −2.0 pts |

**Takeaway:** On CHAIR, textual steering produces only small changes in hallucination rate (~0.5–2 percentage points). No dramatic mitigation at MLP/layer additive sites.

### AMBER discriminative (n=450, June 22)

**LLaVA** (baseline acc 76.7%):
- Additive methods: acc rises to ~78% at β≥0.8, mainly by saying “no” more often (yes_ratio drops from 40% → 34%).
- Uniform rotation @ MLP: acc peaks ~77.8% at β=0.4–0.5; higher β hurts.

**Qwen2.5-VL** (baseline acc 64.2%, but 82/450 responses unparsed):
- Uniform rotation @ MLP @ β=0.4: acc **69.1%** (+4.9 pts), largest gain seen in this grid.
- Additive methods: essentially flat.

### Layer-site rotation diagnostic (Experiment 2)

`uniform_rotation @ layer` behaves very differently from MLP-site steering:
- **LLaVA CHAIR:** caption length collapses above β≈0.5 (avg length 270 → 7 chars at β=0.6); CHAIRᵢ goes to 0% only because captions are nearly empty.
- **LLaVA AMBER:** accuracy drops as β increases; model becomes overly conservative (yes_ratio 40% → 3%).
- **Qwen2.5 AMBER:** interesting trade-off — acc rises to **85.6%** at β=0.35, then collapses at β=0.6 (induced hallucinations h+=127).

**Takeaway:** Layer-site rotation is not a viable production intervention; it is mainly a diagnostic for understanding where steering breaks.

---

## What to send your manager

**Quickest read:** `evaluation/results/2026-06-22/_diagnostic_summary_chair_amber.md` (full tables, CHAIR+AMBER)

**For POPE numbers:** browse `evaluation/results/2026-06-19/<model>/pope_*/vti_textual_*__b*/metric_summary.json`

**One-line status:** Textual β sweeps are complete for POPE (4 models, full grid) and mostly complete for CHAIR/AMBER (LLaVA full; Qwen2.5 partial). Gains on presence benchmarks (POPE) are modest; gains on caption hallucination (CHAIR) are small; layer-site rotation is unstable. Vision-encoder steering (α) is a separate, newer line of work under `evaluation/results/2026-07-02/`.
