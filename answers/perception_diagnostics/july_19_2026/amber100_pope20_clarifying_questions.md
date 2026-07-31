# Clarifying questions — AMBER-100 + POPE-20 dumps

**Date:** 2026-07-19

Before building pins / leading-prompt JSONLs / steered dumps.

## Confirmed from current AMBER-25

25 items = 5×{existence×no, attribute×yes, attribute×no, relation×yes, relation×no}. No existence×yes in AMBER.

---

## Questions

### 1) AMBER-100 stratum sizes
Keep all 25 AMBER-25 IDs and add 75 more. Preferred balance?

- **A (recommended):** 20 per stratum × 5 strata (existence×no, attr×yes/no, rel×yes/no) = 100  
- **B:** something else (specify counts)

### 2) Models + steering settings for the new dumps
Existing dumps: LLaVA all 13 settings; Qwen2.5 default subset (11). For AMBER-100 and POPE-20, run:

- **A:** both models, same settings as each model’s existing dump  
- **B:** LLaVA only / Qwen2.5 only  
- **C:** baseline only first (no steering) for a cheaper gate  

### 3) POPE-20 selection
Existence × gold=yes, n=20. Prefer:

- **A:** 20 from `random` split of pinned POPE-600 (seed fixed)  
- **B:** stratified across random/popular/adversarial (e.g. 7+7+6)  
- **C:** other source / hand-picked list  

### 4) Generation budget
Match existing dumps (LLaVA 512, Qwen2.5 128) or use 128 for both?

### 5) Dump layout
Separate run tags, e.g. `amber100_all_steering_settings` and `pope20_existence_yes_…`, each with the five leading-prompt conditions — OK?
