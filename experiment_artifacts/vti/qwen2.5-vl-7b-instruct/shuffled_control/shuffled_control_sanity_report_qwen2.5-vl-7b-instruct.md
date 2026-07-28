# Shuffled-control sanity report — `qwen2.5-vl-7b-instruct`

Date: 2026-07-22
Control for: `demosv2_9a44f4af_all_nd200_s42_r2_prefix`
Derangement: `experiment_artifacts/vti/shuffled_control_image_derangement_nd200_s1234.json` (seed=1234)
Direction dir: `experiment_artifacts/vti/qwen2.5-vl-7b-instruct/shuffled_control/all_nd200/`
Act cache: `experiment_artifacts/vti/qwen2.5-vl-7b-instruct/shuffled_control/_act_cache/`

## Gating checks

| Check | Observed | Expect | Pass |
|---|---|---|---|
| Derangement validity (fixed points) | 0 | 0 | PASS |
| Captions unchanged | 200 of 200 | 200 of 200 | PASS |
| Same demos / order (positional id mismatches) | 0 | 0 | PASS |
| Direction array shape | [28, 3584] | [28, 3584] | PASS |
| No cache reuse (forward passes executed) | 400 | 400 | PASS |

**All gating checks pass:** yes

## Non-gating: per-layer magnitude (L2)

Shuffled-control `direction_layer_norms` alongside the deployed direction's norms (includes embedding row at index 0; same layout as deployed `metadata.json`).

| Layer | Deployed norm | Shuffled-control norm |
|---:|---:|---:|
| 0 | 2.3042025532049593e-07 | 2.0007475143302145e-07 |
| 1 | 0.16701559722423553 | 0.16859105229377747 |
| 2 | 0.2584514319896698 | 0.25898846983909607 |
| 3 | 0.28893670439720154 | 0.29116833209991455 |
| 4 | 0.3396773934364319 | 0.34427985548973083 |
| 5 | 0.41352760791778564 | 0.4093784987926483 |
| 6 | 0.6106672883033752 | 0.558964729309082 |
| 7 | 0.6630118489265442 | 0.5998844504356384 |
| 8 | 0.758107602596283 | 0.673991322517395 |
| 9 | 0.8898553848266602 | 0.785962700843811 |
| 10 | 1.2251393795013428 | 1.0822432041168213 |
| 11 | 1.2625982761383057 | 1.1000183820724487 |
| 12 | 1.4948725700378418 | 1.2847144603729248 |
| 13 | 1.5787676572799683 | 1.3095684051513672 |
| 14 | 1.6952762603759766 | 1.429743766784668 |
| 15 | 2.1534998416900635 | 1.7527401447296143 |
| 16 | 2.4456353187561035 | 1.9395253658294678 |
| 17 | 2.715235948562622 | 1.979090690612793 |
| 18 | 3.6382975578308105 | 2.1866607666015625 |
| 19 | 9.424370765686035 | 2.9809181690216064 |
| 20 | 13.492837905883789 | 3.404026508331299 |
| 21 | 19.939393997192383 | 4.161954879760742 |
| 22 | 26.17023468017578 | 5.143865585327148 |
| 23 | 34.710330963134766 | 6.173048496246338 |
| 24 | 42.70281219482422 | 7.500365734100342 |
| 25 | 50.95137023925781 | 8.637744903564453 |
| 26 | 57.91352081298828 | 10.493996620178223 |
| 27 | 65.53219604492188 | 11.827380180358887 |
| 28 | 36.309783935546875 | 5.7023468017578125 |

All shuffled norms finite: True
Deployed direction shape (decoder layers): [28, 3584]
Cache files after extraction: 400

