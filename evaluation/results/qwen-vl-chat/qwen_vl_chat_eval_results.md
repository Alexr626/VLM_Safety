# Qwen-VL-Chat Evaluation Results

Results from merged 10-shard evaluation. ASR is attack success rate; lower is safer.

Classifier caveat: these numbers use the current keyword classifier. It may misclassify harmful responses that mention words like `illegal` or `harmful` as refusals, so final reporting should use the planned LLM judge or a repaired keyword rule.

## Overall ASR

| Benchmark | N | Vanilla | Adashield_s | Change |
|---|---:|---:|---:|---:|
| MM-SafetyBench | 5,040 | 48.3% | 7.9% | -40.4 pp |
| FigStep | 500 | 65.0% | 0.0% | -65.0 pp |
| MSSBench | 1,200 | 64.6% | 61.9% | -2.7 pp |
| Weighted total | 6,740 | 52.4% | 16.9% | -35.5 pp |

## MM-SafetyBench By Image Type

| Image type | N | Vanilla | Adashield_s | Change |
|---|---:|---:|---:|---:|
| OCR | 1,680 | 46.9% | 8.3% | -38.6 pp |
| SD | 1,680 | 21.2% | 9.5% | -11.7 pp |
| SD_TYPO | 1,680 | 76.8% | 5.9% | -70.9 pp |

## MSSBench By Safety Label

| Label | N | Vanilla | Adashield_s | Change |
|---|---:|---:|---:|---:|
| SSS | 600 | 67.5% | 65.2% | -2.3 pp |
| SSU | 600 | 61.7% | 58.7% | -3.0 pp |

## MM-SafetyBench By Scenario

| Scenario | N | Vanilla | Adashield_s | Change |
|---|---:|---:|---:|---:|
| Illegal Activity | 291 | 29.2% | 2.1% | -27.1 pp |
| HateSpeech | 489 | 44.0% | 3.5% | -40.5 pp |
| Malware Generation | 132 | 43.9% | 3.0% | -40.9 pp |
| Physical Harm | 432 | 41.9% | 3.0% | -38.9 pp |
| Economic Harm | 366 | 59.3% | 19.4% | -39.9 pp |
| Fraud | 462 | 41.8% | 3.9% | -37.9 pp |
| Pornography | 327 | 46.5% | 6.1% | -40.4 pp |
| Political Lobbying | 459 | 64.7% | 16.6% | -48.1 pp |
| Privacy Violence | 417 | 46.0% | 7.9% | -38.1 pp |
| Legal Opinion | 390 | 56.9% | 13.8% | -43.1 pp |
| Financial Advice | 501 | 60.7% | 10.0% | -50.7 pp |
| Health Consultation | 327 | 42.2% | 5.5% | -36.7 pp |
| Government Decision | 447 | 40.3% | 4.0% | -36.2 pp |
