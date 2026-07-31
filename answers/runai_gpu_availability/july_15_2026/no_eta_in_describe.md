# Do RunAI `describe` outputs show remaining runtime?

**Date:** 2026-07-15  
**Jobs:** `hal44` (training), `llama3-70b8` (inference)  
**Owner:** chun-nam.yu@nokia-bell-labs.com

## Short answer

**No.** Neither `runai training describe` nor `runai inference describe` reports expected duration, remaining time, progress %, or a TTL/deadline. Ask the submitter, or inspect the job’s script/logs.

## What the descriptions *do* show (useful for ETA intuition)

### `hal44` (Training) — finite job, but no end time in metadata

| Field | Value | Implication |
|-------|--------|-------------|
| Created At | 2026-07-15T13:49:22Z | Started ~today (UTC); still Running |
| Command | `.../logistic5_llama3_gepa_old.sh` | Batch script — will finish when the script exits |
| Preemptible | true | Could be preempted; not a long-lived service |
| GPUs | 2 × H100 (`h100-pool`) | |

No wall-clock limit or progress field. Duration is whatever that shell script does.

### `llama3-70b8` (Inference) — likely indefinite until stopped

| Field | Value | Implication |
|-------|--------|-------------|
| Created At | 2026-07-14T14:28:55Z | Running since ~yesterday |
| Type / Category | Inference / Deploy | Serving deployment, not a one-shot train |
| Model args | vLLM Llama-3.1-70B, `--tensor-parallel-size 2` | Persistent API server |
| Min/Max replicas | 1 / 1 | Keeps one replica up |
| Preemptible | false | Priority very-high (150) | Harder to displace; won’t “complete” on its own |

This one typically occupies 2 GPUs until Chun-Nam stops/deletes it — not until a training epoch finishes.

## How to learn more without guessing

1. Ask Chun-Nam (email above) for ETA / whether either can be paused.
2. Optional: `runai training logs hal44 -p nlm-mh` for script progress (if the script prints it).
3. Inference jobs rarely “finish”; only stop frees those 2 GPUs.

## Security note (from the pasted describe)

`llama3-70b8` describe printed an `HF_TOKEN` in Environment Variables. Prefer rotating that token and avoiding pasting full `describe` output into shared chats.
