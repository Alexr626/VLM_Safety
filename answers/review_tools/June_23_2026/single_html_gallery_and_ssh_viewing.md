# Single HTML gallery + viewing via Cursor SSH

## Single HTML file

`review_responses.py` and `sample_responses.py` now write **one self-contained gallery page** (images embedded as base64, table of contents with anchor links):

| Script | Default output |
|--------|----------------|
| `review_responses.py --html` | `evaluation/results/{run_date}/_review/gallery.html` |
| `sample_responses.py` | `evaluation/results/{run_date}/_samples/review.html` |

### `review_responses.py` — multiple ids, one file

```bash
python helper_scripts/review_responses.py \
  --ids chair_000000357659 chair_000000256343 \
  evaluation/results/2026-06-22/llava-1.5-7b-hf/chair/no_intervention/responses.json \
  --html --run-date 2026-06-22
```

Ids can also come from a positional id, `--ids`, or `--ids-file` (one id per line).

### `sample_responses.py` — batch run

```bash
python helper_scripts/sample_responses.py --run_date 2026-06-22
```

Produces `review.html` with all model × benchmark × sample panels (20 panels for the default 5 CHAIR + 5 AMBER × 2 models).

## Viewing rendered HTML through Cursor SSH (lambdab2)

`file://` URLs on the remote host do not open in your **local** browser over SSH. Three practical options:

### 1. Download and open locally (simplest)

In Cursor’s file explorer, right-click `review.html` → **Download**, then open the file in Chrome/Firefox on your laptop. Because images are base64-embedded, the file works fully offline.

### 2. Port-forward + local HTTP server (best in-browser experience)

On lambdab2:

```bash
cd evaluation/results/2026-06-22/_samples
python -m http.server 8765 --bind 127.0.0.1
```

In Cursor: **Ports** panel → **Forward a Port** → `8765` (or it may auto-forward). Open on your local machine:

`http://localhost:8765/review.html`

Bind to `127.0.0.1` only; do not expose `0.0.0.0` on a shared machine unless you intend to.

### 3. Cursor Simple Browser

With port 8765 forwarded, **Simple Browser: Show** (`Ctrl/Cmd+Shift+P`) and navigate to `http://localhost:8765/review.html`. This uses the forwarded tunnel like a normal local URL.

### What does not work well

- **`xdg-open` / `--open` on the remote** — opens a viewer on lambdab2’s display, not your laptop.
- **Opening `file:///home/romanus/...` in Simple Browser** — resolves on the remote filesystem context; unreliable over SSH. Prefer HTTP or download.
