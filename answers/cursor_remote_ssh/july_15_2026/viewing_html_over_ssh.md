# Viewing HTML files over Cursor Remote SSH (lambdab2)

**Question:** Easiest way to open/view HTML files while SSH'd into Cursor on the lambdab2 repo copy from a laptop.

## Recommended (easiest reliable workflow)

Serve the directory with a tiny HTTP server on lambdab2; Cursor forwards the port to the laptop, then open in the local browser.

```bash
cd /home/romanus/dev/vlm_hallucination_mitigation_summer_2026/evaluation/results/2026-07-13/_samples
python -m http.server 8765
```

Then either:

1. Accept Cursor's "Port 8765 is available" / open the **Ports** panel and open the forwarded URL, or
2. On the laptop browser go to `http://localhost:8765/...path.../file.html`

Example for an AMBER sample report:

`http://localhost:8765/llava-1.5-7b-hf/amber/additive_mlp/all/500/llava-1.5-7b-hf_amber_additive_mlp_all_nd500.html`

Stop the server with Ctrl+C when done.

## One-off alternative (no server)

Right-click the `.html` file in the Cursor explorer → **Download...** → open the downloaded file in the laptop's browser. Fine for a single file; awkward for browsing many sample reports.

## Optional IDE preview

Install Microsoft's **Live Preview** extension in the remote window, then right-click the HTML → **Show Preview**. Useful if you prefer an in-editor pane; the HTTP-server + localhost approach is usually simpler and more reliable for static result HTML.

## Note

Opening a remote `file://` path in the laptop browser does not work — the browser runs on the laptop and cannot see lambdab2's filesystem. Port forwarding (or Download) is required.
