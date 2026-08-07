#!/usr/bin/env python3
"""Cut over RunAI AMBER Qwen grid: when β=0.5 finishes, launch β=0.9 cell triple.

Polls the live ``amber-qwen-beta-triple`` pod until all β=0.5 steered cells have
``responses.json`` (or the β=0.5 worker log shows WORKER_OK). Then:

  1. delete ``amber-qwen-beta-triple`` (stops the lone β=0.9 worker; checkpoint kept)
  2. submit ``amber-qwen-beta09-triple`` with the cell-partitioned launcher,
     patching the launcher onto NFS via quote-safe ``echo b64 | base64 -d``

Does not interpret metrics. Idempotent if the cutover job already exists/Running.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "helper_scripts/runai/run_amber_expanded_qwen_beta09_cell_triple_one_h100.sh"
DEFAULT_LOG = REPO / "logs" / "babysit_amber_qwen_cutover_beta09_triple.log"

IMG = "blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1"
NFS = (
    "path=/volume1/airl-datalake,"
    "server=gpustorage-1.cloud.bell-labs.com,"
    "mountpath=/home/datalake,readwrite"
)
SRC_JOB = "amber-qwen-beta-triple"
DST_JOB = "amber-qwen-beta09-triple"
PROJECT = "nlm-mh"
RUN_DATE = "2026-08-05"
MODEL = "qwen2.5-vl-7b-instruct"

B05_CELLS = [
    f"vti_textual_additive_mlp__b0.5__dall__nd500__meandiff__{w}"
    for w in ("layers_all", "layers_5_14", "layers_15_24")
] + [
    f"vti_textual_additive_mlp__b0.5__dall__nd500__pc1_plus_mean__{w}"
    for w in ("layers_all", "layers_5_14", "layers_15_24")
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(path: Path, msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


def _runai(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    home = Path.home()
    env["PATH"] = f"{home}/.runai/bin:" + env.get("PATH", "")
    return subprocess.run(
        ["runai", *args],
        capture_output=True,
        text=True,
        check=check,
        env=env,
    )


def _workload_status(name: str) -> str | None:
    r = _runai("workload", "list", "-p", PROJECT)
    for line in r.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == name and len(parts) >= 4:
            return parts[3]
    return None


def _exec(job: str, remote_bash: str) -> subprocess.CompletedProcess:
    return _runai(
        "training",
        "exec",
        job,
        "-p",
        PROJECT,
        "--",
        "bash",
        "-c",
        remote_bash,
    )


def _beta05_done_via_exec(job: str) -> tuple[bool, str]:
    """Return (done, detail) by inspecting NFS result dirs inside the pod."""
    # Embed cell names as a Python literal; remote script is not an f-string body for quotes.
    cells_lit = repr(B05_CELLS)
    script = f"""
python3 - <<'PY'
from pathlib import Path
import json, subprocess
root = Path("/home/datalake/romanus/vlm_hallucination/evaluation/results/{RUN_DATE}/{MODEL}/amber")
cells = {cells_lit}
done, prog, todo = [], [], []
for name in cells:
    d = root / name
    if (d / "responses.json").exists():
        done.append(name)
    elif (d / "responses.checkpoint.json").exists():
        p = d / "responses.checkpoint.json"
        try:
            data = json.loads(p.read_text())
            n = len(data) if isinstance(data, list) else len(data.get("responses", []))
        except Exception:
            n = "?"
        prog.append(f"{{name}}:{{n}}/1500")
    else:
        todo.append(name)
ps = subprocess.check_output(
    ["bash", "-lc", "ps -eo args | grep run_eval.py | grep -v grep || true"],
    text=True,
)
has_b05 = any("--beta 0.5" in ln or "--beta=0.5" in ln for ln in ps.splitlines())
# All six steered β=0.5 cells on disk => ready to cut over (process may still be exiting).
print("DONE_COUNT", len(done))
print("PROG", ";".join(prog) if prog else "-")
print("TODO", ";".join(todo) if todo else "-")
print("HAS_B05_EVAL", has_b05)
print("ALL_DONE", len(done) == len(cells))
PY
"""
    r = _exec(job, script)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 and "ALL_DONE" not in out:
        return False, f"exec_rc={r.returncode} out={out[-500:]}"
    all_done = any(line.strip() == "ALL_DONE True" for line in out.splitlines())
    detail_lines = [
        ln for ln in out.splitlines()
        if ln.startswith(("DONE_COUNT", "PROG", "TODO", "HAS_B05_EVAL", "ALL_DONE"))
    ]
    return all_done, " | ".join(detail_lines) if detail_lines else out[-300:]


def _wait_job_gone(name: str, log: Path, timeout_s: int = 300) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = _workload_status(name)
        if st is None:
            _log(log, f"{name} gone")
            return
        _log(log, f"waiting for {name} to disappear (status={st})")
        time.sleep(5)
    raise RuntimeError(f"timeout waiting for {name} delete")


def _build_inner_command() -> str:
    b64 = base64.b64encode(LAUNCHER.read_bytes()).decode("ascii")
    assert "'" not in b64
    tgt = (
        "/home/datalake/romanus/vlm_hallucination/helper_scripts/runai/"
        "run_amber_expanded_qwen_beta09_cell_triple_one_h100.sh"
    )
    lf = "/home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_bash_lf.py"
    inner = (
        f"echo {b64} | base64 -d > {tgt}"
        f" && chmod 755 {tgt}"
        " && echo LAUNCHER_PATCHED_BETA09"
        f" && grep -n meandiff-5-14 {tgt} | head -1"
        " && export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        f" && python3 {lf} {tgt}"
    )
    assert "'" not in inner
    return inner


def _submit_beta09(log: Path) -> None:
    st = _workload_status(DST_JOB)
    if st in {"Running", "Pending", "Creating", "ContainerCreating"}:
        _log(log, f"{DST_JOB} already {st}; skip submit")
        return
    if st is not None:
        _log(log, f"deleting stale {DST_JOB} status={st}")
        _runai("training", "delete", DST_JOB, "-p", PROJECT)
        _wait_job_gone(DST_JOB, log)

    inner = _build_inner_command()
    _log(log, f"submitting {DST_JOB} (inner_len={len(inner)})")
    r = _runai(
        "training",
        "submit",
        DST_JOB,
        "-p",
        PROJECT,
        "--nfs",
        NFS,
        "-i",
        IMG,
        "--gpu-devices-request",
        "1",
        "--node-pools",
        "h100-pool",
        "--command",
        "--",
        "bash",
        "-c",
        inner,
    )
    _log(log, f"submit_stdout: {(r.stdout or '').strip()}")
    _log(log, f"submit_stderr: {(r.stderr or '').strip()}")
    if r.returncode != 0:
        raise RuntimeError(f"submit failed rc={r.returncode}")


def cutover(log: Path) -> None:
    _log(log, f"cutover: deleting {SRC_JOB}")
    _runai("training", "delete", SRC_JOB, "-p", PROJECT)
    _wait_job_gone(SRC_JOB, log)
    time.sleep(3)
    _submit_beta09(log)
    # write marker
    marker = REPO / "logs" / "amber_qwen_cutover_beta09_done.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "cutover_at_utc": _now(),
                "src_job": SRC_JOB,
                "dst_job": DST_JOB,
                "launcher": str(LAUNCHER.relative_to(REPO)),
            },
            indent=2,
        )
        + "\n"
    )
    _log(log, f"wrote {marker}")
    _log(log, "CUTOVER_OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll_sec", type=int, default=60)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument(
        "--force_cutover_now",
        action="store_true",
        help="Skip wait; delete src and submit β=0.9 triple immediately",
    )
    ap.add_argument(
        "--check_once",
        action="store_true",
        help="Print β=0.5 status once and exit (no cutover)",
    )
    args = ap.parse_args()
    log: Path = args.log

    if not LAUNCHER.is_file():
        _log(log, f"missing launcher {LAUNCHER}")
        return 2

    marker = REPO / "logs" / "amber_qwen_cutover_beta09_done.json"
    if marker.is_file() and not args.force_cutover_now:
        _log(log, f"marker exists {marker}; nothing to do")
        return 0

    if args.force_cutover_now:
        cutover(log)
        return 0

    _log(log, f"watching {SRC_JOB} for β=0.5 completion; poll={args.poll_sec}s")
    while True:
        st = _workload_status(SRC_JOB)
        if st is None:
            _log(log, f"{SRC_JOB} missing; checking whether to submit {DST_JOB} alone")
            # If src vanished without cutover, still try submit if β=0.5 complete on... can't exec.
            # Fall through to submit only if dst not running — operator can --force.
            dst = _workload_status(DST_JOB)
            if dst is None:
                _log(log, "src gone and dst absent; submitting β=0.9 triple")
                _submit_beta09(log)
                _log(log, "CUTOVER_OK (src already gone)")
                return 0
            _log(log, f"dst already {dst}")
            return 0

        if st not in {"Running", "Pending", "Creating"}:
            _log(log, f"{SRC_JOB} status={st}; keep polling")
            time.sleep(args.poll_sec)
            continue

        done, detail = _beta05_done_via_exec(SRC_JOB)
        _log(log, f"status={st} beta05_done={done} {detail}")
        if args.check_once:
            return 0 if done else 1
        if done:
            cutover(log)
            return 0
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
