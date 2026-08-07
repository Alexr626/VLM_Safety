#!/usr/bin/env python3
"""Overnight crash babysitter for the AMBER expanded steering-direction grid.

Watches the two model driver processes. On a true crash (driver exits before
the finished banner), re-launches that model's driver with the same RUN_DATE
so ``--skip_if_exists`` resumes from unfinished cells.

OOM policy: if a process dies with CUDA/OOM evidence and the sibling is still
running, wait for the sibling to finish before restarting.

Does not interpret metrics. Caps restarts per model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

FINISHED_RE = re.compile(r"=== grid process finished for .+ ===")
OOM_PATTERNS = (
    re.compile(r"CUDA out of memory", re.I),
    re.compile(r"torch\.cuda\.OutOfMemoryError", re.I),
    re.compile(r"OutOfMemoryError", re.I),
    re.compile(r"CUBLAS_STATUS_ALLOC_FAILED", re.I),
    re.compile(r"CUDA error: out of memory", re.I),
)

DRIVER = "evaluation/run_scripts/run_amber_expanded_steering_direction_grid.sh"

MODELS = {
    "llava": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "model_short": "llava-1.5-7b-hf",
        "default_log": "logs/amber_expanded_steering_direction_grid_llava_{run_date}.log",
        "max_pixels": None,
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_short": "qwen2.5-vl-7b-instruct",
        "default_log": "logs/amber_expanded_steering_direction_grid_qwen25_{run_date}.log",
        "max_pixels": 1003520,
    },
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _tail_text(path: Path, max_bytes: int = 120_000) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


def _log_finished(path: Path) -> bool:
    return bool(FINISHED_RE.search(_tail_text(path)))


def _log_suggests_oom(path: Path) -> bool:
    text = _tail_text(path)
    return any(p.search(text) for p in OOM_PATTERNS)


def _find_driver_pid(model_id: str) -> Optional[int]:
    try:
        out = subprocess.check_output(["ps", "-ef"], text=True)
    except Exception:
        return None
    needle = f"run_amber_expanded_steering_direction_grid.sh {model_id}"
    for line in out.splitlines():
        if "grep" in line:
            continue
        if needle in line and "bash" in line:
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def _count_complete_cells(output_dir: str, run_date: str, model_short: str) -> int:
    amber = Path(output_dir) / run_date / model_short / "amber"
    if not amber.is_dir():
        return 0
    n = 0
    for cell in amber.iterdir():
        if cell.is_dir() and (cell / "metric_summary.json").is_file():
            n += 1
    return n


def _launch_driver(
    *,
    model_id: str,
    log_path: Path,
    run_date: str,
    max_pixels: Optional[int],
) -> int:
    env = os.environ.copy()
    env["HF_HOME"] = env.get("HF_HOME", "/data/romanus/huggingface")
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONUNBUFFERED"] = "1"
    env["RUN_DATE"] = run_date
    if max_pixels is not None:
        env["MAX_PIXELS"] = str(max_pixels)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as logf:
        logf.write(f"\n\n===== babysitter relaunch at {_now()} =====\n")
        logf.flush()
        proc = subprocess.Popen(
            ["bash", DRIVER, model_id],
            cwd=str(_PROJECT_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_date", default="2026-08-05")
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument("--poll_sec", type=int, default=60)
    p.add_argument("--max_restarts_per_model", type=int, default=8)
    p.add_argument("--wait_for_grids_sec", type=int, default=900)
    p.add_argument("--oom_wait_sibling_sec", type=int, default=72 * 3600)
    p.add_argument("--llava_pid", type=int, default=None)
    p.add_argument("--qwen_pid", type=int, default=None)
    p.add_argument("--llava_log", type=str, default=None)
    p.add_argument("--qwen_log", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(_PROJECT_ROOT)
    status_dir = (
        Path(args.output_dir)
        / args.run_date
        / "_analysis_steering_vector_validation_continuation"
    )
    launch_path = status_dir / "launch_status.json"
    babysit_path = status_dir / "babysitter_status.json"

    state = {
        "started_at": _now(),
        "updated_at": _now(),
        "run_date": args.run_date,
        "stage": "waiting_for_grids",
        "models": {
            name: {
                "model_id": cfg["model_id"],
                "model_short": cfg["model_short"],
                "pid": None,
                "log": None,
                "finished": False,
                "restarts": 0,
                "last_event": None,
                "n_complete_cells": 0,
            }
            for name, cfg in MODELS.items()
        },
        "events": [],
    }

    def event(msg: str) -> None:
        state["events"].append({"at": _now(), "msg": msg})
        state["events"] = state["events"][-200:]
        state["updated_at"] = _now()
        print(f"[babysitter {_now()}] {msg}", flush=True)
        _write_json(babysit_path, state)

    # Seed from CLI / launch_status.
    launch = _read_json(launch_path)
    seeds = {
        "llava": {
            "pid": args.llava_pid or launch.get("llava_pid"),
            "log": args.llava_log or launch.get("llava_log"),
        },
        "qwen": {
            "pid": args.qwen_pid or launch.get("qwen_pid"),
            "log": args.qwen_log or launch.get("qwen_log"),
        },
    }
    for name, cfg in MODELS.items():
        log = seeds[name]["log"] or cfg["default_log"].format(run_date=args.run_date)
        pid = seeds[name]["pid"] or _find_driver_pid(cfg["model_id"])
        state["models"][name]["log"] = log
        state["models"][name]["pid"] = int(pid) if pid else None

    event("babysitter started; waiting for grid PIDs")

    deadline = time.time() + args.wait_for_grids_sec
    while time.time() < deadline:
        for name, cfg in MODELS.items():
            m = state["models"][name]
            if not m["pid"] or not _pid_alive(m["pid"]):
                found = _find_driver_pid(cfg["model_id"])
                if found:
                    m["pid"] = found
        if (
            state["models"]["llava"]["pid"]
            and _pid_alive(state["models"]["llava"]["pid"])
        ):
            # Accept once LLaVA is up; Qwen may still be in the 5-min stagger.
            break
        _write_json(babysit_path, state)
        time.sleep(min(30, args.poll_sec))

    state["stage"] = "watching"
    event(
        f"watching grids "
        f"llava_pid={state['models']['llava']['pid']} "
        f"qwen_pid={state['models']['qwen']['pid']}"
    )

    stop = {"flag": False}

    def _handle_signal(signum, _frame):
        stop["flag"] = True
        event(f"received signal {signum}; shutting down babysitter")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while not stop["flag"]:
            all_done = True
            for name, cfg in MODELS.items():
                m = state["models"][name]
                log_path = Path(
                    m["log"] or cfg["default_log"].format(run_date=args.run_date)
                )
                m["log"] = str(log_path)
                m["n_complete_cells"] = _count_complete_cells(
                    args.output_dir, args.run_date, cfg["model_short"]
                )

                if m["finished"]:
                    continue
                all_done = False

                pid = m["pid"]
                if not _pid_alive(pid):
                    found = _find_driver_pid(cfg["model_id"])
                    if found and _pid_alive(found):
                        m["pid"] = found
                        pid = found

                if _log_finished(log_path):
                    m["finished"] = True
                    m["last_event"] = "finished"
                    event(
                        f"{name}: finished banner seen "
                        f"(cells={m['n_complete_cells']}/19)"
                    )
                    continue

                if _pid_alive(pid):
                    continue

                # Process dead without finished banner → crash.
                oom = _log_suggests_oom(log_path)
                sibling = "qwen" if name == "llava" else "llava"
                sibling_m = state["models"][sibling]
                sibling_alive = (
                    not sibling_m["finished"] and _pid_alive(sibling_m["pid"])
                )

                if m["restarts"] >= args.max_restarts_per_model:
                    m["last_event"] = "max_restarts"
                    event(f"{name}: crashed; max restarts reached — giving up")
                    m["finished"] = True
                    continue

                if oom and sibling_alive:
                    event(
                        f"{name}: OOM crash while sibling alive; "
                        f"waiting up to {args.oom_wait_sibling_sec}s"
                    )
                    wait_deadline = time.time() + args.oom_wait_sibling_sec
                    while time.time() < wait_deadline and not stop["flag"]:
                        if sibling_m["finished"] or _log_finished(
                            Path(sibling_m["log"] or "")
                        ):
                            sibling_m["finished"] = True
                            break
                        if not _pid_alive(sibling_m["pid"]):
                            # Sibling also dead; proceed to restart.
                            break
                        time.sleep(args.poll_sec)
                        _write_json(babysit_path, state)

                m["restarts"] += 1
                new_pid = _launch_driver(
                    model_id=cfg["model_id"],
                    log_path=log_path,
                    run_date=args.run_date,
                    max_pixels=cfg["max_pixels"],
                )
                m["pid"] = new_pid
                m["last_event"] = "restarted"
                event(
                    f"{name}: restarted pid={new_pid} "
                    f"(restart #{m['restarts']}, oom={oom})"
                )

            if all_done or (
                state["models"]["llava"]["finished"]
                and state["models"]["qwen"]["finished"]
            ):
                state["stage"] = "done"
                event("both models finished")
                break

            _write_json(babysit_path, state)
            time.sleep(args.poll_sec)
    finally:
        state["updated_at"] = _now()
        _write_json(babysit_path, state)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
