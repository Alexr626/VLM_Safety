#!/usr/bin/env python3
"""Overnight crash babysitter for steering visual reasoning validation grids.

Watches the two model driver processes. On a true crash (driver exits before
the finished banner), re-launches that model's driver with the same RUN_DATE /
CHAIR_CAP so ``--skip_if_exists`` resumes from unfinished cells.

OOM policy (Alex 2026-07-30): per-sample OOMs that the runner already
continues past are ignored. If a process *dies* with CUDA/OOM evidence in the
log and the sibling model is still running, wait for the sibling to finish
before restarting — restarting into the same concurrent footprint would likely
OOM again.

Does not interpret metrics. Does not patch code. Caps restarts per model.
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

MODELS = {
    "llava": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "model_short": "llava-1.5-7b-hf",
        "log_key": "llava_log",
        "pid_key": "llava_pid",
        "default_log": "logs/steering_visual_reasoning_validation_llava_{run_date}.log",
        "max_pixels": None,
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_short": "qwen2.5-vl-7b-instruct",
        "log_key": "qwen_log",
        "pid_key": "qwen_pid",
        "default_log": "logs/steering_visual_reasoning_validation_qwen25_{run_date}.log",
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
    """Best-effort: bash driver whose argv contains the model id."""
    try:
        out = subprocess.check_output(["ps", "-ef"], text=True)
    except Exception:
        return None
    needle = f"run_steering_visual_reasoning_validation.sh {model_id}"
    for line in out.splitlines():
        if "grep" in line:
            continue
        if needle in line and "bash" in line:
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def _launch_driver(
    *,
    model_id: str,
    log_path: Path,
    chair_cap: int,
    run_date: str,
    max_pixels: Optional[int],
) -> int:
    env = os.environ.copy()
    env["HF_HOME"] = env.get("HF_HOME", "/data/romanus/huggingface")
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONUNBUFFERED"] = "1"
    env["CHAIR_CAP"] = str(chair_cap)
    env["RUN_DATE"] = run_date
    if max_pixels is not None:
        env["MAX_PIXELS"] = str(max_pixels)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as logf:
        logf.write(f"\n\n===== babysitter relaunch at {_now()} CHAIR_CAP={chair_cap} =====\n")
        logf.flush()
        proc = subprocess.Popen(
            [
                "bash",
                "evaluation/run_scripts/run_steering_visual_reasoning_validation.sh",
                model_id,
            ],
            cwd=str(_PROJECT_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_date", default="2026-07-30")
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument("--poll_sec", type=int, default=60)
    p.add_argument("--max_restarts_per_model", type=int, default=8)
    p.add_argument(
        "--wait_for_grids_sec",
        type=int,
        default=3600,
        help="How long to wait for orchestrator to record both grid PIDs.",
    )
    p.add_argument(
        "--oom_wait_sibling_sec",
        type=int,
        default=72 * 3600,
        help="Max wait for sibling to finish before OOM-restart.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(_PROJECT_ROOT)
    status_dir = (
        Path(args.output_dir)
        / args.run_date
        / "_analysis_steering_visual_reasoning_validation"
    )
    orch_path = status_dir / "overnight_orchestrator_status.json"
    decision_path = status_dir / "chair_cap_probe_decision.json"
    babysit_path = status_dir / "babysitter_status.json"

    state = {
        "started_at": _now(),
        "updated_at": _now(),
        "run_date": args.run_date,
        "stage": "waiting_for_grids",
        "chair_cap": None,
        "models": {
            name: {
                "model_id": cfg["model_id"],
                "pid": None,
                "log": None,
                "finished": False,
                "restarts": 0,
                "last_event": None,
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

    event("babysitter started; waiting for grid PIDs")

    # Wait until orchestrator has launched grids (or logs/pids appear).
    deadline = time.time() + args.wait_for_grids_sec
    while time.time() < deadline:
        orch = _read_json(orch_path)
        decision = _read_json(decision_path)
        chair_cap = orch.get("chair_cap") or decision.get("chair_max_new_tokens_chosen")
        if chair_cap is not None:
            state["chair_cap"] = int(chair_cap)

        ready = True
        for name, cfg in MODELS.items():
            pid = orch.get(cfg["pid_key"])
            log = orch.get(cfg["log_key"]) or cfg["default_log"].format(
                run_date=args.run_date
            )
            if pid is None:
                pid = _find_driver_pid(cfg["model_id"])
            state["models"][name]["pid"] = int(pid) if pid else None
            state["models"][name]["log"] = log
            if not pid or not _pid_alive(int(pid)):
                # Qwen launches 5 min after LLaVA — tolerate missing qwen briefly
                if name == "qwen" and orch.get("llava_pid"):
                    ready = False
                elif name == "llava":
                    ready = False
                else:
                    ready = False
        if orch.get("stage") in ("grids_running", "grids_launched") and orch.get(
            "llava_pid"
        ):
            # Accept once LLaVA is up; keep polling for Qwen pid separately.
            if state["models"]["llava"]["pid"] and _pid_alive(
                state["models"]["llava"]["pid"]
            ):
                break
        if (
            state["models"]["llava"]["pid"]
            and _pid_alive(state["models"]["llava"]["pid"])
            and state["models"]["qwen"]["pid"]
            and _pid_alive(state["models"]["qwen"]["pid"])
        ):
            break
        _write_json(babysit_path, state)
        time.sleep(min(30, args.poll_sec))

    if state["chair_cap"] is None:
        # Fallback if status never wrote chair_cap but decision exists.
        decision = _read_json(decision_path)
        if "chair_max_new_tokens_chosen" in decision:
            state["chair_cap"] = int(decision["chair_max_new_tokens_chosen"])
        else:
            state["chair_cap"] = 256

    state["stage"] = "watching"
    event(
        f"watching grids chair_cap={state['chair_cap']} "
        f"llava_pid={state['models']['llava']['pid']} "
        f"qwen_pid={state['models']['qwen']['pid']}"
    )

    try:
        while True:
            orch = _read_json(orch_path)
            # Refresh pids from orch if present and alive.
            for name, cfg in MODELS.items():
                m = state["models"][name]
                if m["finished"]:
                    continue
                log_path = Path(m["log"] or cfg["default_log"].format(run_date=args.run_date))
                m["log"] = str(log_path)

                pid = m["pid"]
                if not _pid_alive(pid):
                    # Try rediscovery (orchestrator may have updated, or restart).
                    found = orch.get(cfg["pid_key"]) or _find_driver_pid(cfg["model_id"])
                    if found and _pid_alive(int(found)):
                        m["pid"] = int(found)
                        pid = m["pid"]

                if _log_finished(log_path):
                    if not m["finished"]:
                        m["finished"] = True
                        m["last_event"] = "finished"
                        event(f"{name}: finished banner seen in {log_path}")
                    continue

                if _pid_alive(pid):
                    continue

                # Dead without finished banner → crash.
                m["last_event"] = "crashed"
                oom = _log_suggests_oom(log_path)
                event(
                    f"{name}: driver dead without finished banner "
                    f"(oom_evidence={oom}) restarts={m['restarts']}"
                )

                if m["restarts"] >= args.max_restarts_per_model:
                    event(f"{name}: restart budget exhausted; leaving down")
                    state["stage"] = "failed_restart_budget"
                    _write_json(babysit_path, state)
                    continue

                sibling = "qwen" if name == "llava" else "llava"
                sib = state["models"][sibling]
                if oom and not sib["finished"] and _pid_alive(sib.get("pid")):
                    event(
                        f"{name}: OOM crash while {sibling} still running — "
                        f"waiting for sibling before restart"
                    )
                    wait_deadline = time.time() + args.oom_wait_sibling_sec
                    while time.time() < wait_deadline:
                        if _log_finished(Path(sib["log"] or "")) or not _pid_alive(
                            sib.get("pid")
                        ):
                            # Refresh sibling finished flag.
                            if sib["log"] and _log_finished(Path(sib["log"])):
                                sib["finished"] = True
                            break
                        # Also allow sibling rediscovery death.
                        found = _find_driver_pid(MODELS[sibling]["model_id"])
                        if found:
                            sib["pid"] = found
                        time.sleep(args.poll_sec)
                        state["updated_at"] = _now()
                        _write_json(babysit_path, state)
                    event(f"{name}: sibling wait done; restarting alone/after sibling")

                # Relaunch
                m["restarts"] += 1
                new_pid = _launch_driver(
                    model_id=cfg["model_id"],
                    log_path=log_path,
                    chair_cap=int(state["chair_cap"]),
                    run_date=args.run_date,
                    max_pixels=cfg["max_pixels"],
                )
                m["pid"] = new_pid
                m["last_event"] = "relaunched"
                # Persist pid into orch status if possible (additive).
                orch2 = _read_json(orch_path)
                orch2[cfg["pid_key"]] = new_pid
                orch2[cfg["log_key"]] = str(log_path)
                orch2["babysitter_last_relaunch"] = {
                    "model": name,
                    "pid": new_pid,
                    "at": _now(),
                    "restart_n": m["restarts"],
                    "oom_evidence": oom,
                }
                _write_json(orch_path, orch2)
                event(f"{name}: relaunched pid={new_pid} restart#{m['restarts']}")

            if all(state["models"][n]["finished"] for n in MODELS):
                state["stage"] = "both_grids_finished"
                event("both grids finished; babysitter exiting")
                return 0

            if state["stage"] == "failed_restart_budget" and all(
                state["models"][n]["finished"]
                or state["models"][n]["restarts"] >= args.max_restarts_per_model
                for n in MODELS
            ):
                event("giving up: restart budgets exhausted for unfinished models")
                return 1

            state["updated_at"] = _now()
            _write_json(babysit_path, state)
            time.sleep(args.poll_sec)
    except KeyboardInterrupt:
        event("interrupted")
        return 130


if __name__ == "__main__":
    # Ignore SIGHUP so SSH drop does not kill the babysitter.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    raise SystemExit(main())
