#!/usr/bin/env python3
"""
Submit training job to the first available GPU partition via Slurm.

This script cycles through GPU partitions (gpu1..gpu6), submits
the training job, and waits briefly for a node allocation. If no
allocation occurs, it falls back to gpu1 and leaves the job pending.
"""

import subprocess
import time
import os
from pathlib import Path

# --- 설정 ---
BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "src" / "train" / "train_regression.py"
USER = "won0316"
CONDA_ENV = "toxcast_env"
PYTHON_BIN = Path(os.getenv("CONDA_PREFIX", f"/home1/{USER}/anaconda3/envs/{CONDA_ENV}")) / "bin" / "python"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# GPU 수 (gres gpu:<count>)
GPU_COUNT = 1
# Slurm 대기 상태 체크 반복 횟수
CHECK_ITER = 10
# 체크 간격(초)
CHECK_SLEEP = 1


def submit_job(partition: str) -> str | None:
    """Submit the training job to `partition` and return job_id if running."""
    wrap_cmd = (
        f"bash -lc 'source ~/.bashrc && conda activate {CONDA_ENV} && "
        f"{PYTHON_BIN} {SCRIPT_PATH}'"
    )
    cmd = [
        "sbatch",
        "--parsable",
        f"-p{partition}",
        f"--gres=gpu:{GPU_COUNT}",
        f"--output={LOG_DIR}/{partition}_%j.out",
        "--wrap",
        wrap_cmd,
    ]
    job_id = subprocess.check_output(cmd, text=True).strip()

    for _ in range(CHECK_ITER):
        time.sleep(CHECK_SLEEP)
        try:
            status = subprocess.check_output(
                ["squeue", "-u", USER, "-j", job_id, "-h", "-o", "%T %R"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            continue
        if not status:
            continue
        state, nodelist = status.split(maxsplit=1)
        if state != "PD" and nodelist and nodelist != "(null)":
            return job_id

    # 할당 안 됐으면 취소
    subprocess.run(["scancel", job_id])
    return None


def main() -> None:
    partitions = [f"gpu{i}" for i in range(6, 7)]
    for part in partitions:
        print(f"Trying partition {part}...")
        job_id = submit_job(part)
        if job_id:
            print(f"Job {job_id} is running on {part}")
            return

    print("No immediate allocation found; falling back to gpu1 and leaving pending")
    fallback_id = submit_job("gpu1")
    if fallback_id:
        print(f"Fallback job {fallback_id} submitted to gpu1 and is pending.")
    else:
        print("Failed to submit fallback job.")


if __name__ == "__main__":
    main()
