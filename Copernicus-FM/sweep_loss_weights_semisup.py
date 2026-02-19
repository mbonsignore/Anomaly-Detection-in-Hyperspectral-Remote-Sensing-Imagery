#!/usr/bin/env python3
"""
Grid search over loss weights for semi-supervised fine-tuning of Copernicus-FM on FLOGA.

This script repeatedly calls:
    train_finetune_copernicus_floga_semisupervised.py

for a grid of:
    - lambda-center  ∈ {0.1, 0.3, 0.5}
    - lambda-distill ∈ {0.3, 0.5}
    - lambda-anom    ∈ {1.0, 3.0}
    - anom-margin    = FIXED_ANOM_MARGIN (see below, default 1.3)

For each run:
    - redirects stdout+stderr into logs/<run_name>.log
    - parses ALL lines containing "PatchCore_auroc=" and takes the MAX AUROC over epochs
    - appends one row to sweep_results_semisup.csv with:
        * config
        * best_val_patchcore_auroc
        * epoch_best
        * process return_code

After each run:
    - deletes the checkpoint "copernicus_floga_semisupervised_best.pt" to avoid
      filling the disk (only logs + CSV remain).

-------------------------------------------------------
Example usage with tmux to keep the sweep running
even if your SSH connection / laptop dies:

1) On the remote machine (after SSH):
       tmux new -s floga_sweep

2) Inside tmux, launch the sweep (and optionally tee logs):
       conda activate copernicusfm
       export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
       python3 sweep_loss_weights_semisup.py | tee logs_semisup_sweep/main.log

3) Detach from tmux (leaving it running):
       Ctrl + B, then D

4) Later, reattach to see progress:
       tmux attach -t floga_sweep

5) List or kill sessions if needed:
       tmux ls
       tmux kill-session -t floga_sweep
-------------------------------------------------------
"""

import csv
import subprocess
from typing import Optional
from pathlib import Path
from typing import Dict, Any, List, Tuple

# =======================
#  GLOBAL PARAMETERS
# =======================

# Training script (your existing fine-tuning script)
TRAIN_SCRIPT = "train_finetune_copernicus_floga_semisupervised.py"

# Dataset / checkpoint paths (adapt if you move things)
DATA_ROOT = "../FLOGA"
SPLITS_ROOT = "../FLOGA/finetuning data/floga_splits"
TEACHER_CKPT = "ckpt/CopernicusFM_ViT_base_varlang_e100.pth"

# Directory where the training script saves the BEST checkpoint.
# The sweep script will delete this after every run.
OUT_CKPT_DIR = Path("ckpt")

# In this sweep we use only 5 epochs per configuration
EPOCHS = 5
BATCH_SIZE = 4
MIN_BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4

# Use the teacher cache you already computed
USE_TEACHER_CACHE = True

# Fixed anomaly margin for this sweep
FIXED_ANOM_MARGIN = 1.3

# Directories for logs and CSV results
LOG_DIR = Path("logs_semisup_sweep")
RESULTS_CSV = Path("sweep_results_semisup.csv")

LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_CKPT_DIR.mkdir(parents=True, exist_ok=True)


# =======================
#  GRID OF CONFIGURATIONS
# =======================

lambda_center_list = [0.1, 0.3, 0.5]
lambda_distill_list = [0.3, 0.5]
lambda_anom_list = [1.0, 3.0]

# This produces 3 * 2 * 2 = 12 configurations
def build_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for lc in lambda_center_list:
        for ld in lambda_distill_list:
            for la in lambda_anom_list:
                configs.append(
                    {
                        "lambda_center": lc,
                        "lambda_distill": ld,
                        "lambda_anom": la,
                        "anom_margin": FIXED_ANOM_MARGIN,
                    }
                )
    return configs


CONFIGS = build_configs()


# =======================
#  UTILITIES
# =======================

def make_run_name(cfg: Dict[str, Any]) -> str:
    """Compact name for a configuration, used for log filenames."""
    def f(x: float) -> str:
        return str(x).replace(".", "p")
    return (
        f"lc{f(cfg['lambda_center'])}_"
        f"ld{f(cfg['lambda_distill'])}_"
        f"la{f(cfg['lambda_anom'])}_"
        f"m{f(cfg['anom_margin'])}"
    )


def build_command(cfg: Dict[str, Any]) -> List[str]:
    """Build the python command to run for a given configuration."""
    cmd = [
        "python3", TRAIN_SCRIPT,
        "--data-root", DATA_ROOT,
        "--splits-root", SPLITS_ROOT,
        "--teacher-ckpt", TEACHER_CKPT,
        "--out-ckpt-dir", str(OUT_CKPT_DIR),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--min-batch-size", str(MIN_BATCH_SIZE),
        "--lr", str(LR),
        "--weight-decay", str(WEIGHT_DECAY),
        "--lambda-center", str(cfg["lambda_center"]),
        "--lambda-distill", str(cfg["lambda_distill"]),
        "--lambda-anom", str(cfg["lambda_anom"]),
        "--anom-margin", str(cfg["anom_margin"]),
    ]
    if USE_TEACHER_CACHE:
        cmd.append("--load-teacher-cache")
    return cmd


def parse_best_auroc_from_log(
        log_path: Path
    ) -> Tuple[Optional[float], Optional[int]]:
    """
    Parse ALL lines with 'PatchCore_auroc=' and return:
        (best_auroc, epoch_best)

    We assume lines like:
        [Epoch 3/5] loss=... PatchCore_auroc=0.7421 best_f1=...
    """
    if not log_path.is_file():
        return None, None

    best_auroc = None
    best_epoch = None

    with log_path.open("r") as f:
        for line in f:
            if "PatchCore_auroc=" not in line:
                continue
            try:
                # Extract epoch number
                # e.g. "[Epoch 3/5] loss=..."
                if "[Epoch" in line and "/" in line:
                    # part between "[Epoch " and "/"
                    prefix = line.split("[Epoch", 1)[1]
                    epoch_str = prefix.split("/", 1)[0].strip()
                    epoch_num = int(epoch_str)
                else:
                    epoch_num = -1  # if not parseable, use -1

                # Extract AUROC
                after = line.split("PatchCore_auroc=")[1]
                auroc_str = after.split()[0]
                auroc_val = float(auroc_str)
            except Exception:
                continue

            if (best_auroc is None) or (auroc_val > best_auroc):
                best_auroc = auroc_val
                best_epoch = epoch_num

    return best_auroc, best_epoch


def append_result_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append one row (dict) to the CSV, creating the header if needed."""
    file_exists = csv_path.is_file()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def delete_checkpoint_if_exists() -> None:
    """
    Delete 'copernicus_floga_semisupervised_best.pt' in OUT_CKPT_DIR
    if it exists, to avoid accumulating models on disk.
    """
    ckpt_path = OUT_CKPT_DIR / "copernicus_floga_semisupervised_best.pt"
    if ckpt_path.is_file():
        try:
            ckpt_path.unlink()
            print(f"[INFO] Deleted checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"[WARN] Could not delete checkpoint {ckpt_path}: {e}")


# =======================
#  MAIN SWEEP
# =======================

def main() -> None:
    print(f"[INFO] Starting semi-supervised loss-weight sweep.")
    print(f"[INFO] Number of configs: {len(CONFIGS)}")
    print(f"[INFO] Logs dir: {LOG_DIR}")
    print(f"[INFO] Results CSV: {RESULTS_CSV}")
    print(f"[INFO] Epochs per run: {EPOCHS}")
    print(f"[INFO] Fixed anomaly margin: {FIXED_ANOM_MARGIN}")

    for idx, cfg in enumerate(CONFIGS, start=1):
        run_name = make_run_name(cfg)
        log_path = LOG_DIR / f"{run_name}.log"

        print(f"\n========== RUN {idx}/{len(CONFIGS)}: {run_name} ==========")
        print(f"[INFO] Logging to: {log_path}")

        cmd = build_command(cfg)
        print("[INFO] Command:", " ".join(cmd))

        # Launch training and write output to log
        with log_path.open("w") as log_f:
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )

        # After the run, try to delete the checkpoint to free space
        delete_checkpoint_if_exists()

        if proc.returncode != 0:
            print(f"[WARN] Run {run_name} exited with code {proc.returncode}")
            best_auroc, best_epoch = None, None
        else:
            best_auroc, best_epoch = parse_best_auroc_from_log(log_path)
            if best_auroc is None:
                print(f"[WARN] Could not parse AUROC from log {log_path}")
            else:
                if best_epoch is not None and best_epoch > 0:
                    print(f"[INFO] Best PatchCore AUROC: {best_auroc:.4f} at epoch {best_epoch}")
                else:
                    print(f"[INFO] Best PatchCore AUROC: {best_auroc:.4f} (epoch unknown)")

        # Save results to CSV
        row = {
            "run_name": run_name,
            "lambda_center": cfg["lambda_center"],
            "lambda_distill": cfg["lambda_distill"],
            "lambda_anom": cfg["lambda_anom"],
            "anom_margin": cfg["anom_margin"],
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "return_code": proc.returncode,
            "best_val_patchcore_auroc": best_auroc,
            "epoch_best": best_epoch,
        }
        append_result_row(RESULTS_CSV, row)

    print("\n[INFO] Sweep finished.")
    print(f"[INFO] Results in: {RESULTS_CSV}")


if __name__ == "__main__":
    main()