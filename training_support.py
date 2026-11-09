"""Small runtime helpers used directly by the original imdb_*.py scripts.

This is not a model wrapper: each original script still builds and trains its
own network.  These helpers only add atomic checkpoints, metrics files, real
PyCharm screenshots, and early-stopping bookkeeping.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from managed.screenshot import capture_console

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
RESULT = ROOT / "result"


def setup(model_name: str):
    base = EXPERIMENTS / model_name
    dirs = {
        "base": base,
        "logs": base / "logs",
        "checkpoints": base / "checkpoints",
        "screenshots": base / "screenshots",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"imdb_{model_name}_{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    handler = logging.FileHandler(dirs["logs"] / f"{model_name}_{run_id}.log", mode="w", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    dirs["run_id"] = run_id
    dirs["logger"] = logger
    return dirs


def atomic_torch_save(payload, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_latest(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def try_resume(model, optimizer, latest_path: Path, config: dict):
    checkpoint = load_latest(latest_path)
    if checkpoint is None:
        return None
    old_config = dict(checkpoint.get("config") or {})
    new_config = dict(config)
    # Allow increasing/reducing the epoch budget and early-stop patience
    # without discarding completed-epoch weights.
    for key in ("num_epochs", "max_epochs", "patience"):
        old_config.pop(key, None)
        new_config.pop(key, None)
    if old_config != new_config:
        return None
    model.load_state_dict(checkpoint["model_state"])
    if checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def save_runtime(model, optimizer, model_name, config, history, epoch,
                 best_val_loss, best_val_acc, best_epoch, dirs, is_best: bool):
    base = dirs["base"]
    payload = {
        "format_version": 1,
        "model_name": model_name,
        "run_id": dirs["run_id"],
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "config": config,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_torch_save(payload, dirs["checkpoints"] / f"epoch_{epoch:03d}.pt")
    atomic_torch_save(payload, dirs["checkpoints"] / "latest.pt")
    if is_best:
        atomic_torch_save({
            "model_state": model.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "config": config,
        }, dirs["checkpoints"] / "best_model.pt")
    frame = pd.DataFrame(history)
    frame.to_csv(base / "metrics.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(RESULT / f"{model_name}_training_metrics.csv", index=False, encoding="utf-8-sig")


def screenshot(model_name: str, suffix: str, label: str, dirs):
    if os.environ.get("IMDB_CAPTURE_MODE", "").lower() == "none":
        dirs["logger"].info("screenshot skipped (IMDB_CAPTURE_MODE=none); no image fabricated")
        return
    try:
        path = capture_console(dirs["screenshots"] / f"{model_name}_{suffix}.png", label=label)
        if path is None:
            dirs["logger"].warning("real PyCharm screenshot unavailable: %s", suffix)
        else:
            dirs["logger"].info("screenshot saved: %s", path)
    except Exception as exc:
        dirs["logger"].warning("screenshot failed (%s): %s", suffix, exc)


def write_summary(model_name, display_name, config, history, best_epoch,
                  best_val_loss, best_val_acc, early_stopped, dirs,
                  prediction_file, test_metrics=None):
    payload = {
        "model_name": model_name,
        "display_name": display_name,
        "run_id": dirs["run_id"],
        "status": "completed",
        "config": config,
        "max_epochs": config.get("num_epochs"),
        "actual_epochs": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "final_val_acc": history[-1]["val_acc"] if history else None,
        "total_train_seconds": sum(float(row.get("runtime", 0.0)) for row in history),
        "early_stopping": bool(early_stopped),
        "history": history,
        "prediction_file": str(prediction_file),
        "checkpoint_best": str(dirs["checkpoints"] / "best_model.pt"),
        "checkpoint_latest": str(dirs["checkpoints"] / "latest.pt"),
        "test_metrics": test_metrics or {},
    }
    (dirs["base"] / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_prediction(preds, test_frame, model_name: str):
    RESULT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"id": test_frame["id"], "sentiment": preds})
    path = RESULT / f"{model_name}.csv"
    frame.to_csv(path, index=False, quoting=3)
    return path


class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0

    def update(self, epoch, val_loss):
        improved = val_loss < self.best - self.min_delta or self.best_epoch == 0
        if improved:
            self.best = val_loss
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience
