"""GRU sentiment classifier — original teacher file with runtime protection.

The network definition, hidden size, embedding size, learning rate, batch size
and epoch count are unchanged.  Only checkpoint/resume, per-epoch metrics,
per-epoch PyCharm screenshots, and a graceful OOM stop were added so a long CPU
run cannot lose already completed epochs.
"""

import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch import optim
from tqdm import tqdm

os.environ.setdefault("IMDB_CAPTURE_MODE", "pycharm")
os.environ.setdefault("IMDB_CAPTURE_TITLE_SUBSTR", "imdb_gru.py")

from managed.screenshot import capture_console

torch.set_num_threads(12)

test = pd.read_csv(
    "./corpus/imdb/testData.tsv",
    header=0,
    delimiter="\t",
    quoting=3,
)

num_epochs = 10
embed_size = 300
num_hiddens = 120
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.001
grad_clip = 1.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = Path("experiments") / "gru"
ckpt_dir = base_dir / "checkpoints"
log_dir = base_dir / "logs"
shot_dir = base_dir / "screenshots"
for directory in (ckpt_dir, log_dir, shot_dir, Path("result"), Path("logs")):
    directory.mkdir(parents=True, exist_ok=True)
latest_path = ckpt_dir / "latest.pt"
best_path = ckpt_dir / "best_model.pt"

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logger.setLevel(logging.INFO)
logger.handlers.clear()
formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler(
    log_dir / f"gru_{run_id}.log", mode="w", encoding="utf-8"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

config = {
    "model_name": "gru",
    "embed_size": embed_size,
    "num_hiddens": num_hiddens,
    "num_layers": num_layers,
    "bidirectional": bidirectional,
    "batch_size": batch_size,
    "lr": lr,
    "num_epochs": num_epochs,
    "grad_clip": grad_clip,
    "optimizer": "Adam",
    "seed": 42,
    "device": str(device),
}

print("=" * 55)
print("当前运行设备：", device)
print("CUDA 是否可用：", use_gpu)
print("训练轮数：", num_epochs)
print("Batch Size：", batch_size)
print("学习率：", lr)
print("优化器：Adam")
print("=" * 55)

logger.info("running %s", " ".join(sys.argv))
logger.info("device: %s", device)
logger.info("run_id: %s", run_id)
logger.info("config: %s", json.dumps(config, ensure_ascii=False, sort_keys=True))


class SentimentNet(nn.Module):

    def __init__(
        self,
        embed_size,
        num_hiddens,
        num_layers,
        bidirectional,
        weight,
        labels,
        use_gpu,
        **kwargs
    ):

        super(SentimentNet, self).__init__(**kwargs)

        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False

        self.encoder = nn.GRU(
            input_size=embed_size,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=0
        )

        if bidirectional:
            self.decoder = nn.Linear(
                num_hiddens * 4,
                labels
            )
        else:
            self.decoder = nn.Linear(
                num_hiddens * 2,
                labels
            )

    def forward(self, inputs):

        embeddings = self.embedding(inputs)

        states, hidden = self.encoder(
            embeddings.permute(1, 0, 2)
        )

        encoding = torch.cat(
            [states[0], states[-1]],
            dim=1
        )

        outputs = self.decoder(encoding)

        return outputs


def atomic_save(payload, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def screenshot(name: str, label: str) -> None:
    try:
        path = capture_console(shot_dir / name, label=label)
        if path is None:
            logger.warning("真实 PyCharm 截图不可用：%s", name)
        else:
            logger.info("screenshot saved: %s", path)
    except Exception as exc:
        logger.warning("screenshot failed (%s): %s", name, exc)


if __name__ == "__main__":

    torch.manual_seed(config["seed"])
    logger.info("loading data...")

    pickle_file = os.path.join("pickle", "imdb_glove.pickle3")
    with open(pickle_file, "rb") as f:
        (
            train_features,
            train_labels,
            val_features,
            val_labels,
            test_features,
            weight,
            word_to_idx,
            idx_to_word,
            vocab
        ) = pickle.load(f)

    logger.info("data loaded!")
    train_labels = train_labels.long()
    val_labels = val_labels.long()

    net = SentimentNet(
        embed_size=embed_size,
        num_hiddens=num_hiddens,
        num_layers=num_layers,
        bidirectional=bidirectional,
        weight=weight,
        labels=labels,
        use_gpu=use_gpu
    )
    net.to(device)
    logger.info("model created!")
    logger.info("device: %s", device)
    logger.info("model parameters: %s", sum(p.numel() for p in net.parameters()))

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features)

    train_iter = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_iter = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_iter = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    history = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    start_epoch = 0

    if latest_path.exists():
        try:
            checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
            if checkpoint.get("config") == config:
                net.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                history = checkpoint.get("history", [])
                best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                best_val_acc = checkpoint.get("best_val_acc", 0.0)
                best_epoch = checkpoint.get("best_epoch", 0)
                start_epoch = checkpoint.get("epoch", 0)
                logger.info("RESUMED from %s at completed epoch %d", latest_path, start_epoch)
            else:
                logger.warning("checkpoint config mismatch; starting a new run")
        except Exception as exc:
            logger.error("checkpoint load failed (%s); starting a new run", exc)

    logger.info("start training...")
    screenshot("gru_start.png", "PyCharm GRU start")

    oom_stopped = False
    for epoch in range(start_epoch, num_epochs):
        start = time.time()
        net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        logger.info("Epoch %d/%d started | lr=%s | batch_size=%d | max_len=256",
                    epoch + 1, num_epochs, lr, batch_size)

        with tqdm(total=len(train_iter), desc="Epoch %d/%d" % (epoch + 1, num_epochs)) as pbar:
            for feature, label in train_iter:
                try:
                    feature = feature.to(device)
                    label = label.to(device).long()
                    optimizer.zero_grad(set_to_none=True)
                    score = net(feature)
                    loss = loss_function(score, label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
                    optimizer.step()
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() or "alloc_cpu" in str(exc).lower():
                        logger.error("OOM at epoch %d batch %d; last complete checkpoint is preserved",
                                     epoch + 1, pbar.n + 1)
                        oom_stopped = True
                        break
                    raise

                batch_num = label.size(0)
                train_loss += loss.item() * batch_num
                predictions = torch.argmax(score, dim=1)
                train_correct += (predictions == label).sum().item()
                train_total += batch_num
                pbar.set_postfix({
                    "loss": "%.4f" % (train_loss / train_total),
                    "acc": "%.4f" % (train_correct / train_total),
                })
                pbar.update(1)

        if oom_stopped:
            break

        net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_feature, val_label in val_iter:
                val_feature = val_feature.to(device)
                val_label = val_label.to(device).long()
                val_score = net(val_feature)
                loss = loss_function(val_score, val_label)
                batch_num = val_label.size(0)
                val_loss += loss.item() * batch_num
                predictions = torch.argmax(val_score, dim=1)
                val_correct += (predictions == val_label).sum().item()
                val_total += batch_num

        runtime = time.time() - start
        train_loss_avg = train_loss / train_total
        train_acc = train_correct / train_total
        val_loss_avg = val_loss / val_total
        val_acc = val_correct / val_total

        print("\nEpoch: %d/%d" % (epoch + 1, num_epochs))
        print("Train Loss: %.4f" % train_loss_avg)
        print("Train Acc:  %.4f" % train_acc)
        print("Val Loss:   %.4f" % val_loss_avg)
        print("Val Acc:    %.4f" % val_acc)
        print("Runtime:    %.2f seconds" % runtime)
        print("-" * 55)

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f | Runtime: %.2f seconds",
            epoch + 1, num_epochs, train_loss_avg, train_acc, val_loss_avg, val_acc, runtime,
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "train_acc": train_acc,
            "val_loss": val_loss_avg,
            "val_acc": val_acc,
            "runtime": runtime,
            "learning_rate": lr,
            "memory_usage": round(torch.cuda.memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
            "effective_batch_size": batch_size,
            "micro_batch_size": batch_size,
            "grad_accum_steps": 1,
        }
        history.append(record)
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_val_acc = val_acc
            best_epoch = epoch + 1

        # A single compact line is printed immediately before the screenshot so
        # even a short PyCharm Run tool window contains model, epoch, all four
        # metrics and runtime in the captured frame.
        print(
            ">>> Epoch %d/%d RESULT | Train Loss %.4f | Train Acc %.4f | "
            "Val Loss %.4f | Val Acc %.4f | Runtime %.2f s"
            % (epoch + 1, num_epochs, train_loss_avg, train_acc, val_loss_avg, val_acc, runtime)
        )
        screenshot(f"gru_epoch_{epoch + 1:02d}.png", f"GRU PyCharm epoch {epoch + 1}")

        pd.DataFrame(history).to_csv("result/gru_training_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(history).to_csv("experiments/gru/metrics.csv", index=False, encoding="utf-8-sig")

        payload = {
            "format_version": 1,
            "model_name": "gru",
            "run_id": run_id,
            "epoch": epoch + 1,
            "model_state": net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "config": config,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        atomic_save(payload, latest_path)
        if best_epoch == epoch + 1:
            atomic_save({
                "model_state": net.state_dict(),
                "epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "config": config,
            }, best_path)
        logger.info("checkpoint saved: %s", latest_path)

    if oom_stopped:
        logger.error("training stopped by OOM; resume from %s", latest_path)
        screenshot("gru_oom_stopped.png", "GRU OOM stopped")
        raise SystemExit(1)

    logger.info("start predicting...")
    if best_path.exists():
        best = torch.load(best_path, map_location=device, weights_only=False)
        net.load_state_dict(best["model_state"])
        logger.info("loaded best model from %s (epoch %d)", best_path, best.get("epoch", -1))

    net.eval()
    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc="Prediction") as pbar:
            for test_feature, in test_iter:
                test_feature = test_feature.to(device)
                test_score = net(test_feature)
                predictions = torch.argmax(test_score, dim=1)
                test_pred.extend(predictions.cpu().numpy().tolist())
                pbar.update(1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/gru.csv", index=False, quoting=3)
    logger.info("result saved! samples=%d", len(test_pred))
    screenshot("gru_final.png", "GRU final prediction")

    summary = {
        "model_name": "gru",
        "run_id": run_id,
        "status": "completed",
        "max_epochs": num_epochs,
        "actual_epochs": len(history),
        "total_train_seconds": sum(float(row.get("runtime", 0.0)) for row in history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "final_val_acc": history[-1]["val_acc"] if history else None,
        "history": history,
        "config": config,
        "prediction_file": "result/gru.csv",
        "checkpoint_best": str(best_path),
        "checkpoint_latest": str(latest_path),
        "early_stopping": len(history) < num_epochs,
    }
    (base_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n训练和预测全部完成！")
    print("训练轮数：", len(history))
    print("预测样本数量：", len(test_pred))
    print("预测结果：./result/gru.csv")
    print("训练指标：./result/gru_training_metrics.csv")
    print("完整日志：", log_dir / f"gru_{run_id}.log")
    logger.info("all tasks completed!")
