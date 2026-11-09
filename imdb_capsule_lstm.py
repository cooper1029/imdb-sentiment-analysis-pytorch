"""Capsule-LSTM — original teacher file with the capsule head made valid.

Structure: 2-layer bidirectional LSTM (hidden 128, frozen GloVe) + dynamic
routing capsules (5 capsules x width 5, 4 routing iterations) -> Linear(25, 2).
The original Capsule parameter shape was incompatible with its own view and
indexed the batch dimension as if it were the capsule dimension; this version
routes over sequence positions and flattens the capsule output.  Runtime: CPU,
Adam lr=0.001, checkpoint/resume, early stopping, metrics and real PyCharm
screenshots.
"""

import os
import pickle
import sys
import time

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from training_support import (
    EarlyStopper, save_prediction, save_runtime, screenshot, setup,
    try_resume, write_summary,
)

os.environ.setdefault("IMDB_CAPTURE_MODE", "pycharm")
os.environ.setdefault("IMDB_CAPTURE_TITLE_SUBSTR", "imdb_capsule_lstm.py")
torch.set_num_threads(12)

test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

num_epochs = 10
embed_size = 300
num_hiddens = 128
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.001
patience = 3
min_delta = 0.001
num_capsule = 5
dim_capsule = 5
routings = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()


class Capsule(nn.Module):
    def __init__(self, num_hiddens, bidirectional, num_capsule=5, dim_capsule=5, routings=4, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.bidirectional = bidirectional
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        in_dim = num_hiddens * (2 if bidirectional else 1)
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, in_dim, num_capsule * dim_capsule)))

    def forward(self, inputs, lengths=None):
        # inputs: (seq, batch, in_dim)
        x = inputs.permute(1, 0, 2)  # (batch, seq, in_dim)
        u_hat = torch.matmul(x, self.W)  # (batch, seq, capsules*width)
        batch, seq_len, _ = u_hat.shape
        u_hat = u_hat.view(batch, seq_len, self.num_capsule, self.dim_capsule)
        u_hat = u_hat.permute(0, 2, 1, 3).contiguous()  # (batch, caps, seq, width)
        b = torch.zeros(batch, self.num_capsule, seq_len, device=u_hat.device)
        mask = None
        if lengths is not None:
            mask = torch.arange(seq_len, device=u_hat.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1)  # (batch,1,seq)
        for i in range(self.routings):
            if mask is not None:
                b = b.masked_fill(~mask, -1e9)
            c = F.softmax(b, dim=2)
            outputs = self.squash(torch.sum(c.unsqueeze(-1) * u_hat, dim=2))
            if i < self.routings - 1:
                b = b + torch.sum(outputs.unsqueeze(2) * u_hat, dim=-1)
        return outputs.flatten(start_dim=1)

    @staticmethod
    def squash(x, axis=-1):
        squared = (x ** 2).sum(dim=axis, keepdim=True)
        return x / torch.sqrt(squared + 1e-7)


class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers, bidirectional, weight,
                 labels, use_gpu, num_capsule=5, dim_capsule=5, routings=4, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=self.embed_size, hidden_size=self.num_hiddens,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                               dropout=0)
        self.capsule = Capsule(num_hiddens, bidirectional, num_capsule, dim_capsule, routings)
        self.decoder = nn.Linear(num_capsule * dim_capsule, labels)

    def forward(self, inputs, lengths=None):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute(1, 0, 2))
        capsule = self.capsule(states, lengths)
        return self.decoder(capsule)


if __name__ == "__main__":
    dirs = setup("capsule_lstm")
    logger = dirs["logger"]
    logger.info("running %s", " ".join(sys.argv))
    logger.info("device: %s | threads: %d", device, torch.get_num_threads())

    with open(os.path.join("pickle", "imdb_glove.pickle3"), "rb") as handle:
        [train_features, train_labels, val_features, val_labels, test_features,
         weight, word_to_idx, idx_to_word, vocab] = pickle.load(handle)
    logger.info("data loaded: train=%s val=%s test=%s", tuple(train_features.shape),
                tuple(val_features.shape), tuple(test_features.shape))

    net = SentimentNet(embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers,
                       bidirectional=bidirectional, weight=weight, labels=labels,
                       use_gpu=use_gpu, num_capsule=num_capsule,
                       dim_capsule=dim_capsule, routings=routings).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    config = {
        "model_name": "capsule_lstm", "embed_size": embed_size,
        "num_hiddens": num_hiddens, "num_layers": num_layers,
        "bidirectional": bidirectional, "num_capsule": num_capsule,
        "dim_capsule": dim_capsule, "routings": routings,
        "batch_size": batch_size, "lr": lr, "num_epochs": num_epochs,
        "optimizer": "Adam", "source_lr": 0.01, "patience": patience,
        "min_delta": min_delta, "device": str(device),
    }
    history = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    start_epoch = 0
    checkpoint = try_resume(net, optimizer, dirs["checkpoints"] / "latest.pt", config)
    if checkpoint:
        history = checkpoint.get("history", [])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_epoch = checkpoint.get("best_epoch", 0)
        start_epoch = checkpoint.get("epoch", 0)
        logger.info("RESUMED at completed epoch %d", start_epoch)

    stopper = EarlyStopper(patience, min_delta)
    stopper.best = best_val_loss
    stopper.best_epoch = best_epoch
    stopped = False
    screenshot("capsule_lstm", "start", "Capsule-LSTM start", dirs)
    logger.info("start training")

    for epoch in range(start_epoch, num_epochs):
        start = time.time()
        net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        with tqdm(total=len(train_iter), desc="Epoch %d/%d" % (epoch + 1, num_epochs)) as pbar:
            for feature, label in train_iter:
                feature = feature.to(device)
                label = label.to(device).long()
                optimizer.zero_grad(set_to_none=True)
                score = net(feature)
                loss = loss_function(score, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                batch_num = label.size(0)
                train_loss += loss.item() * batch_num
                train_correct += (torch.argmax(score, dim=1) == label).sum().item()
                train_total += batch_num
                pbar.set_postfix({"loss": "%.4f" % (train_loss / train_total),
                                  "acc": "%.4f" % (train_correct / train_total)})
                pbar.update(1)

        net.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feature, label in val_iter:
                feature = feature.to(device)
                label = label.to(device).long()
                score = net(feature)
                loss = loss_function(score, label)
                val_loss_sum += loss.item() * label.size(0)
                val_correct += (torch.argmax(score, dim=1) == label).sum().item()
                val_total += label.size(0)

        runtime = time.time() - start
        train_loss_avg = train_loss / train_total
        train_acc = train_correct / train_total
        val_loss_avg = val_loss_sum / val_total
        val_acc = val_correct / val_total
        improved, should_stop = stopper.update(epoch + 1, val_loss_avg)
        if improved:
            best_val_loss, best_val_acc, best_epoch = val_loss_avg, val_acc, epoch + 1

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss_avg, "train_acc": train_acc,
            "val_loss": val_loss_avg, "val_acc": val_acc, "runtime": runtime,
            "learning_rate": lr, "effective_batch_size": batch_size,
            "micro_batch_size": batch_size, "grad_accum_steps": 1,
        })
        print(">>> [Capsule-LSTM] Epoch %d/%d RESULT | Train Loss %.4f | Train Acc %.4f | "
              "Val Loss %.4f | Val Acc %.4f | Runtime %.2f s"
              % (epoch + 1, num_epochs, train_loss_avg, train_acc,
                 val_loss_avg, val_acc, runtime))
        screenshot("capsule_lstm", f"epoch_{epoch + 1:02d}", f"Capsule-LSTM epoch {epoch + 1}", dirs)
        save_runtime(net, optimizer, "capsule_lstm", config, history, epoch + 1,
                     best_val_loss, best_val_acc, best_epoch, dirs, improved)
        logger.info("Epoch %d/%d | Train Loss %.4f | Train Acc %.4f | Val Loss %.4f | Val Acc %.4f | Runtime %.2f",
                    epoch + 1, num_epochs, train_loss_avg, train_acc, val_loss_avg, val_acc, runtime)
        if should_stop:
            stopped = True
            logger.info("EARLY STOPPING after epoch %d", epoch + 1)
            break

    if (dirs["checkpoints"] / "best_model.pt").exists():
        best = torch.load(dirs["checkpoints"] / "best_model.pt", map_location=device, weights_only=False)
        net.load_state_dict(best["model_state"])
        logger.info("loaded best model from epoch %d", best.get("epoch", -1))

    net.eval()
    preds = []
    with torch.no_grad():
        for test_feature, in tqdm(test_iter, desc="Prediction"):
            test_feature = test_feature.to(device)
            preds.extend(torch.argmax(net(test_feature), dim=1).cpu().tolist())

    path = save_prediction(preds, test, "capsule_lstm")
    logger.info("result saved: %s samples=%d", path, len(preds))
    write_summary("capsule_lstm", "Capsule-LSTM", config, history, best_epoch,
                  best_val_loss, best_val_acc, stopped, dirs, path,
                  {"prediction_count": len(preds), "positive_rate": float(sum(preds) / len(preds))})
    screenshot("capsule_lstm", "final", "Capsule-LSTM final", dirs)
    print("Capsule-LSTM completed: best epoch %d val acc %.4f, result %s" % (best_epoch, best_val_acc, path))
