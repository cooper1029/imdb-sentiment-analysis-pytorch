"""Transformer encoder from scratch — original teacher file with runtime fixes.

Structure: 300-d trainable token embedding -> Linear(300,120) projection ->
positional encoding -> 2 TransformerEncoder layers (2 heads, FFN 512) ->
masked mean pooling -> Linear(120,2).  The original fed 300-d embeddings into a
120-d layer without projection and called validation/test without lengths.
Runtime: CPU, Adam lr=0.0005, checkpoint/resume, early stopping, metrics and real
PyCharm screenshots.
"""

import math
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
os.environ.setdefault("IMDB_CAPTURE_TITLE_SUBSTR", "imdb_transformer.py")
torch.set_num_threads(12)

test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

num_epochs = 10
embed_size = 300
num_hiddens = 120
num_layers = 2
num_head = 2
dim_feedforward = 512
dropout = 0.1
batch_size = 64
labels = 2
lr = 0.0005
patience = 3
min_delta = 0.001
max_len = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, max_len=256):
        super(Transformer, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.position_embedding = PositionalEncoding(hidden_dim, dropout, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_head, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths=None):
        if lengths is None:
            lengths = (inputs != 0).sum(dim=1).clamp(min=1)
        hidden = self.embeddings(inputs)
        hidden = self.projection(hidden)
        hidden = self.position_embedding(hidden)
        mask = torch.arange(inputs.size(1), device=inputs.device).unsqueeze(0) < lengths.unsqueeze(1)
        hidden = self.transformer(hidden, src_key_padding_mask=~mask)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.output(pooled)


def lengths_of(features):
    return (features != 0).sum(dim=1).clamp(min=1)


if __name__ == "__main__":
    dirs = setup("transformer")
    logger = dirs["logger"]
    logger.info("running %s", " ".join(sys.argv))
    logger.info("device: %s | threads: %d", device, torch.get_num_threads())

    with open(os.path.join("pickle", "imdb_glove.pickle3"), "rb") as handle:
        [train_features, train_labels, val_features, val_labels, test_features,
         weight, word_to_idx, idx_to_word, vocab] = pickle.load(handle)
    logger.info("data loaded: train=%s val=%s test=%s", tuple(train_features.shape),
                tuple(val_features.shape), tuple(test_features.shape))

    net = Transformer(vocab_size=weight.shape[0], embedding_dim=embed_size,
                      hidden_dim=num_hiddens, num_class=labels,
                      dim_feedforward=dim_feedforward, num_head=num_head,
                      num_layers=num_layers, dropout=dropout,
                      max_len=max_len).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    config = {
        "model_name": "transformer", "embed_size": embed_size,
        "num_hiddens": num_hiddens, "num_layers": num_layers,
        "num_head": num_head, "dim_feedforward": dim_feedforward,
        "dropout": dropout, "batch_size": batch_size, "lr": lr,
        "num_epochs": num_epochs, "optimizer": "Adam", "source_lr": 0.05,
        "patience": patience, "min_delta": min_delta, "device": str(device),
        "embedding": "random 300d, trainable",
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
    screenshot("transformer", "start", "Transformer start", dirs)
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
                score = net(feature, lengths_of(feature))
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
                score = net(feature, lengths_of(feature))
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
        print(">>> [Transformer] Epoch %d/%d RESULT | Train Loss %.4f | Train Acc %.4f | "
              "Val Loss %.4f | Val Acc %.4f | Runtime %.2f s"
              % (epoch + 1, num_epochs, train_loss_avg, train_acc,
                 val_loss_avg, val_acc, runtime))
        screenshot("transformer", f"epoch_{epoch + 1:02d}", f"Transformer epoch {epoch + 1}", dirs)
        save_runtime(net, optimizer, "transformer", config, history, epoch + 1,
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
            preds.extend(torch.argmax(net(test_feature, lengths_of(test_feature)), dim=1).cpu().tolist())

    path = save_prediction(preds, test, "transformer")
    logger.info("result saved: %s samples=%d", path, len(preds))
    write_summary("transformer", "Transformer (from scratch)", config, history, best_epoch,
                  best_val_loss, best_val_acc, stopped, dirs, path,
                  {"prediction_count": len(preds), "positive_rate": float(sum(preds) / len(preds))})
    screenshot("transformer", "final", "Transformer final", dirs)
    print("Transformer completed: best epoch %d val acc %.4f, result %s" % (best_epoch, best_val_acc, path))
