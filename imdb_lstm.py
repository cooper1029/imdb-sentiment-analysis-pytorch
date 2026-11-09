
import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm


# ==================== 1. 基本参数配置 ====================

num_epochs = 10
embed_size = 300
num_hiddens = 120
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.001

# 自动检测运行设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()

print("=" * 50)
print("当前运行设备：", device)
print("CUDA 是否可用：", use_gpu)
print("=" * 50)


# ==================== 2. 定义 LSTM 网络 ====================

class SentimentNet(nn.Module):

    def __init__(self, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, use_gpu, **kwargs):

        super(SentimentNet, self).__init__(**kwargs)

        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional

        # 使用预训练 GloVe 词向量
        self.embedding = nn.Embedding.from_pretrained(weight)

        # 冻结预训练词向量，不参与梯度更新
        self.embedding.weight.requires_grad = False

        # 双向 LSTM 编码器
        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=self.num_hiddens,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            dropout=0
        )

        # 全连接分类层
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)

    def forward(self, inputs):

        embeddings = self.embedding(inputs)

        states, hidden = self.encoder(
            embeddings.permute([1, 0, 2])
        )

        # 拼接序列首尾状态
        encoding = torch.cat(
            [states[0], states[-1]], dim=1
        )

        outputs = self.decoder(encoding)

        return outputs


# ==================== 3. 主程序 ====================

if __name__ == '__main__':

    # 日志配置
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: %(message)s'
    )

    logging.root.setLevel(level=logging.INFO)

    logger.info("running %s" % ''.join(sys.argv))

    # ==================== 4. 加载预处理数据 ====================

    logging.info("loading data...")

    pickle_file = os.path.join(
        'pickle', 'imdb_glove.pickle3'
    )

    with open(pickle_file, 'rb') as f:

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

    logging.info("data loaded!")

    # ==================== 5. 加载测试集 ====================

    test = pd.read_csv(
        "./corpus/imdb/testData.tsv",
        header=0,
        delimiter="\t",
        quoting=3
    )

    # ==================== 6. 创建模型 ====================

    net = SentimentNet(
        embed_size=embed_size,
        num_hiddens=num_hiddens,
        num_layers=num_layers,
        bidirectional=bidirectional,
        weight=weight,
        labels=labels,
        use_gpu=use_gpu
    )

    # 将模型移动到对应设备
    net.to(device)

    logging.info("model created!")
    logging.info("device: %s" % device)

    # 损失函数
    loss_function = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(
        net.parameters(),
        lr=lr
    )

    # ==================== 7. 构建数据集 ====================

    train_set = torch.utils.data.TensorDataset(
        train_features, train_labels
    )

    val_set = torch.utils.data.TensorDataset(
        val_features, val_labels
    )

    test_set = torch.utils.data.TensorDataset(
        test_features
    )

    # 数据加载器
    train_iter = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    val_iter = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )

    test_iter = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )

    # ==================== 8. 模型训练 ====================

    for epoch in range(num_epochs):

        start = time.time()

        net.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 训练过程
        with tqdm(
            total=len(train_iter),
            desc='Epoch %d/%d' % (epoch + 1, num_epochs)
        ) as pbar:

            for feature, label in train_iter:

                # 移动数据到设备
                feature = feature.to(device)
                label = label.to(device).long()

                # 梯度清零
                optimizer.zero_grad(set_to_none=True)

                # 前向传播
                score = net(feature)

                # 计算损失
                loss = loss_function(score, label)

                # 反向传播
                loss.backward()

                # 更新参数
                optimizer.step()

                # 统计损失和准确率
                batch_num = label.size(0)

                train_loss += loss.item() * batch_num

                predictions = torch.argmax(score, dim=1)

                train_correct += (
                    predictions == label
                ).sum().item()

                train_total += batch_num

                # 更新进度条
                pbar.set_postfix({
                    'loss': '%.4f' % (
                        train_loss / train_total
                    ),
                    'acc': '%.4f' % (
                        train_correct / train_total
                    )
                })

                pbar.update(1)

        # ==================== 9. 验证集评估 ====================

        net.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for val_feature, val_label in val_iter:

                val_feature = val_feature.to(device)
                val_label = val_label.to(device).long()

                val_score = net(val_feature)

                loss = loss_function(
                    val_score, val_label
                )

                batch_num = val_label.size(0)

                val_loss += loss.item() * batch_num

                predictions = torch.argmax(
                    val_score, dim=1
                )

                val_correct += (
                    predictions == val_label
                ).sum().item()

                val_total += batch_num

        # ==================== 10. 输出训练结果 ====================

        end = time.time()

        runtime = end - start

        train_loss_avg = train_loss / train_total
        train_acc = train_correct / train_total

        val_loss_avg = val_loss / val_total
        val_acc = val_correct / val_total

        print("\nEpoch: %d/%d" % (
            epoch + 1, num_epochs
        ))

        print("Train Loss: %.4f" % train_loss_avg)
        print("Train Acc:  %.4f" % train_acc)

        print("Val Loss:   %.4f" % val_loss_avg)
        print("Val Acc:    %.4f" % val_acc)

        print("Runtime:    %.2f seconds" % runtime)

        print("-" * 50)

    # ==================== 11. 测试集预测 ====================

    logging.info("start predicting...")

    net.eval()

    test_pred = []

    with torch.no_grad():

        with tqdm(
            total=len(test_iter),
            desc='Prediction'
        ) as pbar:

            for test_feature, in test_iter:

                test_feature = test_feature.to(device)

                test_score = net(test_feature)

                predictions = torch.argmax(
                    test_score, dim=1
                )

                test_pred.extend(
                    predictions.cpu().numpy().tolist()
                )

                pbar.update(1)

    # ==================== 12. 保存预测结果 ====================

    os.makedirs("./result", exist_ok=True)

    result_output = pd.DataFrame(
        data={
            "id": test["id"],
            "sentiment": test_pred
        }
    )

    result_output.to_csv(
        "./result/lstm.csv",
        index=False,
        quoting=3
    )

    logging.info("result saved!")

    print("\n预测完成！")
    print("预测样本数量：", len(test_pred))
    print("结果保存位置：./result/lstm.csv")

