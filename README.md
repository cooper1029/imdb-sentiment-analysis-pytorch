# IMDB 电影评论情感分析（PyTorch）

基于 PyTorch 的 IMDB 电影评论情感二分类项目，使用 GloVe 词向量及多种深度学习模型，对电影评论进行正面与负面情感分类。实验涵盖循环神经网络、卷积神经网络、注意力机制、胶囊网络、Transformer 及预训练语言模型。

项目共实现 **10 个独立模型**，其中包括 7 个自建模型和 3 个预训练模型。完成了数据预处理、模型构建、训练验证、测试集预测及 Kaggle 提交，并保存了相应的训练指标、模型权重和实验记录。

## 1. 项目概况

| 项目 | 实验配置 |
|:---|:---|
| 任务类型 | 二分类（正面 / 负面） |
| 数据集 | IMDB Movie Reviews |
| 深度学习框架 | PyTorch |
| 独立模型 | 10 个（7 个自建 + 3 个预训练） |
| 本地最高验证准确率 | **88.72%** |
| Kaggle 最高得分 | **0.88040** |
| 训练设备 | CPU |

## 2. 数据集与预处理

实验使用 IMDB 电影评论数据集，包含有标签训练数据、测试数据及无标签评论。

| 数据集 | 样本数量 | 用途 |
|:---|---:|:---|
| 有标签评论 | 25,000 | 训练与验证 |
| 训练集 | 20,000 | 模型参数训练 |
| 验证集 | 5,000 | 模型评估与选择 |
| 测试集 | 25,000 | Kaggle 预测 |
| 无标签评论 | 50,000 | 本实验未使用 |

将有标签数据按照 8:2 划分为训练集和验证集。

文本预处理包括 HTML 标签清洗、标点处理、统一小写及分词。词表由训练集与测试集共同构建，每条评论统一截断或补齐至 256 个 Token。

自建模型采用 **300 维 GloVe 词向量**进行初始化，并在训练过程中冻结词向量参数。预处理结果保存为文件，供各模型直接读取。

## 3. 模型设计

本项目按照网络结构与预训练方式，将模型分为两类。

**自建模型（7 个）**

- **LSTM**：双向两层 LSTM，用于学习评论文本的上下文依赖关系。
- **GRU**：双向两层 GRU，通过门控机制提取序列特征。
- **Attention-LSTM**：在双向 LSTM 基础上加入注意力机制，增强对重要文本特征的关注。
- **Capsule-LSTM**：结合 LSTM 与胶囊网络，使用动态路由机制进行特征建模。
- **CNN**：利用一维卷积提取局部文本特征，并通过全局最大池化完成特征聚合。
- **CNN-LSTM**：结合卷积网络与双向 LSTM，提取局部及上下文特征。
- **Transformer**：使用多头自注意力机制和编码器结构进行文本建模。

**预训练模型（3 个）**

- **BERT-base**：基于预训练双向 Transformer 进行微调。
- **DistilBERT-base**：采用轻量化 Transformer 结构进行微调。
- **RoBERTa-base**：使用改进的预训练语言模型进行微调。

各模型的主要结构参数如下：

| 模型 | 主要结构参数 | 特征表示 |
|:---|:---|:---|
| LSTM | 双向 2 层，hidden=120 | GloVe 300 维 |
| GRU | 双向 2 层，hidden=120 | GloVe 300 维 |
| Attention-LSTM | 双向 2 层，hidden=128 | GloVe 300 维 |
| Capsule-LSTM | 双向 LSTM，5 个胶囊，路由 4 次 | GloVe 300 维 |
| CNN | 128 个卷积核，kernel size=3 | GloVe 300 维 |
| CNN-LSTM | 卷积、池化、双向 2 层 LSTM | GloVe 300 维 |
| Transformer | 2 层编码器，2 个注意力头，d_model=120 | 随机初始化 |
| DistilBERT | distilbert-base-uncased | 预训练权重 |
| BERT | bert-base-uncased | 预训练权重 |
| RoBERTa | roberta-base | 预训练权重 |

注：自建模型均使用冻结的 GloVe 词向量，Transformer 使用随机初始化的 300 维嵌入。三个预训练模型加载官方权重进行微调，不改变网络主体结构。

## 4. 实验环境与训练配置

实验在 Windows 11 环境下完成，使用 PyCharm 作为开发工具，所有模型均通过 CPU 进行训练。

| 环境 | 配置 |
|:---|:---|
| 操作系统 | Windows 11 |
| 处理器 | Intel Core i7-14650HX |
| 内存 | 32 GB |
| Python | 3.14 |
| PyTorch | 2.11.0+cpu |
| CUDA | 不可用 |
| 训练设备 | CPU |

主要依赖包括 PyTorch、NumPy、Pandas、Scikit-learn、Transformers、Gensim、BeautifulSoup、tqdm 和 Matplotlib 等。

**训练参数**

| 参数 | 自建模型 | 预训练模型 |
|:---|:---:|:---:|
| 优化器 | Adam | AdamW |
| 学习率 | 0.001（Transformer 为 0.0005） | 2e-5 / 5e-5 / 1e-5 |
| Batch Size | 64 | 按各模型训练配置 |
| 最大序列长度 | 256 | BERT / DistilBERT：128；RoBERTa：64 |
| 最大训练轮数 | 10 | 2～3 |
| 早停机制 | 部分模型启用 | 本次按设定轮数训练 |

自建模型以验证损失作为模型选择依据，连续 3 轮未达到 0.001 的有效改善条件时触发早停。预训练模型使用 AdamW 和线性学习率调度。

## 5. 模型训练结果

所有模型均完成相应的训练与验证实验，并记录逐轮训练损失、训练准确率、验证损失、验证准确率及运行时间。

| 模型 | 最佳 Epoch | Best Val Loss | Best Val Acc | Final Val Acc | 实际轮数 |
|:---|---:|---:|---:|---:|---:|
| LSTM | 8 | 0.3141 | 86.94% | 86.62% | 10/10 |
| GRU | 4 | 0.2858 | 88.56% | 87.86% | 10/10 |
| Attention-LSTM | 5 | 0.2752 | **88.72%** | 87.80% | 8/10 |
| Capsule-LSTM | 9 | 0.3452 | 85.56% | 82.74% | 10/10 |
| CNN | 3 | 0.2959 | 87.46% | 88.02% | 6/10 |
| CNN-LSTM | 4 | 0.3380 | 85.72% | 84.18% | 7/10 |
| Transformer | 4 | 0.3522 | 86.56% | 85.68% | 7/10 |
| DistilBERT | 2 | 0.3516 | 87.16% | 87.34% | 3/3 |
| BERT | 1 | 0.3245 | 87.30% | 88.82% | 2/2 |
| RoBERTa | 1 | 0.3376 | 85.62% | 86.50% | 2/2 |

**运行时间统计**

| 模型 | 运行时间 |
|:---|---:|
| LSTM | 2h 10m 08s |
| GRU | 50m 21s |
| Attention-LSTM | 11m 30s |
| Capsule-LSTM | 14m 09s |
| CNN | 43s |
| CNN-LSTM | 2m 59s |
| Transformer | 27m 44s |
| DistilBERT | 1h 43m |
| BERT | 2h 37m |
| RoBERTa | 1h 29m |

**训练结果分析**

从本地验证结果来看，Attention-LSTM 获得了最高的最佳验证准确率，为 88.72%，GRU 达到 88.56%，表现同样突出。

CNN 具有较短的训练时间，在较低计算成本下取得了不错的分类效果。Capsule-LSTM 的验证准确率相对较低，说明不同网络结构在当前数据集上的适应能力存在差异。

需要注意，Best Val Acc 与 Final Val Acc 分别代表训练过程中记录的最佳验证准确率和最后一轮验证准确率，二者并不一定相同。模型最佳权重按照验证损失进行选择。

**早停情况**

| 模型 | 实际轮数 | 训练结束情况 |
|:---|:---:|:---|
| LSTM | 10/10 | 完成设定轮数 |
| GRU | 10/10 | 完成设定轮数 |
| Attention-LSTM | 8/10 | 触发早停 |
| Capsule-LSTM | 10/10 | 完成设定轮数 |
| CNN | 6/10 | 触发早停 |
| CNN-LSTM | 7/10 | 触发早停 |
| Transformer | 7/10 | 触发早停 |
| DistilBERT | 3/3 | 完成设定轮数 |
| BERT | 2/2 | 完成设定轮数 |
| RoBERTa | 2/2 | 完成设定轮数 |

## 6. Kaggle 提交结果

完成模型训练后，将各模型在测试集上的预测结果生成 CSV 文件，并提交至 Kaggle 的 **Bag of Words Meets Bags of Popcorn** 竞赛。

| 模型 | 提交文件 | Kaggle Score |
|:---|:---|---:|
| LSTM | `lstm.csv` | 0.86540 |
| GRU | `gru.csv` | **0.88040** |
| Attention-LSTM | — | 未提交 |
| Capsule-LSTM | `capsule_lstm.csv` | 0.85460 |
| CNN | `cnn.csv` | 0.87512 |
| CNN-LSTM | `cnnlstm.csv` | 0.85284 |
| Transformer | `transformer.csv` | 0.84284 |
| DistilBERT | `distilbert.csv` | 0.87104 |
| BERT | `bert.csv` | 0.87292 |
| RoBERTa | `roberta.csv` | 0.85604 |

<summary></summary>

Kaggle 最高成绩为 **GRU：0.88040**。从提交结果来看，GRU、CNN 和 BERT 均取得了较好的成绩。各模型的本地验证表现与 Kaggle 评分并不完全一致，说明模型在不同数据划分上的泛化表现存在差异。

注：Kaggle 评分采用 ROC AUC 指标，而本地验证准确率采用 Accuracy 指标，两者衡量角度不同，不能直接进行数值比较。LSTM 曾重复提交，两个提交记录的成绩均为 0.86540。

## 7. 项目文件结构

项目主要文件按照功能划分为以下几个部分：

| 文件或目录 | 主要内容 |
|:---|:---|
| `corpus/` | 原始数据集及语料 |
| `pickle/` | GloVe 词向量及预处理缓存 |
| `logs/` | 模型训练日志 |
| `result/` | 预测结果及训练指标 |
| `对测试集运行结果/` | 各模型测试集预测文件 |
| `所有训练模型每轮结果/` | 各模型逐轮训练指标 |
| `所有的模型在kaggle上提交的得分/` | Kaggle 提交成绩记录 |
| `原来没修改前的代码/` | 原始代码备份 |

主要 Python 文件：

| 文件 | 功能 |
|:---|:---|
| `imdb_process.py` | 数据预处理 |
| `imdb_lstm.py` | LSTM 模型 |
| `imdb_gru.py` | GRU 模型 |
| `imdb_attention_lstm.py` | Attention-LSTM 模型 |
| `imdb_capsule_lstm.py` | Capsule-LSTM 模型 |
| `imdb_cnn.py` | CNN 模型 |
| `imdb_cnnlstm.py` | CNN-LSTM 模型 |
| `imdb_transformer.py` | Transformer 模型 |
| `imdb_bert_native.py` | BERT 模型 |
| `imdb_distilbert_native.py` | DistilBERT 模型 |
| `imdb_roberta_trainer.py` | RoBERTa 模型 |
| `training_support.py` | 训练辅助模块 |

## 8. 实验说明

本次实验在 CPU 环境下完成。由于不同模型的计算复杂度和训练时间存在差异，各模型采用了相应的训练轮数及参数配置。

LSTM 为前期完成的历史实验，本次未重新训练，未保存模型权重。BERT 和 RoBERTa 实际训练 2 轮，DistilBERT 实际训练 3 轮。BERT 和 RoBERTa 的原始配置为最多 3 轮，本次受 CPU 训练时间限制完成了 2 轮训练。

BERT 第 1 轮验证损失最低，因此最佳模型权重选择第 1 轮；第 2 轮虽然取得了更高的验证准确率，但验证损失有所上升。

BERT、DistilBERT 的输入长度为 128 Token，RoBERTa 为 64 Token。BERT 和 DistilBERT 存在多个代码入口，本项目实际训练时各保留一个代表模型。其中名称包含 `scratch` 的 BERT 入口实际加载了预训练权重，并非从头训练。

## 9. 项目总结

本项目围绕 IMDB 电影评论情感分类任务，完成了从文本预处理、词向量构建、深度学习模型设计到模型训练与 Kaggle 提交的完整实验流程。

通过对不同网络结构的实现与比较，进一步理解了循环神经网络、卷积神经网络、注意力机制和 Transformer 在文本分类任务中的应用，也积累了预训练语言模型微调及实验结果分析的实践经验。

---

**项目关键词：** Python · PyTorch · NLP · IMDB · GloVe · LSTM · GRU · CNN · Transformer · BERT · Kaggle
