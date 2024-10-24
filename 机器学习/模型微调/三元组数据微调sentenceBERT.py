import torch
import numpy as np
import random
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from enum import Enum
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    models,
    evaluation,
)
from torch.utils.data import DataLoader, random_split


# 设置随机种子以便实验复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(2)


# 从三个句子列表中创建三元组数据集
def create_triplet_dataset(anchor_sentences, positive_sentences, negative_sentences):
    """
    创建三元组数据集

    :param anchor_sentences: 锚定句子列表
    :param positive_sentences: 正向句子列表
    :param negative_sentences: 负向句子列表
    :return: 返回 InputExample 对象列表
    """
    if not (
        len(anchor_sentences) == len(positive_sentences) == len(negative_sentences)
    ):
        raise ValueError("锚定、正向、负向句子列表必须长度一致")

    triplet_examples = []
    for anchor, positive, negative in zip(
        anchor_sentences, positive_sentences, negative_sentences
    ):
        triplet_examples.append(InputExample(texts=[anchor, positive, negative]))

    return triplet_examples


# 定义数据集文件路径
import os

dataset = "code_complex"
data_type = "_all"
file_path = (
    f"result/{dataset}_dataset/{dataset}_contrastive_dataset_train{data_type}.jsonl"
)
test_file_path = (
    f"result/{dataset}_dataset/{dataset}_contrastive_dataset_test{data_type}.jsonl"
)
file_path = os.path.join(os.path.dirname(__file__), file_path)
test_file_path = os.path.join(os.path.dirname(__file__), test_file_path)


# 初始化用于存储数据的列表
anchor_sentences = []
positive_sentences = []
negative_sentences = []
test_anchor_sentences = []
test_positive_sentences = []
test_negative_sentences = []

# 从 JSONL 文件读取训练数据并填充列表
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        anchor_sentences.append(data["anchor"])
        positive_sentences.append(data["positive"])
        negative_sentences.append(data["negative"])

# 从 JSONL 文件读取测试数据并填充列表
with open(test_file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        test_anchor_sentences.append(data["anchor"])
        test_positive_sentences.append(data["positive"])
        test_negative_sentences.append(data["negative"])

# 从加载的数据创建训练和测试数据集
triplet_dataset = create_triplet_dataset(
    anchor_sentences, positive_sentences, negative_sentences
)
test_examples = create_triplet_dataset(
    test_anchor_sentences, test_positive_sentences, test_negative_sentences
)


# 定义枚举类用于模型的距离度量
class TripletDistanceMetric(Enum):
    """三元组损失的度量标准"""

    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


# 使用特定模型配置初始化 Sentence Transformer 模型
model_name = "microsoft/codebert-base"
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# 设置模型的运行设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model.to(device)

# 将数据集分为训练和评估子集
train_size = int(0.8 * len(triplet_dataset))
eval_size = len(triplet_dataset) - train_size
train_examples, eval_examples = random_split(triplet_dataset, [train_size, eval_size])

# 为训练创建 DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
train_loss = losses.TripletLoss(
    model, TripletDistanceMetric.COSINE
)  # 设置训练的三元组损失

# 初始化评估器以监控训练进度
evaluator = evaluation.TripletEvaluator.from_input_examples(eval_examples, name="eval")

# 配置训练参数
num_epochs = 10
# warmup_steps 代表一个在训练开始阶段逐渐增加学习率的步骤数
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
# 表明在总训练步骤数的前 10% 期间，学习率将从一个较低的值逐渐增加到设定的最大值
print(warmup_steps)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    evaluator=evaluator,
)

# 在测试数据上评估模型
test_evaluator = evaluation.TripletEvaluator.from_input_examples(
    test_examples, name="test"
)
output = model.evaluate(test_evaluator)
print(output)

# 保存训练好的模型
model_path = f"result/{dataset}_contrastive{data_type}_model"
model_path = os.path.join(os.path.dirname(__file__), model_path)
model.save(model_path)
