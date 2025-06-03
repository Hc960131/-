import numpy as np

# 初始化参数
vocab_size, embed_dim = 10, 3
np.random.seed(42)  # 设置随机种子以便结果可复现

# 初始化Embedding权重（对应PyTorch中的embedding.weight）
# PyTorch默认使用均匀分布初始化，范围为[-sqrt(1/embed_dim), sqrt(1/embed_dim)]
scale = np.sqrt(1.0 / embed_dim)
embedding_weights = np.random.uniform(-scale, scale, (vocab_size, embed_dim))

# 模拟输入（单词ID=1, 2）
input_ids = np.array([1, 2])


# 前向传播
def embedding_forward(weights, ids):
    # 从权重矩阵中选取对应ID的向量
    return weights[ids]


output = embedding_forward(embedding_weights, input_ids)  # 形状 [2, 3]
print("前向传播输出形状:", output.shape)

# 模拟损失函数（假设损失为所有输出元素的和）
loss = output.sum()
print("损失值:", loss)

# 反向传播：计算损失对输出的梯度（d_loss/d_output）
# 由于loss = sum(output)，所以d_loss/d_output = 全1矩阵
d_output = np.ones_like(output)  # 形状 [2, 3]


# 计算损失对Embedding权重的梯度
def embedding_backward(ids, d_output, vocab_size, embed_dim):
    # 初始化梯度矩阵（与Embedding权重形状相同）
    d_weights = np.zeros((vocab_size, embed_dim))

    # 对于每个输入ID，将对应的梯度累加到权重梯度中
    for i, idx in enumerate(ids):
        d_weights[idx] += d_output[i]

    return d_weights


d_weights = embedding_backward(input_ids, d_output, vocab_size, embed_dim)

# 打印结果
print("\n梯度是否非零？", np.any(d_weights))  # True
print("ID=1的向量梯度：", d_weights[1])  # 非零
print("ID=3的向量梯度：", d_weights[3])  # 零（未被查询）