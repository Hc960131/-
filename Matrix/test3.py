import numpy as np
import math


def softmax(x):
    e_x = np.exp(x - np.max(x))  # 防止数值溢出
    return e_x / e_x.sum(axis=-1, keepdims=True)


def attention(query, keys, values):
    """
    简单的注意力机制实现

    参数:
    query - 查询向量, shape (d_k,)
    keys - 键矩阵, shape (n, d_k)
    values - 值矩阵, shape (n, d_v)

    返回:
    加权求和后的值, shape (d_v,)
    注意力权重, shape (n,)
    """
    # 计算query和每个key的点积分数
    scores = np.dot(keys, query) / math.sqrt(len(query))  # 缩放点积注意力

    # 计算注意力权重
    weights = softmax(scores)
    print(weights.shape)
    print(values.shape)

    # 加权求和values
    output = np.dot(weights, values)

    return output, weights


# 示例使用
d_k = 64  # 查询和键的维度
d_v = 64  # 值的维度

# 随机初始化
query = np.random.rand(d_k)
keys = np.random.rand(5, d_k)  # 5个键
values = np.random.rand(5, d_v)  # 5个值
print(query.shape)
print(keys.shape)
print(values.shape)

output, attn_weights = attention(query, keys, values)
print("注意力权重:", attn_weights)
print("输出:", output.shape)