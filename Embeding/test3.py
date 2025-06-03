import torch
import torch.nn as nn

# 初始化Embedding层
vocab_size, embed_dim = 10, 3
embedding = nn.Embedding(vocab_size, embed_dim)

# 模拟输入（单词ID=1, 2）
input_ids = torch.tensor([1, 2])
output = embedding(input_ids)  # 形状 [2, 3]

# 模拟损失函数（实际任务中为交叉熵等）
loss = output.sum()  # 假设损失为所有输出元素的和
loss.backward()      # 反向传播

print(embedding.weight.grad)
# 查看Embedding矩阵的梯度
print("梯度是否非零？", embedding.weight.grad is not None)  # True
print("ID=1的向量梯度：", embedding.weight.grad[1])        # 非零
print("ID=3的向量梯度：", embedding.weight.grad[3])        # 零（未被查询）