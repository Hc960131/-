import torch
import torch.nn as nn
import math


class TransformerEmbedding(nn.Module):
    """
    结合token embedding和positional embedding
    """

    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('position_ids', torch.arange(max_len).unsqueeze(0))

    def forward(self, x):
        # x: [batch_size, seq_len]
        token_embeddings = self.token_embedding(x)  # [batch_size, seq_len, d_model]

        seq_len = x.size(1)
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embedding(position_ids)  # [1, seq_len, d_model]

        embeddings = token_embeddings + position_embeddings
        return self.dropout(embeddings)


# 定义参数
vocab_size = 10  # 假设词汇表有10个单词
d_model = 4      # 嵌入维度=4
max_len = 5      # 最大句子长度=5

# 初始化嵌入层
embedding_layer = TransformerEmbedding(vocab_size, d_model, max_len)

# 输入数据（2个句子，每个句子3个单词）
input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape [2, 3]

# 前向传播
output = embedding_layer(input_ids)
print("输入形状:", input_ids.shape)
print("输出形状:", output.shape)  # [2, 3, 4]
print("输出示例:", output[0])