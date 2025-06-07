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
        self.position_embedding = PositionEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('position_ids', torch.arange(max_len).unsqueeze(0))

    def forward(self, x):
        # x: [batch_size, seq_len]
        token_embeddings = self.token_embedding(x)  # [batch_size, seq_len, d_model]

        position_embeddings = self.position_embedding(x)  # [1, seq_len, d_model]

        embeddings = token_embeddings + position_embeddings
        return self.dropout(embeddings)  # [1, seq_len, d_model]


class PositionEncoding(nn.Module):
    def __init__(self, d_module, max_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        position = torch.arange(max_length).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_module, 2) * (-math.log(10000.0) / d_module))
        pe = torch.zeros(max_length, d_module)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列用sin
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].detach()


# 定义参数
vocab_size = 10  # 假设词汇表有10个单词
d_model = 4      # 嵌入维度=4
max_len = 5      # 最大句子长度=5

# 初始化嵌入层
embedding_layer = TransformerEmbedding(vocab_size, d_model, max_len)

# 输入数据（2个句子，每个句子3个单词）
input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])  # shape [2, 3]

# 前向传播
output = embedding_layer(input_ids)
print("输入形状:", input_ids.shape)
print("输出形状:", output.shape)  # [2, 3, 4]
print("输出示例:", output[0])