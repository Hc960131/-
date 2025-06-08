import math

import torch.nn as nn
import torch
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, seq_len=100, n_module=512, n_head=8, vocab_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = TransEmbedding(vocab_size, n_module, seq_len)
        self.self_attention = SelfAttention(n_module)
        self.multi_attention = MultiHeadAttention(n_module, n_head)

    def forward(self, x):
        x = self.embedding(x)
        q, k, v = self.self_attention(x)
        output = self.multi_attention(q, k, v)
        return output + x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_module, n_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_module = n_module
        self.n_head = n_head
        assert n_module % n_head == 0
        self.d_k = n_module // n_head
        self.w_o = nn.Linear(self.n_module, self.n_module)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        multi_q = q.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        multi_k = k.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        multi_v = v.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        scores = torch.matmul(multi_q, multi_k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k))
        scores = F.softmax(scores, dim=-1)
        out_put = torch.matmul(scores, multi_v)
        out_put = out_put.transpose(1, 2).contiguous().view(batch_size, -1, self.n_module)
        return self.w_o(out_put)


class SelfAttention(nn.Module):
    def __init__(self, n_module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_q = nn.Linear(n_module, n_module)
        self.w_k = nn.Linear(n_module, n_module)
        self.w_v = nn.Linear(n_module, n_module)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        return q, k, v


class TransEmbedding(nn.Module):
    def __init__(self, vocab_size=256, n_module=512, seq_len=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_module = n_module  # 词向量的维度
        self.seq_len = seq_len  # 单个句子的最大长度
        self.embedding = nn.Embedding(vocab_size, n_module)
        self.position_embedding = PositionEmbedding(n_module, seq_len)

    def forward(self, x):
        embedding_x = self.embedding(x)
        position_embedding_x = self.position_embedding(x)
        return embedding_x + position_embedding_x


class PositionEmbedding(nn.Module):
    def __init__(self, n_module, seq_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_module, 2) * (-math.log(10000) / n_module))
        pe = torch.zeros((seq_length, n_module))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].detach()


if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    transformer = Transformer()
    output = transformer.forward(x)
    print(output.shape)