import math

import torch.nn as nn
import torch

from Trans.structure import TransEmbedding


class TransEncodingWithEmbedding(nn.Module):
    def __init__(self, vocab_size=256, seq_len=100, n_module=512, n_head=8, n_dim=2048, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = TransEmbedding(vocab_size, n_module, seq_len)
        self.encoding = TransEncoding(n_module, n_head, n_dim, dropout)

    def forward(self, x, attn_mask=None, padding_mask=None):
        x = self.embedding(x)
        return self.encoding(x, attn_mask, padding_mask)


class TransEncoding(nn.Module):
    def __init__(self, n_module=512, n_head=8, n_dim=2048, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normal1 = nn.LayerNorm(n_module)
        self.dropout1 = nn.Dropout(dropout)
        # nn.MultiheadAttention默认情况是batch_size=False，这种情况下，attn_mask矩阵应该是batch_size, batch_size，
        # 只有在指定batch_size=True的情况下，attn_mask矩阵才是(seq_len, seq_len)
        self.multi_attn = nn.MultiheadAttention(n_module, n_head, dropout=dropout, batch_first=True)

        self.normal2 = nn.LayerNorm(n_module)
        self.linear1 = nn.Linear(n_module, n_dim)
        self.linear2 = nn.Linear(n_dim, n_module)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, attn_mask=None, padding_mask=None):
        src2 = self.normal1(src)
        src2 = self.multi_attn(src2, src2, src2, attn_mask=attn_mask, key_padding_mask=padding_mask)[0]
        src2 = self.dropout1(src2)
        src = src2 + src

        src2 = self.normal2(src)
        src2 = self.linear2(self.dropout2(torch.relu(self.linear1(src2))))
        return src2 + src


if __name__ == '__main__':
    batch_size = 4
    seq_length = 10
    vocab_size = 20

    input = torch.randint(0, 10, (batch_size, seq_length))

    padding_mask = torch.zeros(batch_size, seq_length).bool()
    padding_mask[:, -3:] = True

    attn_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

    network = TransEncodingWithEmbedding(vocab_size=vocab_size, seq_len=seq_length)
    output = network.forward(input, attn_mask, padding_mask)
