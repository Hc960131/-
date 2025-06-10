import math

import torch
import torch.nn as nn

from Trans.structure import TransEmbedding


class Encoder(nn.Module):
    def __init__(self, vocab_size=256, seq_len=100, n_module=512, n_head=8, forward_dim=2048, dropout=0.1, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = TransEmbedding(vocab_size, n_module, seq_len)
        self.multi_attn = nn.MultiheadAttention(n_module, n_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.normal1 = nn.LayerNorm(n_module)
        self.normal2 = nn.LayerNorm(n_module)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(n_module, forward_dim)
        self.linear2 = nn.Linear(forward_dim, n_module)

    def forward(self, x, sequence_mask=None, padding_mask=None):
        src = self.embedding(x)
        src2 = self.multi_attn(x, x, x, attn_mask=sequence_mask, key_padding_mask=padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.normal1(src)

        src2 = self.linear2(self.dropout1(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.normal2(src)

