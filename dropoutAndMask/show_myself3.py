import math

import torch.nn as nn
import torch

from Trans.structure import TransEmbedding


class TransEncoderLayer(nn.Module):
    def __init__(self, n_module=512, n_head=8, n_dim=2048, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_attn = nn.MultiheadAttention(n_module, n_head, dropout)
        self.normal = nn.LayerNorm(n_module)
        self.dropout = nn.Dropout(0.1)
        self.forward = TransForward(n_module, n_dim, dropout)

    def forward(self, src, attn_mask, padding_mask):
        src2 = self.normal(src)
        src2 = self.multi_attn(src2, src2, src2, attn_mask=attn_mask, key_padding_mask=padding_mask)
        src = src + self.dropout(src2)
        return self.forward(src)


class TransForward(nn.Module):
    def __init__(self, n_module=512, n_dim=2048, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normal = nn.LayerNorm(n_module)
        self.linear1 = nn.Linear(n_module, n_dim)
        self.linear2 = nn.Linear(n_dim, n_module)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        src = self.normal(x)
        src = self.linear1(src)
        src = nn.ReLU(src)
        src = self.dropout(src)
        src = self.linear2(src)
        return x + self.dropout2(src)

