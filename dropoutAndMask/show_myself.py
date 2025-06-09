import math

import torch
import torch.nn as nn

from Trans.structure import TransEmbedding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_module=512, vocab_size=256, seq_len=100, n_head=8, dim_feed_forward=2048, dropout=0.1, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = TransEmbedding(vocab_size, n_module, seq_len)
        self.multi_attention = nn.MultiheadAttention(n_module, n_head, dropout=dropout)
        self.linear1 = nn.Linear(n_module, dim_feed_forward)
        self.linear2 = nn.Linear(dim_feed_forward, n_module)
        self.dropout = nn.Dropout(dropout)
        self.normal1 = nn.LayerNorm(n_module)
        self.normal2 = nn.LayerNorm(n_module)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        embedding_result = self.embedding(x)
        src = self.multi_attention(embedding_result, embedding_result, embedding_result, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)[0]
        src = embedding_result + self.dropout(src)
        src = self.normal1(src)
        src2 = self.linear2(self.dropout1(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.normal2(src)
