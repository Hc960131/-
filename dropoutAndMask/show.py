import torch
import torch.nn as nn
import math


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: 输入序列 (seq_len, batch_size, d_model)
        src_mask: 注意力mask (seq_len, seq_len)
        src_key_padding_mask: 序列padding mask (batch_size, seq_len)
        """
        # 第一部分: 自注意力 + Add & Norm
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 第二部分: 前馈网络 + Add & Norm
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


if __name__ == '__main__':
    # 参数设置
    batch_size = 4
    seq_len = 10
    d_model = 512
    nhead = 8

    # 创建模型
    encoder_layer = TransformerEncoderLayer()

    # 模拟输入 (seq_len, batch_size, d_model)
    src = torch.randn(seq_len, batch_size, d_model)

    # 创建padding mask (假设后3个token是padding)
    src_key_padding_mask = torch.zeros(batch_size, seq_len).bool()
    src_key_padding_mask[:, -3:] = True  # 最后3个位置是padding

    # 创建注意力mask (可选，这里创建一个三角形mask)
    src_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'))
    src_mask = src_mask.masked_fill(src_mask == 0, float(0.0))

    # 前向传播
    output = encoder_layer(
        src,
        src_mask=src_mask,
        src_key_padding_mask=src_key_padding_mask
    )

    print("输入形状:", src.shape)
    print("输出形状:", output.shape)
    print("Padding mask形状:", src_key_padding_mask.shape)
    print("注意力mask形状:", src_mask.shape)