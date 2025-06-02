import torch
import torch.nn as nn
from torch import Tensor


class ManualLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ManualLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 初始化权重（输入门、遗忘门、输出门、候选记忆）
        scale = 1.0 / (hidden_dim ** 0.5)
        self.W_ii = nn.Parameter(torch.randn(input_dim, hidden_dim) * scale)
        self.W_if = nn.Parameter(torch.randn(input_dim, hidden_dim) * scale)
        self.W_io = nn.Parameter(torch.randn(input_dim, hidden_dim) * scale)
        self.W_ig = nn.Parameter(torch.randn(input_dim, hidden_dim) * scale)

        self.W_hi = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * scale)
        self.W_hf = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * scale)
        self.W_ho = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * scale)
        self.W_hg = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * scale)

        self.b_i = nn.Parameter(torch.zeros(hidden_dim))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))
        self.b_g = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: Tensor, h_0: Tensor = None, c_0: Tensor = None):
        batch_size, seq_len, _ = x.shape

        # 初始化隐藏状态和细胞状态
        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h_t = h_0

        if c_0 is None:
            c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            c_t = c_0

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步的输入

            # LSTM 计算步骤
            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)  # 输入门
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)  # 遗忘门
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)  # 输出门
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)  # 候选记忆

            c_t = f_t * c_t + i_t * g_t  # 更新细胞状态
            h_t = o_t * torch.tanh(c_t)  # 更新隐藏状态

            outputs.append(h_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), (h_t, c_t)