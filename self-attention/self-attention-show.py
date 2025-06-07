import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_module=512, q_size=128, v_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_size = q_size
        self.w_q = nn.Linear(d_module, q_size)
        self.w_k = nn.Linear(d_module, q_size)
        self.w_v = nn.Linear(d_module, v_size)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores / self.q_size ** 0.5
        scores = F.softmax(scores, dim=-1)

        return torch.bmm(scores, v)


if __name__ == '__main__':
    input = torch.randn(3, 100, 512)
    attention = SelfAttention()
    output = attention(input)
    print(output.shape)

    d_k = 128
    q = torch.randn(512, 128)
    k = torch.randn(512, 128)
    s = (q * k).sum(dim=1)
    s2 = torch.matmul(q, k.T)
    print(s2.sum(dim=1).var().item() / 512)
    print(s2.shape)
    print(s.var().item())

