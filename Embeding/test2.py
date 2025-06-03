import torch
import torch.nn as nn


input1 = torch.ones(1000, 512, dtype=torch.long)
input2 = torch.ones(2, 3, dtype=torch.long)
embedding = nn.Embedding(512, 64)
output1 = embedding(input1)
output2 = embedding(input2)
print(output1.shape)
print(output2.shape)
