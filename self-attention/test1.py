import matplotlib.pyplot as plt
import torch
import math


def plot_positional_encoding(d_model=512, max_len=50):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    plt.figure(figsize=(10, 6))
    plt.imshow(pe.numpy().T, aspect='auto', cmap='viridis')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.colorbar()
    plt.title('Positional Encoding')
    plt.show()


plot_positional_encoding()