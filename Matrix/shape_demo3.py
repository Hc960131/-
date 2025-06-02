import numpy as np


seq_year = np.arange(12)
seq_month = np.arange(12)
np1 = np.repeat(seq_year, 12)
np2 = np.tile(seq_month, 12)
print(seq_year)
print(seq_month)
print(np1)
print(np2)
np3 = np.array([np1, np2])
print(np3.shape)
print(np.transpose(np3))