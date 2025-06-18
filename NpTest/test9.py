import numpy as np
from numpy.lib.stride_tricks import as_strided


sequences = [
    np.array([1, 3, 4]),
    np.array([7, 8, 11, 13]),
    np.array([17, 19])
]
max_length = 4
embedding_dim = 8


def batch_sequences(sequences, max_length, embedding_dim):
    result = [np.pad(sequence, (0, max_length - len(sequence)), mode='constant', constant_values=(0, 0))
                       for sequence in sequences]
    padded = np.array(result)
    padded = np.expand_dims(padded, axis=-1)
    padded = np.tile(padded, (1, 1, embedding_dim))

    return as_strided(padded,
                      shape=(len(sequences), max_length, embedding_dim),
                      strides=padded.strides)

batch = batch_sequences(sequences, max_length, embedding_dim)
print("____________________")