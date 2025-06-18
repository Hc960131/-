import sys

import numpy as np
from numpy.lib.stride_tricks import as_strided

from NpTest.AS_STRIDED_STUDY import public_check


def sequence_stride(sequence, sub_sequence_length):
    shape = (len(sequence) - sub_sequence_length + 1, sub_sequence_length)
    strides = (sequence.strides[0], sequence.strides[0])
    public_check(sequence, shape, strides)
    return as_strided(sequence,
                      shape=shape,
                      strides=strides)


def sequence_self(sequence, sub_sequence_length):
    shape = (len(sequence) - sub_sequence_length - 1, sub_sequence_length)
    result = np.zeros(shape)
    for i in range(shape[0]):
        result[i, :] = sequence[i:i + sub_sequence_length]
    return result


data = np.arange(100)
sub_sequence_length = 4
result1 = sequence_stride(data, sub_sequence_length)
result2 = sequence_self(data, sub_sequence_length)