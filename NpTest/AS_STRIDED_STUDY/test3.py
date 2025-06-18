import sys

import numpy as np
from numpy.lib.stride_tricks import as_strided

from NpTest.AS_STRIDED_STUDY import public_check


def self_j(image, stride_length):
    padding_size = stride_length // 2
    padding_image = np.pad(image, ((padding_size, padding_size), (padding_size, padding_size)))
    h, w = padding_image.shape
    shape = (h - stride_length + 1, w - stride_length + 1, stride_length, stride_length)
    strides = padding_image.strides * 2
    public_check(padding_image, shape, strides)
    kernel = np.random.randn(stride_length, stride_length)
    return np.sum(as_strided(padding_image,
                      shape=shape,
                      strides=strides) * kernel, axis=(2, 3))


if __name__ == '__main__':
    image = np.arange(64).reshape(8, 8)
    result = self_j(image, stride_length=3)
    print("__________________")