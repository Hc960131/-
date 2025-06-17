import numpy as np
from numpy.lib.stride_tricks import as_strided


def image_stride(image, kernel, stride_size=3):
    image_padding = np.pad(image, pad_width=stride_size // 2, mode='reflect')
    h, w = image_padding.shape
    shape = (h - stride_size + 1, w - stride_size + 1, stride_size, stride_size)
    strides = image_padding.strides * 2
    stride_image = check(image_padding, shape, strides)
    stride_result = stride_image * kernel
    return np.sum(stride_result, axis=(2, 3))


def check(image, shape, stride):
    needed = (np.array(shape) - 1) * np.array(stride)
    if (needed >= image.nbytes).any():
        raise ValueError("数组越界")
    return as_strided(image,
                      shape=shape,
                      strides=stride)



# image = np.random.randn(512, 512)
# stride_size = 5
# kernel = np.random.rand(stride_size, stride_size)
# print(image_stride(image, kernel, stride_size).shape)

arr = np.arange(12)
print(arr.dtype)
print(arr.nbytes)