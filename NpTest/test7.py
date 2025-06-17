import numpy as np
from numpy.lib.stride_tricks import as_strided


# arr = np.arange(12).reshape(3, 4)
# print(arr.strides)
# arr = np.arange(12, dtype=np.int16).reshape(3, 4)
# print(arr.strides)

# data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16)
# window_view = as_strided(data,
#                          (4, 3),
#                          (2, 2))
# print(window_view)
# window_view[1][1] = 100
# print(data)
# print(data.dtype.itemsize)


def image_to_patches(image, patch_size=8):
    h, w = image.shape
    return as_strided(image,
                      shape=(h - patch_size + 1, w - patch_size + 1, patch_size, patch_size),
                      strides=image.strides * 2)


image = np.arange(25).reshape(5, 5)
print(image.dtype)
print(image.strides)
print(image.strides * 2)
print(image)
print(image_to_patches(image, patch_size=2).shape)
print(image_to_patches(image, patch_size=2))


# arr = np.arange(12).reshape(3, 4)
# print(arr.strides)
# transposed = as_strided(arr,
#                         shape=(4, 3),
#                         strides=(arr.strides[1], arr.strides[0]))
# print(arr)
# print(transposed.strides)
# print(transposed)

# ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# print(ts.strides)
# window_size = 3
# sequence = as_strided(ts,
#                       shape=(len(ts) - window_size - 1, window_size),
#                       strides=(ts.strides[0], ts.strides[0] * 2))
# print(sequence.strides)
# print(sequence)

# 传统方法 vs as_strided
import time

arr = np.random.rand(10000)
# 方法1：循环切片
t1 = time.time()
windows = np.array([arr[i:i+100] for i in range(9901)])
print("循环耗时:", time.time() - t1)

# 方法2：as_strided
t2 = time.time()
windows = as_strided(arr, shape=(9901,100), strides=(8,8))
print("as_strided耗时:", time.time() - t2)