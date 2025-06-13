import numpy as np
from numpy.lib.stride_tricks import as_strided

# arr = np.arange(12).reshape(3,4)  # 3行4列
# print(arr)
# print(arr.strides)  # 输出: (32, 8)
# # 解释：行间步幅32字节（4元素×8字节），列间步幅8字节
#
# from numpy.lib.stride_tricks import as_strided
#
# arr = np.array([1,2,3,4,5,6], dtype=np.int16)
# view = as_strided(arr, shape=(3,2), strides=(4,2))
# print(view)# 零拷贝创建3x2视图
# """
# [[1 2]
#  [3 4]
#  [5 6]]
# """
#
# def split_blocks(img, block_size=8):
#     h, w = img.shape
#     return as_strided(img,
#                      shape=(h//block_size, w//block_size, block_size, block_size),
#                      strides=(w*block_size, block_size, w, 1))
#
# # 使用示例
# gray_img = np.random.rand(512,512)
# print(gray_img.dtype)# 假设是512x512灰度图
# blocks = split_blocks(gray_img)
# print(blocks)# 形状(64,64,8,8)


# ts = np.arange(100)
# window_size = 10
# step = 2
#
# # 计算窗口数量
# num_windows = (len(ts) - window_size) // step + 1
#
# # 零拷贝创建视图
# window_view = as_strided(ts,
#                         shape=(num_windows, window_size),
#                         strides=(ts.strides[0] * step,  # 每次跳2个元素
#                                 ts.strides[0]))         # 正常元素间隔
# print(window_view)  # 输出: (46, 10)

data = np.arange(16).reshape(4, 4)
data[data % 2 != 0] = 100
print(data)


