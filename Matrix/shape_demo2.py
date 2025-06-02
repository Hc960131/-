import numpy as np


arr = np.arange(24).reshape(2, 3, 4)
print("原始数组:\n", arr)
# print("默认转置:\n", np.transpose(arr), np.transpose(arr).shape)
print("指定轴顺序(0,1,2)->(1,0,2):\n", np.transpose(arr, (2, 0, 1)), np.transpose(arr, (2, 0, 1)).shape)

# arr2 = np.arange(9).reshape(3, 3)
# print(arr2)
# print(np.transpose(arr2))

# # 图像数据通道顺序转换 (H,W,C) -> (C,H,W)
# image = np.random.rand(256, 256, 3)
# transposed_image = np.transpose(image, (2, 0, 1))
# print(transposed_image.shape)  # 输出: (3, 256, 256)