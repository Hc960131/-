import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义变换矩阵 (2D→3D)
A = np.array([
    [1, 0],  # x' = x
    [0, 1],  # y' = y
    [1, 1]   # z' = x + y
])

# 原始2D点（单位圆上的点）
theta = np.linspace(0, 2*np.pi, 50)
X_2d = np.column_stack([np.cos(theta), np.sin(theta)])

# 变换到3D
X_3d = np.dot(X_2d, A.T)  # 等价于 A·X_2d.T

# 可视化
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax1.scatter(X_2d[:,0], X_2d[:,1])
ax1.set_title("原始2D空间")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2])
ax2.set_title("变换后的3D空间")
plt.show()