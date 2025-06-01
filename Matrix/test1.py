import numpy as np
import matplotlib.pyplot as plt

# 定义变换矩阵（剪切+缩放）
A = np.array([[1.5, 0.5],
              [0, 0.8]])

# 绘制单位圆及其变换
theta = np.linspace(0, 2*np.pi, 100)
circle = np.column_stack([np.cos(theta), np.sin(theta)])
transformed_circle = np.dot(circle, A.T)

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(circle[:,0], circle[:,1], 'b')
plt.title("单位圆")

plt.subplot(122)
plt.plot(transformed_circle[:,0], transformed_circle[:,1], 'r')
plt.title("变换后的椭圆")
plt.show()