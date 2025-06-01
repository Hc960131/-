import numpy as np
import matplotlib.pyplot as plt


theta = np.linspace(0, 2*np.pi, 100)
circle = np.column_stack([np.sin(theta), np.cos(theta)])
shearing_metrix = np.array([[1, 0.5], [0, 1]])
rotation_metrix = np.array([[0, -1], [1, 0]])
trans_circle = np.dot(circle, shearing_metrix.T)
trans_circle1 = np.dot(np.dot(circle, shearing_metrix.T), rotation_metrix.T)
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(circle[:, 0], circle[:, 1], 'b')
plt.title("单位圆")
plt.subplot(132)
plt.plot(trans_circle[:, 0], trans_circle[:, 1], 'r')
plt.title("变换后的椭圆")
plt.subplot(133)
plt.plot(trans_circle1[:, 0], trans_circle1[:, 1], 'r')
plt.title("变换后的椭圆")
plt.show()