import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 100)
circle = np.column_stack([np.sin(theta), np.cos(theta)])
shearing_metrix1 = np.array([[1, 0.5], [0, 1]])
shearing_metrix2 = np.array([[1, 0.5], [0, 0]])
shearing_metrix3 = np.array([[0, 0], [1, 0.5]])
shearing_circle1 = np.dot(circle, shearing_metrix1.T)
shearing_circle2 = np.dot(circle, shearing_metrix2.T)
shearing_circle3 = np.dot(circle, shearing_metrix3.T)
plt.figure(figsize=(16, 4))
plt.subplot(141)
plt.plot(circle[:, 0], circle[:, 1], 'b')
plt.title("单位圆")
plt.subplot(142)
plt.plot(shearing_circle1[:,0], shearing_circle1[:,1], 'r')
plt.title("变换后的椭圆")

plt.subplot(143)
plt.plot(shearing_circle2[:,0], shearing_circle2[:,1], 'r')
plt.title("变换后的椭圆")

plt.subplot(144)
plt.plot(shearing_circle3[:,0], shearing_circle3[:,1], 'r')
plt.title("变换后的椭圆")
plt.show()
