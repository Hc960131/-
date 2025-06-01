import numpy as np
import matplotlib.pyplot as plt


theta = np.linspace(0, 2*np.pi, 100)
circle = np.column_stack([np.sin(theta), np.cos(theta)])
sacling_metrix = np.array([[2, 0], [0, 0.5]])
transformer_circle = np.dot(circle, sacling_metrix.T)
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot(circle[:, 0], circle[:, 1], 'b')
plt.title("单位圆")
plt.subplot(122)
plt.plot(transformer_circle[:,0], transformer_circle[:,1], 'r')
plt.title("变换后的椭圆")
plt.show()