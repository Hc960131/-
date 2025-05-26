import matplotlib.pyplot as plt
import numpy as np

z = np.linspace(-2, 2, 100)
hinge = np.maximum(0, 1 - z)
log_loss = np.log(1 + np.exp(-z))

plt.plot(z, hinge, label='Hinge Loss (SVM)')
plt.plot(z, log_loss, label='Log Loss (Logistic Regression)')
plt.legend()
plt.xlabel('y * f(x)')
plt.ylabel('Loss')
plt.title('SVM vs Logistic Regression Loss')
plt.show()