import numpy as np
import numpy.random as random

x_dog = np.array([1.0, 0.5, 0.2])
x_bite = np.array([-0.5, 1.2, 0.3])

w_q = random.rand(3, 3)
w_k = random.rand(3, 3)
w_share = random.rand(3, 3)

q_dog = np.dot(x_dog, w_q)
q_bite = np.dot(x_bite, w_q)
k_dog = np.dot(x_dog, w_k)
k_bite = np.dot(x_bite, w_k)

score1 = np.dot(q_dog, k_bite.T)
score2 = np.dot(q_bite, k_dog.T)
print(score1)
print(score2)

q_dog_share = np.dot(x_dog, w_share)
q_bite_share = np.dot(x_bite, w_share)
k_dog_share = np.dot(x_dog, w_share)
k_bite_share = np.dot(x_bite, w_share)
score3 = np.dot(q_dog_share, k_bite_share.T)
score4 = np.dot(q_bite_share, k_dog_share.T)
print(score3)
print(score4)


