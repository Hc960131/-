import numpy as np

# 初始化词向量和矩阵
x_trading = np.random.randn(1, 2).reshape(2,) # "炒股"
x_money = np.random.randn(1, 2).reshape(2,)  # "赚钱"
print(x_trading)
print(x_money)
W_Q = np.random.randn(2, 2)  # 初始随机 Q 矩阵
W_K = np.random.randn(2, 2)  # 初始随机 K 矩阵

# 模拟训练过程
learning_rate = 0.01
for step in range(1000):
    # 前向传播
    Q = x_trading @ W_Q  # "炒股" 的 Query 向量
    K = x_money @ W_K  # "赚钱" 的 Key 向量
    score = Q @ K.T  # 注意力分数

    # 定义损失函数：目标是最大化 "炒股→赚钱" 的分数
    loss = -score  # 最小化 -score 等价于最大化 score

    # 反向传播计算梯度
    dscore = -1  # d(loss)/d(score) = -1
    dQ = dscore * K  # d(loss)/d(Q) = dscore * K
    dK = dscore * Q  # d(loss)/d(K) = dscore * Q

    # 计算 W_Q 和 W_K 的梯度（链式法则）
    dW_Q = np.outer(x_trading, dQ)  # d(loss)/d(W_Q) = x_trading^T @ dQ
    dW_K = np.outer(x_money, dK)  # d(loss)/d(W_K) = x_money^T @ dK

    # 更新参数
    W_Q -= learning_rate * dW_Q
    W_K -= learning_rate * dW_K

    # 打印训练过程（可选）
    if step % 200 == 0:
        print(f"Step {step}, Loss: {loss:.4f}, Score: {score:.4f}")

# 训练后的注意力分数
final_score = (x_trading @ W_Q) @ (x_money @ W_K).T
print(f"\n训练后 '炒股→赚钱' 的注意力分数: {final_score:.4f}")
# 模拟训练过程（通过梯度下降更新 W_Q 和 W_K）
# learning_rate = 0.001
# for _ in range(10000):  # 假设训练1000步
#     # 计算当前注意力分数
#     Q = x_trading @ W_Q
#     K = x_money @ W_K
#     score = Q @ K.T  # 当前分数
#
#     # 目标：让 "炒股→赚钱" 的分数最大化（假设这是正确关系） # 损失函数（简单示例）
#
#     # 反向传播更新 W_Q 和 W_K
#     grad_W_Q = np.outer(x_trading, x_money @ W_K.T)  # Q的梯度
#     grad_W_K = np.outer(x_money, x_trading @ W_Q.T)  # K的梯度
#     W_Q -= learning_rate * grad_W_Q
#     W_K -= learning_rate * grad_W_K
#
print(np.dot(x_trading, W_Q))
print(np.dot(x_money, W_K))
# # 训练后的注意力分数
# final_score = (x_trading @ W_Q) @ (x_money @ W_K).T
# print(f"训练后 '炒股→赚钱' 的注意力分数: {final_score:.4f}")  # 输出接近正无穷大（实际任务会约束范围）