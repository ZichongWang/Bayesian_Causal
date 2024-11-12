# test_SVI.py
import numpy as np
from SVI_py import svi

# 模拟数据
Y = np.random.rand(1000)
LS = np.random.rand(1000)
LF = np.random.rand(1000)
BD = np.random.randint(0, 2, size=1000)
LOCAL = np.random.randint(0, 7, size=1000)
w = np.random.rand(15)
Nq = 10
rho = 1e-3
delta = 1e-3
eps_0 = 0.001
lambda_term = 0
regu_type = 1
sigma = 0.5
prune_type = 'double'

# 调用 SVI 函数
try:
    opt_w, opt_QBD, opt_QLS, opt_QLF, QLS, QLF, QBD, final_loss, best_loss, local = svi(
        Y, LS, LF, w, Nq, rho, delta, eps_0, LOCAL, lambda_term, regu_type, sigma, prune_type
    )
    print("SVI function executed successfully.")
    print(f"opt_w shape: {opt_w.shape}, opt_QBD shape: {opt_QBD.shape}, opt_QLS shape: {opt_QLS.shape}, opt_QLF shape: {opt_QLF.shape}")
    print(f"final_loss length: {len(final_loss)}, best_loss: {best_loss}")
except Exception as e:
    print(f"Error occurred: {e}")
