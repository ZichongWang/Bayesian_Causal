import torch
import numpy as np
def f(a):
    """
    计算 f(a) 函数的值
    参数：
    a: 输入值
    返回：
    f: 函数的输出值
    """
    return -np.log(1 + np.exp(a))
def Tfxn(y, qBD, qLS, qLF, alLS, alLF, w, local, delta):
    """
    Nonlinear function T in PyTorch with Tensor-based implementation.
    :param y: observation vector (torch.Tensor, shape [N])
    :param qBD, qLS, qLF: local estimates (torch.Tensor, shape [N])
    :param alLS, alLF: transformed local prior estimates (torch.Tensor, shape [N])
    :param w: weight vector (torch.Tensor, shape [15])
    :param local: local information matrix (torch.Tensor, shape [N], dtype=torch.int)
    :param delta: parameter for nonlinear function (float)
    :return: g, a tensor containing gBD, gLS, and gLF (torch.Tensor, shape [N, 3])
    """

    # Safe log transformation of y to avoid small values
    y = torch.log(1e-6 + y)  # Log transformation ensures numerical stability

    # Compute shared terms
    w12_sq_half = (w[12] ** 2) / 2
    w10_sq_half = (w[10] ** 2) / 2
    w11_sq_half = (w[11] ** 2) / 2

    # Generate masks for local conditions
    mask_3 = (local == 3).float()  # Mask for local == 3
    mask_4 = (local == 4).float()  # Mask for local == 4
    mask_5 = (local == 5).float()  # Mask for local == 5
    mask_6 = (local == 6).float()  # Mask for local == 6
    mask_1_3_5_6 = ((local == 1) | (local == 3) | (local == 5) | (local == 6)).float()
    mask_2_4_5_6 = ((local == 2) | (local == 4) | (local == 5) | (local == 6)).float()

    # Compute gBD
    gBD = (
        mask_3 * (
            qLS * f(-w[2] - w[5] + w12_sq_half) +
            (1 - qLS) * f(-w[2] + w12_sq_half) -
            qLS * f(w[2] + w[5] + w12_sq_half) -
            (1 - qLS) * f(w[2] + w12_sq_half)
        ) +
        mask_4 * (
            qLF * f(-w[2] - w[6] + w12_sq_half) +
            (1 - qLF) * f(-w[2] + w12_sq_half) -
            qLF * f(w[2] + w[6] + w12_sq_half) -
            (1 - qLF) * f(w[2] + w12_sq_half)
        ) -
        (mask_3 + mask_4 + mask_6) * ((w[7] - 2 * y + 2 * w[0]) / (2 * (w[1] ** 2))) * w[7] -
        mask_3 * (w[8] * w[7] * qLS) / (w[1] ** 2) -
        mask_4 * (w[9] * w[7] * qLF) / (w[1] ** 2) -
        mask_6 * (w[8] * w[9] * w[7] * qLS * qLF) / (w[1] ** 2) -
        mask_6 * (
            qLS * qLF * f(-w[2] - w[5] - w[6] + w12_sq_half) +
            qLS * (1 - qLF) * f(-w[2] - w[5] + w12_sq_half) +
            (1 - qLS) * qLF * f(-w[2] - w[6] + w12_sq_half) -
            (1 - qLS) * (1 - qLF) * f(w[2] + w12_sq_half)
        )
    )

    # Compute gLS
    gLS = (
        mask_1_3_5_6 * (
            f(-w[3] - w[13] * alLS + w10_sq_half) -
            f(w[3] + w[13] * alLS + w10_sq_half)
        )
    )
    gLS += (
        mask_3 * (
            qBD * (f(-w[2] - w[5] + w12_sq_half) - f(-w[2] + w12_sq_half)) +
            (1 - qBD) * (f(w[2] + w[5] + w12_sq_half) - f(w[2] + w12_sq_half))
        )
    )
    gLS -= (
        mask_1_3_5_6 * ((w[8] - 2 * y + 2 * w[0]) / (2 * (w[1] ** 2))) * w[8]
    )
    gLS -= mask_3 * ((w[8] * w[9] * qBD) / (w[1] ** 2))
    gLS -= mask_5 * ((w[9] * w[10] * qLF) / (w[1] ** 2))
    gLS -= mask_6 * ((w[8] * w[9] * w[10] * qLF * qBD) / (w[1] ** 2))
    gLS -= ((mask_5 + mask_6) * (qLF / (2 * delta)))

    # Compute gLF
    gLF = (
        mask_2_4_5_6 * (
            f(-w[4] - w[14] * alLF + w11_sq_half) -
            f(w[4] + w[14] * alLF + w11_sq_half)
        )
    )
    gLF += (
        mask_4 * (
            qBD * (f(-w[2] - w[6] + w12_sq_half) - f(-w[2] + w12_sq_half)) +
            (1 - qBD) * (f(w[2] + w[6] + w12_sq_half) - f(w[2] + w12_sq_half))
        )
    )
    gLF -= (
        mask_2_4_5_6 * ((w[9] - 2 * y + 2 * w[0]) / (2 * (w[1] ** 2))) * w[9]
    )
    gLF -= mask_4 * ((w[8] * w[10] * qBD) / (w[1] ** 2))
    gLF -= mask_5 * ((w[9] * w[10] * qLS) / (w[1] ** 2))
    gLF -= mask_6 * ((w[8] * w[9] * w[10] * qLS * qBD) / (w[1] ** 2))
    gLF -= ((mask_5 + mask_6) * (qLS / (2 * delta)))

    # Combine results into a tensor of shape [N, 3]
    g = torch.stack([gBD, gLS, gLF], dim=-1)

    return g

# 假设输入数据为 PyTorch Tensor
y = torch.rand(1000)         # 观测值
qBD = torch.rand(1000)       # 建筑估计
qLS = torch.rand(1000)       # 滑坡估计
qLF = torch.rand(1000)       # 液化估计
alLS = torch.rand(1000)      # 滑坡先验
alLF = torch.rand(1000)      # 液化先验
w = torch.rand(15)           # 权重向量
local = torch.randint(0, 7, (1000,))  # 局部模型
delta = 1e-3                 # 参数

# 调用函数
g = Tfxn(y, qBD, qLS, qLF, alLS, alLF, w, local, delta)

# 输出形状: [1000, 3]
print(g.shape)
