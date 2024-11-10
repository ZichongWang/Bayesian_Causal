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

def loss_fn(y, qBD, qLS, qLF, alLS, alLF, w, local, delta):
    """
    损失函数

    参数：
    y: 目标值
    qBD, qLS, qLF: 后验估计
    alLS, alLF: 先验估计的对数值
    w: 权重向量
    local: LOCAL 矩阵中的局部信息
    delta: 参数 delta
    
    返回：
    L: 损失值
    """
    # 对数转换
    y = np.log(1e-6 + y)
    f_qBD = np.minimum(qBD + 1e-6, 1 - 1e-6)
    f_qLS = np.minimum(qLS + 1e-6, 1 - 1e-6)
    f_qLF = np.minimum(qLF + 1e-6, 1 - 1e-6)

    # LBD 计算
    LBD = ((local == 3) * qBD * (qLS * f(-w[2] - w[5] + (w[12] ** 2) / 2) + (1 - qLS) * f(-w[2] + (w[12] ** 2) / 2))) + \
          ((local == 4) * qBD * (qLF * f(-w[2] - w[6] + (w[12] ** 2) / 2) + (1 - qLF) * f(-w[2] + (w[12] ** 2) / 2))) + \
          ((local == 3) * (1 - qBD) * (qLS * f(w[2] + w[5] + (w[12] ** 2) / 2) + (1 - qLS) * f(w[2] + (w[12] ** 2) / 2))) + \
          ((local == 4) * (1 - qBD) * (qLF * f(w[2] + w[6] + (w[12] ** 2) / 2) + (1 - qLF) * f(w[2] + (w[12] ** 2) / 2))) + \
          ((local == 6) * (qBD * (qLS * qLF * f(-w[2] - w[5] - w[6] + (w[12] ** 2) / 2) +
                                  qLS * (1 - qLF) * f(-w[2] - w[5] + (w[12] ** 2) / 2) +
                                  (1 - qLS) * qLF * f(-w[2] - w[6] + (w[12] ** 2) / 2) +
                                  (1 - qLS) * (1 - qLF) * f(-w[2] + (w[12] ** 2) / 2))) +
           (1 - qBD) * (qLS * qLF * f(w[2] + w[5] + w[6] + (w[12] ** 2) / 2) +
                        qLS * (1 - qLF) * f(w[2] + w[5] + (w[12] ** 2) / 2) +
                        (1 - qLS) * qLF * f(w[2] + w[6] + (w[12] ** 2) / 2) +
                        (1 - qLS) * (1 - qLF) * f(w[2] + (w[12] ** 2) / 2))) - \
          ((local == 3) | (local == 4) | (local == 6)) * (qBD * np.log(f_qBD) + (1 - qBD) * np.log(1 - f_qBD))
    LBD[(local == 0) | (local == 1) | (local == 2) | (local == 5)] = 0

    # LLS 计算
    LLS = ((local == 1) | (local == 3) | (local == 5) | (local == 6)) * (qLS * f(-w[3] - w[13] * alLS + (w[10] ** 2) / 2) +
                                                                         (1 - qLS) * f(w[3] + w[13] * alLS + (w[10] ** 2) / 2) -
                                                                         qLS * np.log(f_qLS) - (1 - qLS) * np.log(1 - f_qLS))
    LLS[(local == 0) | (local == 2) | (local == 4)] = 0

    # LLF 计算
    LLF = ((local == 2) | (local == 4) | (local == 5) | (local == 6)) * (qLF * f(-w[4] - w[14] * alLF + (w[11] ** 2) / 2) +
                                                                         (1 - qLF) * f(w[4] + w[14] * alLF + (w[11] ** 2) / 2) -
                                                                         qLF * np.log(f_qLF) - (1 - qLF) * np.log(1 - f_qLF))
    LLF[(local == 0) | (local == 1) | (local == 3)] = 0

    # Epsilon 计算
    Leps = - (0.5 * np.log(w[1] ** 2) + (y - w[0]) * (y - w[0]) / (2 * (w[1] ** 2))) - \
           (1 / (2 * (w[1] ** 2))) * ((local == 3) | (local == 4) | (local == 6)) * w[7] * qBD * (w[7] - 2 * y + 2 * w[0]) - \
           (1 / (2 * (w[1] ** 2))) * ((local == 1) | (local == 3) | (local == 5) | (local == 6)) * w[8] * qLS * (w[8] - 2 * y + 2 * w[0]) - \
           (1 / (2 * (w[1] ** 2))) * ((local == 2) | (local == 4) | (local == 5) | (local == 6)) * w[9] * qLF * (w[9] - 2 * y + 2 * w[0]) - \
           (1 / (w[1] ** 2)) * ((local == 3) * w[7] * w[8] * qBD * qLS +
                                (local == 4) * w[7] * w[9] * qBD * qLF +
                                (local == 5) * w[8] * w[9] * qLS * qLF +
                                (local == 6) * w[7] * w[8] * w[9] * qBD * qLS * qLF) - y

    # L_ex 计算
    L_ex = - ((local == 5) | (local == 6)) * (f_qLS * f_qLF) / (2 * delta)
    L_ex = np.maximum(L_ex, -1e+2)

    # 总损失
    L = LBD + LLS + LLF + Leps + L_ex
    return L