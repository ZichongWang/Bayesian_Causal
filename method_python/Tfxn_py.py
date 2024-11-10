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

# 非线性函数 T
def Tfxn(y, qBD, qLS, qLF, alLS, alLF, w, local, delta):
    # 输入参数：
    # y: 数值数据
    # qBD, qLS, qLF: 各种概率参数
    # alLS, alLF: 线性参数
    # w: 参数向量 [w0, weps, w0BD, w0LS, w0LF, wLSBD, wLFBD, wBDy, wLSy, wLFy, weLS, weLF, weBD]
    # local: 整数值，用于选择不同的公式计算方式
    # delta: 一个常数参数

    # 为防止计算中出现 log(0)，对 y 加上一个小常数后再取对数
    y = np.log(1e-6 + y)

    # gBD 部分计算
    gBD = (
        (local == 3) * (
            qLS * f(-w[2] - w[5] + (w[12] ** 2) / 2)
            + (1 - qLS) * f(-w[2] + (w[12] ** 2) / 2)
            - qLS * f(w[2] + w[5] + (w[12] ** 2) / 2)
            - (1 - qLS) * f(w[2] + (w[12] ** 2) / 2)
        )
        + (local == 4) * (
            qLF * f(-w[2] - w[6] + (w[12] ** 2) / 2)
            + (1 - qLF) * f(-w[2] + (w[12] ** 2) / 2)
            - qLF * f(w[2] + w[6] + (w[12] ** 2) / 2)
            - (1 - qLF) * f(w[2] + (w[12] ** 2) / 2)
        )
        - (local == 3) * (w[8] - 2 * y + 2 * w[0]) / (2 * (w[1] ** 2)) * w[8]
        - (local == 4) * (1 / (w[1] ** 2)) * (w[9] * w[8] * qLS)
        - (local == 4) * (1 / (w[1] ** 2)) * (w[10] * w[8] * qLF)
    )

    # gLS 部分计算
    gLS = (
        (local == 1) * (f(-w[3] - w[13] * alLS + (w[10] ** 2) / 2) - f(w[3] + w[13] * alLS + (w[10] ** 2) / 2))
        - (local == 1) * ((w[8] - 2 * y + 2 * w[0]) / (2 * (w[1] ** 2))) * w[8]
        - (local == 3) * (w[7] * w[8] * qBD) / (w[1] ** 2)
    )

    # gLF 部分计算
    gLF = (
        (local == 2) * (f(-w[4] - w[14] * alLF + (w[11] ** 2) / 2) - f(w[4] + w[14] * alLF + (w[11] ** 2) / 2))
        - (local == 2) * ((w[9] - 2 * y + 2 * w[0]) / (2 * (w[1] ** 2))) * w[9]
        - (local == 4) * (w[8] * w[9] * qBD) / (w[1] ** 2)
    )

    # 将 gBD, gLS, gLF 合并成矩阵
    g = np.zeros((len(y), 3))
    g[:, 0] = gBD
    g[:, 1] = gLS
    g[:, 2] = gLF

    return g

