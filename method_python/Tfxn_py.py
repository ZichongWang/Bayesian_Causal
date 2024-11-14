import numpy as np
from Loss_py import f  # 假设 f 函数在 Loss_py 模块中定义

def Tfxn(y, qBD, qLS, qLF, alLS, alLF, w, local, delta):
    """
    Nonlinear function T represented in Python
    :param y: observation vector
    :param qBD, qLS, qLF: local estimates
    :param alLS, alLF: transformed local prior estimates
    :param w: weight vector
    :param local: local information matrix
    :param delta: parameter for nonlinear function
    :return: g, containing gBD, gLS, and gLF
    """
    
    # Log transformation of y to avoid very small values
    # y = np.where(y < 0, 0, y)
    y = np.log(1e-6 + y)# log之后全是负的

    # gBD computation
    gBD = ((local == 3) * (qLS * f(-w[2] - w[5] + (w[12] ** 2) / 2)
                           + (1 - qLS) * f(-w[2] + (w[12] ** 2) / 2)
                           - qLS * f(w[2] + w[5] + (w[12] ** 2) / 2)
                           - (1 - qLS) * f(w[2] + (w[12] ** 2) / 2))
          + (local == 4) * (qLF * f(-w[2] - w[6] + (w[12] ** 2) / 2)
                           + (1 - qLF) * f(-w[2] + (w[12] ** 2) / 2)
                           - qLF * f(w[2] + w[6] + (w[12] ** 2) / 2)
                           - (1 - qLF) * f(w[2] + (w[12] ** 2) / 2))
          - ((local == 3) | (local == 4) | (local == 6)) * ((w[7] - 2 * y + 2 * w[0]) / (2 * (w[1] ** 2))) * w[7]
          - (local == 3) * (w[8] * w[7] * qLS) / (w[1] ** 2)
          - (local == 4) * (w[9] * w[7] * qLF) / (w[1] ** 2)
          - (local == 6) * (w[8] * w[9] * w[7] * qLS * qLF) / (w[1] ** 2)
          - (local == 6) * (qLS * qLF * f(-w[2] - w[5] - w[6] + (w[12] ** 2) / 2)
                           + qLS * (1 - qLF) * f(-w[2] - w[5] + (w[12] ** 2) / 2)
                           + (1 - qLS) * qLF * f(-w[2] - w[6] + (w[12] ** 2) / 2)
                           - (1 - qLS) * (1 - qLF) * f(w[2] + (w[12] ** 2) / 2)
                           - (qLS * qLF * f(w[2] + w[5] + w[6] + (w[12] ** 2) / 2)
                           - qLS * (1 - qLF) * f(w[2] + w[5] + (w[12] ** 2) / 2)
                           - (1 - qLS) * qLF * f(w[2] + w[6] + (w[12] ** 2) / 2)
                           - (1 - qLS) * (1 - qLF) * f(w[2] + (w[12] ** 2) / 2))))

    # gLS computation
    gLS = ((local == 1) | (local == 3) | (local == 5) | (local == 6)) * (
            f(-w[3] - w[13] * alLS + (w[10] ** 2) / 2) - f(w[3] + w[13] * alLS + (w[10] ** 2) / 2))
    gLS += ((local == 3) * (qBD * (f(-w[2] - w[5] + (w[12] ** 2) / 2) - f(-w[2] + (w[12] ** 2) / 2))
             + (1 - qBD) * (f(w[2] + w[5] + (w[12] ** 2) / 2) - f(w[2] + (w[12] ** 2) / 2))))
    gLS += ((local == 6) * (qBD * (qLF * f(-w[2] - w[5] - w[6] + (w[12] ** 2) / 2)
                                  + (1 - qLF) * f(-w[2] - w[5] + (w[12] ** 2) / 2)
                                  - qLF * f(-w[2] - w[6] + (w[12] ** 2) / 2)
                                  - (1 - qLF) * f(-w[2] + (w[12] ** 2) / 2))))
    gLS += ((local == 6) * ((1 - qBD) * (qLF * f(w[2] + w[5] + w[6] + (w[12] ** 2) / 2)
                                        + (1 - qLF) * f(w[2] + w[5] + (w[12] ** 2) / 2)
                                        - qLF * f(w[2] + w[6] + (w[12] ** 2) / 2)
                                        - (1 - qLF) * f(w[2] + (w[12] ** 2) / 2))))
    gLS -= ((local == 1) | (local == 3) | (local == 5) | (local == 6)) * ((w[8] - 2 * y + 2 * w[0]) / (2 * (w[1] ** 2))) * w[8]
    gLS -= (local == 3) * ((w[8] * w[9] * qBD) / (w[1] ** 2))
    gLS -= (local == 5) * ((w[9] * w[10] * qLF) / (w[1] ** 2))
    gLS -= (local == 6) * ((w[8] * w[9] * w[10] * qLF * qBD) / (w[1] ** 2))
    gLS -= ((local == 5) | (local == 6)) * (qLF / (2 * delta))

    # gLF computation
    gLF = ((local == 2) | (local == 4) | (local == 5) | (local == 6)) * (
            f(-w[4] - w[14] * alLF + (w[11] ** 2) / 2) - f(w[4] + w[14] * alLF + (w[11] ** 2) / 2))
    gLF += ((local == 4) | (local == 6)) * (qBD * (f(-w[2] - w[6] + (w[12] ** 2) / 2) - f(-w[2] + (w[12] ** 2) / 2))
                                           + (1 - qBD) * (f(w[2] + w[6] + (w[12] ** 2) / 2) - f(w[2] + (w[12] ** 2) / 2)))
    gLF += ((local == 6) * (qBD * (qLS * f(-w[2] - w[5] - w[6] + (w[12] ** 2) / 2)
                                  + (1 - qLS) * f(-w[2] - w[6] + (w[12] ** 2) / 2)
                                  - qLS * f(-w[2] - w[5] + (w[12] ** 2) / 2)
                                  - (1 - qLS) * f(-w[2] + (w[12] ** 2) / 2))))
    gLF += ((local == 6) * ((1 - qBD) * (qLS * f(w[2] + w[5] + w[6] + (w[12] ** 2) / 2)
                                        + (1 - qLS) * f(w[2] + w[6] + (w[12] ** 2) / 2)
                                        - qLS * f(w[2] + w[5] + (w[12] ** 2) / 2)
                                        - (1 - qLS) * f(w[2] + (w[12] ** 2) / 2))))
    gLF -= ((local == 2) | (local == 4) | (local == 5) | (local == 6)) * ((w[9] - 2 * y + 2 * w[0]) / (2 * (w[1] ** 2))) * w[9]
    gLF -= (local == 4) * ((w[8] * w[10] * qBD) / (w[1] ** 2))
    gLF -= (local == 5) * ((w[9] * w[10] * qLS) / (w[1] ** 2))
    gLF -= (local == 6) * ((w[8] * w[9] * w[10] * qLS * qBD) / (w[1] ** 2))
    gLF -= ((local == 5) | (local == 6)) * (qLS / (2 * delta))

    # Combine results
#     g = np.zeros((len(y), 3))
#     g[:, 0] = gBD
#     g[:, 1] = gLS
#     g[:, 2] = gLF

    g = np.hstack((gBD, gLS, gLF))

    # return gBD, gLS, gLF
    return g
