# 改好了。
import numpy as np
from Loss_py import df

#   Guide 
# %   w has [w0;weps;w0BD;w0LS;w0LF;wLSBD;wLFBD;wBDy;wLSy;wLFy;weLS;weLF;weBD;waLS;waLF] 
# %   grad is a matrix of row locations for the batch size and 15 weights

def parder(y, qBD, qLS, qLF, alLS, alLF, w, local):
    """
    Parameters:
        y : ndarray
            输入数据，形状为(batch_size, )，表示目标值。
        qBD, qLS, qLF : binary values
            二元值，分别表示不同的条件。
        alLS, alLF : float
            权重相关的浮点数。
        w : list of 15 weights
            长度为15的权重列表。
        local : ndarray
            表示局部条件的数组。
        df : function
            传入的导数函数，用于计算梯度。
    Returns:
        grad : ndarray
            梯度矩阵，形状为(batch_size, 15)。
    """
    # 初始化梯度矩阵，大小为 (batch_size, 15)
    w = w.reshape(-1, 1)
    grad = np.zeros((y.shape[0], 15))
    # 对输入 y 取对数，以增加数值稳定性
    y = np.log(1e-6 + y)

    # 计算 w0y 对应的梯度 (grad[:, 0])
    test = (1 / (w[1] ** 2)) * (y - w[0]) \
                - ((local == 3) | (local == 4) | (local == 6)) * (1 / (w[1] ** 2)) * (w[7] * qBD) \
                - ((local == 1) | (local == 3) | (local == 5) | (local == 6)) * (1 / (w[1] ** 2)) * (w[8] * qLS) \
                - ((local == 2) | (local == 4) | (local == 5) | (local == 6)) * (1 / (w[1] ** 2)) * (w[9] * qLF)[:, 0] 
    test1 = (1 / (w[1] ** 2)) * (y - w[0]) \
                - ((local == 3) | (local == 4) | (local == 6)) * (1 / (w[1] ** 2)) * (w[7] * qBD) \
                - ((local == 1) | (local == 3) | (local == 5) | (local == 6)) * (1 / (w[1] ** 2)) * (w[8] * qLS) \
                - ((local == 2) | (local == 4) | (local == 5) | (local == 6)) * (1 / (w[1] ** 2)) * (w[9] * qLF)   
    grad[:, 0] = ((1 / (w[1] ** 2)) * (y - w[0]) \
                - ((local == 3) | (local == 4) | (local == 6)) * (1 / (w[1] ** 2)) * (w[7] * qBD) \
                - ((local == 1) | (local == 3) | (local == 5) | (local == 6)) * (1 / (w[1] ** 2)) * (w[8] * qLS) \
                - ((local == 2) | (local == 4) | (local == 5) | (local == 6)) * (1 / (w[1] ** 2)) * (w[9] * qLF))[:, 0] 

    # 计算 wey 对应的梯度 (grad[:, 1])
    grad[:, 1] = ((-1 / w[1]) - (1 / w[1] ** 3) * (
        - y ** 2 - w[0] ** 2 + 2 * w[0] * y
        - ((local == 1) | (local == 3) | (local == 5) | (local == 6)) * (w[8] ** 2 - 2 * y * w[8] + 2 * w[0] * w[8]) * qLS
        - ((local == 2) | (local == 4) | (local == 5) | (local == 6)) * (w[9] ** 2 - 2 * y * w[9] + 2 * w[0] * w[9]) * qLF
        - ((local == 3) | (local == 4) | (local == 6)) * (w[7] ** 2 - 2 * y * w[7] + 2 * w[0] * w[7]) * qBD
        - (local == 3) * 2 * (w[7] * w[8] * qLS * qBD)
        - (local == 4) * 2 * (w[7] * w[9] * qLF * qBD)
        - (local == 6) * 2 * (w[7] * w[7] * w[9] * qLS * qLF * qBD)
    ))[:, 0] 

    # 计算 w0BD 对应的梯度 (grad[:, 2])
    grad[:, 2] = ((local == 3) * (
        qBD * qLS * df(w[2] + w[5] - (w[12] ** 2) / 2)
        - (1 - qBD) * qLS * df(-w[2] - w[5] - (w[12] ** 2) / 2)
        + qBD * (1 - qLS) * df(w[2] - (w[12] ** 2) / 2)
        - (1 - qBD) * (1 - qLS) * df(-w[2] - (w[12] ** 2) / 2)
    ) + (local == 4) * (
        qBD * qLF * df(w[2] + w[6] - (w[12] ** 2) / 2)
        - (1 - qBD) * qLF * df(-w[2] - w[6] - (w[12] ** 2) / 2)
        + qBD * (1 - qLF) * df(w[2] - (w[12] ** 2) / 2)
        - (1 - qBD) * (1 - qLF) * df(-w[2] - (w[12] ** 2) / 2)
    ) + (local == 6) * (
        qBD * qLS * qLF * df(w[2] + w[5] + w[6] - (w[12] ** 2) / 2)
        + qBD * qLS * (1 - qLF) * df(w[2] + w[5] - (w[12] ** 2) / 2)
        + qBD * (1 - qLS) * qLF * df(w[2] + w[6] - (w[12] ** 2) / 2)
        + qBD * (1 - qLS) * (1 - qLF) * df(w[2] - (w[12] ** 2) / 2)
    ) + (local == 6) * (
        -(1 - qBD) * qLS * qLF * df(-w[2] - w[5] - w[6] - (w[12] ** 2) / 2)
        - (1 - qBD) * qLS * (1 - qLF) * df(-w[2] - w[5] - (w[12] ** 2) / 2)
        - (1 - qBD) * (1 - qLS) * qLF * df(-w[2] - w[6] - (w[12] ** 2) / 2)
        - (1 - qBD) * (1 - qLS) * (1 - qLF) * df(-w[2] - (w[12] ** 2) / 2)
    ))[:, 0] 

    # 计算 w0LS 对应的梯度 (grad[:, 3])
    # w0LS
    grad[:, 3] = (((local == 1) | (local == 3) | (local == 5) | (local == 6)) * qLS * df(w[3] + w[13] * alLS - (w[10] ** 2) / 2) \
                - ((local == 1) | (local == 3) | (local == 5) | (local == 6)) * (1 - qLS) * df(-w[3] - w[13] * alLS - (w[10] ** 2) / 2))[:, 0] 

    # w0LF
    grad[:, 4] = (((local == 2) | (local == 4) | (local == 5) | (local == 6)) * qLF * df(w[4] + w[14] * alLF - (w[11] ** 2) / 2) \
                - ((local == 2) | (local == 4) | (local == 5) | (local == 6)) * (1 - qLF) * df(-w[4] - w[14] * alLF - (w[11] ** 2) / 2))[:, 0] 

    # wLSBD
    grad[:, 5] =( ((local == 3) * (qBD * qLS * df(w[2] + w[5] - (w[12] ** 2) / 2)
                                - (1 - qBD) * qLS * df(-w[2] - w[5] - (w[12] ** 2) / 2))
                + (local == 6) * (qBD * qLS * qLF * df(w[2] + w[5] + w[6] - (w[12] ** 2) / 2)
                                + qBD * qLS * (1 - qLF) * df(w[2] + w[5] - (w[12] ** 2) / 2)
                                - (1 - qBD) * qLS * qLF * df(-w[2] - w[5] - w[6] - (w[12] ** 2) / 2)
                                - (1 - qBD) * qLS * (1 - qLF) * df(-w[2] - w[5] - (w[12] ** 2) / 2))))[:, 0] 

    # wLFBD
    grad[:, 6] =(((local == 4) * (qBD * qLF * df(w[2] + w[6] - (w[12] ** 2) / 2)
                                - (1 - qBD) * qLF * df(-w[2] - w[6] - (w[12] ** 2) / 2))
                + (local == 6) * (qBD * qLF * qLS * df(w[2] + w[5] + w[6] - (w[12] ** 2) / 2)
                                + qBD * qLF * (1 - qLS) * df(w[2] + w[6] - (w[12] ** 2) / 2)
                                - (1 - qBD) * qLF * qLS * df(-w[2] - w[5] - w[6] - (w[12] ** 2) / 2)
                                - (1 - qBD) * qLF * (1 - qLS) * df(-w[2] - w[6] - (w[12] ** 2) / 2))))[:, 0] 

    # wBDy
    grad[:, 7] = (((local == 3) | (local == 4) | (local == 6)) * (-1 / (w[1] ** 2)) * qBD * (w[7] - y + w[0]) \
                 - (1 / (w[1] ** 2)) * (
                     (local == 3) * w[8] * qBD * qLS
                     + (local == 4) * w[9] * qBD * qLF
                     + (local == 6) * w[8] * w[9] * qBD * qLS * qLF
                 ))[:, 0] 

    # wLSy
    grad[:, 8] = (((local == 1) | (local == 3) | (local == 5) | (local == 6)) * (-1 / (w[1] ** 2)) * qLS * (w[8] - y + w[0]) \
                 - (1 / (w[1] ** 2)) * (
                     (local == 3) * w[7] * qBD * qLS
                     + (local == 6) * w[7] * w[9] * qBD * qLS * qLF
                 ))[:, 0] 

    # wLFy
    grad[:, 9] = (((local == 2) | (local == 4) | (local == 5) | (local == 6)) * (-1 / (w[1] ** 2)) * qLF * (w[9] - y + w[0]) \
                 - (1 / (w[1] ** 2)) * (
                     (local == 4) * w[7] * qBD * qLF
                     + (local == 6) * w[7] * w[8] * qBD * qLF * qLS
                 ))[:, 0] 

    # weLS
    grad[:, 10] = (((local == 1) | (local == 3) | (local == 5) | (local == 6)) * (
        -qLS * w[10] * df(w[3] + w[13] * alLS - (w[10] ** 2) / 2)
        - (1 - qLS) * w[10] * df(-w[3] - w[13] * alLS - (w[10] ** 2) / 2)
    ))[:, 0] 

    # weLF
    grad[:, 11] = (((local == 2) | (local == 4) | (local == 5) | (local == 6)) * (
        -qLF * w[11] * df(w[4] + w[14] * alLF - (w[11] ** 2) / 2)
        - (1 - qLF) * w[11] * df(-w[4] - w[14] * alLF - (w[11] ** 2) / 2)
    ))[:, 0] 

    # weBD
    grad[:, 12] = ((local == 3) * (
        -qBD * qLS * w[12] * df(w[2] + w[5] - (w[12] ** 2) / 2)
        - (1 - qBD) * qLS * w[12] * df(-w[2] - w[5] - (w[12] ** 2) / 2)
        - qBD * (1 - qLS) * w[12] * df(w[2] - (w[12] ** 2) / 2)
        - (1 - qBD) * (1 - qLS) * w[12] * df(-w[2] - (w[12] ** 2) / 2)
    ) + (local == 4) * (
        -qBD * qLF * w[12] * df(w[2] + w[6] - (w[12] ** 2) / 2)
        - (1 - qBD) * qLF * w[12] * df(-w[2] - w[6] - (w[12] ** 2) / 2)
        - qBD * (1 - qLF) * w[12] * df(w[2] - (w[12] ** 2) / 2)
        - (1 - qBD) * (1 - qLF) * w[12] * df(-w[2] - (w[12] ** 2) / 2)
    ) + (local == 6) * (
        -qBD * qLS * qLF * w[12] * df(w[2] + w[5] + w[6] - (w[12] ** 2) / 2)
        - qBD * qLS * (1 - qLF) * w[12] * df(w[2] + w[5] - (w[12] ** 2) / 2)
        - qBD * (1 - qLS) * qLF * w[12] * df(w[2] + w[6] - (w[12] ** 2) / 2)
        - qBD * (1 - qLS) * (1 - qLF) * w[12] * df(w[2] - (w[12] ** 2) / 2)
        - (1 - qBD) * qLS * qLF * w[12] * df(-w[2] - w[5] - w[6] - (w[12] ** 2) / 2)
        - (1 - qBD) * qLS * (1 - qLF) * w[12] * df(-w[2] - w[5] - (w[12] ** 2) / 2)
        - (1 - qBD) * (1 - qLS) * qLF * w[12] * df(-w[2] - w[6] - (w[12] ** 2) / 2)
        - (1 - qBD) * (1 - qLS) * (1 - qLF) * w[12] * df(-w[2] - (w[12] ** 2) / 2)
    ))[:, 0] 

        # waLS
    grad[:, 13] = (((local == 1) | (local == 3) | (local == 5) | (local == 6)) * (
        qLS * alLS * df(w[3] + w[13] * alLS - (w[10] ** 2) / 2)
        - (1 - qLS) * alLS * df(-w[3] - w[13] * alLS - (w[10] ** 2) / 2)
    ))[:, 0] 

    # waLF
    grad[:, 14] = (((local == 2) | (local == 4) | (local == 5) | (local == 6)) * (
        qLF * alLF * df(w[4] + w[14] * alLF - (w[11] ** 2) / 2)
        - (1 - qLF) * alLF * df(-w[4] - w[14] * alLF - (w[11] ** 2) / 2)
    ))[:, 0] 

    # 检查 grad 中是否有 NaN 值
    # if np.sum(np.isnan(grad)) > 0:
    #     print(grad)

    return grad

