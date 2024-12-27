import numpy as np
import pandas as pd
import time

from Tfxn_py import Tfxn
from parder_py import parder
from Loss_py import loss_fn
from Loss_py import df

def svi(Y, LS, LF, w, Nq, rho, delta, eps_0, LOCAL, lambda_, regu_type, sigma, prune_type):
    """
    随机变分推断函数

    参数：
    Y: 输入矩阵
    LS: LS 概率矩阵
    LF: LF 概率矩阵
    w: 权重向量
    Nq: 迭代次数
    rho: 学习率
    delta: 参数 delta
    eps_0: 停止条件的阈值
    LOCAL: LOCAL 矩阵
    lambda_: 正则化参数
    regu_type: 正则化类型
    sigma: 剪枝的阈值
    prune_type: 剪枝类型（"single" 或其他）
    
    返回：
    final_w, final_QBD, final_QLS, final_QLF, QLS, QLF, QBD, final_loss, best_loss, LOCAL
    QBD: 建筑物的后验概率矩阵
    QLS: 滑坡的后验概率矩阵
    QLF: 液化的后验概率矩阵
    """
    # 初始化
    # w = w.flatten()
    # Y = Y.astype(np.float64)
    # LS = LS.astype(np.float64)
    # LF = LF.astype(np.float64)
    # LOCAL = LOCAL.astype(np.int32)
    time_record = pd.DataFrame(columns=["epoch", "time"])

    QBD = 0.001 * np.random.rand(*Y.shape)
    QLS = LS.copy()
    QLF = LF.copy()
    loss = 1e+5
    loss_old = 0
    best_loss = -1e+6
    final_loss = []
    grad = np.zeros(len(w))
    epoches = 0

    # 计算 alpha
    t_alLS = np.log(LS / (1 - LS + 1e-4))
    t_alLF = np.log(LF / (1 - LF + 1e-4))
    t_alLS = np.clip(t_alLS, -6, 6)
    t_alLF = np.clip(t_alLF, -6, 6)
    final_w = w.copy()

    # 自定义停止函数
    if np.sum(LOCAL >= 5) == 0:
        my_condition_func = lambda x, y: y > 0
    else:
        my_condition_func = lambda x, y: (x > 0) or (y > 0)

    # 主循环
    start_time = time.time()
    while epoches < 30 and my_condition_func(np.sum(LOCAL == 5), abs(loss_old - loss) - eps_0):

        # 创建一个大小为 bsize 的位置样本批次
        totalnum = Y.size
        bsize = 500
        bnum = totalnum // bsize
        iD_sample = np.random.permutation(Y.size)
        iD = iD_sample[:bnum * bsize].reshape(bsize, -1)

        # 重置损失参数
        loss_old = loss
        loss = 0
        tmp_final_loss = []

        # 学习率衰减
        if epoches > 1 and epoches % 10 == 0:
            rho = max(rho * 0.1, 1e-4)

        # 对每个样本批次运行
        for i in range(bnum):
            #* 这里MATLAB和Python有较大差别，需要flatten()函数
            idx = iD[:, i].reshape(-1, 1)
            y = Y.flatten()[idx]          # 从 Y 中提取当前批次的样本
            qBD = QBD.flatten()[idx]      # 从 QBD 中提取当前批次的样本
            qLS = QLS.flatten()[idx]      # 从 QLS 中提取当前批次的样本
            qLF = QLF.flatten()[idx]      # 从 QLF 中提取当前批次的样本
            local = LOCAL.flatten()[idx]  # 从 LOCAL 中提取当前批次的样本
            alLS = t_alLS.flatten()[idx]  # 从 t_alLS 中提取当前批次的样本
            alLF = t_alLF.flatten()[idx]  # 从 t_alLF 中提取当前批次的样本

            # 迭代
            for nq in range(Nq):# Nq = 10, 是设定的后验概率迭代次数
                q = 1 / (1 + np.exp(-Tfxn(y, qBD, qLS, qLF, alLS, alLF, w, local, delta)))
                # 使用Tfxn函数计算后验概率更新
                qBD, qLS, qLF = q[:, 0], q[:, 1], q[:, 2]
                qBD = qBD.reshape(-1, 1)
                qLS = qLS.reshape(-1, 1)
                qLF = qLF.reshape(-1, 1)

                # 应用剪枝
                qBD[local < 3] = 0
                qBD[local == 5] = 0
                qLS[np.isin(local, [0, 2, 4])] = 0
                qLF[np.isin(local, [0, 1, 3])] = 0

                # 去除非常小的估计值
                qLS[qLS < 1e-6] = 0
                qLF[qLF < 1e-6] = 0

                # 应用 sigma 进行剪枝
                tqLS, tqLF = qLS.copy(), qLF.copy()
                if prune_type == "single":
                    qLS[tqLS < tqLF - sigma] = 0
                    qLF[tqLF < tqLS] = 0
                else:
                    qLS[tqLS < tqLF - sigma] = 0
                    qLF[tqLF < tqLS - sigma] = 0

                # 将后验估计合并到主矩阵中
                QBD.flat[iD[:, i]] = qBD
                QLS.flat[iD[:, i]] = qLS
                QLF.flat[iD[:, i]] = qLF
            #* 这个batch的迭代结束，得到一个batch的QBD, QLS, QLF

            # 修剪后的分类
            if np.sum(local >= 5) > 0:
                if prune_type == "single":
                    local[(tqLF < tqLS) & (local == 5)] = 1
                    local[(tqLF < tqLS) & (local == 6)] = 3
                    local[(tqLS < tqLF - sigma) & (local == 5)] = 2
                    local[(tqLS < tqLF - sigma) & (local == 6)] = 4
                else:
                    local[(tqLF < tqLS - sigma) & (local == 5)] = 1
                    local[(tqLF < tqLS - sigma) & (local == 6)] = 3
                    local[(tqLS < tqLF - sigma) & (local == 5)] = 2
                    local[(tqLS < tqLF - sigma) & (local == 6)] = 4
                LOCAL.flat[iD[:, i]] = local

            # 计算偏导数
            identity = np.column_stack([local >= 0, local >= 0, np.isin(local, [3, 4, 5]), np.isin(local, [1, 3, 5, 6]),
                                        np.isin(local, [2, 4, 5, 6]), np.isin(local, [3, 6]), np.isin(local, [4, 6]),
                                        np.isin(local, [3, 4, 6]), np.isin(local, [1, 3, 5, 6]),
                                        np.isin(local, [2, 4, 5, 6]), np.isin(local, [1, 3, 5, 6]),
                                        np.isin(local, [2, 4, 5, 6]), np.isin(local, [3, 4, 6]),
                                        np.isin(local, [1, 3, 5, 6]), np.isin(local, [2, 4, 5, 6])])
            grad_D = parder(y, qBD, qLS, qLF, alLS, alLF, w, local)
            assert identity.shape == grad_D.shape, "Shape mismatch between identity and grad_D"
            tmp_count = np.sum(identity, axis=0)
            grad = (np.sum(identity * grad_D, axis=0) / tmp_count)
            grad[tmp_count == 0] = 0

            # 计算正则化梯度
            regu_grad = lambda_ * 100 * (1 / (1 + np.exp(-0.01 * w)) - 1 / (1 + np.exp(0.01 * w)))

            # 确保 grad 和 regu_grad 的广播匹配
            if regu_type == 1:
                grad[[7, 8, 9]] -= regu_grad[[7, 8, 9]]
            else:
                grad[[13, 14]] -= regu_grad[[13, 14]]


            # 替换 NaN 值，防止错误传播
            grad = np.nan_to_num(grad)


            # 计算新的权重
            wnext = w + rho * grad
            wnext[[0, 2]] = np.minimum(wnext[[0, 2]], -1e-6)
            wnext[[13, 14]] = np.maximum(0, wnext[[13, 14]])
            wnext[1] = np.clip(wnext[1], 1e-3, 1)

            # 计算损失
            tmp_loss = np.mean(loss_fn(y, qBD, qLS, qLF, alLS, alLF, wnext, local, delta))
            loss += tmp_loss
            if (i+1) % 20 == 0:
                tmp_final_loss.append(tmp_loss)
            if (i+1) % 100 == 0:
                c_loss = np.mean(loss_fn(Y.flatten(), QBD.flatten(), QLS.flatten(), QLF.flatten(),
                                         t_alLS.flatten(), t_alLF.flatten(), wnext, LOCAL.flatten(), delta))
                if c_loss > best_loss:
                    final_QLS, final_QLF, final_QBD, final_w = QLS.copy(), QLF.copy(), QBD.copy(), wnext
                    best_loss = c_loss

            # 更新权重
            w = wnext

        end_time = time.time()
        time_record = pd.concat(
            [time_record, pd.DataFrame({"epoch": [epoches], "time": [end_time - start_time]})]
        )
        # 显示进度
        print(f"epoch loss is {loss / bnum}")
        epoches += 1
        final_loss.extend(tmp_final_loss)

    time_record.to_csv("time_record.csv", index=False)
    return final_w, QBD, QLS, QLF, final_QLS, final_QLF, final_QBD, final_loss, best_loss, LOCAL

# 需要定义的其他函数：
# - tfxn
# - parder
# - loss_fn