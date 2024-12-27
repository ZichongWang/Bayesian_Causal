################################################################################
# 完整示例：将 SVI/Tfxn/剪枝 等逻辑迁移到 PyTorch 中
################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import time

###############################################################################
# 1. 定义一个 PyTorch 模型 SVIModel
###############################################################################
class SVIModel(nn.Module):
    """
    将 15 维权重 w、后验概率 QBD/QLS/QLF 以及 Tfxn/loss_fn 等逻辑
    全部封装进一个 nn.Module，使用自动求导来更新参数。
    """

    def __init__(self, shape, init_w=None):
        """
        参数:
            shape: (H, W)，用于初始化 QBD/QLS/QLF 大小
            init_w: (可选) 初始 w(15,) 数组；若不指定则随机初始化
        """
        super(SVIModel, self).__init__()

        # 1) 15维可学习权重
        if init_w is not None:
            # 若用户指定了初始 w，则使用
            assert len(init_w) == 15, "w 必须是长度15的向量"
            w_init = torch.tensor(init_w, dtype=torch.float32)
        else:
            # 否则随机一个
            w_init = torch.rand(15, dtype=torch.float32)*0.1
        
        # 将 w_init 作为可学习参数
        self.w = nn.Parameter(w_init)

        # 2) 后验概率矩阵 QBD, QLS, QLF
        #    Kaiming 初始化, 大小和 shape 一致
        #    注意: 这些原本在老代码中会被迭代更新
        #          我们将它们视为模型参数(可学的)或可被在 forward 中重写
        self.QBD = nn.Parameter(torch.empty(*shape))
        nn.init.kaiming_normal_(self.QBD, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.QLS = nn.Parameter(torch.empty(*shape))
        nn.init.kaiming_normal_(self.QLS, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.QLF = nn.Parameter(torch.empty(*shape))
        nn.init.kaiming_normal_(self.QLF, a=0, mode='fan_in', nonlinearity='leaky_relu')


    ###########################################################################
    #  Tfxn: 复刻原来 Txfn_py.py 中的逻辑，但用 torch 实现
    ###########################################################################
    def Tfxn(self, y, qBD, qLS, qLF, alLS, alLF, local, delta):
        """
        对照 Txfn_py.py 完整逻辑，将 numpy 改为 torch。
        这里的 y, qBD, qLS, qLF, alLS, alLF, local 都是同尺寸的 torch.Tensor

        公式见原 Txfn_py.py: 
        Tfxn(y, qBD, qLS, qLF, alLS, alLF, w, local, delta) -> gBD, gLS, gLF
        不同之处在于，这里我们可以直接返回 [gBD, gLS, gLF] 合并后的 shape
        方便后面做 sigmoid。
        """
        w = self.w  # 取出 15维权重

        # 先对 y 做 log
        y = torch.log(1e-6 + y)

        # (与原文相同) 下方是完全照搬 Txfn_py 的运算，只是用 torch.where 或布尔索引
        # 计算 gBD, gLS, gLF
        # --------------------------------------------------------------------
        # 由于太长，这里原封不动搬运，并把 numpy 数组逻辑改为 torch 张量逻辑:
        
        # 为简洁，这里直接把 local==3 等做成布尔mask:
        loc_eq_1 = (local == 1)
        loc_eq_2 = (local == 2)
        loc_eq_3 = (local == 3)
        loc_eq_4 = (local == 4)
        loc_eq_5 = (local == 5)
        loc_eq_6 = (local == 6)

        # gBD 计算
        # --------------------------------------------------------------------
        # 这里的 f(...) 等价于在 Loss_py.py 里定义的: f(a) = -log(1+exp(a))
        # 也可以内联写:
        def f(a):
            return -torch.log(1.0 + torch.exp(a))

        gBD = (
            (loc_eq_3 * (
                qLS * f(-w[2] - w[5] + (w[12]**2)/2)
                + (1 - qLS) * f(-w[2] + (w[12]**2)/2)
                - qLS * f(w[2] + w[5] + (w[12]**2)/2)
                - (1 - qLS) * f(w[2] + (w[12]**2)/2)
            ))
            + (loc_eq_4 * (
                qLF * f(-w[2] - w[6] + (w[12]**2)/2)
                + (1 - qLF) * f(-w[2] + (w[12]**2)/2)
                - qLF * f(w[2] + w[6] + (w[12]**2)/2)
                - (1 - qLF) * f(w[2] + (w[12]**2)/2)
            ))
            # - ((local==3)|(local==4)|(local==6)) * ((w[7] - 2*y + 2*w[0]) / (2 * (w[1]**2))) * w[7]
            - ((loc_eq_3) | (loc_eq_4) | (loc_eq_6)) * ((w[7] - 2*y + 2*w[0])/(2*(w[1]**2))) * w[7]
            - (loc_eq_3) * (w[8]*w[7]*qLS)/(w[1]**2)
            - (loc_eq_4) * (w[9]*w[7]*qLF)/(w[1]**2)
            - (loc_eq_6) * (w[8]*w[9]*w[7]*qLS*qLF)/(w[1]**2)
            - (loc_eq_6)*(
                qLS*qLF*f(-w[2]-w[5]-w[6] + (w[12]**2)/2)
                + qLS*(1-qLF)*f(-w[2]-w[5] + (w[12]**2)/2)
                + (1-qLS)*qLF*f(-w[2]-w[6] + (w[12]**2)/2)
                - (1-qLS)*(1-qLF)*f(w[2] + (w[12]**2)/2)
                - (
                    qLS*qLF*f(w[2]+w[5]+w[6] + (w[12]**2)/2)
                    - qLS*(1-qLF)*f(w[2]+w[5] + (w[12]**2)/2)
                    - (1-qLS)*qLF*f(w[2]+w[6] + (w[12]**2)/2)
                    - (1-qLS)*(1-qLF)*f(w[2] + (w[12]**2)/2)
                )
            )
        )

        # gLS 计算 (同理)
        # --------------------------------------------------------------------
        gLS = ((loc_eq_1 | loc_eq_3 | loc_eq_5 | loc_eq_6) * (
            f(-w[3] - w[13]*alLS + (w[10]**2)/2)
            - f(w[3] + w[13]*alLS + (w[10]**2)/2)
        ))
        gLS += (
            (loc_eq_3)*(
                qBD*(f(-w[2] - w[5] + (w[12]**2)/2) - f(-w[2] + (w[12]**2)/2))
                + (1-qBD)*(f(w[2] + w[5] + (w[12]**2)/2) - f(w[2] + (w[12]**2)/2))
            )
        )
        gLS += (
            (loc_eq_6)*(
                qBD*(
                    qLF*f(-w[2]-w[5]-w[6] + (w[12]**2)/2)
                    + (1-qLF)*f(-w[2]-w[5] + (w[12]**2)/2)
                    - qLF*f(-w[2]-w[6] + (w[12]**2)/2)
                    - (1-qLF)*f(-w[2] + (w[12]**2)/2)
                )
            )
        )
        gLS += (
            (loc_eq_6)*(
                (1-qBD)*(
                    qLF*f(w[2]+w[5]+w[6] + (w[12]**2)/2)
                    + (1-qLF)*f(w[2]+w[5] + (w[12]**2)/2)
                    - qLF*f(w[2]+w[6] + (w[12]**2)/2)
                    - (1-qLF)*f(w[2] + (w[12]**2)/2)
                )
            )
        )
        gLS -= ((loc_eq_1 | loc_eq_3 | loc_eq_5 | loc_eq_6)) * ((w[8] - 2*y + 2*w[0])/(2*(w[1]**2))) * w[8]
        gLS -= (loc_eq_3)*( (w[8]*w[9]*qBD)/(w[1]**2) )
        gLS -= (loc_eq_5)*( (w[9]*w[10]*qLF)/(w[1]**2) )
        gLS -= (loc_eq_6)*( (w[8]*w[9]*w[10]*qLF*qBD)/(w[1]**2) )
        gLS -= ((loc_eq_5 | loc_eq_6))*( qLF/(2*delta) )

        # gLF 计算 (同理)
        # --------------------------------------------------------------------
        gLF = ((loc_eq_2 | loc_eq_4 | loc_eq_5 | loc_eq_6)*(
            f(-w[4] - w[14]*alLF + (w[11]**2)/2)
            - f(w[4] + w[14]*alLF + (w[11]**2)/2)
        ))
        gLF += ((loc_eq_4 | loc_eq_6)*(
            qBD*(f(-w[2] - w[6] + (w[12]**2)/2) - f(-w[2] + (w[12]**2)/2))
            + (1-qBD)*(f(w[2] + w[6] + (w[12]**2)/2) - f(w[2] + (w[12]**2)/2))
        ))
        gLF += (
            (loc_eq_6)*(
                qBD*(
                    qLS*f(-w[2]-w[5]-w[6] + (w[12]**2)/2)
                    + (1-qLS)*f(-w[2]-w[6] + (w[12]**2)/2)
                    - qLS*f(-w[2]-w[5] + (w[12]**2)/2)
                    - (1-qLS)*f(-w[2] + (w[12]**2)/2)
                )
            )
        )
        gLF += (
            (loc_eq_6)*(
                (1-qBD)*(
                    qLS*f(w[2]+w[5]+w[6] + (w[12]**2)/2)
                    + (1-qLS)*f(w[2]+w[6] + (w[12]**2)/2)
                    - qLS*f(w[2]+w[5] + (w[12]**2)/2)
                    - (1-qLS)*f(w[2] + (w[12]**2)/2)
                )
            )
        )
        gLF -= ((loc_eq_2 | loc_eq_4 | loc_eq_5 | loc_eq_6)) * ((w[9] - 2*y + 2*w[0])/(2*(w[1]**2))) * w[9]
        gLF -= (loc_eq_4)*((w[8]*w[10]*qBD)/(w[1]**2))
        gLF -= (loc_eq_5)*((w[9]*w[10]*qLS)/(w[1]**2))
        gLF -= (loc_eq_6)*((w[8]*w[9]*w[10]*qLS*qBD)/(w[1]**2))
        gLF -= ((loc_eq_5 | loc_eq_6))*( qLS/(2*delta) )

        # 拼成 (N, 3) 形状: [gBD, gLS, gLF]
        # 但这里我们在和原代码保持： 只是在前向中得到 T
        # 下游可以再做 sigmoid
        g = torch.stack([gBD, gLS, gLF], dim=-1)

        return g


    ###########################################################################
    #  loss_fn: 复刻原 Loss_py.py 逻辑，但用 torch 实现 + 自动求导
    ###########################################################################
    def loss_fn(self, Y, qBD, qLS, qLF, alLS, alLF, local, delta):
        """
        这里对照 Loss_py.py:
         LBD + LLS + LLF + Leps + L_ex

        由于原逻辑非常长，这里直接搬运 + torch 化
        """
        w = self.w
        y = torch.log(1e-6 + Y)

        # 为简化: 先定义 f(a) = -log(1 + exp(a))
        def f(a): return -torch.log(1.0 + torch.exp(a))

        # 先把 qBD,qLS,qLF clamp到 (1e-6,1-1e-6) 避免 log(0)
        f_qBD = torch.clamp(qBD, 1e-6, 1.0-1e-6)
        f_qLS = torch.clamp(qLS, 1e-6, 1.0-1e-6)
        f_qLF = torch.clamp(qLF, 1e-6, 1.0-1e-6)

        loc_eq_0 = (local == 0)
        loc_eq_1 = (local == 1)
        loc_eq_2 = (local == 2)
        loc_eq_3 = (local == 3)
        loc_eq_4 = (local == 4)
        loc_eq_5 = (local == 5)
        loc_eq_6 = (local == 6)

        # --- LBD ---------------------------------------------------------
        LBD = torch.zeros_like(Y)

        # (local == 3)
        mask3 = loc_eq_3
        # qBD * ...
        part1 = mask3 * qBD * (qLS*f(-w[2]-w[5] + (w[12]**2)/2) + (1-qLS)*f(-w[2] + (w[12]**2)/2))
        part2 = mask3 * (1-qBD)*( qLS*f(w[2]+w[5] + (w[12]**2)/2) + (1-qLS)*f(w[2] + (w[12]**2)/2) )
        # 和原本的顺序略有区别

        # (local == 4)
        mask4 = loc_eq_4
        part3 = mask4 * qBD*(qLF*f(-w[2]-w[6] + (w[12]**2)/2) + (1-qLF)*f(-w[2] + (w[12]**2)/2))
        part4 = mask4 * (1-qBD)*(qLF*f(w[2]+w[6] + (w[12]**2)/2) + (1-qLF)*f(w[2] + (w[12]**2)/2))

        # (local == 6)
        mask6 = loc_eq_6
        part5 = mask6*(
            qBD*(
                qLS*qLF*f(-w[2]-w[5]-w[6] + (w[12]**2)/2)
                + qLS*(1-qLF)*f(-w[2]-w[5] + (w[12]**2)/2)
                + (1-qLS)*qLF*f(-w[2]-w[6] + (w[12]**2)/2)
                + (1-qLS)*(1-qLF)*f(-w[2] + (w[12]**2)/2)
            )
            + (1-qBD)*(
                qLS*qLF*f(w[2]+w[5]+w[6] + (w[12]**2)/2)
                + qLS*(1-qLF)*f(w[2]+w[5] + (w[12]**2)/2)
                + (1-qLS)*qLF*f(w[2]+w[6] + (w[12]**2)/2)
                + (1-qLS)*(1-qLF)*f(w[2] + (w[12]**2)/2)
            )
        )
        # - ((local==3)|(local==4)|(local==6)) * (qBD * log(...) + (1-qBD)*log(...))
        part6 = (-1) * ((loc_eq_3 | loc_eq_4 | loc_eq_6)) * (
            qBD*torch.log(f_qBD+1e-6) + (1-qBD)*torch.log(1-f_qBD+1e-6)
        )

        LBD = part1 + part2 + part3 + part4 + part5 + part6
        # 其他local==0,1,2,5 的地方 LBD=0 (默认)
        #* 为了确保可导，将特定部位缩小，而非直接设置成0

        lbd_mask = (loc_eq_0 | loc_eq_1 | loc_eq_2 | loc_eq_5).float()
        LBD = LBD * (1 - lbd_mask + lbd_mask * 1e-6) 

        # --- LLS ---------------------------------------------------------
        LLS = torch.zeros_like(Y)
        mask_ls = (loc_eq_1 | loc_eq_3 | loc_eq_5 | loc_eq_6)
        part_ls1 = mask_ls*(
            qLS*f(-w[3]-w[13]*alLS + (w[10]**2)/2)
            + (1-qLS)*f(w[3] + w[13]*alLS + (w[10]**2)/2)
            - qLS*torch.log(f_qLS+1e-6) - (1-qLS)*torch.log(1-f_qLS+1e-6)
        )
        LLS = part_ls1
        lls_mask = (loc_eq_0 | loc_eq_2 | loc_eq_4).float()
        LLS = LLS * (1 - lls_mask + lls_mask * 1e-6)

        # --- LLF ---------------------------------------------------------
        LLF = torch.zeros_like(Y)
        mask_lf = (loc_eq_2 | loc_eq_4 | loc_eq_5 | loc_eq_6)
        part_lf1 = mask_lf*(
            qLF*f(-w[4]-w[14]*alLF + (w[11]**2)/2)
            + (1-qLF)*f(w[4] + w[14]*alLF + (w[11]**2)/2)
            - qLF*torch.log(f_qLF+1e-6) - (1-qLF)*torch.log(1-f_qLF+1e-6)
        )
        LLF = part_lf1
        llf_mask = (loc_eq_0 | loc_eq_1 | loc_eq_3).float()
        LLF = LLF * (1 - llf_mask + llf_mask * 1e-6)

        # --- Leps (仿照原 ep 系数) ----------------------------------------
        # 参考: - (0.5*log(w[1]^2) + (y-w[0])^2/(2*w[1]^2)) + ...
        Leps = -(
            0.5*torch.log(w[1]**2 + 1e-6)
            + (y - w[0])*(y - w[0])/(2*(w[1]**2)+1e-6)
        )
        Leps -= (1/(2*(w[1]**2))) * ((loc_eq_3|loc_eq_4|loc_eq_6))*w[7]*qBD*(w[7]-2*y+2*w[0])
        Leps -= (1/(2*(w[1]**2))) * ((loc_eq_1|loc_eq_3|loc_eq_5|loc_eq_6))*w[8]*qLS*(w[8]-2*y+2*w[0])
        Leps -= (1/(2*(w[1]**2))) * ((loc_eq_2|loc_eq_4|loc_eq_5|loc_eq_6))*w[9]*qLF*(w[9]-2*y+2*w[0])
        Leps -= (1/(w[1]**2))*(
            (loc_eq_3)*w[7]*w[8]*qBD*qLS
            + (loc_eq_4)*w[7]*w[9]*qBD*qLF
            + (loc_eq_5)*w[8]*w[9]*qLS*qLF
            + (loc_eq_6)*w[7]*w[8]*w[9]*qBD*qLS*qLF
        )
        Leps -= y  # 原代码里最后多了个 -y

        # --- L_ex ( (local==5)|(local==6) ) * -(f_qLS*f_qLF)/(2*delta ) ----
        mask_56 = (loc_eq_5 | loc_eq_6)
        L_ex = -1.0 * mask_56 * (f_qLS*f_qLF)/(2*delta)
        L_ex = torch.clamp(L_ex, min=-1e2)

        # 总
        L = LBD + LLS + LLF + Leps + L_ex
        #* 为了最小化，加了一个负号！！！！
        return -L


    ###########################################################################
    #  forward: 将 QBD, QLS, QLF 激活到 [0,1]，再计算 loss
    ###########################################################################
    def forward(self, Y, LS, LF, LOCAL, alLS, alLF, delta):
        """
        一次前向: 
          1) 先把 QBD, QLS, QLF 用 sigmoid 映射到 [0,1] 
          2) 调用 loss_fn
        返回: mean loss (标量)
        """
        # sigmoid 映射 -> [0,1]
        qBD = torch.sigmoid(self.QBD)
        qLS = torch.sigmoid(self.QLS)
        qLF = torch.sigmoid(self.QLF)

        # 计算 loss
        L = self.loss_fn(Y, qBD, qLS, qLF, alLS, alLF, LOCAL, delta)
        loss_mean = torch.mean(L)
        return loss_mean


###############################################################################
# 2. 主函数: svi(...)，与原函数签名保持一致
###############################################################################
def svi(Y, LS, LF,
        w,           # 初始 w
        Nq,          # 后验概率迭代次数 (原代码: 10)
        rho,         # 学习率
        delta,       # 参数 delta
        eps_0,       # 收敛阈值
        LOCAL,       # local矩阵
        lambda_,     # 正则
        regu_type,   # 正则化类型
        sigma,       # 剪枝sigma
        prune_type   # 剪枝类型
    ):
    """
    说明:
      - 该函数在原代码里会做多次 epoch，并在每个 epoch 内对 (QBD, QLS, QLF) 做局部更新、剪枝等。
      - 在 PyTorch 中, 我们把 (QBD, QLS, QLF, w) 都放到 model 里用自动求导。
      - 下面的流程示范了如何大体迁移原先 SVI_py.py 的训练循环:
        1) 构建 SVIModel
        2) 准备 alLS, alLF (和原similar)
        3) for epoch in range(30): 
             forward -> backward -> step
             做一些剪枝(可使用 in-place 或 mask 方式)
             监控loss等
        4) 返回最终的 w, QBD, QLS, QLF, loss等
    """

    # 把 Numpy array 转成 torch.Tensor
    device = torch.device("cuda:0")  # 你可以改成 "cuda" 用GPU
    Y_t   = torch.from_numpy(Y).float().to(device)
    LS_t  = torch.from_numpy(LS).float().to(device)
    LF_t  = torch.from_numpy(LF).float().to(device)
    LOC_t = torch.from_numpy(LOCAL).float().to(device)

    # 计算 alpha
    # 原代码: t_alLS = log(LS/(1-LS+1e-4)), clip在[-6,6]
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp_alLS = np.log(LS/(1-LS+1e-4))
    tmp_alLS = np.clip(tmp_alLS, -6, 6)
    alLS_t = torch.from_numpy(tmp_alLS).float().to(device)

    with np.errstate(divide='ignore', invalid='ignore'):
        tmp_alLF = np.log(LF/(1-LF+1e-4))
    tmp_alLF = np.clip(tmp_alLF, -6, 6)
    alLF_t = torch.from_numpy(tmp_alLF).float().to(device)

    # 构建 PyTorch 模型
    shape = Y.shape
    model = SVIModel(shape, init_w=w).to(device)

    # 这里构建一个简单的优化器 (SGD 或 Adam)
    # 学习率用 rho
    optimizer = optim.Adam(model.parameters(), lr=rho)

    # 记录过程
    time_record = []
    final_loss = []
    best_loss = -1e9
    best_epoch = -1
    start_time = time.time()

    # 主循环, 最多 30 epoch
    max_epoch = 30
    epoches = 0
    loss_old = 1e5
    loss_val = 0

    # 当 (local>=5) == 0 时, my_condition_func = y>0
    # 当 (local>=5) != 0 时, my_condition_func = (x>0) or (y>0)
    # 下面只是把原文的逻辑做对应
    if (LOCAL >= 5).sum() == 0:
        def my_condition_func(x, y):
            return (y > 0)
    else:
        def my_condition_func(x, y):
            return (x > 0) or (y > 0)

    while epoches < max_epoch and my_condition_func(np.sum(LOCAL==5), abs(loss_old - loss_val) - eps_0):
        
        # 在每个epoch内，先进行后验更新，使用Tfxn函数，然后进行剪枝等操作

        for nq in range(Nq):
            g = model.Tfxn(
                Y_t, model.QBD, model.QLS, model.QLF, alLS_t, alLF_t, LOC_t, delta
            )
            q_next = torch.sigmoid(g)
            qBD_new, qLS_new, qLF_new = q_next[...,0], q_next[...,1], q_next[...,2]

            #* 剪枝，为了保证可导，这里不直接置零，而是缩小
            mask_qBD_zero = (LOC_t < 3) | (LOC_t == 5)
            qBD_new = torch.where(mask_qBD_zero, qBD_new / 1e6, qBD_new)
            mask_qLS_zero = (LOC_t == 0) | (LOC_t == 2) | (LOC_t == 4)
            qLS_new = torch.where(mask_qLS_zero, qLS_new / 1e6, qLS_new)
            mask_qLF_zero = (LOC_t == 0) | (LOC_t == 1) | (LOC_t == 3)
            qLF_new = torch.where(mask_qLF_zero, qLF_new / 1e6, qLF_new)

            tqLS = qLS_new
            tqLF = qLF_new


            if prune_type == 'single':
                # single: qLS < qLF - sigma => qLS /1e6
                #         qLF < qLS => qLF /1e6
                cond1 = (tqLS < tqLF - sigma)
                qLS_new = torch.where(cond1, qLS_new / 1e6, qLS_new)
                cond2 = (tqLF < tqLS)
                qLF_new = torch.where(cond2, qLF_new / 1e6, qLF_new)
            else:
                # double: qLS < qLF - sigma => qLS /1e6
                #         qLF < qLS - sigma => qLF /1e6
                cond3 = (tqLS < tqLF - sigma)
                qLS_new = torch.where(cond3, qLS_new / 1e6, qLS_new)
                cond4 = (tqLF < tqLS - sigma)
                qLF_new = torch.where(cond4, qLF_new / 1e6, qLF_new)
            
            with torch.no_grad():
                model.QBD = nn.Parameter(qBD_new)
                model.QLS = nn.Parameter(qLS_new)
                model.QLF = nn.Parameter(qLF_new)

        if torch.sum(LOC_t >= 5) > 0:
            with torch.no_grad():
                if prune_type == "single":
                    # 更新 LOC_t 值（单一剪枝规则）
                    LOC_t[(tqLF < tqLS) & (LOC_t == 5)] = 1
                    LOC_t[(tqLF < tqLS) & (LOC_t == 6)] = 3
                    LOC_t[(tqLS < tqLF - sigma) & (LOC_t == 5)] = 2
                    LOC_t[(tqLS < tqLF - sigma) & (LOC_t == 6)] = 4
                else:
                    # 更新 LOC_t 值（其他剪枝规则）
                    LOC_t[(tqLF < tqLS - sigma) & (LOC_t == 5)] = 1
                    LOC_t[(tqLF < tqLS - sigma) & (LOC_t == 6)] = 3
                    LOC_t[(tqLS < tqLF - sigma) & (LOC_t == 5)] = 2
                    LOC_t[(tqLS < tqLF - sigma) & (LOC_t == 6)] = 4

        
        # 每个 epoch
        optimizer.zero_grad()
        # 前向
        loss = model(Y_t, LS_t, LF_t, LOC_t, alLS_t, alLF_t, delta)
        loss_val = loss.item()

        # 反向
        loss.backward()

        # 如果需要正则化(比如对 w某些维度), 也可在此处手动调 w.grad
        if regu_type == 1:
            # 例如 regu_grad 只对 w[7,8,9]
            with torch.no_grad():
                regu_grad = lambda_*100*(1/(1+torch.exp(-0.01*model.w)) - 1/(1+torch.exp(0.01*model.w)))
                model.w.grad[7] -= regu_grad[7]
                model.w.grad[8] -= regu_grad[8]
                model.w.grad[9] -= regu_grad[9]
        else:
            # 对 w[13,14] 做同理
            with torch.no_grad():
                regu_grad = lambda_*100*(1/(1+torch.exp(-0.01*model.w)) - 1/(1+torch.exp(0.01*model.w)))
                model.w.grad[13] -= regu_grad[13]
                model.w.grad[14] -= regu_grad[14]

        # 更新
        optimizer.step()

        # 一些后处理(如剪枝, 或对 w[0], w[2], w[13], w[14], w[1] 的约束)
        # 原代码: wnext[[0,2]] = min(wnext[[0,2]], -1e-6)
        with torch.no_grad():
            model.w[0]  = torch.clamp(model.w[0], max=-1e-6)
            model.w[2]  = torch.clamp(model.w[2], max=-1e-6)
            model.w[13] = torch.clamp(model.w[13], min=0.)
            model.w[14] = torch.clamp(model.w[14], min=0.)
            model.w[1]  = torch.clamp(model.w[1], 1e-3, 1.)


        # 统计
        end_t = time.time()
        time_record.append((epoches, end_t - start_time))
        final_loss.append(loss_val)

        if loss_val > best_loss:
            best_loss = loss_val
            best_epoch = epoches

        print(f"Epoch {epoches}, loss={loss_val:.4f}, best={best_loss:.4f}")
        epoches += 1
        loss_old = loss_val

    # 训练结束, 返回和原函数同样的若干值
    # -----------------------------------------------------------
    # 取最终的 w
    final_w = model.w.detach().cpu().numpy().copy()

    # 取最终的 QBD/QLS/QLF
    QBD_ = torch.sigmoid(model.QBD).detach().cpu().numpy()
    QLS_ = torch.sigmoid(model.QLS).detach().cpu().numpy()
    QLF_ = torch.sigmoid(model.QLF).detach().cpu().numpy()

    # (原函数里返回 10 项):
    # final_w, QBD, QLS, QLF, final_QLS, final_QLF, final_QBD, final_loss, best_loss, LOCAL
    # 这里为了与之对齐:
    final_QBD = QBD_.copy()
    final_QLS = QLS_.copy()
    final_QLF = QLF_.copy()
    best_loss_ = best_loss
    local_ = LOC_t.cpu()  # 未在这段示例中修改 local
    print(f"type of local_: {type(local_)}, \n type of best loss: {type(best_loss_)}")

    return (
        final_w, 
        QBD_, QLS_, QLF_,
        final_QLS, final_QLF, final_QBD,
        final_loss, best_loss_, local_
    )


###############################################################################
# 3. 测试代码
###############################################################################
def test_svi():
    """
    简单地随机生成 Y/LS/LF/LOCAL, 
    然后调用 svi(...) 进行测试，打印最后输出。
    """
    # 随机生成数据
    H, W = 4816, 5647

    np.random.seed(42)
    Y = np.random.rand(H, W).astype(np.float32)
    LS = np.random.rand(H, W).astype(np.float32)*0.5
    LF = np.random.rand(H, W).astype(np.float32)*0.5
    LOCAL = np.random.randint(0, 7, size=(H, W)).astype(np.int32)  # 0~6

    # 初始 w
    w_init = np.random.randn(15).astype(np.float32)*0.01
    # 调用 svi
    out = svi(
        Y, LS, LF,
        w_init,
        Nq=10,
        rho=1e-3,
        delta=0.001,
        eps_0=1e-3,
        LOCAL=LOCAL,
        lambda_=0.0,
        regu_type=2,
        sigma=0.1,
        prune_type='double'
    )

    # 解包
    final_w, QBD_, QLS_, QLF_, final_QLS, final_QLF, final_QBD, final_loss, best_loss, local_ = out

    print("============== SVI Finished ==============")
    print("final_w:", final_w)
    print("best_loss:", best_loss)
    print("len(final_loss):", len(final_loss))
    print("QBD_.shape:", QBD_.shape)
    print("QLS_.shape:", QLS_.shape)
    print("QLF_.shape:", QLF_.shape)


# 如果直接执行本脚本，可在此处调用测试
if __name__ == "__main__":
    test_svi()
