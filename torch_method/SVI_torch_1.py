################################################################################
# 完整示例：将 SVI/Tfxn/剪枝 等逻辑迁移到 PyTorch 中
################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
import os
import argparse

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
# 2. 定义 Dataset 类
###############################################################################
class SVIDataset(Dataset):
    """
    自定义 Dataset 类，用于包装 Y, LS, LF, LOCAL 等数据
    """
    def __init__(self, Y, LS, LF, LOCAL):
        """
        初始化数据

        参数:
            Y, LS, LF, LOCAL: np.ndarray，形状 (H, W)
        """
        super(SVIDataset, self).__init__()
        self.Y = torch.from_numpy(Y).float()
        self.LS = torch.from_numpy(LS).float()
        self.LF = torch.from_numpy(LF).float()
        self.LOC = torch.from_numpy(LOCAL).float()

        # Flatten 栅格数据
        self.Y = self.Y.view(-1)
        self.LS = self.LS.view(-1)
        self.LF = self.LF.view(-1)
        self.LOC = self.LOC.view(-1)

    def __len__(self):
        return self.Y.size(0)

    def __getitem__(self, idx):
        return self.Y[idx], self.LS[idx], self.LF[idx], self.LOC[idx]



###############################################################################
# 3. 定义 svi 函数
###############################################################################
def svi(
    Y, LS, LF,               # 数据
    w,                       # 初始 w (长度15)
    Nq,                      # 后验概率迭代次数 (原代码: 10)
    rho,                     # 学习率
    delta,                   # 参数 delta
    eps_0,                   # 收敛阈值
    LOCAL,                   # local矩阵
    lambda_,                 # 正则化参数
    regu_type,               # 正则化类型
    sigma,                   # 剪枝sigma
    prune_type,              # 剪枝类型 ("single"或"double")
    num_epochs=30,           # 最大训练轮数
    batch_size=5000000          # 每个批次大小
):
    """
    与原始SVI_py逻辑基本一致的PyTorch单卡训练版本

    1. 分批读取 (batch)
    2. 在每个batch内, 做 Nq次后验概率更新(基于 Tfxn)
    3. 做剪枝(将值设为 x/1e6 以替代设0, 保证可导)
    4. 汇总并在每个 epoch 计算一次 loss/backward/step
    5. 重复 epoch, 直到满足停止条件

    参数:
        Y, LS, LF, LOCAL: 原栅格数据, np.ndarray
        w: 初始15维权重
        Nq, rho, delta, eps_0, lambda_, regu_type, sigma, prune_type: 同原逻辑
        num_epochs: 最大训练轮数
        batch_size: 每个批次大小

    返回:
        final_w, QBD, QLS, QLF, final_QLS, final_QLF, final_QBD, final_loss, best_loss, LOCAL
        (与原svi函数相同)
    """

    device = torch.device("cuda:0")
    # 计算 alpha
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp_alLS = np.log(LS / (1 - LS + 1e-4))
        tmp_alLF = np.log(LF / (1 - LF + 1e-4))
    tmp_alLS = np.clip(tmp_alLS, -6, 6)
    tmp_alLF = np.clip(tmp_alLF, -6, 6)
    alLS_t = torch.from_numpy(tmp_alLS).float().to(device)
    alLF_t = torch.from_numpy(tmp_alLF).float().to(device)

    # 创建 Dataset 和 DataLoader
    dataset = SVIDataset(Y, LS, LF, LOCAL)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = SVIModel(Y.shape, init_w=w).to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=rho)

    # 记录过程
    final_loss = []
    best_loss = -1e9
    best_epoch = -1
    start_time = time.time()

    # 定义停止条件函数
    local_sum = (LOCAL == 5).sum()
    if local_sum == 0:
        def my_condition_func(x, y):
            return y > 0
    else:
        def my_condition_func(x, y):
            return (x > 0) or (y > 0)
        
    total_num = Y.size
    num_batches = len(dataloader)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(dataloader):
            y_batch, ls_batch, lf_batch, loc_batch = batch
            y_batch = y_batch.to(device).unsqueeze(1)  # shape: (batch_size, 1)
            ls_batch = ls_batch.to(device).unsqueeze(1)
            lf_batch = lf_batch.to(device).unsqueeze(1)
            loc_batch = loc_batch.to(device).unsqueeze(1)

            alLS_batch = torch.log(1e-6 + ls_batch / (1 - ls_batch + 1e-4)).clamp(-6, 6)
            alLF_batch = torch.log(1e-6 + lf_batch / (1 - lf_batch + 1e-4)).clamp(-6, 6)

            start_idx = batch_count * batch_size
            end_idx = start_idx + y_batch.size(0)
            flat_indices = torch.arange(start_idx, end_idx, device=device)

            # 在每个 batch 中进行 Nq 次后验概率更新
            for nq in range(Nq):
                qBD_batch = torch.sigmoid(model.QBD.view(-1)[flat_indices].unsqueeze(1))
                qLS_batch = torch.sigmoid(model.QLS.view(-1)[flat_indices].unsqueeze(1))
                qLF_batch = torch.sigmoid(model.QLF.view(-1)[flat_indices].unsqueeze(1))
                # 1) 调用 Tfxn
                g = model.Tfxn(
                    y_batch, 
                    qBD_batch,
                    qLS_batch,
                    qLF_batch,
                    alLS_batch, 
                    alLF_batch, 
                    loc_batch, 
                    delta
                )

                # 2) 计算 q = 1/(1+exp(-g))
                print(f"q_next shape: {g.shape}")
                q_next = torch.sigmoid(g)  # shape: (batch_size, 3)
                qBD_new, qLS_new, qLF_new = q_next[...,0], q_next[...,1], q_next[...,2]
                print(f"qBD_new shape: {qBD_new.shape}")
                # 3) 剪枝逻辑: 将需要置零的部分除以1e6
                mask_qBD_zero = (loc_batch < 3) | (loc_batch == 5)
                qBD_new = torch.where(mask_qBD_zero, qBD_new / 1e6, qBD_new)

                mask_qLS_zero = (loc_batch == 0) | (loc_batch == 2) | (loc_batch == 4)
                qLS_new = torch.where(mask_qLS_zero, qLS_new / 1e6, qLS_new)

                mask_qLF_zero = (loc_batch == 0) | (loc_batch == 1) | (loc_batch == 3)
                qLF_new = torch.where(mask_qLF_zero, qLF_new / 1e6, qLF_new)

                # 4) 根据 prune_type 和 sigma 进行进一步剪枝
                if prune_type == "single":
                    # single: qLS < qLF - sigma => qLS /1e6
                    #        qLF < qLS => qLF /1e6
                    cond1 = (qLS_new < qLF_new - sigma)
                    qLS_new = torch.where(cond1, qLS_new / 1e6, qLS_new)
                    cond2 = (qLF_new < qLS_new)
                    qLF_new = torch.where(cond2, qLF_new / 1e6, qLF_new)
                else:
                    # double: qLS < qLF - sigma => qLS /1e6
                    #         qLF < qLS - sigma => qLF /1e6
                    cond3 = (qLS_new < qLF_new - sigma)
                    qLS_new = torch.where(cond3, qLS_new / 1e6, qLS_new)
                    cond4 = (qLF_new < qLS_new - sigma)
                    qLF_new = torch.where(cond4, qLF_new / 1e6, qLF_new)

                # 5) 将更新后的 q 写回全局 QBD, QLS, QLF
                with torch.no_grad():
                    # 确保 flat_indices 是一维的
                    model.QBD.data.view(-1)[flat_indices] = torch.logit(qBD_new.squeeze(1).clamp(1e-6, 1-1e-6))
                    model.QLS.data.view(-1)[flat_indices] = torch.logit(qLS_new.squeeze(1).clamp(1e-6, 1-1e-6))
                    model.QLF.data.view(-1)[flat_indices] = torch.logit(qLF_new.squeeze(1).clamp(1e-6, 1-1e-6))

            # 计算当前 batch 的损失
            optimizer.zero_grad()
            loss = model.forward(
                Y=y_batch.view(-1, 1),
                LS=ls_batch.view(-1, 1),
                LF=lf_batch.view(-1, 1),
                LOCAL=loc_batch.view(-1, 1),
                alLS=alLS_batch,
                alLF=alLF_batch,
                delta=delta
            )
            loss.backward()

            # 正则化梯度
            if regu_type == 1:
                with torch.no_grad():
                    regu_grad = lambda_ * 100 * (1 / (1 + torch.exp(-0.01 * model.w)) - 1 / (1 + torch.exp(0.01 * model.w)))
                    model.w.grad[7:10] -= regu_grad[7:10]
            else:
                with torch.no_grad():
                    regu_grad = lambda_ * 100 * (1 / (1 + torch.exp(-0.01 * model.w)) - 1 / (1 + torch.exp(0.01 * model.w)))
                    model.w.grad[13:15] -= regu_grad[13:15]

            # 优化器更新
            optimizer.step()

            # Clamp 权重
            with torch.no_grad():
                model.w[0].clamp_(max=-1e-6)
                model.w[2].clamp_(max=-1e-6)
                model.w[13].clamp_(min=0.0)
                model.w[14].clamp_(min=0.0)
                model.w[1].clamp_(min=1e-3, max=1.0)

            # 累加损失
            epoch_loss += loss.item()
            batch_count += 1

        # 计算平均损失
        avg_loss = epoch_loss / batch_count
        final_loss.append(avg_loss)

        # 更新最佳损失
        if avg_loss > best_loss:
            best_loss = avg_loss
            best_epoch = epoch

        # 输出当前 epoch 信息
        print(f"[Epoch {epoch}] loss={avg_loss:.4f}, best={best_loss:.4f} at {best_epoch}")

        # 检查停止条件
        if epoch > 0 and abs(final_loss[-2] - final_loss[-1]) < eps_0:
            print("Convergence reached.")
            break

    # 训练结束, 导出结果
    final_w = model.w.detach().cpu().numpy().copy()

    # 将 QBD, QLS, QLF 转为概率
    QBD_ = torch.sigmoid(model.QBD).detach().cpu().numpy()
    QLS_ = torch.sigmoid(model.QLS).detach().cpu().numpy()
    QLF_ = torch.sigmoid(model.QLF).detach().cpu().numpy()

    # 返回与原函数相同的结果
    return (
        final_w,
        QBD_, QLS_, QLF_,
        QLS_.copy(), QLF_.copy(), QBD_.copy(),
        final_loss, best_loss, LOCAL
    )

###############################################################################
# 4. 测试代码
###############################################################################
def test_svi():
    """
    简单的测试函数，生成随机数据并运行 svi 函数
    """
    # 随机生成数据
    H, W = 50, 50

    np.random.seed(42)
    Y = np.random.rand(H, W).astype(np.float32)
    LS = np.random.rand(H, W).astype(np.float32) * 0.5
    LF = np.random.rand(H, W).astype(np.float32) * 0.5
    LOCAL = np.random.randint(0, 7, size=(H, W)).astype(np.int32)  # 0~6

    # 初始 w
    w_init = np.random.randn(15).astype(np.float32) * 0.01

    # 调用 svi
    out = svi(
        Y, LS, LF,
        w=w_init,
        Nq=10,
        rho=1e-3,
        delta=0.001,
        eps_0=1e-3,
        LOCAL=LOCAL,
        lambda_=0.0,
        regu_type=2,
        sigma=0.1,
        prune_type='double',
        num_epochs=30
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