import numpy as np

def pruning(BD, LS, LF, sigma, side):
    """
    剪枝函数

    参数：
    BD: {0 - 没有建筑, 1 - 有建筑} 的矩阵
    LS: 概率矩阵，范围 [0, 1]
    LF: 概率矩阵，范围 [0, 1]
    sigma: 阈值，用于判断差距
    side: 选择 'single' 或其他值来选择不同的剪枝逻辑
    
    返回：
    model: 模型分类的矩阵，值为 {1, 2, 3, 4, 5, 6}
    """
    # 初始化一个与BD矩阵相同大小的全零矩阵
    model = np.zeros_like(BD, dtype=np.int32)  # 使用 int32 类型

    # 判断 side 是单一模型的情况
    if side == "single":
        # 1 - 只有 LS
        model += ((BD == 0) & (LF <= sigma)).astype(np.int32)

        # 2 - 只有 LF
        model += 2 * ((BD == 0) & (LF > sigma) & (LF > LS)).astype(np.int32)

        # 3 - LS 和 BD
        model += 3 * ((BD == 1) & (LF <= sigma)).astype(np.int32)

        # 4 - LF 和 BD
        model += 4 * ((BD == 1) & (LF > sigma) & (LF > LS)).astype(np.int32)

    else:
        # 1 - 只有 LS
        model += ((BD == 0) & (LS > LF + sigma) & (LS > 0)).astype(np.int32)

        # 2 - 只有 LF
        model += 2 * ((BD == 0) & (LF > LS + sigma) & (LF > 0)).astype(np.int32)

        # 3 - LS 和 BD
        model += 3 * ((BD == 1) & (LS > LF + sigma) & (LS > 0)).astype(np.int32)

        # 4 - LF 和 BD
        model += 4 * ((BD == 1) & (LF > LS + sigma) & (LF > 0)).astype(np.int32)

        # 5 - LF 和 LS
        model += 5 * ((BD == 0) & (np.abs(LF - LS) <= sigma)).astype(np.int32)

        # 6 - LF 和 LS 和 BD
        model += 6 * ((BD == 1) & (np.abs(LF - LS) <= sigma)).astype(np.int32)
    
    return model

# 示例使用
BD = np.array([[0, 1], [1, 0]])
LS = np.array([[0.3, 0.7], [0.2, 0.5]])
LF = np.array([[0.4, 0.6], [0.5, 0.3]])
sigma = 0.1
side = "single"

# result = pruning(BD, LS, LF, sigma, side)
# print(result)