import numpy as np
import rasterio
import scipy.io as sio
from rasterio.enums import Resampling
import os
from pruning_py import pruning
from SVI_py import svi
import datetime
np.random.seed(42)
# 初始化
import time
start_time = time.time()  # 开始计时，记录代码执行时间

# 导入地理数据
location = 'data'  # 文件所在的位置

event = '2024_japan2'
# event = '2024_japan2'
if event == '2023_turkey_new':
    event_1 = '2023_turkey'
else:
    event_1 = event

def read_raster(file_path):
    """
    读取栅格数据文件，并返回数据和其元数据配置
    参数:
        file_path: 栅格文件路径
    返回:
        data: 栅格数据
        profile: 栅格文件的元数据配置
    """
    with rasterio.open(file_path) as src:
        data = src.read(1, resampling=Resampling.nearest)  # 读取数据，并使用最近邻重采样
        profile = src.profile  # 获取文件元数据
    return data, profile

# 读取多个栅格文件
Y, Y_profile = read_raster(os.path.join(location, event, 'damage_proxy_map', f'{event_1}_damage_proxy_map.tif'))
BD, BD_profile = read_raster(os.path.join(location, event, 'building_footprint', f'{event_1}_building_footprint_rasterized.tif'))
LS, LS_profile = read_raster(os.path.join(location, event, 'prior_models', f'{event_1}_prior_landslide_model.tif'))
LF, LF_profile = read_raster(os.path.join(location, event, 'prior_models', f'{event_1}_prior_liquefaction_model.tif'))

# 数据修正
BD[BD > 0] = 1  # 将基础数据 BD 中所有大于 0 的值设为 1
Y[np.isnan(Y)] = 0  # 将 Y 数据中的 NaN 值设为 0
BD[np.isnan(BD)] = 0  # 将 BD 数据中的 NaN 值设为 0
LS[np.isnan(LS)] = 0  # 将滑坡数据 LS 中的 NaN 值设为 0
LF[np.isnan(LF)] = 0  # 将液化数据 LF 中的 NaN 值设为 0
Y = (Y + 11) / 20

# 将滑坡区域百分比转换为概率
new_LS = np.copy(LS)  # 创建滑坡数据的副本
index = np.where(LS > 0)  # 找到 LS 中大于 0 的元素索引
for i in range(len(index[0])):
    idx = (index[0][i], index[1][i])  # 获取当前元素的索引
    p = [4.035, -3.042, 5.237, (-7.592 - np.log(LS[idx]))]  # 构建多项式方程
    tmp_root = np.roots(p)  # 求解方程的根
    real_roots = tmp_root[np.iscomplex(tmp_root) == False]  # 筛选出实数根
    new_LS[idx] = np.real(real_roots)  # 将实数根存储到 new_LS，不取最大值
print('Converted Landslide Areal Percentages to Probabilities')  # 输出转换完成的信息
print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))  # 输出当前已耗时间

# 将液化区域百分比转换为概率
new_LF = np.copy(LF)  # 创建液化数据的副本
index = np.where(LF > 0)  # 找到 LF 中大于 0 的元素索引
for i in range(len(index[0])):
    idx = (index[0][i], index[1][i])  # 获取当前元素的索引
    new_LF[idx] = (np.log((np.sqrt(0.4915 / LF[idx]) - 1) / 42.40)) / (-9.165)  # 根据给定公式计算概率
    # new_LF[i] = (np.log((np.sqrt(0.4915 / LF[i]) - 1) / 42.40)) / (-9.165)
print('Converted Liquefaction Areal Percentages to Probabilities')  # 输出转换完成的信息
print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))  # 输出当前已耗时间

# 将概率值转换为非负数
new_LF[new_LF < 0] = 0  # 将 new_LF 中小于 0 的值设为 0
new_LS[new_LS < 0] = 0  # 将 new_LS 中小于 0 的值设为 0
new_LS[np.isnan(new_LS)] = 0  # 将 new_LS 中的 NaN 值设为 0
new_LF[np.isnan(new_LF)] = 0  # 将 new_LF 中的 NaN 值设为 0
tmp_LF = new_LF  # 临时存储液化数据
tmp_LS = new_LS  # 临时存储滑坡数据

#* 根据剪枝对局部模型进行分类 相当于删除一些图的
prune_type = 'double'  # 剪枝类型
# sigma = np.median(np.abs(new_LS[(LS > 0) & (LF > 0)] - new_LF[(LS > 0) & (LF > 0)]))  # 计算 LS 和 LF 之间差值的中位数
sigma = 0
LOCAL = pruning(BD, tmp_LS, tmp_LF, sigma, prune_type)  # 调用剪枝函数（需要实现） 数值为1-6
tmp_LS[(LOCAL == 5) | (LOCAL == 6)] = np.min(new_LS[new_LS > 0])  # 修改剪枝后的滑坡数据
tmp_LF[(LOCAL == 5) | (LOCAL == 6)] = np.min(new_LF[new_LF > 0])  # 修改剪枝后的液化数据
# 剪枝得到的矩阵和BD长得一样，但是值是1-6


# 设置 Lambda 项
lambda_term = 0  # Lambda 参数

# 初始化权重向量 w
# [w0;weps;w0BD;w0LS;w0LF;wLSBD;wLFBD;wBDy;wLSy;wLFy;weLS;weLF;weBD;waLS;waLF]
w = np.random.rand(15)  # 随机初始化权重向量，在01之间
w[[3, 4]] = 0  # 将第 4 和第 5 个权重设为 0
w[[0, 2]] = -1 * w[[0, 2]]  # 将第 1 和第 3 个权重取反
regu_type = 2  # 正则化类型

# 设置变分超参数
Nq = 10  # 后验概率迭代次数

# 设置权重更新参数
rho = 1e-3  # 学习步长
delta = 1e-3  # 权重优化的容差
eps_0 = 0.001  # 非负权重的下界

# 输出结果
# 注意：需要实现 SVI 函数
opt_w, opt_QBD, opt_QLS, opt_QLF, QLS, QLF, QBD, final_loss, best_loss, local = svi(
    Y, tmp_LS, tmp_LF, w, Nq, rho, delta, eps_0, LOCAL, lambda_term, regu_type, sigma, prune_type
)

# 将概率转换回区域百分比
final_QLS = np.exp(-7.592 + 5.237 * opt_QLS - 3.042 * opt_QLS**2 + 4.035 * opt_QLS**3)  # 根据公式转换滑坡概率
final_QLF = 0.4915 / (1 + 42.40 * np.exp(-9.165 * opt_QLF))**2  # 根据公式转换液化概率

# 将非常小的区域百分比舍入为零
final_QLS[final_QLS <= np.exp(-7.592)] = 0  # 滑坡概率小于阈值的设为 0
final_QLF[final_QLF <= 0.4915 / (1 + 42.40)**2] = 0  # 液化概率小于阈值的设为 0

# 移除水体区域的概率
final_QLS[(LS == 0) & (LF == 0)] = 0  # 将水体区域的滑坡概率设为 0
final_QLF[(LS == 0) & (LF == 0)] = 0  # 将水体区域的液化概率设为 0

output_dir = os.path.join(location, event, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

os.makedirs(output_dir, exist_ok =True)  # 创建输出文件夹
# 导出 GeoTIFF 文件
def write_raster(filename, data, profile):
    """
    将数据写入 GeoTIFF 文件
    参数:
        filename: 文件名
        data: 要写入的栅格数据
        profile: 栅格文件的元数据配置
    """
    filepath = os.path.join(output_dir, filename)  # 文件路径
    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(data, 1)  # 写入数据

# 导出滑坡、液化和基础数据的 GeoTIFF 文件
write_raster('QLS.tif', final_QLS, LS_profile)
write_raster('QLF.tif', final_QLF, LF_profile)
write_raster('QBD.tif', opt_QBD, BD_profile)

# 将所有数据导出到文件
filename = f"{location}lambda{lambda_term}_sigma{sigma}_prune{prune_type}.mat"
sio.savemat(os.path.join(output_dir, filename), {
    'opt_w': opt_w,
    'opt_QBD': opt_QBD,
    'opt_QLS': opt_QLS,
    'opt_QLF': opt_QLF,
    'QLS': QLS,
    'QLF': QLF,
    'QBD': QBD,
    'final_loss': final_loss,
    'best_loss': best_loss,
    'local': local
})

print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))  # 输出总耗时间