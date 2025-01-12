import numpy as np
import rasterio
import scipy.io as sio
from rasterio.enums import Resampling
import os
from pruning_py import pruning
from SVI_py import svi
np.random.seed(42)
# 初始化
import time
from multiprocessing import Pool, cpu_count

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


def landslide_conversion_chunk(chunk_data):
    """
    子进程执行的函数:
    - 输入 chunk_data: (LS_sub, top_left_x, top_left_y)
        其中 LS_sub 是原始数组的一部分
    - 返回和 LS_sub 同维度的结果局部矩阵
    """
    LS_sub, offset_x, offset_y = chunk_data
    
    # --- 这里可以直接复用原始的 landslide_conversion 逻辑 ---
    # 但要改成仅对 LS_sub 进行处理
    index = LS_sub > 0
    log_LS = np.log(LS_sub[index])
    n = len(log_LS)

    # 构造多项式系数
    p = np.column_stack((
        np.full(n, 4.035),
        np.full(n, -3.042),
        np.full(n, 5.237),
        -7.592 - log_LS
    ))

    # 批量求根
    roots = np.array([np.roots(poly) for poly in p])
    real_roots = np.real(roots)

    # 写回结果
    sub_result = np.zeros_like(LS_sub, dtype=float)
    coords = np.argwhere(index)  # 在子块坐标系下
    for i, (x, y) in enumerate(coords):
        valid_vals = real_roots[i][np.isfinite(real_roots[i])]
        if valid_vals.size > 0:
            sub_result[x, y] = valid_vals.min()
        else:
            sub_result[x, y] = 0.0

    return (sub_result, offset_x, offset_y)


def parallel_landslide_conversion(LS, num_workers=None, chunks=4):
    """
    使用多进程并行对 LS 做 landslide_conversion
    - num_workers: 并行进程数(默认是 cpu_count())
    - chunks: 在维度上分块的数量(示例简化为对行方向分块)
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    # 1) 将 LS 在行方向做拆分
    rows = LS.shape[0]
    chunk_size = rows // chunks
    chunked_data = []
    for i in range(chunks):
        start = i * chunk_size
        # 最后一块可能包含多余行
        end = rows if (i == chunks - 1) else ((i+1) * chunk_size)
        LS_sub = LS[start:end, :]
        # 把子块和它在原图中的起始偏移量一起传递
        chunked_data.append((LS_sub, start, 0))

    # 2) 建立进程池并行执行
    with Pool(processes=num_workers) as p:
        results = p.map(landslide_conversion_chunk, chunked_data)

    # 3) 合并各子块结果
    final_result = np.zeros_like(LS, dtype=float)
    for sub_result, offset_x, offset_y in results:
        h, w = sub_result.shape
        final_result[offset_x:offset_x+h, offset_y:offset_y+w] = sub_result

    return final_result


def liquefaction_conversion(LF):
    """
    液化概率转换函数（保持不变）
    """
    index = LF > 0
    result = np.zeros_like(LF, dtype=float)
    result[index] = (
        np.log( (np.sqrt(0.4915 / LF[index]) - 1) / 42.40 )
        / (-9.165)
    )
    # 去掉负值、无穷大和 NaN
    result[result < 0] = 0
    result[~np.isfinite(result)] = 0
    return result

# ------------------------------#
#      主程序: 使用并行加速
# ------------------------------#
if __name__ == "__main__":
        # 导入地理数据
    location = '/Users/lixintong/Desktop/大三上/大数据并行计算/并行大作业/第三次作业/Bayesian_Causal-main/data'  # 文件所在的位置

    event = '2023_turkey_new'
    # event = '2024_japan2'
    if event == '2023_turkey_new':
        event_1 = '2023_turkey'
    else:
        event_1 = event

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

    start_time = time.time()
    new_LS = parallel_landslide_conversion(LS, num_workers=4, chunks=4)
    print("Converted Landslide Areal Percentages to Probabilities (Parallel)")
    print(f"Elapsed time for Landslide: {time.time() - start_time:.6f} seconds")

    start_time = time.time()
    new_LF = liquefaction_conversion(LF)
    print("Converted Liquefaction Areal Percentages to Probabilities")
    print(f"Elapsed time for Liquefaction: {time.time() - start_time:.6f} seconds")

    # 清理结果
    new_LF[new_LF < 0] = 0
    new_LS[new_LS < 0] = 0
    new_LS[np.isnan(new_LS)] = 0
    new_LF[np.isnan(new_LF)] = 0

    print(f"Total elapsed time: {time.time() - start_time:.6f} seconds")


'''
(three_env) (base) lixintong@bogon 大数据并行计算 % python 并行大作业/第三次作业/Bayesian_Causal-main/method_python/预处理CPU_1.py
Converted Landslide Areal Percentages to Probabilities (Parallel)
Elapsed time for Landslide: 131.904359 seconds
Converted Liquefaction Areal Percentages to Probabilities
Elapsed time for Liquefaction: 0.145399 seconds
Total elapsed time: 0.185044 seconds
'''