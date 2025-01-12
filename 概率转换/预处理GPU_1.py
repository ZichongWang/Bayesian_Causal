import os
import time
import numpy as np
import rasterio
from rasterio.enums import Resampling
from multiprocessing import Pool, cpu_count

# ------------------------
# 1) 数据读取相关函数
# ------------------------
def read_raster(file_path):
    """
    读取栅格数据文件，并返回数据和其元数据配置
    参数:
        file_path: 栅格文件路径
    返回:
        data: 栅格数据 (numpy 数组)
        profile: 栅格文件的元数据配置 (dict)
    """
    with rasterio.open(file_path) as src:
        data = src.read(1, resampling=Resampling.nearest)  # 读取第一波段
        profile = src.profile
    return data, profile


# ------------------------
# 2) 滑坡概率转换：子进程执行函数
# ------------------------
def landslide_conversion_chunk(chunk_data):
    """
    子进程执行的函数:
    输入: chunk_data = (LS_sub, offset_x, offset_y)
        - LS_sub: 原数据 LS 对应的子块
        - offset_x, offset_y: 该子块在原图中的左上角坐标(行、列)
    返回: (sub_result, offset_x, offset_y)
        - sub_result: 子块计算后的结果矩阵 (与 LS_sub 同维度)
    """
    LS_sub, offset_x, offset_y = chunk_data
    
    # 找到 LS_sub 中 > 0 的像元
    index = LS_sub > 0
    if not np.any(index):
        # 如果这个子块里全部 <= 0，直接返回 0 矩阵
        return (np.zeros_like(LS_sub, dtype=float), offset_x, offset_y)

    # 求 log(LS_sub)；只对 >0 的位置求
    log_LS = np.log(LS_sub[index])
    n = len(log_LS)

    # 构造多项式系数 (n x 4)
    # 形如: 4.035*x^3 - 3.042*x^2 + 5.237*x + (-7.592 - ln(LS))
    p = np.column_stack((
        np.full(n, 4.035),         # t^3
        np.full(n, -3.042),        # t^2
        np.full(n, 5.237),         # t^1
        -7.592 - log_LS            # 常数项
    ))

    # 对每个多项式求根 (n x 3)
    roots = np.array([np.roots(poly) for poly in p])
    # 提取实部
    real_roots = np.real(roots)

    # 分配结果子矩阵
    sub_result = np.zeros_like(LS_sub, dtype=float)
    coords = np.argwhere(index)  # 子块坐标系下的 (x, y)
    for i, (x, y) in enumerate(coords):
        valid_vals = real_roots[i][np.isfinite(real_roots[i])]
        if valid_vals.size > 0:
            sub_result[x, y] = valid_vals.min()
        else:
            sub_result[x, y] = 0.0

    return (sub_result, offset_x, offset_y)


# ------------------------
# 3) 滑坡概率转换：并行主函数
# ------------------------
def parallel_landslide_conversion(LS, num_workers=None, chunks=4):
    """
    使用多进程并行对 LS 做 landslide_conversion
    - num_workers: 并行进程数 (默认使用 cpu_count())
    - chunks: 在行方向分块的数量 (示例只做简单行切分)
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    rows, cols = LS.shape
    chunk_size = rows // chunks
    
    # 1) 构造每个子块数据
    chunked_data = []
    for i in range(chunks):
        start = i * chunk_size
        end = rows if (i == chunks - 1) else (start + chunk_size)
        LS_sub = LS[start:end, :]
        # (子块, 子块在原始数组中的起始行, 起始列=0)
        chunked_data.append((LS_sub, start, 0))

    # 2) 并行计算
    with Pool(processes=num_workers) as p:
        results = p.map(landslide_conversion_chunk, chunked_data)

    # 3) 合并结果
    final_result = np.zeros_like(LS, dtype=float)
    for sub_result, offset_x, offset_y in results:
        h, w = sub_result.shape
        final_result[offset_x:offset_x + h, offset_y:offset_y + w] = sub_result

    return final_result


# ------------------------
# 4) 液化概率转换函数 (CPU 端，非并行示例)
# ------------------------
def liquefaction_conversion(LF):
    """
    液化概率转换函数
    公式: result = ln( ( sqrt(0.4915 / LF) - 1 ) / 42.40 ) / (-9.165)
    """
    result = np.zeros_like(LF, dtype=float)
    index = LF > 0
    tmp = (np.sqrt(0.4915 / LF[index]) - 1) / 42.40
    result[index] = np.log(tmp) / (-9.165)
    # 去掉负值、无穷大和 NaN
    result[result < 0] = 0
    result[~np.isfinite(result)] = 0
    return result


# ------------------------
# 5) 主程序
# ------------------------
if __name__ == "__main__":
    # =============== 这里修改为你本地/服务器上的路径 ===============
    # location = "/Users/lixintong/Desktop/大三上/大数据并行计算/并行大作业/第三次作业/Bayesian_Causal-main/data"
    # event = '2023_turkey_new'
    # # event = '2024_japan2'   # 你也可以切换事件
    
    # # 路径逻辑: 如果是 2023_turkey_new, 就用 '2023_turkey' 做子文件夹名
    # if event == '2023_turkey_new':
    #     event_1 = '2023_turkey'
    # else:
    #     event_1 = event
    
    # # 读取数据
    # print("Reading input rasters...")
    # Y, Y_profile = read_raster(os.path.join(location, event, 'damage_proxy_map', f'{event_1}_damage_proxy_map.tif'))
    # BD, BD_profile = read_raster(os.path.join(location, event, 'building_footprint', f'{event_1}_building_footprint_rasterized.tif'))
    # LS, LS_profile = read_raster(os.path.join(location, event, 'prior_models', f'{event_1}_prior_landslide_model.tif'))
    # LF, LF_profile = read_raster(os.path.join(location, event, 'prior_models', f'{event_1}_prior_liquefaction_model.tif'))
    
    # # 数据预处理
    # BD[BD > 0] = 1
    # Y[np.isnan(Y)] = 0
    # BD[np.isnan(BD)] = 0
    # LS[np.isnan(LS)] = 0
    # LF[np.isnan(LF)] = 0
    # # 对 Y 做简单归一化或偏移
    # Y = (Y + 11) / 20
    location = "/mnt"
    
    # 读取数据 (只需 LS 和 LF)
    print("Reading LS and LF rasters...")
    LS, LS_profile = read_raster(os.path.join(location, "2023_turkey_prior_landslide_model.tif"))
    LF, LF_profile = read_raster(os.path.join(location, "2023_turkey_prior_liquefaction_model.tif"))

    # 数据预处理
    LS[np.isnan(LS)] = 0
    LF[np.isnan(LF)] = 0

    
    print("Starting parallel landslide conversion...")
    start_time = time.time()
    new_LS = parallel_landslide_conversion(LS, num_workers=4, chunks=4)
    print("Landslide probabilities computed via parallel processing.")
    print(f"Elapsed time for Landslide: {time.time() - start_time:.6f} seconds")

    print("Starting liquefaction conversion (single-thread)...")
    start_time2 = time.time()
    new_LF = liquefaction_conversion(LF)
    print("Liquefaction probabilities computed.")
    print(f"Elapsed time for Liquefaction: {time.time() - start_time2:.6f} seconds")

    # 最终清理
    new_LF[new_LF < 0] = 0
    new_LS[new_LS < 0] = 0
    new_LS[np.isnan(new_LS)] = 0
    new_LF[np.isnan(new_LF)] = 0

    total_time = time.time() - start_time
    print(f"Total elapsed time (including parallel LS + LF): {total_time:.2f} seconds")

    # 如果需要保存结果到新 tif 文件，参考以下示例：
    # with rasterio.open('new_LS_output.tif', 'w', **LS_profile) as dst:
    #     dst.write(new_LS, 1)
    #
    # with rasterio.open('new_LF_output.tif', 'w', **LF_profile) as dst:
    #     dst.write(new_LF, 1)

    print("Done.")

