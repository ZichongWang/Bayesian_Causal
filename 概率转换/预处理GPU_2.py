#!/usr/bin/env python3
# coding: utf-8
import time
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
import cupy as cp

def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1, resampling=Resampling.nearest)
        profile = src.profile
    return data, profile

def cubic_cardano(a, b, c, d):
    alpha = b / (3*a)
    p = (c/a) - (b**2)/(3*(a**2))
    q = (2*b**3)/(27*(a**3)) - (b*c)/(3*(a**2)) + (d/a)

    half_q = q/2.0
    p_over_3 = p/3.0
    disc = half_q**2 + p_over_3**3

    sqrt_disc = cp.sqrt(disc)

    def cbrt(z):
        return cp.sign(z) * cp.abs(z)**(1/3)

    term1 = -half_q + sqrt_disc
    term2 = -half_q - sqrt_disc
    t = cbrt(term1) + cbrt(term2)
    x = t - alpha
    return x

def landslide_conversion_gpu(LS):
    LS_gpu = cp.array(LS, dtype=cp.float32)
    mask_gpu = (LS_gpu > 0)
    new_LS_gpu = cp.zeros_like(LS_gpu, dtype=cp.float32)

    valid_vals = LS_gpu[mask_gpu]
    log_vals = cp.log(valid_vals)

    a, b, c = 4.035, -3.042, 5.237
    d_vals = -7.592 - log_vals

    x_vals = cubic_cardano(a, b, c, d_vals)
    new_LS_gpu[mask_gpu] = x_vals

    new_LS_gpu = cp.maximum(new_LS_gpu, 0)
    new_LS_gpu = cp.nan_to_num(new_LS_gpu, 0.0)

    return new_LS_gpu.get()

def liquefaction_conversion(LF):
    # 液化部分若已在 CPU 上实现得很快，也可以保持不变
    result = np.zeros_like(LF, dtype=float)
    index = LF > 0
    tmp = (np.sqrt(0.4915 / LF[index]) - 1)/42.40
    val = np.log(tmp)/(-9.165)
    val[~np.isfinite(val)] = 0
    val[val < 0] = 0
    result[index] = val
    return result


if __name__ == "__main__":
    # 1) 读取栅格
    tif_path_ls = "/mnt/2023_turkey_prior_landslide_model.tif"
    tif_path_lf = "/mnt/2023_turkey_prior_liquefaction_model.tif"

    print("Reading LS and LF ...")
    LS, LS_profile = read_raster(tif_path_ls)
    LF, LF_profile = read_raster(tif_path_lf)

    LS[np.isnan(LS)] = 0
    LF[np.isnan(LF)] = 0

    # 2) GPU 滑坡
    start_time = time.time()
    new_LS = landslide_conversion_gpu(LS)
    print(f"GPU Landslide time: {time.time()-start_time:.6f} s")

    # 3) CPU 液化
    start_time = time.time()
    new_LF = liquefaction_conversion(LF)
    print(f"CPU Liquefaction time: {time.time()-start_time:.6f} s")

    # 如果需要保存结果:
    # with rasterio.open("/mnt/new_LS_gpu.tif", "w", **LS_profile) as dst:
    #     dst.write(new_LS, 1)
    #
    # with rasterio.open("/mnt/new_LF_cpu.tif", "w", **LF_profile) as dst:
    #     dst.write(new_LF, 1)

    print("All done.")