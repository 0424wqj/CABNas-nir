#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
近红外光谱预处理方法分别绘制 - 修复版
模仿SG.py的风格，将所有预处理方法放在一张画布上
包含：SG, SNV, MSC, FD, SD, LG, SG+SNV, SG+MSC, OSC
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import os
import string
import warnings
warnings.filterwarnings('ignore')

# 设置全局字体为Times New Roman
plt.rcParams.update({
    'font.size': 12,  # 基础字体大小
    'axes.titlesize': 14,  # 标题字体大小
    'axes.labelsize': 12,  # 坐标轴标签字体大小
    'xtick.labelsize': 10,  # x轴刻度字体大小
    'ytick.labelsize': 10,  # y轴刻度字体大小
    'font.family': 'Times New Roman',  # 使用Times New Roman字体
    'axes.unicode_minus': False
})

class SpectrumPreprocessor:
    """光谱预处理类"""
    
    def __init__(self):
        pass
    
    def sg_transform(self, x):
        """SG平滑"""
        return savgol_filter(x, window_length=11, polyorder=2)
    
    def snv_transform(self, x):
        """SNV标准正态变量变换"""
        mean_val = np.mean(x)
        std_val = np.std(x)
        if std_val == 0:
            return x - mean_val
        return (x - mean_val) / std_val
    
    def msc_transform(self, spectra_matrix, spectrum):
        """MSC多元散射校正"""
        mean_spectrum = np.mean(spectra_matrix, axis=0)
        coeffs = np.polyfit(mean_spectrum, spectrum, 1)
        if coeffs[0] != 0:
            return (spectrum - coeffs[1]) / coeffs[0]
        else:
            return spectrum
    
    def first_derivative(self, x):
        """一阶导数"""
        return np.diff(x)
    
    def second_derivative(self, x):
        """二阶导数"""
        return np.diff(x, n=2)
    
    def log_transform(self, x):
        """对数变换"""
        x_positive = np.where(x <= 0, 1e-10, x)
        return np.log10(x_positive)

def load_data():
    """加载数据并去除异常样本"""
    data = pd.read_csv(
        r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv')
    
    # 去除异常样本（索引25，即第26行）
    print("正在去除异常样本...")
    print(f"原始数据形状: {data.shape}")
    
    # 删除异常样本
    anomaly_index = 25  # 之前检测到的异常样本索引
    data_clean = data.drop(data.index[anomaly_index]).reset_index(drop=True)
    print(f"去除异常样本后数据形状: {data_clean.shape}")
    print(f"已去除第 {anomaly_index + 1} 行异常数据（标签为 {data.iloc[anomaly_index, -1]}）")
    
    # 获取波段列和标签列
    band_columns = data_clean.columns[:-1]
    spectral_data = data_clean[band_columns]
    labels = data_clean.iloc[:, -1].values
    
    # 获取波长值
    wavelengths = [float(col) for col in band_columns]
    
    return spectral_data, wavelengths, labels

def create_save_directory():
    """创建保存目录"""
    save_dir = r'C:\Users\Administrator\Desktop\管道淤泥项目\论文\实验图片\预处理'
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def plot_preprocessing_subplot(ax, processed_data, wavelengths, method_name, subplot_letter, y_label="Reflectance Intensity"):
    """在子图上绘制预处理结果"""
    for row in processed_data:
        ax.plot(wavelengths, row, linewidth=0.5, alpha=0.5)
    
    # 设置标题和坐标轴标签
    ax.set_title(method_name, fontsize=14)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(y_label)
    
    # 添加子图标记 (a), (b), ... 放在子图下方中央
    ax.text(0.5, -0.25, f'({subplot_letter})', transform=ax.transAxes, 
            fontsize=24, ha='center', va='top')

def main():
    """主函数"""
    print("="*80)
    print("近红外光谱预处理方法绘制程序 - 九个子图版")
    print("="*80)
    
    # 加载数据
    spectral_data, wavelengths, labels = load_data()
    print(f"✓ 成功加载数据: {spectral_data.shape[0]}个样本，{spectral_data.shape[1]}个波长点")
    
    # 创建保存目录
    save_dir = create_save_directory()
    print(f"✓ 保存目录已创建: {save_dir}")
    
    # 创建预处理器
    preprocessor = SpectrumPreprocessor()
    
    # 准备预处理数据
    print("正在进行各种预处理...")
    
    # 1. SG平滑
    sg_data = []
    for spectrum in spectral_data.values:
        sg_spectrum = preprocessor.sg_transform(spectrum)
        sg_data.append(sg_spectrum)
    sg_data = np.array(sg_data)
    
    # 2. SNV标准正态变量变换
    snv_data = []
    for spectrum in spectral_data.values:
        snv_spectrum = preprocessor.snv_transform(spectrum)
        snv_data.append(snv_spectrum)
    snv_data = np.array(snv_data)
    
    # 3. MSC多元散射校正
    msc_data = []
    spectral_matrix = spectral_data.values
    for i, spectrum in enumerate(spectral_matrix):
        msc_spectrum = preprocessor.msc_transform(spectral_matrix, spectrum)
        msc_data.append(msc_spectrum)
    msc_data = np.array(msc_data)
    
    # 4. 一阶导数
    fd_data = []
    fd_wavelengths = wavelengths[1:]  # 导数后波长减少1个
    for spectrum in spectral_data.values:
        fd_spectrum = preprocessor.first_derivative(spectrum)
        fd_data.append(fd_spectrum)
    fd_data = np.array(fd_data)
    
    # 5. 二阶导数
    sd_data = []
    sd_wavelengths = wavelengths[2:]  # 二阶导数后波长减少2个
    for spectrum in spectral_data.values:
        sd_spectrum = preprocessor.second_derivative(spectrum)
        sd_data.append(sd_spectrum)
    sd_data = np.array(sd_data)
    
    # 6. 对数变换
    lg_data = []
    for spectrum in spectral_data.values:
        lg_spectrum = preprocessor.log_transform(spectrum)
        lg_data.append(lg_spectrum)
    lg_data = np.array(lg_data)
    
    # 7. SG+SNV组合
    sg_snv_data = []
    for spectrum in sg_data:
        sg_snv_spectrum = preprocessor.snv_transform(spectrum)
        sg_snv_data.append(sg_snv_spectrum)
    sg_snv_data = np.array(sg_snv_data)
    
    # 8. SG+MSC组合
    sg_msc_data = []
    sg_matrix = np.array(sg_data)
    for i, spectrum in enumerate(sg_matrix):
        sg_msc_spectrum = preprocessor.msc_transform(sg_matrix, spectrum)
        sg_msc_data.append(sg_msc_spectrum)
    sg_msc_data = np.array(sg_msc_data)
    
    # 9. OSC正交信号校正（简化版）
    mean_spectrum = np.mean(spectral_data.values, axis=0)
    centered_spectra = spectral_data.values - mean_spectrum
    
    # 使用PCA进行正交校正
    pca = PCA(n_components=min(5, spectral_data.shape[1]//10))
    pca.fit(centered_spectra)
    transformed = pca.transform(centered_spectra)
    reconstructed = pca.inverse_transform(transformed)
    osc_data = spectral_data.values - 0.05 * reconstructed
    
    # 创建3x3网格布局图像
    print("正在创建九宫格预处理图...")
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()
    
    # 绘制9个子图
    plot_preprocessing_subplot(axs[0], sg_data, wavelengths, "SG Spectra", 'a')
    plot_preprocessing_subplot(axs[1], snv_data, wavelengths, "SNV Spectra", 'b')
    plot_preprocessing_subplot(axs[2], msc_data, wavelengths, "MSC Spectra", 'c')
    plot_preprocessing_subplot(axs[3], fd_data, fd_wavelengths, "FD Spectra", 'd')
    plot_preprocessing_subplot(axs[4], sd_data, sd_wavelengths, "SD Spectra", 'e')
    plot_preprocessing_subplot(axs[5], lg_data, wavelengths, "LG Spectra", 'f')
    plot_preprocessing_subplot(axs[6], sg_snv_data, wavelengths, "SG+SNV Spectra", 'g')
    plot_preprocessing_subplot(axs[7], sg_msc_data, wavelengths, "SG+MSC Spectra", 'h')
    plot_preprocessing_subplot(axs[8], osc_data, wavelengths, "OSC Spectra", 'i')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # 保存图像
    save_path = os.path.join(save_dir, '九种预处理方法对比.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    
    print(f"✓ 九宫格预处理图已保存到: {save_path}")
    print("\n" + "="*80)
    print("✓ 九种预处理方法对比图已生成完成！")
    print(f"✓ 图片保存位置: {save_path}")
    print("="*80)

if __name__ == "__main__":
    main() 