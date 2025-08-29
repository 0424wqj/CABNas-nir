import matplotlib.pyplot as plt
import numpy as np

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
methods = ["SG", "SNV", "MSC", "FD", "SD", "LG", "SG+SNV", "SG+MSC", "OSC"]
snr_values = [20.68, 15.98, 19.41, 13.27, 18.62, 12.38, 15.80, 19.77, 17.69]

# 设置柱状图位置
x = np.arange(len(methods))
width = 0.7

# 创建画布
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制信噪比柱状图
rects = ax.bar(x, snr_values, width, color='#6FB07F', edgecolor='black')

# 绘制小圆圈和连接线
ax.plot(x, snr_values, 'o-', color='#FC5B3F', markersize=8, linewidth=4, markeredgecolor='black', markeredgewidth=1)

# 设置坐标轴和标题
ax.set_ylabel('SNR (dB)', fontsize=16)  # 增大y轴标签字体大小
ax.set_title('Comparison of SNR for Different Preprocessing Methods', fontsize=18, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=14)  # 增大x轴标签字体大小
# 移除图例
# ax.legend(fontsize=16)

# 定义添加数值标签函数，调整字体大小
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        y_offset = 3 if height >= 0 else -15
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16, color='darkred')  # 增大柱形标注字体大小

autolabel(rects)

# 调整布局，增大纵坐标刻度字体大小
plt.xticks(rotation=45)
# 设置y轴范围
y_min = min(snr_values) - 5 if min(snr_values) >= 0 else min(snr_values) - 5
y_max = max(snr_values) + 5
ax.set_ylim(y_min, y_max)
# 增大纵坐标刻度字体大小
ax.tick_params(axis='y', labelsize=16)

# 添加水平参考线
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('SNR_comparison.png', dpi=600, bbox_inches='tight')
plt.show()