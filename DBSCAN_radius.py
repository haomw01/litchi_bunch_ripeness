import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict5\\216_centers.csv").values

print(data.shape)

# 计算每个点的k-距离
def select_MinPts(data, k):
    k_dist = []
    for i in range(data.shape[0]):
        dist = (((data[i] - data)**2).sum(axis=1)**0.5)
        dist.sort()
        k_dist.append(dist[k])
    return np.array(k_dist)

k = 1  # k值
k_dist = select_MinPts(data, k)
k_dist.sort()

# 绘制散点图
fig, ax = plt.subplots()

# 示例散点数据
x = np.arange(k_dist.shape[0])
y = k_dist[::1]

# 设置散点的颜色、大小和形状
scatter = ax.scatter(x, y, c='blue', s=50, marker='o', edgecolor='k')

# 设置坐标轴的刻度线方向
ax.tick_params(axis='both', direction='in', labelsize=10)

# 设置坐标轴的粗细
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# 设置坐标轴刻度范围、大小、间隔
ax.set_xlim(left=-1, right=len(x))
ax.set_ylim(bottom=-50, top=max(y)*1.1)  # 增加10%以保证数据不被截断
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.yaxis.set_major_locator(plt.MultipleLocator(200))
ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")

# 设置坐标轴标签的字体大小
ax.set_xlabel('Number of points', fontproperties='Times New Roman', fontsize=18)
ax.set_ylabel('Nearest distance', fontproperties='Times New Roman', fontsize=18)

# 设置刻度标签的字体大小
# ax.tick_params(axis='both', labelsize=10)
plt.xticks(fontproperties='Times New Roman', size=18)
plt.yticks(fontproperties='Times New Roman', size=18)  # 设置大小及加粗, weight='bold'

# 添加标题
ax.set_title('The nearest distance from each data point', fontproperties='Times New Roman', fontsize=20)

# 保存图片
output_path = "E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict5\\scatter_plot_216.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight')

# 显示图像
plt.show()
