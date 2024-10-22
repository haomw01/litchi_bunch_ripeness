import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import time

# 开始计时
start_time = time.time()

# 读取数据
data = pd.read_csv("E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict5\\216_centers.csv").values

# 设置DBSCAN模型的参数
model = DBSCAN(eps=400, min_samples=1)
model.fit(data)
result = model.fit_predict(data)

# 确定聚类的数量
num_clusters = len(set(result))  # 确定聚类的数量
colors = plt.cm.get_cmap('hsv', num_clusters)  # 生成颜色映射

# 创建自定义的数字列表，假设长度和data相同
# custom_labels = np.arange(1, len(data) + 1)  # 示例：从1到数据点数量的序号
# custom_labels = [0.97, 0.99, 0.99]

# 创建一个图形，设置背景颜色为灰白色
fig, ax = plt.subplots(figsize=(12, 8))  # 根据需要修改图像尺寸
# fig.patch.set_facecolor('whitesmoke')  # 图形背景颜色
ax.set_facecolor('whitesmoke')  # 散点图背景颜色

# 设置坐标轴的粗细、刻度、标签字体大小等
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# 设置坐标轴范围
ax.set_xlim([0, 4096])  # x轴范围
ax.set_ylim([0, 3072])  # y轴范围

# 设置坐标轴刻度
ax.xaxis.set_ticks(np.arange(0, 4097, 500))  # x轴刻度间隔
ax.yaxis.set_ticks(np.arange(0, 3073, 500))  # y轴刻度间隔

# 设置刻度标签的大小
plt.xticks(fontproperties='Times New Roman', size=18)
plt.yticks(fontproperties='Times New Roman', size=18)  # 设置大小及加粗, weight='bold'

# 添加网格线
# ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.grid(linewidth=1)

# 画出散点，并为每个点添加数字标注
for i, d in enumerate(data):
    ax.scatter(d[0], 3072 - d[1], c=[colors(result[i] % num_clusters)], s=200, edgecolor='black', marker='o')  # 散点颜色、大小、形状
    # ax.text(d[0], 3072 - d[1], f'{custom_labels[i]}', fontsize=10, color='black', ha='center', va='center')  # 每个点标注数字

# 隐藏坐标轴
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)

# 保存图片并显示
plt.savefig('E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict5\\216_dbscan_1400.jpg', dpi=600)
plt.show()

# 结束计时
end_time = time.time()
print(f"Processing time for one image: {end_time - start_time} seconds")
print(end_time)

'''
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(result)) - (1 if -1 in result else 0)
# n_noise_ = list(result).count(-1)
n_noise_ = 0

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(data, result))
print("Completeness: %0.3f" % metrics.completeness_score(data, result))
print("V-measure: %0.3f" % metrics.v_measure_score(data, result))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(data, result))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(data, result))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(data, result))
结果：
Estimated number of clusters: 3
Estimated number of noise points: 18
Homogeneity: 0.953
Completeness: 0.883
V-measure: 0.917
Adjusted Rand Index: 0.952
Adjusted Mutual Information: 0.916
Silhouette Coefficient: 0.626
'''
########################################################################################################################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN  # DBSCAN API
# import time
#
# start_time = time.time()
#
# plt.figure(figsize=(6, 4.5))
#
# data = pd.read_csv("E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict5\\45.csv").values
#
# model = DBSCAN(eps=400, min_samples=1)  # Set radius to eps and Mpts to min_samples
# model.fit(data)
#
# result = model.fit_predict(data)
#
# # Generate a list of colors for the clusters
# num_clusters = len(set(result))  # The number of unique clusters
# colors = plt.cm.get_cmap('hsv', num_clusters)  # Use a colormap to generate colors
#
# # Visualize: Different colors represent points in the same cluster
# for i, d in enumerate(data):
#     # Choose a color based on the cluster label
#     plt.plot(d[0], 3072 - d[1], 'o', color=colors(result[i] % num_clusters))
#
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# plt.xlim(0, 4096)
# plt.ylim(0, 3072)
#
# plt.savefig('E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict5\\45_dbscan_1200.jpg', dpi=600)
# plt.show()
#
# end_time = time.time()
# print(f"Processing time for one image: {end_time - start_time} seconds")
########################################################################################################################

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN  # DBSCAN API
# import time
#
# start_time = time.time()
#
# plt.figure(figsize=(6, 4.5))
#
# data = pd.read_csv("E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict5\\22.csv").values
# # plt.scatter(data[:,0],data[:,1])
# # plt.show()
#
# model = DBSCAN(eps=250, min_samples=1)  # 设半径为eps Mpts为min_samples
# model.fit(data)
#
# result = model.fit_predict(data)
#
# # 可视化 不同颜色表示同簇的点
# # 构建点的样式列表
# mark = ['or', 'ob', 'og', 'oy', 'ok', 'om']
#
# # 输出数据的 索引 以及 数据
# for i, d in enumerate(data):
#     # print(i,d)
#     # 可视化
#     plt.plot(d[0], 3072 - d[1], mark[result[i]])
#
# plt.xlim(0, 4096)
# # # x_ticks = np.linspace(0, 4096, 2)
# # print(x_ticks)
# # plt.xticks(x_ticks)
# plt.ylim(0, 3072)
# # y_ticks = np.linspace(0, 3072, 2)
# # print(y_ticks)
# # plt.yticks(y_ticks)

# plt.xlim(0, 3072)
# plt.xticks(range(0, 3072, 3072))
# plt.ylim(0, 4096)
# plt.yticks(range(0, 4096, 4096))
# plt.axis('equal')
# plt.savefig('E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict5\\22_dbscan.jpg', dpi=600)
# plt.show()
#
# end_time = time.time()
# print(f"Processing time for one image: {end_time - start_time} seconds")


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN  # DBSCAN API
# import time
#
# start_time = time.time()
#
# plt.figure(figsize=(4.5, 6))
#
# data = pd.read_csv("E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict5\\01.csv").values
# # plt.scatter(data[:,0],data[:,1])
# # plt.show()
#
# model = DBSCAN(eps=400,min_samples=1)  # 设半径为eps Mpts为min_samples
# model.fit(data)
#
# result = model.fit_predict(data)
#
# # 可视化 不同颜色表示同簇的点
# # 构建点的样式列表
# mark = ['or', 'ob', 'og', 'oy', 'ok', 'om']
#
# # 输出数据的 索引 以及 数据
# for i, d in enumerate(data):
#     # print(i,d)
#     # 可视化
#     plt.plot(d[0], 4096 - d[1], mark[result[i]])
#
# plt.xlim(0, 3072)
# x_ticks = np.linspace(0, 3072, 2)
# print(x_ticks)
# plt.xticks(x_ticks)
# plt.ylim(0, 4096)
# y_ticks = np.linspace(0, 4096, 2)
# print(y_ticks)
# plt.yticks(y_ticks)
#
# # plt.xlim(0, 3072)
# # plt.xticks(range(0, 3072, 3072))
# # plt.ylim(0, 4096)
# # plt.yticks(range(0, 4096, 4096))
# # plt.axis('equal')
# plt.savefig('E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict5\\01_dbscan.jpg',dpi=600)
# plt.show()
#
# end_time = time.time()
# print(f"Processing time for one image: {end_time - start_time} seconds")