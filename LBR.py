import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# 定义图片文件夹路径
folder_path = r"E:\yolov8\v8-litchi-WHM\ultralytics\runs\segment\predict2\crops22"
output_folder = r"E:\yolov8\v8-litchi-WHM\ultralytics\runs\segment\predict2\output_histograms"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有图片
for image_file in os.listdir(folder_path):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):  # 检查是否为图片文件
        image_path = os.path.join(folder_path, image_file)

        # 读取图片并转换为Lab颜色空间
        image = cv2.imread(image_path)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        # 提取a通道
        a_channel = lab_image[:, :, 1]

        # 计算并绘制直方图
        plt.hist(a_channel.ravel(), bins=256, range=[0, 255], color='red', alpha=0.7)
        plt.title(f'a-channel Histogram for {image_file}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        # 保存直方图
        output_path = os.path.join(output_folder, f"{image_file}_a_histogram.png")
        plt.savefig(output_path)
        plt.clf()  # 清除绘图以便绘制下一个直方图

results = []  # 保存结果

for image_file in os.listdir(folder_path):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        image_path = os.path.join(folder_path, image_file)

        # 读取图片并转换为Lab颜色空间
        image = cv2.imread(image_path)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        # 提取a通道
        a_channel = lab_image[:, :, 1]

        # 计算总像素数
        total_pixels = a_channel.size

        # 计算a分量在128到255之间的像素数量
        red_pixels = np.sum((a_channel >= 128) & (a_channel <= 255))

        # 计算红色像素的占比r
        r = red_pixels / total_pixels

        # 确定类别
        if 0 <= r < 0.1:
            category = 'Unripe'
        elif 0.1 <= r < 0.9:
            category = 'Turning'
        else:
            category = 'Fully ripe'

        # 将结果保存到表格中
        results.append([image_file, r, category])

# 使用pandas将结果保存为表格
import pandas as pd

df = pd.DataFrame(results, columns=['Image File', 'Red Pixel Ratio (r)', 'Category'])
df.to_csv(os.path.join(output_folder, 'red_pixel_ratios.csv'), index=False)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.patches as patches

# 读取数据
data = pd.read_csv(r"E:\yolov8\v8-litchi-WHM\ultralytics\runs\segment\predict2\22.csv").values

# 定义参数，r值和其对应的类别
r_values = df['Red Pixel Ratio (r)'].values  # 假设r的值和data的点一一对应
categories = df['Category'].values

# 设置DBSCAN模型的参数
model = DBSCAN(eps=200, min_samples=1)
result = model.fit_predict(data)

# 确定聚类的数量
num_clusters = len(set(result)) - (1 if -1 in result else 0)
colors = plt.cm.get_cmap('hsv', num_clusters)

# 创建一个图形，设置背景颜色为灰白色
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_facecolor('whitesmoke')

# 画出散点，并为每个点添加数字标注
for i, d in enumerate(data):
    ax.scatter(d[0], 3072 - d[1], c=[colors(result[i] % num_clusters)], s=200, edgecolor='black', marker='o')
    ax.text(d[0], 3072 - d[1], f'{r_values[i]:.2f}', fontsize=10, ha='center', va='center', color='black')

# 计算bi参数，并为每个聚类画出最小外接矩形框
for cluster_label in np.unique(result):
    if cluster_label == -1:  # 忽略噪声点
        continue

    cluster_points = data[result == cluster_label]
    cluster_r_values = r_values[result == cluster_label]
    cluster_categories = categories[result == cluster_label]

    # 计算bi值
    unripe_count = np.sum(cluster_categories == 'Unripe')
    turning_count = np.sum(cluster_categories == 'Turning')
    ripe_count = np.sum(cluster_categories == 'Fully ripe')
    bi = unripe_count / 27.5 + turning_count / 18 + ripe_count / 3.5

    # 计算最小外接矩形框
    x_min, y_min = np.min(cluster_points, axis=0)
    x_max, y_max = np.max(cluster_points, axis=0)

    # 根据bi的值设置颜色
    if 0 <= bi < 0.3:
        rect_color = (112 / 255, 173 / 255, 71 / 255)
    elif 0.3 <= bi < 0.6:
        rect_color = (169 / 255, 209 / 255, 142 / 255)
    elif 0.6 <= bi < 0.9:
        rect_color = (255 / 255, 217 / 255, 102 / 255)
    elif 0.9 <= bi < 1.2:
        rect_color = (237 / 255, 125 / 255, 49 / 255)
    else:
        rect_color = (192 / 255, 0 / 255, 0 / 255)

    # 画出矩形框
    rect = patches.Rectangle((x_min, 3072 - y_max), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=rect_color,
                             facecolor='none')
    ax.add_patch(rect)

    # 在矩形框上方添加bi值
    ax.text((x_min + x_max) / 2, 3072 - y_max - 10, f'bi={bi:.2f}', fontsize=12, ha='center', va='center',
            color='white', backgroundcolor=rect_color)

# 保存图片
plt.savefig(r'E:\yolov8\v8-litchi-WHM\ultralytics\runs\segment\predict2\22_dbscan_improved.jpg', dpi=600)
plt.show()

# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.cluster import DBSCAN
# from matplotlib.patches import Rectangle
#
# # 路径定义
# folder_path = "E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict2\\crops22"
# output_csv = "22_output_r_values.csv"
#
# # 初始化结果表格
# results = []
#
# # 遍历文件夹中的每张图片
# for img_file in os.listdir(folder_path):
#     if img_file.endswith(".png"):
#         img_path = os.path.join(folder_path, img_file)
#
#         # 读取图像
#         img = cv2.imread(img_path)
#         img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#
#         # 提取Lab颜色空间中的a分量
#         a_channel = img_lab[:, :, 1]
#
#         # 创建a分量直方图并保存
#         plt.figure()
#         plt.hist(a_channel.ravel(), bins=256, range=[0, 256], color='r', alpha=0.75)
#         plt.title(f'a Channel Histogram of {img_file}')
#         plt.xlabel('Pixel Value (a)')
#         plt.ylabel('Frequency')
#         hist_img_path = os.path.join(folder_path, f"{img_file}_a_histogram.png")
#         plt.savefig(hist_img_path)
#         plt.close()
#
#         # 计算a分量中128~255的红色像素数
#         mask = (a_channel >= 128) & (a_channel <= 255)
#         pr = np.sum(mask)
#
#         # 计算总的非透明像素数
#         alpha_channel = img[:, :, 3] if img.shape[-1] == 4 else np.ones_like(a_channel) * 255
#         non_transparent_mask = alpha_channel > 0
#         pt = np.sum(non_transparent_mask)
#
#         # 计算红色像素占比 r
#         if pt > 0:
#             r = pr / pt
#         else:
#             r = 0
#
#         # 分类 r
#         if 0 <= r < 0.1:
#             category = "Unripe"
#         elif 0.1 <= r < 0.9:
#             category = "Turning"
#         else:
#             category = "Fully ripe"
#
#         # 记录结果
#         results.append([img_file, r, category])
#
# # 保存结果到表格
# df_results = pd.DataFrame(results, columns=["Image", "Red Pixel Ratio (r)", "Category"])
# df_results.to_csv(output_csv, index=False)
#
# # 步骤 4: 计算参数 a
# unripe_count = sum(df_results['Category'] == "Unripe")
# turning_count = sum(df_results['Category'] == "Turning")
# fully_ripe_count = sum(df_results['Category'] == "Fully ripe")
#
# a_value = unripe_count / 27.5 + turning_count / 18 + fully_ripe_count / 3.5
#
# # 步骤 5: 使用DBSCAN聚类并绘制最小外接矩形框
# data = pd.read_csv("E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict2\\22.csv").values
#
# model = DBSCAN(eps=200, min_samples=1)
# result = model.fit_predict(data)
#
# # 创建聚类颜色
# num_clusters = len(set(result))
# colors = plt.cm.get_cmap('hsv', num_clusters)
#
# # 创建图形
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.set_facecolor('whitesmoke')
# ax.set_xlim([0, 4096])
# ax.set_ylim([0, 3072])
# ax.grid(linewidth=1)
#
# # 为每个聚类绘制最小外接矩形框
# for cluster in set(result):
#     cluster_points = data[result == cluster]
#
#     # 计算最小外接矩形框
#     if len(cluster_points) > 0:
#         x_min, y_min = np.min(cluster_points, axis=0)
#         x_max, y_max = np.max(cluster_points, axis=0)
#
#         # 颜色根据a的值设定
#         if 0 <= a_value < 0.3:
#             rect_color = (112 / 255, 173 / 255, 71 / 255)  # RGB(112,173,71)
#         elif 0.3 <= a_value < 0.6:
#             rect_color = (169 / 255, 209 / 255, 142 / 255)  # RGB(169,209,142)
#         elif 0.6 <= a_value < 0.9:
#             rect_color = (255 / 255, 217 / 255, 102 / 255)  # RGB(255,217,102)
#         elif 0.9 <= a_value < 1.2:
#             rect_color = (237 / 255, 125 / 255, 49 / 255)  # RGB(237,125,49)
#         else:
#             rect_color = (192 / 255, 0, 0)  # RGB(192,0,0)
#
#         # 画出矩形框
#         rect = Rectangle((x_min, 3072 - y_max), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='black',
#                          facecolor=rect_color, alpha=0.5)
#         ax.add_patch(rect)
#
#         # 在矩形框上方写入a的值
#         ax.text(x_min, 3072 - y_max - 10, f'a={a_value:.2f}', color='white', fontsize=10, weight='bold')
#
# plt.savefig('E:\\yolov8\\v8-litchi-WHM\\ultralytics\\runs\\segment\\predict2\\22_clustered_image_with_a_value.jpg',
#             dpi=600)
# plt.show()
