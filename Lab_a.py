import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 文件夹路径
folder_path = r"E:\yolov8\v8-litchi-WHM\ultralytics\runs\segment\predict6\crops\fruit\a_hist"

# 创建一个列表来存储每张图片的红色像素占比和分类
results = []

# 遍历文件夹中的所有png图片
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        # 加载图像
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # 将图像转换为Lab颜色空间
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        # 获取a分量
        a_channel = lab_img[:, :, 1]

        # 将a_channel的值全部减128(不可用)
        # a_channel -= 128

        # 获取alpha通道，如果没有alpha通道，则跳过透明像素的计算
        if img.shape[2] == 4:
            alpha_channel = img[:, :, 3]
            # 忽略alpha通道为0的像素（透明部分）
            mask = alpha_channel > 0
            a_channel = a_channel[mask]
        else:
            a_channel = a_channel.flatten()

        # 绘制并保存a分量的直方图
        plt.figure()
        plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)  # 设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(2)  # 设置左边坐标轴的粗细
        plt.hist(a_channel, bins=256, range=(0, 255), color='blue', alpha=0.7)
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
        plt.title(f'a Channel Histogram for {filename}')
        plt.xlabel('a Value')
        plt.ylabel('Frequency')

        # 设置x轴的最小刻度-128和最大刻度127
        plt.xlim([-10, 265])
        plt.xticks([0, 67, 127, 187, 255], fontproperties='Times New Roman', size=18)  # 设置刻度为0, ……, 255
        plt.xticks(fontproperties='Times New Roman', size=18)
        plt.yticks(fontproperties='Times New Roman', size=18)  # 设置大小及加粗, weight='bold'

        # 画出 x=127 这条垂直线
        plt.axvline(x=127, c="r", alpha=0.8, ls="--", lw=2)

        # 去除上边和右边的边框
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_position(('data', 0))
        hist_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_hist.png")
        plt.savefig(hist_path, dpi=600)
        plt.close()

        # 计算a分量在128~255之间的像素数
        p_r = np.sum((a_channel >= 128) & (a_channel <= 255))
        p_t = len(a_channel)  # 总像素数

        # 计算红色像素占比
        r = p_r / p_t * 100  # 乘以100以获得百分比

        # 分类
        if 0 <= r < 10:
            category = "unripe litchi"
        elif 10 <= r < 90:
            category = "ripe litchi"
        elif 90 <= r <= 100:
            category = "fully ripe litchi"
        else:
            category = "未知"

        # 将结果存储到列表中
        results.append({
            "图片名称": filename,
            "p_r": p_r,
            "p_t": p_t,
            "红色像素占比(%)": r,
            "分类": category
        })

# 将结果写入Excel表格
output_df = pd.DataFrame(results)
output_path = os.path.join(folder_path, "分类结果.xlsx")
output_df.to_excel(output_path, index=False)

print("处理完成，结果已保存到分类结果.xlsx")

########################################################################################################################
# import os
# import cv2
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
#
# # 文件夹路径
# folder_path = r"E:\yolov8\v8-litchi-WHM\ultralytics\runs\segment\predict6\crops\fruit"
# output_csv = "red_pixel_ratios_2.csv"
#
# # 初始化表格数据
# data = []
#
# # 遍历文件夹中的PNG图片
# for filename in os.listdir(folder_path):
#     if filename.endswith(".png"):
#         img_path = os.path.join(folder_path, filename)
#         # 读取图片（包含alpha通道）
#         img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#
#         # 分离alpha通道
#         bgr_img = img[:, :, :3]
#         alpha_channel = img[:, :, 3]
#
#         # 转换为Lab颜色空间
#         lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
#         a_channel = lab_img[:, :, 1]  # 获取a分量
#
#         # 只保留非透明像素
#         non_transparent_mask = alpha_channel > 0
#         a_non_transparent = a_channel[non_transparent_mask]
#
#         # 绘制a分量的直方图
#         plt.figure()
#         plt.hist(a_non_transparent.flatten(), bins=256, range=(0, 255), color='red', alpha=0.7)
#         plt.title(f"Histogram of 'a' channel for {filename}")
#         plt.xlabel("a channel value")
#         plt.ylabel("Pixel Count")
#         plt.savefig(os.path.join(folder_path, f"{filename}_a_histogram.png"))
#         plt.close()
#
#         # 计算红色像素占比
#         p_r = np.sum((a_non_transparent >= 128) & (a_non_transparent <= 255))
#         p_t = a_non_transparent.size
#         r = p_r / p_t if p_t != 0 else 0
#
#         # 添加数据到表格中
#         data.append([filename, r])
#
# # 保存表格数据到CSV文件
# df = pd.DataFrame(data, columns=["Image Name", "Red Pixel Ratio"])
# df.to_csv(output_csv, index=False)
#
# print(f"红色像素占比已保存到 {output_csv}")

########################################################################################################################
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from tqdm import tqdm
#
# # 文件夹路径
# folder_path = r"E:\yolov8\v8-litchi-WHM\ultralytics\runs\segment\predict6\crops\fruit"
#
# # 保存直方图和表格的路径
# histogram_save_path = os.path.join(folder_path, 'a_histograms')
# os.makedirs(histogram_save_path, exist_ok=True)
#
# # 初始化表格
# results = []
#
# for filename in tqdm(os.listdir(folder_path)):
#     if filename.endswith(".png"):
#         # 加载图片并转换为Lab颜色空间
#         image_path = os.path.join(folder_path, filename)
#         image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#
#         # 分离颜色分量和透明度通道
#         if image.shape[2] == 4:
#             bgr_image = image[:, :, :3]
#             alpha_channel = image[:, :, 3]
#             mask = alpha_channel > 0  # 非透明部分的掩码
#         else:
#             bgr_image = image
#             mask = np.ones(bgr_image.shape[:2], dtype=bool)
#
#         # 转换为Lab颜色空间
#         lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)
#
#         # 提取a分量并应用掩码
#         a_channel = lab_image[:, :, 1]
#         a_channel_masked = a_channel[mask]
#
#         # 计算并保存a分量的直方图
#         plt.figure()
#         plt.hist(a_channel_masked, bins=256, range=(-128, 127), color='red', alpha=0.7)
#         plt.title(f'a-channel Histogram for {filename}')
#         plt.xlabel('a-channel value')
#         plt.ylabel('Frequency')
#         plt.xlim([-128, 127])
#         plt.grid(True)
#         histogram_file = os.path.join(histogram_save_path, f'{os.path.splitext(filename)[0]}_a_histogram.png')
#         plt.savefig(histogram_file)
#         plt.close()
#
#         # 计算红色像素占比r
#         p_t = len(a_channel_masked)
#         p_r = np.sum((a_channel_masked >= 0) & (a_channel_masked <= 127))
#         r = p_r / p_t if p_t > 0 else 0
#
#         # 将结果存入列表
#         results.append({"filename": filename, "red_pixel_ratio": r})
#
# # 将结果保存为Excel表格
# df = pd.DataFrame(results)
# output_file = os.path.join(folder_path, 'red_pixel_ratios.xlsx')
# df.to_excel(output_file, index=False)
