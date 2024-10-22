import cv2
import numpy as np
import os

# 文件夹路径
input_folder = r"E:\yolov8\v8-litchi-WHM\ultralytics\runs\segment\predict6\crops\fruit\a"
output_folder = r"E:\yolov8\v8-litchi-WHM\ultralytics\runs\segment\predict6\crops\fruit\a\processed"

# 创建输出文件夹（如果不存在的话）
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中的所有图片文件
file_list = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

for file_name in file_list:
    # 读取图像
    image_path = os.path.join(input_folder, file_name)
    img = cv2.imread(image_path)

    if img is None:
        print(f"无法读取图像: {image_path}")
        continue

    # 转换到Lab颜色空间
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # 获取Lab图像的a分量
    a_channel = lab_img[:, :, 1]

    # 创建一个全透明的图像
    h, w = a_channel.shape
    output_img = np.zeros((h, w, 4), dtype=np.uint8)  # 4通道: B, G, R, A

    # 设置a分量的条件
    output_img[:, :, 3] = np.where(a_channel <= 127, 0, 255)  # 透明度通道

    # 创建黑色图像
    black_img = np.zeros((h, w, 3), dtype=np.uint8)

    # 根据a分量的条件设置颜色
    output_img[:, :, :3] = np.where(a_channel[:, :, np.newaxis] > 127, black_img, black_img)

    # 保存Lab颜色空间的图片
    lab_img_path = os.path.join(output_folder, f"lab_{file_name}")
    cv2.imwrite(lab_img_path, lab_img)

    # 保存处理后的图片
    processed_img_path = os.path.join(output_folder, f"processed_{file_name}")
    cv2.imwrite(processed_img_path, output_img)

print("处理完成。")
