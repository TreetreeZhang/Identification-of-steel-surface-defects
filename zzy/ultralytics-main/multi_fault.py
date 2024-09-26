import numpy as np
from PIL import Image
import os

global flag
flag = 0
def check_defects(image_path):
    # 打开图像并转换为数组
    img = Image.open(image_path)
    img_array = np.array(img)

    # 找到所有的非零值
    unique_values = np.unique(img_array)

    # 检查非零的独特值（即缺陷类型）
    defect_values = unique_values[unique_values != 0]

    if len(defect_values) == 0:
        print(f"{image_path}: 没有检测到缺陷")
    elif len(defect_values) == 1:
        print(f"{image_path}: 只有一种缺陷，值为: {defect_values[0]}")
    else:
        print(f"{image_path}: 有多种缺陷，缺陷值分别为: {defect_values}")
        flag = 1

# 批量处理文件夹中的所有图片
def check_defects_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            check_defects(file_path)

# 调用函数，指定你的文件夹路径
folder_path = 'masks/'
check_defects_in_folder(folder_path)
print(flag)
