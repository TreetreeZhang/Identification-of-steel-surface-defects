# Description: 将掩码图转换为 YOLOv8 分割格式 批处理版

import cv2
import os

def convert_mask_to_yolov8(mask_file, output_txt, image_width, image_height):
    # 读取掩码图像，灰度读取
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    
    # 掩码中可能的类别值：0=background, 1=Inclusion, 2=Patches, 3=Scratches
    class_ids = [1, 2, 3]

    with open(output_txt, 'w') as f:
        for class_id in class_ids:
            # 为每个类别生成单独的掩码
            mask_class = (mask == class_id).astype('uint8') * 255
            
            # 查找该类别的轮廓
            contours, _ = cv2.findContours(mask_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3:
                    continue  # 如果轮廓点过少，跳过
                
                # 归一化多边形坐标
                normalized_points = []
                for point in contour:
                    x_norm = point[0][0] / image_width
                    y_norm = point[0][1] / image_height
                    normalized_points.extend([x_norm, y_norm])
                
                # 输出YOLOv8分割格式：<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
                f.write(f"{class_id} " + " ".join(map(str, normalized_points)) + "\n")

def batch_process_masks(input_folder, output_folder, image_width, image_height):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            mask_file = os.path.join(input_folder, filename)
            output_txt = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")
            
            # 进行转换
            convert_mask_to_yolov8(mask_file, output_txt, image_width, image_height)

# 示例使用
input_folder = 'annotations\\training'  # 掩码图像所在的文件夹
output_folder = 'annotations\\training_txt'  # 输出txt文件的文件夹
image_width = 200  # 图像宽度
image_height = 200  # 图像高度

# 调用函数进行批处理
batch_process_masks(input_folder, output_folder, image_width, image_height)
