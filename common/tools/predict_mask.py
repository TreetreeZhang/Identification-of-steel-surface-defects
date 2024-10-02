# Description: 将yolov8分割模型的预测结果转换为掩码图 批处理版 输出改变了图像大小 弃用

from ultralytics import YOLO
import cv2
import numpy as np
import os

# 加载训练好的模型，替换成你的模型文件路径
model = YOLO('runs\\segment\\train5\\weights\\best.pt')

# 指定图片所在文件夹路径
image_folder = 'test/'
output_folder = 'masks/'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义类别与掩码值的映射，假设类别的索引是 [0, 1, 2]，分别对应 Inclusion, Patches, Scratches
class_to_mask_value = {
    1: 1,  # Inclusion
    2: 2,  # Patches
    3: 3   # Scratches
}

# 进行批量预测
results = model.predict(source=image_folder, save=False)  # 不保存默认输出，自定义掩码保存

# 遍历每张图片的预测结果
for i, result in enumerate(results):
    masks = result.masks.data  # 获取分割掩码 (这是一个张量)
    classes = result.boxes.cls  # 获取每个对象的类别标签

    # 创建一个空白图像来保存所有对象的掩码
    mask_combined = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)  # 用掩码的尺寸创建空白图像
    
    # 遍历每个对象的掩码，并给每个对象分配预定义的掩码值
    for j, mask in enumerate(masks):
        mask_img = mask.cpu().numpy()  # 先将CUDA张量移到CPU，再转换为NumPy格式
        mask_img = (mask_img > 0.5).astype(np.uint8)  # 二值化处理，确保掩码区域为0或1
        class_label = int(classes[j])  # 获取该对象的类别标签
        mask_value = class_to_mask_value.get(class_label, 0)  # 获取对应的掩码值，若不存在，默认为0
        
        # 将掩码值应用到掩码区域
        mask_img = mask_img * mask_value  # 应用类别对应的掩码值

        # 将该对象的掩码叠加到总掩码图中
        mask_combined = np.maximum(mask_combined, mask_img)

    # 保存合并后的掩码图像，命名为 "image_{i}_mask_combined.png"
    mask_filename = os.path.join(output_folder, f'image_{i}_mask_combined.png')
    cv2.imwrite(mask_filename, mask_combined)
    print(f"Saved combined mask: {mask_filename}")