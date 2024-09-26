# Description: 将yolov8分割模型的预测结果转换为掩码图 批处理版 输出改变了图像大小 弃用

from ultralytics import YOLO
import cv2
import numpy as np
import os

# 加载训练好的模型，替换成你的模型文件路径
model = YOLO('runs/segment/train5/weights/best.pt')

# 读取原始图像
original_image_path = '000001.jpg'
original_image = cv2.imread(original_image_path)
original_shape = original_image.shape[:2]  # 获取原图的高度和宽度

# 使用模型进行预测
result = model.predict(source=original_image_path, save=False)

# 获取预测结果
if result[0].masks is not None and len(result[0].masks) > 0:
    masks_data = result[0].masks.data  # 获取掩码数据

    # 初始化一个空白图像，用于叠加所有掩码，大小与原图相同
    combined_mask = np.zeros(original_shape, dtype=np.uint8)

    # 遍历每个掩码并进行叠加
    for index, mask in enumerate(masks_data):
        mask = mask.cpu().numpy()  # 将CUDA张量转换为NumPy数组
        mask = (mask > 0.5).astype(np.uint8) * 255  # 二值化并将掩码转换为0或255

        # 调整掩码大小，使其与原图大小一致
        mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

        # 将每个掩码叠加到总掩码图上
        combined_mask = cv2.add(combined_mask, mask_resized)

    # 保存叠加后的掩码图像
    pred_image_path = 'combined_mask_resized.png'
    cv2.imwrite(pred_image_path, combined_mask)
    print(f"Saved combined mask: {pred_image_path}")

else:
    print("No masks were found.")
