from ultralytics import YOLO
import cv2
import numpy as np
import os

# 加载训练好的模型，替换成你的模型文件路径
model = YOLO('runs/segment/train5/weights/best.pt')

# 指定图片所在文件夹路径
image_folder = 'test/'
output_folder = 'masks/'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义类别与掩码值的映射，假设类别的索引是 [1, 2, 3]，分别对应 Inclusion, Patches, Scratches
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

    # 读取对应原始图像的尺寸
    image_path = os.path.join(image_folder, result.path)  # 获取图片路径
    original_image = cv2.imread(image_path)  # 读取原始图像
    original_shape = original_image.shape[:2]  # 获取原图像的高度和宽度 (height, width)

    # 创建一个空白图像来保存所有对象的掩码，大小与原图一致
    mask_combined = np.zeros(original_shape, dtype=np.uint8)  # 用原始图像的尺寸创建空白图像
    
    # 遍历每个对象的掩码，并给每个对象分配预定义的掩码值
    for j, mask in enumerate(masks):
        mask_img = mask.cpu().numpy()  # 将CUDA张量转换为NumPy数组
        mask_img = (mask_img > 0.5).astype(np.uint8)  # 二值化处理，确保掩码区域为0或1

        # 获取该对象的类别标签并映射为掩码值
        class_label = int(classes[j])  # 获取类别标签
        mask_value = class_to_mask_value.get(class_label, 0)  # 根据类别映射值

        # 将掩码大小调整为与原图像大小一致
        mask_resized = cv2.resize(mask_img, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

        # 将掩码值应用到掩码区域
        mask_resized = mask_resized * mask_value  # 应用类别对应的掩码值

        # 将该对象的掩码叠加到总掩码图中
        mask_combined = np.maximum(mask_combined, mask_resized)

    # 保存合并后的掩码图像，命名为 "image_{i}_mask_combined.png"
    mask_filename = os.path.join(output_folder, f'image_{i+1}_mask_combined.png')
    cv2.imwrite(mask_filename, mask_combined)
    print(f"Saved combined mask: {mask_filename}")
