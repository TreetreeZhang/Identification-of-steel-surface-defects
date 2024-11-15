import os
from PIL import Image
import numpy as np

# 颜色映射：将标签的不同值映射到不同颜色
label_to_color = {
    1: (255, 0, 0),    # 缺陷类型1（红色）
    2: (0, 255, 0),    # 缺陷类型2（绿色）
    3: (0, 0, 255)     # 缺陷类型3（蓝色）
}

def apply_mask_to_image(image_path, mask_path, output_path):
    # 读取原始图片（JPG）和掩码图（PNG）
    image = Image.open(image_path).convert("RGB")  # 将图片转换为RGB模式
    mask = Image.open(mask_path).convert("L")  # 将掩码图转换为灰度模式（L模式）

    # 转换为NumPy数组
    mask_array = np.array(mask)
    image_array = np.array(image)

    # 获取掩码图的宽高
    height, width = mask_array.shape

    # 对掩码图像素进行处理：如果像素值为1、2或3，则替换为对应颜色
    for i in range(height):
        for j in range(width):
            label_value = mask_array[i, j]
            if label_value in label_to_color:
                # 获取颜色
                color = label_to_color[label_value]
                # 用颜色叠加到原图（覆盖掩码部分）
                image_array[i, j] = np.array(color)

    # 将修改后的图片转换回PIL图像并保存
    result_image = Image.fromarray(image_array)
    result_image.save(output_path, format="JPEG")  # 保存为JPEG格式

def process_images(image_folder, mask_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, file_name)
        mask_path = os.path.join(mask_folder, file_name.replace(".jpg", ".png"))  # 假设掩码图的文件名与图像文件名相同，只是扩展名不同

        # 生成输出路径
        output_path = os.path.join(output_folder, file_name)

        # 如果图片和掩码图都存在，进行处理
        if os.path.exists(mask_path):
            apply_mask_to_image(image_path, mask_path, output_path)
            print(f"处理并保存：{file_name}")
        else:
            print(f"缺少掩码图：{file_name}")


# 文件夹路径
image_folder = "/root/autodl-tmp/standard_project/datasets/datasets_A_slice_paste_blend_fuse/train1/image"
mask_folder = "/root/autodl-tmp/standard_project/datasets/datasets_A_slice_paste_blend_fuse/train1/label"
output_folder = "/root/autodl-tmp/standard_project/datasets/datasets_A_slice_paste_blend_fuse/output"

# 处理并保存结果
process_images(image_folder, mask_folder, output_folder)
