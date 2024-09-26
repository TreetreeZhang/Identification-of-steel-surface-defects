# Description: 检查图像和掩码图的大小是否一致

from PIL import Image
import os

def check_image_mask_size(image_folder, mask_folder):
    for img_file in os.listdir(image_folder):
        if img_file.endswith('.jpg'):  # 假设原图为jpg格式
            img_path = os.path.join(image_folder, img_file)
            mask_path = os.path.join(mask_folder, img_file.replace('.jpg', '.png'))

            if os.path.exists(mask_path):
                img = Image.open(img_path)
                mask = Image.open(mask_path)

                if img.size != mask.size:
                    print(f"Size mismatch: {img_file} and its mask")
                else:
                    print(f"{img_file} and its mask are the same size")
            else:
                print(f"No mask found for {img_file}")

# 替换为你的图像和掩码图文件夹路径
image_folder = 'images\\training'
mask_folder = 'annotations\\training'
check_image_mask_size(image_folder, mask_folder)
