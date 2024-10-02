# Function: 检查标签图中是否包含无效类别

from PIL import Image
import numpy as np
import os

def check_mask_classes(mask_folder, num_classes):
    for mask_file in os.listdir(mask_folder):
        if mask_file.endswith('.png'):
            mask_path = os.path.join(mask_folder, mask_file)
            mask = np.array(Image.open(mask_path))
            unique_values = np.unique(mask)
            if any(v >= num_classes for v in unique_values):
                print(f"Error: {mask_file} contains invalid class {unique_values[unique_values >= num_classes]}")

mask_folder = 'annotations\\training'  # 包含 .png 掩码图的文件夹
check_mask_classes(mask_folder, 4)  # 4个类别：0到3
