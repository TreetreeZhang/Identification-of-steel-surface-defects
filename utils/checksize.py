import os
from PIL import Image

def check_image_size(folder_path, target_size=(200, 200)):
    error_file = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with Image.open(file_path) as img:
                if img.size != target_size:
                    error_file.append((file_name, img.size))

    for file_name,img_size in error_file:
        print(f"尺寸不符: {file_name} -> {img_size}")
    
    if(len(error_file) == 0):
        print("所有图片尺寸正确！")

train_image_folder = "/root/autodl-tmp/standard_project/datasets/datasets_AB_blend_with_noise/train/image"
train_label_folder = "/root/autodl-tmp/standard_project/datasets/datasets_AB_blend_with_noise/train/label"

check_image_size(train_image_folder)
check_image_size(train_label_folder)
