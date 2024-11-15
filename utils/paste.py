import cv2
import numpy as np
import random
import os

def blend_images(background_img, foreground_img, background_mask, foreground_mask):
    # 获取背景和前景图像的尺寸
    bg_h, bg_w, _ = background_img.shape
    fg_h, fg_w, _ = foreground_img.shape
    
    # 随机设定正方形的大小，确保大小适中
    square_size = random.randint(50, min(fg_h, fg_w) // 2)
    
    # 随机选取正方形的左上角坐标
    x1 = random.randint(0, fg_w - square_size)
    y1 = random.randint(0, fg_h - square_size)
    x2 = x1 + square_size
    y2 = y1 + square_size

    # 随机选择延伸小矩形的尺寸
    small_rect_size = random.randint(square_size // 4, square_size // 2)
    
    # 随机选择延伸方向（上、下、左、右）
    direction = random.choice(['top', 'bottom', 'left', 'right'])
    if direction == 'top':
        x1_ext, y1_ext = x1 + square_size // 4, max(0, y1 - small_rect_size)
        x2_ext, y2_ext = x1_ext + small_rect_size, y1
    elif direction == 'bottom':
        x1_ext, y1_ext = x1 + square_size // 4, y2
        x2_ext, y2_ext = x1_ext + small_rect_size, min(fg_h, y2 + small_rect_size)
    elif direction == 'left':
        x1_ext, y1_ext = max(0, x1 - small_rect_size), y1 + square_size // 4
        x2_ext, y2_ext = x1, y1_ext + small_rect_size
    elif direction == 'right':
        x1_ext, y1_ext = x2, y1 + square_size // 4
        x2_ext, y2_ext = min(fg_w, x2 + small_rect_size), y1_ext + small_rect_size

    # 裁剪前景图片和掩码区域，包括正方形和延伸矩形
    crop_img = foreground_img[y1:y2, x1:x2].copy()
    crop_ext_img = foreground_img[y1_ext:y2_ext, x1_ext:x2_ext].copy()
    
    crop_mask = foreground_mask[y1:y2, x1:x2].copy()
    crop_ext_mask = foreground_mask[y1_ext:y2_ext, x1_ext:x2_ext].copy()
    
    # 确保粘贴区域不会超出背景边界
    max_paste_x = bg_w - square_size - small_rect_size
    max_paste_y = bg_h - square_size - small_rect_size
    
    paste_x = random.randint(0, max_paste_x)
    paste_y = random.randint(0, max_paste_y)

    # 将裁剪的前景图像区域粘贴到背景图像上
    if((paste_y + (y1_ext - y1)) < 0):
        return None, None
    if((paste_x + (x1_ext - x1)) < 0):
        return None, None
    blended_image = background_img.copy()
    blended_image[paste_y:paste_y + square_size, paste_x:paste_x + square_size] = crop_img
    blended_image[paste_y + (y1_ext - y1):paste_y + (y1_ext - y1) + crop_ext_img.shape[0], 
                  paste_x + (x1_ext - x1):paste_x + (x1_ext - x1) + crop_ext_img.shape[1]] = crop_ext_img

    # 处理标签，将裁剪的掩码图粘贴到背景标签上
    blended_mask = background_mask.copy()
    blended_mask[paste_y:paste_y + square_size, paste_x:paste_x + square_size] = crop_mask
    blended_mask[paste_y + (y1_ext - y1):paste_y + (y1_ext - y1) + crop_ext_mask.shape[0], 
                 paste_x + (x1_ext - x1):paste_x + (x1_ext - x1) + crop_ext_mask.shape[1]] = crop_ext_mask

    return blended_image, blended_mask


def process_dataset(dataset_path,output_path,num):
    image_dir = os.path.join(dataset_path, 'image')
    label_dir = os.path.join(dataset_path, 'label')
    
    augmented_image_dir = os.path.join(dataset_path, f'{output_path}/image')
    augmented_label_dir = os.path.join(dataset_path, f'{output_path}/label')
    
    os.makedirs(augmented_image_dir, exist_ok=True)
    os.makedirs(augmented_label_dir, exist_ok=True)
    
    # 获取所有图像文件名并打乱
    image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random.shuffle(image_names)
    
    # 处理图像对，确保每对图片只处理一次
    i = 1
    while i <= num:
        # 从列表中随机选择背景图和前景图，确保两者不同
        background_name = image_names.pop(0)  # 取出第一个作为背景图
        foreground_name = random.choice(image_names)  # 随机选择一个前景图
        image_names.remove(foreground_name)  # 移除选中的前景图

        # 获取背景图和前景图路径
        background_path = os.path.join(image_dir, background_name)
        foreground_path = os.path.join(image_dir, foreground_name)
        
        # 获取对应的标签路径
        background_mask_path = os.path.join(label_dir, background_name.replace('.jpg', '.png'))
        foreground_mask_path = os.path.join(label_dir, foreground_name.replace('.jpg', '.png'))
        
        # 读取背景图和前景图及其标签
        background_img = cv2.imread(background_path)
        foreground_img = cv2.imread(foreground_path)
        
        background_mask = cv2.imread(background_mask_path, 0)
        foreground_mask = cv2.imread(foreground_mask_path, 0)
        
        # 执行数据增强（裁剪和粘贴）
        augmented_image, augmented_mask = blend_images(background_img, foreground_img, background_mask, foreground_mask)
        if(augmented_image is None):
            continue
        
        # 保存增强后的图像和标签
        augmented_image_path = os.path.join(augmented_image_dir, f"paset_{i}.jpg")
        augmented_mask_path = os.path.join(augmented_label_dir, f"paset_{i}.png")
        
        cv2.imwrite(augmented_image_path, augmented_image)
        cv2.imwrite(augmented_mask_path, augmented_mask)
        
        print(f"Saved augmented image to {augmented_image_path} and label to {augmented_mask_path}")
        
        i += 1  # 增加计数器

# 设置数据集路径
# 粘贴500*2=1000张 共计1000张
paste_num = 1000
dataset_path = '/root/autodl-tmp/standard_project/datasets/datasets_A/train'    # 这个不要动
output_path = '/root/autodl-tmp/standard_project/datasets/double_origin_paste_slice_blend_noise/train'
process_dataset(dataset_path,output_path,paste_num)
